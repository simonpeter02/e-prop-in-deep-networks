"""
E-prop and d=0 gradient computation for single-layer LIF and ALIF networks.

Eligibility trace derivation (Bellec et al. 2020, Supplementary)
-----------------------------------------------------------------
The RTRL membrane trace P_t[i,j] = d(u_t[i]) / d(W[i,j]) satisfies:

  P_t[i,j] = (alpha - v_th * psi_{t-1}[i]) * P_{t-1}[i,j]    ← temporal carry
            + (1-alpha) * presynaptic_{t-1}[j]                 ← instantaneous drive

where psi_{t-1}[i] = gamma * max(0, 1 - |u_{t-1}[i] - v_th| / v_th)  is the
surrogate pseudoderivative at the PREVIOUS step (from the spike-reset term).

The eligibility trace is:
  epsilon_t[i,j] = psi_t[i] * P_t[i,j]

Gradient:
  dL/dW[i,j] ≈ (1/B) * sum_t  sum_b  delta_t[b,i] * epsilon_t[b,i,j]

d=0 variant: set carry to zero (no temporal propagation).

Key difference from tanh e-prop
--------------------------------
  tanh: carry = psi_t * W_rec[i,i]   ≈ 0.07 * 0.07 ≈ 0.005  (negligible)
  LIF:  carry = alpha - v_th * psi   ≈ 0.9  - 0.3  ≈ 0.6    (substantial)

For LIF, e-prop can propagate credit ~15 steps (1 / (1-0.6) time constants),
so e-prop >> d=0 on tasks requiring temporal credit — unlike tanh where e-prop ≈ d=0.
"""

import torch
from torch import Tensor
from typing import Dict, TYPE_CHECKING


def compute_eprop_lif_gradients(
    model,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
    d_zero: bool = False,
) -> Dict[str, Tensor]:
    """
    Compute e-prop (or d=0) gradients for a single-layer LIF network.

    d_zero=True : set temporal carry to zero; retain only the instantaneous term.
    """
    T, B, n_in = inputs.shape
    n     = model.n_rec
    n_out = model.n_out
    dev   = inputs.device

    alpha = model.alpha
    v_th  = model.v_th
    gamma = model.gamma

    W_rec_ = model.W_rec.detach()
    W_in_  = model.W_in.detach()
    W_out_ = model.W_out.detach()

    # Membrane traces  P_t[b, i, j]  (one per parameter group)
    P_rec = torch.zeros(B, n, n,    device=dev)   # for W_rec[i,j]: presynaptic = s[j]
    P_in  = torch.zeros(B, n, n_in, device=dev)   # for W_in[i,k]:  presynaptic = x[k]
    P_b   = torch.zeros(B, n,       device=dev)   # for b[i]:        presynaptic = 1

    # Gradient accumulators
    grad_W_rec = torch.zeros(n, n,    device=dev)
    grad_W_in  = torch.zeros(n, n_in, device=dev)
    grad_b_rec = torch.zeros(n,       device=dev)
    grad_W_out = torch.zeros(n_out, n, device=dev)
    grad_b_out = torch.zeros(n_out,    device=dev)

    u        = torch.zeros(B, n, device=dev)
    s        = torch.zeros(B, n, device=dev)
    psi_prev = torch.zeros(B, n, device=dev)

    # alpha may be a scalar float (uniform LIF) or a (n,) tensor (hetero LIF).
    # Normalise to a tensor so broadcasting is unambiguous in the trace updates.
    alpha_t = (alpha if isinstance(alpha, torch.Tensor)
               else torch.tensor(alpha, device=dev))
    # oma_col: (1-alpha) shaped for the "output neuron i" axis in (B, n, n/n_in)
    # For scalar: shape () broadcasts fine; for vector (n,): reshape to (1, n, 1).
    if alpha_t.dim() > 0:
        oma_col = (1 - alpha_t).view(1, -1, 1)   # (1, n, 1)
        oma_vec = (1 - alpha_t)                   # (n,)  for P_b
    else:
        oma_col = (1 - alpha_t)                   # scalar
        oma_vec = (1 - alpha_t)                   # scalar

    for t in range(T):
        x_t = inputs[t]

        # ── Forward (no autograd) ─────────────────────────────────────────────
        with torch.no_grad():
            u_new = (alpha * u
                     + (1 - alpha) * (s @ W_rec_.T + x_t @ W_in_.T + model.b_rec)
                     - v_th * s)
            psi   = gamma * torch.clamp(1.0 - (u_new - v_th).abs() / v_th, min=0.0)
            s_new = (u_new >= v_th).float()
            o     = s_new @ W_out_.T + model.b_out

        # ── Update membrane traces ────────────────────────────────────────────
        # carry[b,i] = alpha_i - v_th * psi_{t-1}[b,i]   (e-prop)
        #            = 0                                   (d=0)
        if d_zero:
            carry = torch.zeros(B, n, device=dev)
        else:
            carry = alpha - v_th * psi_prev    # (B, n) for scalar α; broadcasts for vector

        # P_t[b,i,j] = carry[b,i] * P_{t-1}[b,i,j] + (1-alpha_i) * presynaptic[b,j]
        # oma_col is (1,n,1) for hetero or scalar for uniform — both broadcast to (B,n,n)
        P_rec = carry.unsqueeze(2) * P_rec + oma_col * s[:, None, :]
        P_in  = carry.unsqueeze(2) * P_in  + oma_col * x_t[:, None, :]
        P_b   = carry * P_b + oma_vec

        # ── Gradient accumulation (only at masked timesteps) ──────────────────
        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t]) * mask[t].unsqueeze(-1)

            grad_W_out += (err_out.T @ s_new).detach() / B
            grad_b_out += err_out.mean(0).detach()

            delta = err_out @ W_out_              # (B, n)  learning signal at hidden layer

            # epsilon_t[b,i,j] = psi_t[b,i] * P_t[b,i,j]
            eps_rec = psi.unsqueeze(2) * P_rec    # (B, n, n)
            eps_in  = psi.unsqueeze(2) * P_in     # (B, n, n_in)
            eps_b   = psi * P_b                   # (B, n)

            grad_W_rec += torch.einsum('bi,bij->ij', delta, eps_rec) / B
            grad_W_in  += torch.einsum('bi,bij->ij', delta, eps_in)  / B
            grad_b_rec += (delta * eps_b).mean(0)

        psi_prev = psi
        u, s = u_new, s_new

    return {
        'W_rec': grad_W_rec,
        'W_in':  grad_W_in,
        'b_rec': grad_b_rec,
        'W_out': grad_W_out,
        'b_out': grad_b_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ALIF e-prop
# ─────────────────────────────────────────────────────────────────────────────

def compute_eprop_alif_gradients(
    model,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
    d_zero: bool = False,
) -> Dict[str, Tensor]:
    """
    E-prop (or d=0) gradients for a single-layer ALIFNetwork.

    ALIF adds a slow adaptation variable:
        a_t   = rho * a_{t-1} + s_{t-1}
        θ_t   = v_th + beta * a_{t-1}
        s_t   = H(u_t - θ_t)

    The eligibility trace splits into two components (Bellec et al. 2020, S27-S30):

      Fast trace (same as LIF, tracks u_t dependence on W[i,j]):
        ε_u_t[i,j] = (1-α) * s_{t-1}[j]
                     + (α - v_th * ψ_{t-1}[i]) * ε_u_{t-1}[i,j]

      Slow trace (tracks a_t dependence on W[i,j] through past spikes):
        ε_a_t[i,j] = ρ * ε_a_{t-1}[i,j]  +  ψ_{t-1}[i] * ε_u_{t-1}[i,j]
        (uses the truncated approximation from Bellec 2020 that ignores
         the adaptation feedback in ∂s_{t-1}/∂W, keeping the update causal)

      Combined:
        ε_t[i,j] = ψ_t[i] * (ε_u_t[i,j] − β * ε_a_t[i,j])

    d=0: drop both carries and the slow trace entirely.
         ε_t[i,j] = ψ_t[i] * (1-α) * s_{t-1}[j]

    The gap vs d=0 grows with delay because ε_a survives ρ^k steps — far
    beyond the fast LIF carry (α - v_th*ψ ≈ 0.6), so ALIF e-prop >> d=0
    at delays > ~3 steps even when LIF e-prop ≈ d=0.
    """
    T, B, n_in = inputs.shape
    n     = model.n_rec
    n_out = model.n_out
    dev   = inputs.device

    alpha = model.alpha
    rho   = model.rho
    beta  = model.beta
    v_th  = model.v_th
    gamma = model.gamma

    W_rec_ = model.W_rec.detach()
    W_in_  = model.W_in.detach()
    W_out_ = model.W_out.detach()

    # Fast membrane traces  ε_u[b, i, j]
    eps_u_rec = torch.zeros(B, n, n,    device=dev)
    eps_u_in  = torch.zeros(B, n, n_in, device=dev)
    eps_u_b   = torch.zeros(B, n,       device=dev)

    # Slow adaptation traces  ε_a[b, i, j]  (only matter when d_zero=False)
    eps_a_rec = torch.zeros(B, n, n,    device=dev)
    eps_a_in  = torch.zeros(B, n, n_in, device=dev)
    eps_a_b   = torch.zeros(B, n,       device=dev)

    # Gradient accumulators
    grad_W_rec = torch.zeros(n, n,    device=dev)
    grad_W_in  = torch.zeros(n, n_in, device=dev)
    grad_b_rec = torch.zeros(n,       device=dev)
    grad_W_out = torch.zeros(n_out, n, device=dev)
    grad_b_out = torch.zeros(n_out,    device=dev)

    u        = torch.zeros(B, n, device=dev)
    s        = torch.zeros(B, n, device=dev)
    a        = torch.zeros(B, n, device=dev)
    psi_prev = torch.zeros(B, n, device=dev)

    for t in range(T):
        x_t   = inputs[t]
        theta = v_th + beta * a                              # (B, n) adaptive threshold

        with torch.no_grad():
            u_new = (alpha * u
                     + (1 - alpha) * (s @ W_rec_.T + x_t @ W_in_.T + model.b_rec)
                     - theta * s)
            a_new = rho * a + s
            psi   = gamma * torch.clamp(1.0 - (u_new - theta).abs() / v_th, min=0.0)
            s_new = (u_new >= theta).float()
            o     = s_new @ W_out_.T + model.b_out

        oma = 1 - alpha  # scalar shorthand

        if d_zero:
            # Instantaneous only: ε = ψ_t * (1-α) * presynaptic
            eps_u_rec = oma * s[:, None, :]
            eps_u_in  = oma * x_t[:, None, :]
            eps_u_b   = torch.full((B, n), oma, device=dev)
            # slow traces left at zero; eps_a stays zero
        else:
            # Fast carry: c_u = α - v_th * ψ_{t-1}
            c_u = alpha - v_th * psi_prev                    # (B, n)

            # Update slow trace BEFORE updating fast trace (uses old eps_u)
            eps_a_rec = rho * eps_a_rec + psi_prev.unsqueeze(2) * eps_u_rec
            eps_a_in  = rho * eps_a_in  + psi_prev.unsqueeze(2) * eps_u_in
            eps_a_b   = rho * eps_a_b   + psi_prev * eps_u_b

            # Update fast trace
            eps_u_rec = oma * s[:, None, :] + c_u.unsqueeze(2) * eps_u_rec
            eps_u_in  = oma * x_t[:, None, :] + c_u.unsqueeze(2) * eps_u_in
            eps_u_b   = oma + c_u * eps_u_b

        # Combined eligibility trace: ε = ψ_t * (ε_u − β * ε_a)
        eps_rec = psi.unsqueeze(2) * (eps_u_rec - beta * eps_a_rec)
        eps_in  = psi.unsqueeze(2) * (eps_u_in  - beta * eps_a_in)
        eps_b   = psi              * (eps_u_b   - beta * eps_a_b)

        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t]) * mask[t].unsqueeze(-1)

            grad_W_out += (err_out.T @ s_new).detach() / B
            grad_b_out += err_out.mean(0).detach()

            delta = err_out @ W_out_                         # (B, n)

            grad_W_rec += torch.einsum('bi,bij->ij', delta, eps_rec) / B
            grad_W_in  += torch.einsum('bi,bij->ij', delta, eps_in)  / B
            grad_b_rec += (delta * eps_b).mean(0)

        psi_prev = psi
        u, s, a = u_new, s_new, a_new

    return {
        'W_rec': grad_W_rec,
        'W_in':  grad_W_in,
        'b_rec': grad_b_rec,
        'W_out': grad_W_out,
        'b_out': grad_b_out,
    }
