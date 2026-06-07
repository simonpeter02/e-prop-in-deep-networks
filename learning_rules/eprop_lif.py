"""
E-prop and d=0 gradient computation for a single-layer LIF network.

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
from models.lif_rnn import LIFNetwork
from typing import Dict


def compute_eprop_lif_gradients(
    model: LIFNetwork,
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
        # carry[b,i] = alpha - v_th * psi_{t-1}[b,i]   (e-prop)
        #            = 0                                 (d=0)
        if d_zero:
            carry = torch.zeros(B, n, device=dev)
        else:
            carry = alpha - v_th * psi_prev    # (B, n); can dip below alpha

        # P_t[b,i,j] = carry[b,i] * P_{t-1}[b,i,j] + (1-alpha) * presynaptic[b,j]
        P_rec = carry.unsqueeze(2) * P_rec + (1 - alpha) * s.unsqueeze(1)
        P_in  = carry.unsqueeze(2) * P_in  + (1 - alpha) * x_t.unsqueeze(1)
        P_b   = carry * P_b + (1 - alpha)

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
