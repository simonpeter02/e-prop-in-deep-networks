"""
Per-layer e-prop and d=0 gradient computation for a deep ALIF network.

ALIF adds a slow adaptation variable a_t that raises the spike threshold,
giving e-prop a second eligibility trace (slow, decay rho) on top of the
fast membrane trace (decay alpha - v_th*psi).  d=0 drops BOTH traces,
creating a large e-prop > d=0 gap at delays beyond ~8 steps — unlike
plain LIF where e-prop ≈ d=0 because the single trace decays too fast.

Two-trace system per layer l
-----------------------------
  psi_l_t   = gamma * clamp(1 - |u_l_t - theta_l_t| / v_th, 0)   [narrow surrogate]
  c_u       = alpha - v_th * psi_l_{t-1}                           [fast carry, e-prop]
            = 0                                                      [d=0]

  # Update slow trace BEFORE fast trace (uses old eps_u — matches eprop_lif.py)
  eps_a[l] = rho * eps_a[l]  +  psi_l_{t-1} * eps_u[l]

  # Update fast trace
  eps_u_rec[l] = (1-alpha) * s_l_{t-1}   +  c_u * eps_u_rec[l]
  eps_u_ff[l]  = (1-alpha) * s_{l-1,t}   +  c_u * eps_u_ff[l]   [l>=1, current-step presynaptic]
  eps_u_in     = (1-alpha) * x_t         +  c_u * eps_u_in
  eps_u_b[l]   = (1-alpha)               +  c_u * eps_u_b[l]

  # Combined eligibility
  eps_rec[l] = psi_l_t * (eps_u_rec[l] - beta * eps_a_rec[l])
  ...

  d=0: skip slow trace updates (eps_a stays 0); set c_u=0 in fast trace.

Learning signal propagation (instantaneous, same as deep_eprop_lif.py):
  delta_{L-1} = err_out @ W_out
  delta_{l-1} = (1-alpha) * (delta_l * psi_l_t) @ W_ff_l

Output dict keys match model.named_parameters() exactly:
  'W_in', 'W_recs.0', 'W_recs.1', ..., 'W_ffs.0', ..., 'b_recs.0', ..., 'W_out', 'b_out'
"""

import torch
from torch import Tensor
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from models.deep_alif import DeepALIFNetwork


def compute_deep_eprop_alif_gradients(
    model: "DeepALIFNetwork",
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
    d_zero: bool = False,
) -> Dict[str, Tensor]:
    """
    Compute per-layer e-prop (or d=0) gradients for a DeepALIFNetwork.

    d_zero=True : set carry=0 in fast trace and skip all slow trace updates.
    """
    L     = model.n_layers
    T, B, n_in = inputs.shape
    n     = model.n_rec
    n_out = model.n_out
    dev   = inputs.device

    alpha = model.alpha
    rho   = model.rho
    beta  = model.beta
    v_th  = model.v_th
    gamma = model.gamma
    oma   = 1 - alpha

    W_recs_ = [model.W_recs[l].detach() for l in range(L)]
    W_ffs_  = [model.W_ffs[l].detach() for l in range(L - 1)]
    W_in_   = model.W_in.detach()
    W_out_  = model.W_out.detach()
    b_recs_ = [model.b_recs[l].detach() for l in range(L)]

    # ── Fast traces (eps_u): carry = alpha - v_th*psi_prev (e-prop) or 0 (d=0) ──
    eu_rec = [torch.zeros(B, n, n,    device=dev) for _ in range(L)]
    eu_ff  = [None] + [torch.zeros(B, n, n, device=dev) for _ in range(1, L)]
    eu_in  = torch.zeros(B, n, n_in, device=dev)
    eu_b   = [torch.zeros(B, n,       device=dev) for _ in range(L)]

    # ── Slow traces (eps_a): carry = rho; skipped entirely for d=0 ───────────────
    ea_rec = [torch.zeros(B, n, n,    device=dev) for _ in range(L)]
    ea_ff  = [None] + [torch.zeros(B, n, n, device=dev) for _ in range(1, L)]
    ea_in  = torch.zeros(B, n, n_in, device=dev)
    ea_b   = [torch.zeros(B, n,       device=dev) for _ in range(L)]

    # ── Gradient accumulators ──────────────────────────────────────────────────────
    grad_W_recs = [torch.zeros(n, n,    device=dev) for _ in range(L)]
    grad_W_ffs  = [torch.zeros(n, n,    device=dev) for _ in range(L - 1)]
    grad_W_in   = torch.zeros(n, n_in,  device=dev)
    grad_b_recs = [torch.zeros(n,        device=dev) for _ in range(L)]
    grad_W_out  = torch.zeros(n_out, n,  device=dev)
    grad_b_out  = torch.zeros(n_out,     device=dev)

    # ── Running state ──────────────────────────────────────────────────────────────
    u        = [torch.zeros(B, n, device=dev) for _ in range(L)]
    s        = [torch.zeros(B, n, device=dev) for _ in range(L)]
    a        = [torch.zeros(B, n, device=dev) for _ in range(L)]
    psi_prev = [torch.zeros(B, n, device=dev) for _ in range(L)]

    for t in range(T):
        x_t   = inputs[t]
        s_new = []
        psi_t = []
        u_new = []
        a_new = []

        # ── Manual forward pass (no autograd) ─────────────────────────────────────
        with torch.no_grad():
            for l in range(L):
                theta = v_th + beta * a[l]

                if l == 0:
                    ff = x_t @ W_in_.T
                else:
                    ff = s_new[l - 1] @ W_ffs_[l - 1].T

                u_l   = (alpha * u[l]
                         + oma * (s[l] @ W_recs_[l].T + ff + b_recs_[l])
                         - theta * s[l])
                a_l   = rho * a[l] + s[l]
                psi_l = gamma * torch.clamp(1.0 - (u_l - theta).abs() / v_th, min=0.0)
                s_l   = (u_l >= theta).float()

                u_new.append(u_l)
                a_new.append(a_l)
                psi_t.append(psi_l)
                s_new.append(s_l)

            o = s_new[-1] @ W_out_.T + model.b_out

        # ── Update traces ──────────────────────────────────────────────────────────
        for l in range(L):
            if d_zero:
                # Instantaneous only: no carry in fast trace, slow trace stays zero
                eu_rec[l] = oma * s[l].unsqueeze(1)
                eu_b[l]   = torch.full((B, n), oma, device=dev)
                if l == 0:
                    eu_in = oma * x_t.unsqueeze(1)
                else:
                    eu_ff[l] = oma * s_new[l - 1].unsqueeze(1)
            else:
                c_u = alpha - v_th * psi_prev[l]   # fast carry (B, n)

                # Slow trace update (uses OLD fast trace before this step)
                ea_rec[l] = rho * ea_rec[l] + psi_prev[l].unsqueeze(2) * eu_rec[l]
                ea_b[l]   = rho * ea_b[l]   + psi_prev[l] * eu_b[l]
                if l == 0:
                    ea_in = rho * ea_in + psi_prev[0].unsqueeze(2) * eu_in
                else:
                    ea_ff[l] = rho * ea_ff[l] + psi_prev[l].unsqueeze(2) * eu_ff[l]

                # Fast trace update
                eu_rec[l] = oma * s[l].unsqueeze(1)          + c_u.unsqueeze(2) * eu_rec[l]
                eu_b[l]   = oma                               + c_u * eu_b[l]
                if l == 0:
                    eu_in = oma * x_t.unsqueeze(1)            + c_u.unsqueeze(2) * eu_in
                else:
                    # W_ff presynaptic: current-step spikes from layer below
                    eu_ff[l] = oma * s_new[l - 1].unsqueeze(1) + c_u.unsqueeze(2) * eu_ff[l]

        # ── Gradient accumulation (masked timesteps only) ──────────────────────────
        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t]) * mask[t].unsqueeze(-1)   # (B, n_out)

            grad_W_out += (err_out.T @ s_new[-1]).detach() / B
            grad_b_out += err_out.mean(0).detach()

            # Combined eligibility and per-layer learning signals
            deltas = [None] * L
            deltas[L - 1] = err_out @ W_out_
            for l in range(L - 1, 0, -1):
                deltas[l - 1] = oma * (deltas[l] * psi_t[l]) @ W_ffs_[l - 1]

            for l in range(L):
                # Combined eligibility: psi * (eps_u - beta * eps_a)
                eps_rec = psi_t[l].unsqueeze(2) * (eu_rec[l] - beta * ea_rec[l])
                eps_b   = psi_t[l]              * (eu_b[l]   - beta * ea_b[l])

                grad_W_recs[l] += torch.einsum('bi,bij->ij', deltas[l], eps_rec) / B
                grad_b_recs[l] += (deltas[l] * eps_b).mean(0)

                if l == 0:
                    eps_in = psi_t[0].unsqueeze(2) * (eu_in - beta * ea_in)
                    grad_W_in += torch.einsum('bi,bij->ij', deltas[0], eps_in) / B
                else:
                    eps_ff = psi_t[l].unsqueeze(2) * (eu_ff[l] - beta * ea_ff[l])
                    grad_W_ffs[l - 1] += torch.einsum('bi,bij->ij', deltas[l], eps_ff) / B

        psi_prev = psi_t
        u = u_new
        s = s_new
        a = a_new

    result: Dict[str, Tensor] = {
        'W_in':  grad_W_in,
        'W_out': grad_W_out,
        'b_out': grad_b_out,
    }
    for l in range(L):
        result[f'W_recs.{l}'] = grad_W_recs[l]
        result[f'b_recs.{l}'] = grad_b_recs[l]
    for l in range(1, L):
        result[f'W_ffs.{l - 1}'] = grad_W_ffs[l - 1]
    return result


def xent_error(o: Tensor, target: Tensor) -> Tensor:
    """Per-sample cross-entropy error signal: softmax(o) - one_hot_target."""
    return torch.softmax(o, dim=-1) - target
