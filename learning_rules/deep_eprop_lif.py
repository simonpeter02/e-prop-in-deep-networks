"""
Per-layer e-prop and d=0 gradient computation for a deep LIF network.

Approximation used
------------------
This is the *per-layer* variant of deep e-prop:
  - Each layer maintains its own membrane traces (self-traces), which carry
    temporal credit with the LIF decay carry ≈ alpha − v_th·ψ ≈ 0.6 per step.
  - The learning signal is propagated instantaneously (same timestep) through
    the feedforward Jacobian (1−alpha)·ψ^l·W_ff^l at each masked timestep.
  - There are no 4-D cross-layer temporal traces (those scale as B·n³ ≈ 2 GB
    for n_rec=256, used in deep_eprop.py for tanh at n=50).

This gives:
  d=0   : carry = 0  →  no temporal credit, spatial credit only
  e-prop: carry ≈ 0.6 →  ~15 step temporal horizon, spatial credit
  BPTT  : full temporal + spatial credit via autograd

The d=0 vs e-prop difference is PURELY temporal: the feedforward learning-signal
propagation is identical in both cases, so the gap is solely due to the eligibility
trace carrying history.

Membrane trace recursion (per-layer)
-------------------------------------
  carry^l_t = alpha − v_th · ψ^l_{t−1}     (e-prop)
            = 0                               (d=0)

  P_rec^l_t[b,i,j] = carry^l_t[b,i] · P_rec^l_{t−1}[b,i,j] + (1−alpha) · s^l_{t−1}[b,j]
  P_ff^l_t [b,i,j] = carry^l_t[b,i] · P_ff^l_{t−1} [b,i,j] + (1−alpha) · s^{l−1}_t[b,j]  [l≥1]
  P_in_t   [b,i,j] = carry^0_t[b,i] · P_in_{t−1}   [b,i,j] + (1−alpha) · x_t[b,j]
  P_b^l_t  [b,i]   = carry^l_t[b,i] · P_b^l_{t−1}  [b,i]   + (1−alpha)

Note: W_ff receives the *current-step* presynaptic (s^{l−1}_t) because the
feedforward drive is u^l_t ∝ W_ff^l @ s^{l−1}_t (same timestep).

Gradient accumulation (at masked timesteps)
-------------------------------------------
  Learning signal at top layer:
    delta^{L−1} = err_out @ W_out                                 (B, n)

  Propagated downward (instantaneous, no temporal carry):
    delta^{l−1} = (1−alpha) · (delta^l * ψ^l) @ W_ff^l           (B, n)

  Eligibility traces:
    eps_rec^l = ψ^l · P_rec^l       (B, n, n)
    eps_ff^l  = ψ^l · P_ff^l        (B, n, n)   [l≥1]
    eps_in    = ψ^0 · P_in          (B, n, n_in)
    eps_b^l   = ψ^l · P_b^l        (B, n)

  Gradient updates:
    grad_W_rec^l += einsum('bi,bij->ij', delta^l, eps_rec^l) / B
    grad_W_ff^l  += einsum('bi,bij->ij', delta^l, eps_ff^l)  / B  [l≥1]
    grad_W_in    += einsum('bi,bij->ij', delta^0, eps_in)    / B
    grad_b_recs^l += (delta^l * eps_b^l).mean(0)
    grad_W_out   += (err_out.T @ s^{L−1}) / B
    grad_b_out   += err_out.mean(0)

Output dict keys match model.named_parameters() exactly:
  'W_in', 'W_recs.0', 'W_recs.1', ..., 'W_ffs.0', ..., 'b_recs.0', ..., 'W_out', 'b_out'
"""

import torch
from torch import Tensor
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from models.deep_lif import DeepLIFNetwork


def compute_deep_eprop_lif_gradients(
    model: "DeepLIFNetwork",
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
    d_zero: bool = False,
) -> Dict[str, Tensor]:
    """
    Compute per-layer e-prop (or d=0) gradients for a DeepLIFNetwork.

    d_zero=True : set carry = 0 in all membrane traces.
    """
    L     = model.n_layers
    T, B, n_in = inputs.shape
    n     = model.n_rec
    n_out = model.n_out
    dev   = inputs.device

    alpha = model.alpha
    v_th  = model.v_th
    gamma = model.gamma
    oma   = 1 - alpha   # (1 - alpha), scalar

    # Detach all weights for manual forward pass
    W_recs_ = [model.W_recs[l].detach() for l in range(L)]
    W_ffs_  = [model.W_ffs[l].detach() for l in range(L - 1)]  # index by l-1, used for layer l≥1
    W_in_   = model.W_in.detach()
    W_out_  = model.W_out.detach()
    b_recs_ = [model.b_recs[l].detach() for l in range(L)]

    # ── Membrane traces (self-traces per layer) ───────────────────────────────
    # P_rec^l : (B, n, n)     presynaptic = s^l_{t-1}
    # P_ff^l  : (B, n, n)     presynaptic = s^{l-1}_t  [l≥1]
    # P_in    : (B, n, n_in)  presynaptic = x_t         [layer 0]
    # P_b^l   : (B, n)        presynaptic = 1
    P_rec = [torch.zeros(B, n, n,    device=dev) for _ in range(L)]
    P_ff  = [None] + [torch.zeros(B, n, n, device=dev) for _ in range(1, L)]
    P_in  = torch.zeros(B, n, n_in, device=dev)
    P_b   = [torch.zeros(B, n,       device=dev) for _ in range(L)]

    # ── Gradient accumulators ─────────────────────────────────────────────────
    grad_W_recs = [torch.zeros(n, n,    device=dev) for _ in range(L)]
    grad_W_ffs  = [torch.zeros(n, n,    device=dev) for _ in range(L - 1)]
    grad_W_in   = torch.zeros(n, n_in,  device=dev)
    grad_b_recs = [torch.zeros(n,        device=dev) for _ in range(L)]
    grad_W_out  = torch.zeros(n_out, n,  device=dev)
    grad_b_out  = torch.zeros(n_out,     device=dev)

    # ── Running state ─────────────────────────────────────────────────────────
    u        = [torch.zeros(B, n, device=dev) for _ in range(L)]
    s        = [torch.zeros(B, n, device=dev) for _ in range(L)]
    psi_prev = [torch.zeros(B, n, device=dev) for _ in range(L)]

    for t in range(T):
        x_t   = inputs[t]
        s_new = []
        psi_t = []
        u_new = []

        # ── Forward pass (no autograd) ────────────────────────────────────────
        with torch.no_grad():
            for l in range(L):
                if l == 0:
                    ff = x_t @ W_in_.T
                else:
                    ff = s_new[l - 1] @ W_ffs_[l - 1].T

                u_l = (alpha * u[l]
                       + oma * (s[l] @ W_recs_[l].T + ff + b_recs_[l])
                       - v_th * s[l])
                psi_l = gamma * torch.clamp(1.0 - (u_l - v_th).abs() / v_th, min=0.0)
                s_l   = (u_l >= v_th).float()

                u_new.append(u_l)
                psi_t.append(psi_l)
                s_new.append(s_l)

            o = s_new[-1] @ W_out_.T + model.b_out

        # ── Update membrane traces ────────────────────────────────────────────
        for l in range(L):
            if d_zero:
                carry = torch.zeros(B, n, device=dev)
            else:
                # carry^l = alpha - v_th * psi^l_{t-1}
                carry = alpha - v_th * psi_prev[l]   # (B, n)

            # W_rec^l: presynaptic = s^l_{t-1}
            P_rec[l] = carry.unsqueeze(2) * P_rec[l] + oma * s[l].unsqueeze(1)

            # b_rec^l: presynaptic = 1
            P_b[l] = carry * P_b[l] + oma

            if l == 0:
                # W_in: presynaptic = x_t (current input)
                P_in = carry.unsqueeze(2) * P_in + oma * x_t.unsqueeze(1)
            else:
                # W_ff^l: presynaptic = s^{l-1}_t (current-step lower layer spikes)
                P_ff[l] = carry.unsqueeze(2) * P_ff[l] + oma * s_new[l - 1].unsqueeze(1)

        # ── Gradient accumulation (masked timesteps only) ─────────────────────
        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t]) * mask[t].unsqueeze(-1)  # (B, n_out)

            grad_W_out += (err_out.T @ s_new[-1]).detach() / B
            grad_b_out += err_out.mean(0).detach()

            # Learning signal at each layer — propagated instantaneously downward
            deltas = [None] * L
            deltas[L - 1] = err_out @ W_out_   # (B, n): d(loss)/d(s^{L-1}_t)
            for l in range(L - 1, 0, -1):
                # delta^{l-1} = (1-alpha) * (delta^l * psi^l) @ W_ff^l
                deltas[l - 1] = oma * (deltas[l] * psi_t[l]) @ W_ffs_[l - 1]

            # Accumulate per-layer gradients
            for l in range(L):
                eps_rec = psi_t[l].unsqueeze(2) * P_rec[l]   # (B, n, n)
                eps_b   = psi_t[l] * P_b[l]                   # (B, n)

                grad_W_recs[l] += torch.einsum('bi,bij->ij', deltas[l], eps_rec) / B
                grad_b_recs[l] += (deltas[l] * eps_b).mean(0)

                if l == 0:
                    eps_in = psi_t[0].unsqueeze(2) * P_in     # (B, n, n_in)
                    grad_W_in += torch.einsum('bi,bij->ij', deltas[0], eps_in) / B
                else:
                    eps_ff = psi_t[l].unsqueeze(2) * P_ff[l]  # (B, n, n)
                    grad_W_ffs[l - 1] += torch.einsum('bi,bij->ij', deltas[l], eps_ff) / B

        psi_prev = psi_t
        u = u_new
        s = s_new

    # ── Build output dict (keys match model.named_parameters()) ──────────────
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
    """Per-sample cross-entropy error signal: softmax(o) − one_hot_target."""
    return torch.softmax(o, dim=-1) - target
