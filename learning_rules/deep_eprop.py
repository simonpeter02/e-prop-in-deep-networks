"""
Deep e-prop for an L-layer vanilla tanh RNN (Millidge 2025, Eq. 10).

Generalisation of the 2-layer implementation to arbitrary depth L >= 1.

Key approximation (same as single-layer e-prop):
  Replace the full recurrent Jacobian J^{rec,l}_t with its DIAGONAL
  when propagating eligibility traces THROUGH TIME.
  Cross-layer (spatial) credit propagation keeps the FULL feedforward Jacobian.

Trace structure for L layers
------------------------------
Self-traces for layer l:
  eps_self_rec[l]  (B, n, n)        how h^l depends on W_rec^l
  eps_self_ff[l]   (B, n, n)        how h^l depends on W_ff^l  [l >= 1]
  eps_self_in      (B, n, n_in)     how h^0 depends on W_in    [layer 0 only]
  eps_self_b[l]    (B, n)           how h^l depends on b^l

Cross-layer traces for l_top > l_src (tracks how h^{l_top} depends on params of l_src):
  eps_cross_rec[l_top][l_src]   (B, n, n, n)
  eps_cross_ff[l_top][l_src]    (B, n, n, n)   [l_src >= 1]
  eps_cross_in[l_top]           (B, n, n, n_in) [always W_in at l_src=0]
  eps_cross_b[l_top][l_src]     (B, n, n)

Update order: bottom to top.
For l_top = 1..L-1, for l_src = 0..l_top-1:
  Adjacent  (l_src == l_top-1): spatial uses self-trace of l_src
  Non-adjacent (l_src <  l_top-1): spatial uses cross-trace from l_top-1 to l_src

Gradient accumulation uses delta = err_out @ W_out projected to top hidden layer:
  Top layer (L-1): via self-traces
  Layer l_src < L-1: via cross-layer traces eps_cross[L-1][l_src]

Memory: O(L^2 * B * n^3) for cross-rec traces.
  L=3, n=50, B=32 → ~48 MB;  L=5, n=50, B=32 → ~160 MB.
"""

import torch
from torch import Tensor
from models.deep_rnn import DeepRNN
from typing import Dict


def compute_deep_eprop_gradients(
    model: DeepRNN,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
    d_zero: bool = False,
    mode: str = "full",
) -> Dict[str, Tensor]:
    """
    Deep e-prop (or ablations) gradients for an L-layer DeepRNN (L >= 1).

    Supports leaky-integrator dynamics:
        h^l_t = (1-α) h^l_{t-1} + α tanh(a^l_t)
    via model.alpha (scalar buffer, default 1.0 ⇒ vanilla tanh).  The leaky
    diagonal temporal carry is  c^l = (1-α) + α ψ^l W_rec^l_diag  and every
    instantaneous derivative is scaled by the drive  α ψ^l.  At α=1 this is
    identical to the original vanilla deep e-prop.

    Controls (act on the cross-layer / hierarchical trace ϵ^z only — the
    object that carries LOWER-layer credit up to the top, where both depth and
    time credit live; the within-layer self-traces ϵ^h are always kept intact):

      mode='full'            : ϵ^z_t = (∂z/∂h)·ϵ^h_t + (∂z/∂z_{t-1})·ϵ^z_{t-1}
      mode='ablate_spatial'  : set ∂z/∂h = 0 ⇒ ϵ^z_t = (∂z/∂z_{t-1})·ϵ^z_{t-1}
                               → lower-layer credit never enters the hierarchy
                               (DEPTH credit removed; lower-layer grads → 0).
      mode='ablate_temporal' : set ∂z/∂z_{t-1} = 0 ⇒ ϵ^z_t = (∂z/∂h)·ϵ^h_t
                               → only same-timestep transmission of the lower
                               trace (cross-layer TEMPORAL credit removed).

    d_zero=True : legacy baseline — drop ALL temporal carry (self + cross),
    keep only instantaneous same-timestep spatial credit.
    """
    assert mode in ("full", "ablate_spatial", "ablate_temporal"), f"bad mode {mode}"
    L     = model.n_layers
    T, B, n_in = inputs.shape
    n     = model.n_rec
    n_out = model.n_out
    dev   = inputs.device
    # Per-layer integration rate α_l (default 1.0 ⇒ vanilla tanh).
    _a = getattr(model, "alpha", 1.0)
    if isinstance(_a, Tensor) and _a.numel() == L:
        alphas = [float(_a[l]) for l in range(L)]
    else:
        alphas = [float(_a)] * L

    # Which terms of the cross-layer (ϵ^z) trace are active:
    spatial_on  = (mode != "ablate_spatial")
    # temporal carry of ϵ^z off if explicitly ablated OR under legacy d_zero
    z_temporal_on = (mode != "ablate_temporal") and (not d_zero)

    # ── Detach weights ─────────────────────────────────────────────────────────
    W_recs  = [model.W_rec(l).detach() for l in range(L)]
    biases  = [model.bias(l).detach()  for l in range(L)]
    W_in_   = model.W_in.detach()
    # W_ffs[i] = feedforward weight into layer i+1  (model.W_ffs[i])
    W_ffs   = [model.W_ff(l).detach() for l in range(1, L)]
    W_out_  = model.W_out.detach()
    w_diags = [W_recs[l].diag() for l in range(L)]   # (n,) each

    # ── Self-traces ────────────────────────────────────────────────────────────
    eps_self_rec = [torch.zeros(B, n, n,     device=dev) for _ in range(L)]
    eps_self_ff  = [None] + [torch.zeros(B, n, n, device=dev) for _ in range(1, L)]
    eps_self_in  = torch.zeros(B, n, n_in,   device=dev)
    eps_self_b   = [torch.zeros(B, n,        device=dev) for _ in range(L)]

    # ── Cross-layer traces (only l_top > l_src) ────────────────────────────────
    eps_cross_rec = [[None] * L for _ in range(L)]
    eps_cross_ff  = [[None] * L for _ in range(L)]   # l_src > 0 only
    eps_cross_in  = [None] * L                        # l_src=0, indexed by l_top
    eps_cross_b   = [[None] * L for _ in range(L)]

    for l_top in range(1, L):
        eps_cross_in[l_top] = torch.zeros(B, n, n, n_in, device=dev)
        for l_src in range(l_top):
            eps_cross_rec[l_top][l_src] = torch.zeros(B, n, n, n,    device=dev)
            eps_cross_b[l_top][l_src]   = torch.zeros(B, n, n,       device=dev)
            if l_src > 0:
                eps_cross_ff[l_top][l_src] = torch.zeros(B, n, n, n, device=dev)

    # ── Gradient accumulators ──────────────────────────────────────────────────
    grad_W_recs = [torch.zeros(n, n,   device=dev) for _ in range(L)]
    grad_W_ffs  = [torch.zeros(n, n,   device=dev) for _ in range(L - 1)]
    grad_W_in   = torch.zeros(n, n_in, device=dev)
    grad_biases = [torch.zeros(n,      device=dev) for _ in range(L)]
    grad_W_out  = torch.zeros_like(W_out_)
    grad_b_out  = torch.zeros(n_out,   device=dev)

    hs = [torch.zeros(B, n, device=dev) for _ in range(L)]

    for t in range(T):
        x_t     = inputs[t]
        hs_prev = [h.clone() for h in hs]

        # ── Forward pass ──────────────────────────────────────────────────────
        with torch.no_grad():
            new_hs    = []
            tanh_vals = []
            for l in range(L):
                inp   = (x_t @ W_in_.T if l == 0 else new_hs[l - 1] @ W_ffs[l - 1].T)
                tv    = torch.tanh(inp + hs_prev[l] @ W_recs[l].T + biases[l])
                h_new = (1.0 - alphas[l]) * hs_prev[l] + alphas[l] * tv
                tanh_vals.append(tv)
                new_hs.append(h_new)
            hs = new_hs
            o  = hs[-1] @ W_out_.T + model.b_out

        # psi_raw = tanh'(a) = 1 - tanh(a)^2  (NOT 1-h^2: h is the leaky state)
        psi_raw = [1.0 - tanh_vals[l] ** 2 for l in range(L)]     # (B, n) each
        # Instantaneous drive  α_l ψ_raw  scales every immediate derivative term.
        drive   = [alphas[l] * psi_raw[l] for l in range(L)]      # (B, n) each
        # Diagonal temporal carry  c^l = (1-α_l) + α_l ψ_raw W_rec_diag.
        carries = [(1.0 - alphas[l]) + drive[l] * w_diags[l] for l in range(L)]

        # ── Update self-traces ϵ^h (bottom to top) ────────────────────────────
        # Always full e-prop (the controls act only on the cross-layer ϵ^z).
        for l in range(L):
            c = carries[l]   # (B, n)
            if d_zero:
                eps_self_rec[l] = drive[l].unsqueeze(2) * hs_prev[l].unsqueeze(1)
                eps_self_b[l]   = drive[l]
                if l == 0:
                    eps_self_in = drive[0].unsqueeze(2) * x_t.unsqueeze(1)
                else:
                    eps_self_ff[l] = drive[l].unsqueeze(2) * hs[l - 1].unsqueeze(1)
            else:
                eps_self_rec[l] = (drive[l].unsqueeze(2) * hs_prev[l].unsqueeze(1)
                                   + c.unsqueeze(2) * eps_self_rec[l])
                eps_self_b[l]   = drive[l] + c * eps_self_b[l]
                if l == 0:
                    eps_self_in = (drive[0].unsqueeze(2) * x_t.unsqueeze(1)
                                   + c.unsqueeze(2) * eps_self_in)
                else:
                    eps_self_ff[l] = (drive[l].unsqueeze(2) * hs[l - 1].unsqueeze(1)
                                      + c.unsqueeze(2) * eps_self_ff[l])

        # ── Update cross-layer traces ──────────────────────────────────────────
        # Process l_top from 1..L-1; within each l_top, l_src from l_top-1..0.
        # This ensures eps_cross[l_top-1][l_src] is already updated before
        # we use it in the non-adjacent (l_src < l_top-1) case.
        for l_top in range(1, L):
            # Spatial Jacobian ∂h^{l_top}/∂h^{l_top-1} = α ψ_raw ⊙ W_ff = drive ⊙ W_ff
            J_ff = drive[l_top].unsqueeze(2) * W_ffs[l_top - 1]   # (B, n, n)
            c    = carries[l_top]                                   # (B, n)

            for l_src in range(l_top):
                adjacent = (l_src == l_top - 1)

                # -- Spatial term  (∂z/∂h)·ϵ^h  ----------------------------
                if adjacent:
                    # Path of length 1: J_ff @ self-trace of l_src
                    sp_rec = torch.einsum('bpq,bqj->bpqj', J_ff, eps_self_rec[l_src])
                    sp_b   = J_ff * eps_self_b[l_src].unsqueeze(1)   # (B, n, n)
                    if l_src == 0:
                        sp_in = torch.einsum('bpq,bqj->bpqj', J_ff, eps_self_in)
                    if l_src > 0:
                        sp_ff = torch.einsum('bpq,bqj->bpqj', J_ff, eps_self_ff[l_src])
                else:
                    # Path of length >1: J_ff @ cross-trace from l_top-1 to l_src
                    sp_rec = torch.einsum('bpk,bkij->bpij', J_ff, eps_cross_rec[l_top - 1][l_src])
                    sp_b   = torch.einsum('bpk,bki->bpi',   J_ff, eps_cross_b[l_top - 1][l_src])
                    if l_src == 0:
                        sp_in = torch.einsum('bpk,bkij->bpij', J_ff, eps_cross_in[l_top - 1])
                    if l_src > 0:
                        sp_ff = torch.einsum('bpk,bkij->bpij', J_ff, eps_cross_ff[l_top - 1][l_src])

                # mode='ablate_spatial': set ∂z/∂h = 0 → drop the spatial seed
                if not spatial_on:
                    sp_rec = torch.zeros_like(sp_rec)
                    sp_b   = torch.zeros_like(sp_b)
                    if l_src == 0:
                        sp_in = torch.zeros_like(sp_in)
                    if l_src > 0:
                        sp_ff = torch.zeros_like(sp_ff)

                # -- Temporal carry (∂z/∂z_{t-1})·ϵ^z_{t-1}  + spatial ------
                # z_temporal_on=False under mode='ablate_temporal' or legacy d_zero.
                if z_temporal_on:
                    eps_cross_rec[l_top][l_src] = (
                        c[:, :, None, None] * eps_cross_rec[l_top][l_src] + sp_rec)
                    eps_cross_b[l_top][l_src] = (
                        c.unsqueeze(2) * eps_cross_b[l_top][l_src] + sp_b)
                    if l_src == 0:
                        eps_cross_in[l_top] = (
                            c[:, :, None, None] * eps_cross_in[l_top] + sp_in)
                    if l_src > 0:
                        eps_cross_ff[l_top][l_src] = (
                            c[:, :, None, None] * eps_cross_ff[l_top][l_src] + sp_ff)
                else:
                    eps_cross_rec[l_top][l_src] = sp_rec
                    eps_cross_b[l_top][l_src]   = sp_b
                    if l_src == 0:
                        eps_cross_in[l_top] = sp_in
                    if l_src > 0:
                        eps_cross_ff[l_top][l_src] = sp_ff

        # ── Gradient accumulation ──────────────────────────────────────────────
        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t]) * mask[t].unsqueeze(-1)

            grad_W_out += (err_out.T @ hs[-1]).detach() / B
            grad_b_out += err_out.mean(0).detach()

            delta = err_out @ W_out_   # (B, n)  learning signal at top hidden layer
            Lt    = L - 1

            # Top layer own params (self-traces)
            grad_W_recs[Lt] += torch.einsum('bi,bij->ij', delta, eps_self_rec[Lt]) / B
            grad_biases[Lt] += (delta * eps_self_b[Lt]).mean(0)
            if Lt > 0:
                grad_W_ffs[Lt - 1] += torch.einsum('bi,bij->ij', delta, eps_self_ff[Lt]) / B
            # BUG FIX: W_in belongs to layer 0. For L=1 (Lt=0), layer 0 is also
            # the top layer, so its W_in gradient comes from the self-trace.
            # For L>=2 (Lt>=1) it is handled via the cross-trace loop below.
            if Lt == 0:
                grad_W_in += torch.einsum('bi,bij->ij', delta, eps_self_in) / B

            # Lower layers (cross-layer traces from top layer Lt to each l_src)
            for l_src in range(L - 1):
                grad_W_recs[l_src] += (
                    torch.einsum('bi,bipj->pj', delta, eps_cross_rec[Lt][l_src]) / B)
                grad_biases[l_src] += (
                    torch.einsum('bi,bip->p', delta, eps_cross_b[Lt][l_src]) / B)
                if l_src == 0:
                    grad_W_in += torch.einsum('bi,bipj->pj', delta, eps_cross_in[Lt]) / B
                if l_src > 0:
                    grad_W_ffs[l_src - 1] += (
                        torch.einsum('bi,bipj->pj', delta, eps_cross_ff[Lt][l_src]) / B)

    # ── Build output dict ──────────────────────────────────────────────────────
    result = {'W_in': grad_W_in, 'W_out': grad_W_out, 'b_out': grad_b_out}
    for l in range(L):
        result[f'W_recs.{l}'] = grad_W_recs[l]
        result[f'biases.{l}'] = grad_biases[l]
    for l in range(1, L):
        result[f'W_ffs.{l - 1}'] = grad_W_ffs[l - 1]
    return result


def mse_error(o: Tensor, target: Tensor) -> Tensor:
    return o - target


def xent_error(o: Tensor, target: Tensor) -> Tensor:
    """Per-sample cross-entropy error signal: softmax(o) − target."""
    return torch.softmax(o, dim=-1) - target
