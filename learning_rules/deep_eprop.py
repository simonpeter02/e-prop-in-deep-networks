"""
Deep e-prop for a 2-layer vanilla tanh RNN (Millidge 2025, Eq. 10).

Compared to deep-RTRL, deep e-prop makes ONE approximation:
  Replace the full recurrent Jacobian J^{rec,l}_t with its diagonal
  when propagating eligibility traces THROUGH TIME.
  Cross-layer (spatial) credit propagation keeps the FULL feedforward Jacobian.

Single-layer e-prop recap (diagonal J^{rec}):
  ε^1_{rec,t}[b,i,j] = psi^1[b,i]*h^1_{t-1}[b,j]
                       + psi^1[b,i]*W_rec^1[i,i] * ε^1_{rec,t-1}[b,i,j]
  (i.e. carry factor = psi^1[b,i]*W_rec^1[i,i], a scalar per neuron)

Deep e-prop for layer-2 own params (same diagonal approximation):
  ε^2_{rec,t}[b,i,j] = psi^2[b,i]*h^2_{t-1}[b,j]
                       + psi^2[b,i]*W_rec^2[i,i] * ε^2_{rec,t-1}[b,i,j]

  ε^2_{ff,t}[b,i,j]  = psi^2[b,i]*h^1_t[b,j]
                       + psi^2[b,i]*W_rec^2[i,i] * ε^2_{ff,t-1}[b,i,j]
  (h^1_t is the instantaneous feedforward input to layer 2)

Cross-layer trace for layer-1 params:  ε^{2←1}_{rec,t}[b, i2, i1, j]
  tracks how h^2_t[i2] depends on W_rec^1[i1, j].

  Temporal carry (diagonal J^{rec,2}):
    psi^2[b,i2]*W_rec^2[i2,i2] * ε^{2←1}_{rec,t-1}[b,i2,i1,j]

  Spatial term (full J^{12}, e-prop P^1):
    J12[b,i2,i1] * ε^1_{rec,t}[b,i1,j]
    = psi^2[b,i2]*W_ff[i2,i1] * ε^1_{rec,t}[b,i1,j]

  Full RTRL would sum over ALL i1 in the spatial term (O(n^4) memory).
  E-prop here uses only the DIAGONAL block of P^1: P^1[i1',(i1,j)] ≈ 0 for i1'≠i1.

d=0 variant: zero all temporal carry terms. Cross-layer spatial term remains.
  ε^{2←1}_{rec,t}[b,i2,i1,j] = J12[b,i2,i1] * ε^1_{rec,t}[b,i1,j]  (spatial only)

Gradient at each masked timestep:
  delta^2_t = W_out^T @ err_out_t                    (learning signal at layer 2)
  dL/dW_rec^2 += einsum(delta^2, ε^2_{rec,t})  / B
  dL/dW_rec^1 += einsum(delta^2, ε^{2←1}_{rec,t}) / B
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
) -> Dict[str, Tensor]:
    """
    Deep e-prop (or d=0) gradients for a 2-layer DeepRNN.

    d_zero=True : drop all temporal carry terms; keep spatial cross-layer term.
    """
    assert model.n_layers == 2, "Deep e-prop implemented for 2-layer networks"

    T, B, n_in = inputs.shape
    n1 = n2 = model.n_rec
    n_out = model.n_out

    W_rec1 = model.W_rec(0).detach()   # (n1, n1)
    W_rec2 = model.W_rec(1).detach()   # (n2, n2)
    W_in_  = model.W_in.detach()        # (n1, n_in)
    W_ff_  = model.W_ff(1).detach()     # (n2, n1)
    W_out_ = model.W_out.detach()       # (n_out, n2)

    w_diag1 = W_rec1.diag()   # (n1,)  W_rec^1[i,i]
    w_diag2 = W_rec2.diag()   # (n2,)  W_rec^2[i,i]

    dev = inputs.device

    # ── Layer-1 eligibility traces  ε^1[b,i,j] ──────────────────────────────
    eps1_rec = torch.zeros(B, n1, n1,  device=dev)
    eps1_in  = torch.zeros(B, n1, n_in, device=dev)
    eps1_b   = torch.zeros(B, n1,       device=dev)

    # ── Layer-2 own-param eligibility traces  ε^2[b,i,j] ────────────────────
    eps2_rec = torch.zeros(B, n2, n2, device=dev)
    eps2_ff  = torch.zeros(B, n2, n1, device=dev)
    eps2_b   = torch.zeros(B, n2,     device=dev)

    # ── Cross-layer traces  ε^{2←1}[b,i,j]  (layer-1 params seen from layer-2)
    eps21_rec = torch.zeros(B, n2, n1, n1,  device=dev)
    eps21_in  = torch.zeros(B, n2, n1, n_in, device=dev)
    eps21_b   = torch.zeros(B, n2, n1,       device=dev)

    # ── Gradient accumulators ─────────────────────────────────────────────────
    grad_W_rec1 = torch.zeros_like(W_rec1)
    grad_W_in   = torch.zeros_like(W_in_)
    grad_b1     = torch.zeros(n1, device=dev)
    grad_W_rec2 = torch.zeros_like(W_rec2)
    grad_W_ff   = torch.zeros_like(W_ff_)
    grad_b2     = torch.zeros(n2, device=dev)
    grad_W_out  = torch.zeros_like(W_out_)
    grad_b_out  = torch.zeros(n_out, device=dev)

    h1 = torch.zeros(B, n1, device=dev)
    h2 = torch.zeros(B, n2, device=dev)

    for t in range(T):
        x_t    = inputs[t]
        h1_prev, h2_prev = h1.clone(), h2.clone()

        with torch.no_grad():
            pre1 = x_t @ W_in_.T + h1_prev @ W_rec1.T + model.bias(0)
            h1   = torch.tanh(pre1)
            pre2 = h1 @ W_ff_.T + h2_prev @ W_rec2.T + model.bias(1)
            h2   = torch.tanh(pre2)
            o    = h2 @ W_out_.T + model.b_out

        psi1 = 1.0 - h1 ** 2   # (B, n1)
        psi2 = 1.0 - h2 ** 2   # (B, n2)

        # Carry factors (scalar per neuron): psi[b,i] * W_rec[i,i]
        carry1 = psi1 * w_diag1   # (B, n1)
        carry2 = psi2 * w_diag2   # (B, n2)

        # ── Layer-1 eligibility traces ────────────────────────────────────────
        if d_zero:
            eps1_rec = psi1.unsqueeze(2) * h1_prev.unsqueeze(1)
            eps1_in  = psi1.unsqueeze(2) * x_t.unsqueeze(1)
            eps1_b   = psi1
        else:
            eps1_rec = (psi1.unsqueeze(2) * h1_prev.unsqueeze(1)
                        + carry1.unsqueeze(2) * eps1_rec)
            eps1_in  = (psi1.unsqueeze(2) * x_t.unsqueeze(1)
                        + carry1.unsqueeze(2) * eps1_in)
            eps1_b   = psi1 + carry1 * eps1_b

        # ── Layer-2 own-param eligibility traces ──────────────────────────────
        if d_zero:
            eps2_rec = psi2.unsqueeze(2) * h2_prev.unsqueeze(1)
            eps2_ff  = psi2.unsqueeze(2) * h1.unsqueeze(1)
            eps2_b   = psi2
        else:
            eps2_rec = (psi2.unsqueeze(2) * h2_prev.unsqueeze(1)
                        + carry2.unsqueeze(2) * eps2_rec)
            eps2_ff  = (psi2.unsqueeze(2) * h1.unsqueeze(1)
                        + carry2.unsqueeze(2) * eps2_ff)
            eps2_b   = psi2 + carry2 * eps2_b

        # ── Cross-layer traces: spatial J^{12} applied to layer-1 traces ──────
        # E-prop approximation:
        #   In full RTRL the spatial term would be:
        #     sum_{i1} J12[b,i2,i1] * P^1[b,i1,p,j]
        #   where P^1[b,i1,p,j] = dh^1_t[i1]/dW_rec^1[p,j] for all i1.
        #   E-prop approximates P^1[b,i1,p,j] ≈ 0 for i1 ≠ p (diagonal only).
        #   So only the i1=p term survives:
        #     spatial_rec[b,i2,p,j] = J12[b,i2,p] * eps1_rec[b,p,j]
        #
        # J12[b, i2, i1] = psi2[b,i2] * W_ff[i2, i1]  shape (B, n2, n1)
        # eps1_rec[b, i1, j]                             shape (B, n1, n1)
        # spatial_rec[b, i2, i1, j] = J12[b,i2,i1] * eps1_rec[b,i1,j]
        # einsum 'bpq,bqj->bpqj':  (B,n2,n1) × (B,n1,n1) → (B,n2,n1,n1)
        J12 = psi2.unsqueeze(2) * W_ff_   # (B, n2, n1)
        spatial_rec = torch.einsum('bpq,bqj->bpqj', J12, eps1_rec)   # (B,n2,n1,n1)
        spatial_in  = torch.einsum('bpq,bqj->bpqj', J12, eps1_in)    # (B,n2,n1,n_in)
        spatial_b   = J12 * eps1_b.unsqueeze(1)                       # (B,n2,n1)

        if d_zero:
            eps21_rec = spatial_rec
            eps21_in  = spatial_in
            eps21_b   = spatial_b
        else:
            eps21_rec = carry2.unsqueeze(2).unsqueeze(3) * eps21_rec + spatial_rec
            eps21_in  = carry2.unsqueeze(2).unsqueeze(3) * eps21_in  + spatial_in
            eps21_b   = carry2.unsqueeze(2)               * eps21_b  + spatial_b

        # ── Gradient accumulation ─────────────────────────────────────────────
        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t])
            err_out = err_out * mask[t].unsqueeze(-1)

            grad_W_out += (err_out.T @ h2).detach() / B
            grad_b_out += err_out.mean(0).detach()

            delta2 = err_out @ W_out_   # (B, n2) — top-layer learning signal

            # Layer-2 own params
            grad_W_rec2 += torch.einsum('bi,bij->ij', delta2, eps2_rec) / B
            grad_W_ff   += torch.einsum('bi,bij->ij', delta2, eps2_ff)  / B
            grad_b2     += (delta2 * eps2_b).mean(0)

            # Layer-1 params (via cross-layer trace)
            grad_W_rec1 += torch.einsum('bi,bipj->pj', delta2, eps21_rec) / B
            grad_W_in   += torch.einsum('bi,bipj->pj', delta2, eps21_in)  / B
            grad_b1     += torch.einsum('bi,bip->p',   delta2, eps21_b)   / B

    return {
        'W_recs.0': grad_W_rec1,
        'W_recs.1': grad_W_rec2,
        'W_ffs.0':  grad_W_ff,
        'W_in':     grad_W_in,
        'biases.0': grad_b1,
        'biases.1': grad_b2,
        'W_out':    grad_W_out,
        'b_out':    grad_b_out,
    }


def mse_error(o: Tensor, target: Tensor) -> Tensor:
    return o - target
