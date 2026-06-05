"""
Deep RTRL — full Real-Time Recurrent Learning extended to deep networks.

Should match BPTT to numerical precision (modulo floating-point order).
Used as a correctness control for deep e-prop.

For a 2-layer tanh RNN:
  h^1_t = tanh(W_rec^1 h^1_{t-1} + W_in  x_t + b^1)
  h^2_t = tanh(W_rec^2 h^2_{t-1} + W_ff  h^1_t + b^2)
  o_t   = W_out h^2_t + b_out

Jacobians at time t:
  J^{rec,l}_{t}[i,k]   = psi^l_t[i] * W_rec^l[i,k]   (recurrent)
  J^{12}_{t}[i,k]      = psi^2_t[i] * W_ff[i,k]       (feedforward l1→l2)

RTRL sensitivity tensors  P[b, k, i, j] = dh^{top}_t[b,k] / dW[i,j]
where "top" is the layer that owns parameter W (or higher).

For parameters IN layer l with total L layers, we need P^{L←l}_t:
  P^{L←l} = dh^L / d(params of layer l)

Recursion (L=2):
  P^{2←2}_{W_rec^2}[b,k,(i,j)] = sum_m J^{rec,2}[b,k,m] * P^{2←2}_{t-1}[b,m,(i,j)]
                                 + psi^2[b,i] * h^2_{t-1}[b,j] * delta_{k,i}

  P^{2←2}_{W_ff}[b,k,(i,j)]    = sum_m J^{rec,2}[b,k,m] * P^{2←2,ff}_{t-1}[b,m,(i,j)]
                                 + psi^2[b,i] * h^1_t[b,j] * delta_{k,i}

  P^{2←1}_{W_rec^1}[b,k,(i,j)] = sum_m J^{rec,2}[b,k,m] * P^{2←1}_{t-1}[b,m,(i,j)]
                                 + sum_m J^{12}[b,k,m] * P^{1←1}_{t}[b,m,(i,j)]

  P^{1←1}_{W_rec^1}[b,m,(i,j)] = sum_n J^{rec,1}[b,m,n] * P^{1←1}_{t-1}[b,n,(i,j)]
                                 + psi^1[b,i] * h^1_{t-1}[b,j] * delta_{m,i}

(and analogously for W_in, b^1, b^2)

Gradient:
  dL/dW_rec^2[i,j] = sum_t mean_b sum_k delta^2[b,k] * P^{2←2}_{W_rec^2,t}[b,k,i,j]
  dL/dW_rec^1[i,j] = sum_t mean_b sum_k delta^2[b,k] * P^{2←1}_{W_rec^1,t}[b,k,i,j]

NOTE: Only feasible for small n_rec (≤20) due to O(n^4) memory/step.
"""

import torch
from torch import Tensor
from models.deep_rnn import DeepRNN
from typing import Dict


def compute_deep_rtrl_gradients(
    model: DeepRNN,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
) -> Dict[str, Tensor]:
    """
    Full RTRL for a 2-layer DeepRNN. Only supports n_layers=2.

    Returns grad dict with keys matching model parameter names:
      'W_recs.0', 'W_recs.1', 'W_ffs.0', 'biases.0', 'biases.1',
      'W_in', 'W_out', 'b_out'
    """
    assert model.n_layers == 2, "Deep RTRL currently implemented for 2 layers"

    T, B, n_in = inputs.shape
    n1 = n2 = model.n_rec
    n_out = model.n_out

    W_rec1 = model.W_rec(0).detach()   # (n1, n1)
    W_rec2 = model.W_rec(1).detach()   # (n2, n2)
    W_in_  = model.W_in.detach()        # (n1, n_in)
    W_ff_  = model.W_ff(1).detach()     # (n2, n1)
    W_out_ = model.W_out.detach()       # (n_out, n2)
    b1     = model.bias(0).detach()
    b2     = model.bias(1).detach()

    dev = inputs.device

    # ── Sensitivity tensors ──────────────────────────────────────────────────
    # Shape convention: P[b, k, i, j] = dh^top_t[b,k] / dW[i,j]
    # Layer 1 sensitivities (how h^1 depends on layer-1 params)
    P1_rec  = torch.zeros(B, n1, n1, n1, device=dev)   # dh1/dW_rec1[i,j]
    P1_in   = torch.zeros(B, n1, n1, n_in, device=dev)  # dh1/dW_in[i,j]
    P1_b    = torch.zeros(B, n1, n1,       device=dev)   # dh1/db1[i]

    # How h^2 depends on layer-1 params (cross-layer propagation)
    P21_rec = torch.zeros(B, n2, n1, n1,  device=dev)   # dh2/dW_rec1[i,j]
    P21_in  = torch.zeros(B, n2, n1, n_in, device=dev)  # dh2/dW_in[i,j]
    P21_b   = torch.zeros(B, n2, n1,      device=dev)   # dh2/db1[i]

    # Layer 2 sensitivities (how h^2 depends on its own params)
    P2_rec  = torch.zeros(B, n2, n2, n2, device=dev)   # dh2/dW_rec2[i,j]
    P2_ff   = torch.zeros(B, n2, n2, n1, device=dev)   # dh2/dW_ff[i,j]
    P2_b    = torch.zeros(B, n2, n2,     device=dev)   # dh2/db2[i]

    # ── Gradient accumulators ─────────────────────────────────────────────────
    grad = {k: torch.zeros_like(p.detach()) for k, p in model.named_parameters()}

    h1 = torch.zeros(B, n1, device=dev)
    h2 = torch.zeros(B, n2, device=dev)

    for t in range(T):
        x_t  = inputs[t]   # (B, n_in)
        h1_prev, h2_prev = h1.clone(), h2.clone()

        with torch.no_grad():
            pre1 = x_t @ W_in_.T + h1_prev @ W_rec1.T + b1
            h1   = torch.tanh(pre1)
            pre2 = h1 @ W_ff_.T + h2_prev @ W_rec2.T + b2
            h2   = torch.tanh(pre2)
            o    = h2 @ W_out_.T + model.b_out

        psi1 = 1.0 - h1 ** 2   # (B, n1)
        psi2 = 1.0 - h2 ** 2   # (B, n2)

        # Jacobians  J[b,k,m] = psi[b,k] * W[k,m]
        J1  = psi1.unsqueeze(2) * W_rec1   # (B, n1, n1)
        J2  = psi2.unsqueeze(2) * W_rec2   # (B, n2, n2)
        J12 = psi2.unsqueeze(2) * W_ff_    # (B, n2, n1)

        # ── Update layer-1 sensitivities ─────────────────────────────────────
        # P1_rec_new[b,k,i,j] = sum_m J1[b,k,m]*P1_rec[b,m,i,j]
        #                       + psi1[b,i]*h1_prev[b,j]*delta_{k,i}
        carry1_rec = torch.einsum('bkm,bmij->bkij', J1, P1_rec)
        carry1_in  = torch.einsum('bkm,bmij->bkij', J1, P1_in)
        carry1_b   = torch.einsum('bkm,bmi->bki',   J1, P1_b)

        # Direct terms: only the k=i diagonal gets a contribution
        # Efficient: create (B, n1, n1, n1) where [b,i,i,j] += psi1[b,i]*h1_prev[b,j]
        direct1_rec = torch.zeros_like(P1_rec)
        # For each neuron i, the direct term is psi1[b,i]*h1_prev[b,j]
        # Equivalent to: psi1[b,i] * h1_prev[b,j] placed at [b,i,i,j]
        eye1 = torch.eye(n1, device=dev)  # (n1, n1) — selects k=i diagonal
        # psi1[b,i]*h1_prev[b,j] broadcast over k via eye: bki,bij -> bkij
        direct1_rec = torch.einsum('bi,bj,ki->bkij', psi1, h1_prev, eye1)
        direct1_in  = torch.einsum('bi,bj,ki->bkij', psi1, x_t,     eye1)
        direct1_b   = torch.einsum('bi,ki->bki',      psi1,          eye1)

        P1_rec = carry1_rec + direct1_rec
        P1_in  = carry1_in  + direct1_in
        P1_b   = carry1_b   + direct1_b

        # ── Update cross-layer sensitivities (how h2 depends on layer-1 params) ─
        # P21_rec_new = J2 @ P21_rec + J12 @ P1_rec
        P21_rec = (torch.einsum('bkm,bmij->bkij', J2, P21_rec) +
                   torch.einsum('bkm,bmij->bkij', J12, P1_rec))
        P21_in  = (torch.einsum('bkm,bmij->bkij', J2, P21_in)  +
                   torch.einsum('bkm,bmij->bkij', J12, P1_in))
        P21_b   = (torch.einsum('bkm,bmi->bki',   J2, P21_b)   +
                   torch.einsum('bkm,bmi->bki',    J12, P1_b))

        # ── Update layer-2 sensitivities ─────────────────────────────────────
        # P2_rec: dh2/dW_rec2[i,j]  — direct at k=i: psi2[b,i]*h2_prev[b,j]
        direct2_rec = torch.einsum('bi,bj,ki->bkij', psi2, h2_prev, torch.eye(n2, device=dev))
        direct2_ff  = torch.einsum('bi,bj,ki->bkij', psi2, h1,      torch.eye(n2, device=dev))
        direct2_b   = torch.einsum('bi,ki->bki',      psi2,          torch.eye(n2, device=dev))

        carry2_rec = torch.einsum('bkm,bmij->bkij', J2, P2_rec)
        carry2_ff  = torch.einsum('bkm,bmij->bkij', J2, P2_ff)
        carry2_b   = torch.einsum('bkm,bmi->bki',   J2, P2_b)

        P2_rec = carry2_rec + direct2_rec
        P2_ff  = carry2_ff  + direct2_ff
        P2_b   = carry2_b   + direct2_b

        # ── Gradient accumulation at masked timesteps ─────────────────────────
        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t])   # (B, n_out)
            err_out = err_out * mask[t].unsqueeze(-1)

            # Output layer
            grad['W_out'] += (err_out.T @ h2).detach() / B
            grad['b_out'] += err_out.mean(0).detach()

            # Top-layer learning signal projected to h2
            delta2 = (err_out @ W_out_)   # (B, n2)

            # Layer 2 params
            # dL/dW_rec2[i,j] = mean_b sum_k delta2[b,k] * P2_rec[b,k,i,j]
            grad['W_recs.1'] += torch.einsum('bk,bkij->ij', delta2, P2_rec)  / B
            grad['W_ffs.0']  += torch.einsum('bk,bkij->ij', delta2, P2_ff)   / B
            grad['biases.1'] += torch.einsum('bk,bki->i',   delta2, P2_b)    / B

            # Layer 1 params (via cross-layer propagation)
            grad['W_recs.0'] += torch.einsum('bk,bkij->ij', delta2, P21_rec) / B
            grad['W_in']     += torch.einsum('bk,bkij->ij', delta2, P21_in)  / B
            grad['biases.0'] += torch.einsum('bk,bki->i',   delta2, P21_b)   / B

    return grad


def mse_error(o: Tensor, target: Tensor) -> Tensor:
    return o - target
