"""
Single-layer e-prop for a vanilla tanh RNN.

Theory (Bellec et al. 2020, single-layer case)
-----------------------------------------------
Full RTRL update for synapse W_rec[i,j]:

  dL/dW_rec[i,j] = sum_t  L_i^t * P_t[i,(i,j)]

where P_t[k,(i,j)] = dh_t[k]/dW_rec[i,j] satisfies the recursion:

  P_t[k,(i,j)] = sum_m J_t[k,m] * P_{t-1}[m,(i,j)]
               + psi_t[k] * h_{t-1}[j] * delta_{k,i}

Full RTRL keeps all k,m — O(n^4) memory, unusable for large n.

E-prop approximation (Bellec et al., eq. S.8):
  Drop all k ≠ i terms in the recursion; keep only k = i:

  epsilon_t[i,j] ≈ psi_t[i] * h_{t-1}[j]
                 + J_t[i,i] * epsilon_{t-1}[i,j]
                 = psi_t[i] * h_{t-1}[j]
                 + psi_t[i] * W_rec[i,i] * epsilon_{t-1}[i,j]

  Gradient: dL/dW_rec[i,j] ≈ sum_t L_i^t * epsilon_t[i,j]

Only the DIAGONAL W_rec[i,i] appears — no cross-neuron mixing.
O(n^2) memory per batch.

d=0 variant: set the carry term to zero (λ=0):

  epsilon_t[i,j] = psi_t[i] * h_{t-1}[j]   (immediate only)

  Gradient: dL/dW_rec[i,j] ≈ sum_t L_i^t * psi_t[i] * h_{t-1}[j]
"""

import torch
from torch import Tensor
from models.vanilla_rnn import VanillaRNN
from typing import Dict


def compute_eprop_gradients(
    model: VanillaRNN,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
    d_zero: bool = False,
) -> Dict[str, Tensor]:
    """
    Compute e-prop parameter gradients without autograd.

    Parameters
    ----------
    model             : VanillaRNN
    inputs            : (T, B, n_in)
    targets           : (T, B, n_out)
    mask              : (T, B)  — 1 where loss is counted
    learning_signal_fn: callable(o_t, target_t) -> (B, n_out) error signal
    d_zero            : if True, use d=0 variant (no trace propagation)

    Returns
    -------
    dict with keys 'W_rec', 'W_in', 'b_rec', 'W_out', 'b_out'
    """
    T, B, n_in = inputs.shape
    n_rec = model.n_rec
    n_out = model.n_out

    W_rec = model.W_rec.detach()  # (n_rec, n_rec)
    W_in  = model.W_in.detach()   # (n_rec, n_in)
    W_out = model.W_out.detach()  # (n_out, n_rec)
    w_diag = W_rec.diag()         # (n_rec,)  diagonal W_rec[i,i]

    # Eligibility traces: epsilon_t[b, i, j]
    eps_rec = torch.zeros(B, n_rec, n_rec, device=inputs.device)
    eps_in  = torch.zeros(B, n_rec, n_in,  device=inputs.device)
    eps_b   = torch.zeros(B, n_rec,        device=inputs.device)

    # Gradient accumulators
    grad_W_rec = torch.zeros_like(W_rec)
    grad_W_in  = torch.zeros_like(W_in)
    grad_b_rec = torch.zeros(n_rec, device=inputs.device)
    grad_W_out = torch.zeros_like(W_out)
    grad_b_out = torch.zeros(n_out, device=inputs.device)

    h = model.init_hidden(B, device=inputs.device)

    for t in range(T):
        x_t = inputs[t]       # (B, n_in)
        h_prev = h.clone()

        with torch.no_grad():
            pre = x_t @ W_in.T + h_prev @ W_rec.T + model.b_rec
            h = torch.tanh(pre)
            o = h @ W_out.T + model.b_out

        # Exact Jacobian diagonal for tanh: psi_t[i] = 1 - h_t[i]^2
        psi = 1.0 - h ** 2   # (B, n_rec)

        if d_zero:
            # No history: epsilon_t = immediate derivative only
            eps_rec = psi.unsqueeze(2) * h_prev.unsqueeze(1)   # (B, n_rec, n_rec)
            eps_in  = psi.unsqueeze(2) * x_t.unsqueeze(1)      # (B, n_rec, n_in)
            eps_b   = psi                                        # (B, n_rec)
        else:
            # E-prop: diagonal Jacobian carry  J_t[i,i] = psi_t[i] * W_rec[i,i]
            # carry[b, i] = psi[b,i] * W_rec[i,i]
            carry = psi * w_diag  # (B, n_rec)

            # epsilon_t[b,i,j] = psi[b,i]*h_prev[b,j] + carry[b,i]*eps_{t-1}[b,i,j]
            eps_rec = psi.unsqueeze(2) * h_prev.unsqueeze(1) + carry.unsqueeze(2) * eps_rec
            eps_in  = psi.unsqueeze(2) * x_t.unsqueeze(1)   + carry.unsqueeze(2) * eps_in
            eps_b   = psi + carry * eps_b

        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t])  # (B, n_out)
            err_out = err_out * mask[t].unsqueeze(-1)

            grad_W_out += (err_out.T @ h).detach() / B
            grad_b_out += err_out.mean(0).detach()

            # Learning signal backprojected to recurrent layer
            delta_h = err_out @ W_out   # (B, n_rec)

            # dL/dW_rec[i,j] += L_i * epsilon_t[i,j]  (averaged over batch)
            grad_W_rec += torch.einsum('bi,bij->ij', delta_h, eps_rec) / B
            grad_W_in  += torch.einsum('bi,bij->ij', delta_h, eps_in)  / B
            grad_b_rec += (delta_h * eps_b).mean(0)

    return {
        'W_rec': grad_W_rec,
        'W_in':  grad_W_in,
        'b_rec': grad_b_rec,
        'W_out': grad_W_out,
        'b_out': grad_b_out,
    }


def mse_error(o: Tensor, target: Tensor) -> Tensor:
    """MSE learning signal: dL/do = o - target."""
    return o - target


def xent_error(o: Tensor, target: Tensor) -> Tensor:
    """Softmax cross-entropy learning signal: dL/do = softmax(o) - target."""
    return torch.softmax(o, dim=-1) - target


# ─────────────────────────────────────────────────────────────────────────────
# Leaky integrator e-prop
# ─────────────────────────────────────────────────────────────────────────────

def compute_eprop_leaky_gradients(
    model,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    learning_signal_fn,
    d_zero: bool = False,
) -> Dict[str, Tensor]:
    """
    E-prop (or d=0) gradients for a LeakyRNN (or VanillaRNN with alpha=1).

    Leaky update:  h_t = (1-α)*h_{t-1} + α*tanh(W_rec h_{t-1} + W_in x_t + b)

    Diagonal Jacobian:
        ∂h_t[i]/∂h_{t-1}[i] = (1-α) + α * psi_raw[i] * W_rec[i,i]

    E-prop carry  (full):  c_t[i] = (1-α) + α * psi_raw_t[i] * W_rec_diag[i]
    E-prop carry  (d=0):   c_t[i] = 0        (instantaneous only)

    Instantaneous drive:   α * psi_raw_t[i] * presynaptic_t[j]

    At α=0.1: dominant carry = 0.9 → trace survives ~10 steps.
    At α=1.0: reduces to standard e-prop (carry = psi * W_diag ≈ 0.005).
    """
    alpha = getattr(model, 'alpha', 1.0)

    T, B, n_in = inputs.shape
    n_rec = model.n_rec
    n_out = model.n_out

    W_rec  = model.W_rec.detach()
    W_in   = model.W_in.detach()
    W_out  = model.W_out.detach()
    w_diag = W_rec.diag()         # (n_rec,)

    eps_rec = torch.zeros(B, n_rec, n_rec, device=inputs.device)
    eps_in  = torch.zeros(B, n_rec, n_in,  device=inputs.device)
    eps_b   = torch.zeros(B, n_rec,        device=inputs.device)

    grad_W_rec = torch.zeros_like(W_rec)
    grad_W_in  = torch.zeros_like(W_in)
    grad_b_rec = torch.zeros(n_rec, device=inputs.device)
    grad_W_out = torch.zeros_like(W_out)
    grad_b_out = torch.zeros(n_out, device=inputs.device)

    h = model.init_hidden(B, device=inputs.device)

    for t in range(T):
        x_t    = inputs[t]
        h_prev = h.clone()

        with torch.no_grad():
            pre      = x_t @ W_in.T + h_prev @ W_rec.T + model.b_rec
            tanh_val = torch.tanh(pre)                       # (B, n_rec)
            h        = (1 - alpha) * h_prev + alpha * tanh_val
            o        = h @ W_out.T + model.b_out

        # Raw tanh derivative (no alpha factor yet)
        psi_raw = 1.0 - tanh_val ** 2                        # (B, n_rec)

        # Instantaneous drive scale = α * ψ_raw  (the ∂tanh/∂W part)
        drive = alpha * psi_raw                              # (B, n_rec)

        # Diagonal Jacobian carry
        if d_zero:
            carry = torch.zeros_like(psi_raw)
        else:
            carry = (1 - alpha) + alpha * psi_raw * w_diag  # (B, n_rec)

        eps_rec = drive.unsqueeze(2) * h_prev.unsqueeze(1) + carry.unsqueeze(2) * eps_rec
        eps_in  = drive.unsqueeze(2) * x_t.unsqueeze(1)   + carry.unsqueeze(2) * eps_in
        eps_b   = drive + carry * eps_b

        if mask[t].any():
            err_out = learning_signal_fn(o, targets[t]) * mask[t].unsqueeze(-1)

            grad_W_out += (err_out.T @ h).detach() / B
            grad_b_out += err_out.mean(0).detach()

            delta_h = err_out @ W_out                        # (B, n_rec)

            grad_W_rec += torch.einsum('bi,bij->ij', delta_h, eps_rec) / B
            grad_W_in  += torch.einsum('bi,bij->ij', delta_h, eps_in)  / B
            grad_b_rec += (delta_h * eps_b).mean(0)

    return {
        'W_rec': grad_W_rec,
        'W_in':  grad_W_in,
        'b_rec': grad_b_rec,
        'W_out': grad_W_out,
        'b_out': grad_b_out,
    }
