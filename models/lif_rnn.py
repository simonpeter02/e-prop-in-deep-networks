"""
Single-layer LIF (Leaky Integrate-and-Fire) RNN.

Membrane dynamics (Bellec et al. 2020 convention):

  u_t[i] = alpha * u_{t-1}[i]
           + (1-alpha) * (W_rec @ s_{t-1} + W_in @ x_t + b)[i]
           - v_th * s_{t-1}[i]     ← soft reset: subtract threshold after spiking
  s_t[i] = H(u_t[i] - v_th)       ← 1 if membrane potential crosses threshold
  o_t    = W_out @ s_t + b_out     ← linear readout from spikes

For BPTT: H is replaced by a piecewise-linear surrogate gradient (SpikeFn).
For e-prop: psi_t[i] = gamma * max(0, 1 - |u_t[i] - v_th| / v_th) is the pseudoderivative.

Parameters
----------
n_in, n_rec, n_out : layer dimensions
alpha  : membrane decay = exp(-dt / tau_m), default 0.9
v_th   : spike threshold (normalised to 1.0 by convention)
gamma  : surrogate gradient peak magnitude, default 0.3
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class SpikeFn(torch.autograd.Function):
    """Heaviside spike with piecewise-linear surrogate gradient."""

    @staticmethod
    def forward(ctx, v_shifted: Tensor, gamma: float) -> Tensor:
        ctx.save_for_backward(v_shifted)
        ctx.gamma = gamma
        return (v_shifted >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (v_shifted,) = ctx.saved_tensors
        surr = ctx.gamma * torch.clamp(1.0 - v_shifted.abs(), min=0.0)
        return grad_out * surr, None


class LIFNetwork(nn.Module):
    """Single-layer LIF recurrent network with a linear spike readout."""

    def __init__(
        self,
        n_in:  int,
        n_rec: int,
        n_out: int,
        alpha: float = 0.9,
        v_th:  float = 0.1,
        gamma: float = 0.3,
    ):
        super().__init__()
        self.n_in  = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.alpha = alpha
        self.v_th  = v_th
        self.gamma = gamma

        self.W_in  = nn.Parameter(torch.randn(n_rec, n_in)  / n_in  ** 0.5)
        self.W_rec = nn.Parameter(torch.randn(n_rec, n_rec) / n_rec ** 0.5)
        self.b_rec = nn.Parameter(torch.zeros(n_rec))
        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / n_rec ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(n_out))

        # Scale W_rec so spectral radius = 0.9
        with torch.no_grad():
            sr = torch.linalg.eigvals(self.W_rec).abs().max().item()
            self.W_rec.data *= 0.9 / sr

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tuple]:
        """
        inputs : (T, B, n_in)

        Returns
        -------
        outputs : (T, B, n_out)   linear readout from spikes at each step
        state   : (u_seq, s_seq)  membrane potentials and spike trains (T, B, n_rec)
        """
        T, B, _ = inputs.shape
        dev = inputs.device

        u = torch.zeros(B, self.n_rec, device=dev)
        s = torch.zeros(B, self.n_rec, device=dev)

        u_list, s_list, o_list = [], [], []

        for t in range(T):
            x = inputs[t]
            u_new = (self.alpha * u
                     + (1 - self.alpha) * (s @ self.W_rec.T + x @ self.W_in.T + self.b_rec)
                     - self.v_th * s)
            s_new = SpikeFn.apply(u_new - self.v_th, self.gamma)
            o     = s_new @ self.W_out.T + self.b_out

            u, s = u_new, s_new
            u_list.append(u_new.detach())
            s_list.append(s_new)
            o_list.append(o)

        outputs = torch.stack(o_list)   # (T, B, n_out)
        u_seq   = torch.stack(u_list)   # (T, B, n_rec)
        s_seq   = torch.stack(s_list)   # (T, B, n_rec)
        return outputs, (u_seq, s_seq)
