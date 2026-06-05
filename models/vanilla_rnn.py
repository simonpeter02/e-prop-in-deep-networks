"""
Single-layer vanilla RNN with tanh activations.

  h_t = tanh(W_rec @ h_{t-1} + W_in @ x_t + b_rec)
  o_t = W_out @ h_t + b_out

All parameters are stored as nn.Parameter so autograd / BPTT work
automatically.  The forward pass also returns per-step hidden states so
that e-prop can access the Jacobian factors.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List


class VanillaRNN(nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out

        # Recurrent weight initialised following Bellec et al.:
        # spectral radius ≈ 0.9 (ensures stable initial dynamics)
        W_rec = torch.randn(n_rec, n_rec) / (n_rec ** 0.5)
        with torch.no_grad():
            sr = torch.linalg.eigvals(W_rec).abs().max().item()
            W_rec *= 0.9 / sr
        self.W_rec = nn.Parameter(W_rec)

        self.W_in = nn.Parameter(torch.randn(n_rec, n_in) / (n_in ** 0.5))
        self.b_rec = nn.Parameter(torch.zeros(n_rec))

        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / (n_rec ** 0.5))
        self.b_out = nn.Parameter(torch.zeros(n_out))

    def init_hidden(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(batch_size, self.n_rec, device=device or self.W_rec.device)

    def step(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Single time step. Returns (h_new, o_new)."""
        pre = x @ self.W_in.T + h @ self.W_rec.T + self.b_rec
        h_new = torch.tanh(pre)
        o_new = h_new @ self.W_out.T + self.b_out
        return h_new, o_new

    def forward(self, inputs: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        inputs : (T, B, n_in)
        returns: outputs (T, B, n_out), hidden_states list of (B, n_rec) length T+1
        """
        T, B, _ = inputs.shape
        h = self.init_hidden(B, device=inputs.device)
        outputs = []
        hiddens = [h]
        for t in range(T):
            h, o = self.step(inputs[t], h)
            outputs.append(o)
            hiddens.append(h)
        return torch.stack(outputs, dim=0), hiddens
