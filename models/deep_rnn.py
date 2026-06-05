"""
Multi-layer vanilla tanh RNN.

Architecture for L layers:
  h^1_t = tanh(W_rec^1 @ h^1_{t-1} + W_in  @ x_t   + b^1)
  h^l_t = tanh(W_rec^l @ h^l_{t-1} + W_ff^l @ h^{l-1}_t + b^l)   l = 2,...,L
  o_t   = W_out @ h^L_t + b_out

Parameters per layer l:
  W_rec^l : (n_rec, n_rec)
  W_in    : (n_rec, n_in)     [layer 1 only]
  W_ff^l  : (n_rec, n_rec)    [layers 2+]
  b^l     : (n_rec,)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


class DeepRNN(nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int, n_layers: int = 2):
        super().__init__()
        assert n_layers >= 1
        self.n_in     = n_in
        self.n_rec    = n_rec
        self.n_out    = n_out
        self.n_layers = n_layers

        # Input weights (layer 1 only)
        self.W_in = nn.Parameter(torch.randn(n_rec, n_in) / (n_in ** 0.5))

        # Per-layer recurrent weights, feedforward weights, biases
        W_recs, W_ffs, biases = [], [], []
        for l in range(n_layers):
            W_rec = torch.randn(n_rec, n_rec) / (n_rec ** 0.5)
            with torch.no_grad():
                sr = torch.linalg.eigvals(W_rec).abs().max().item()
                W_rec *= 0.9 / sr
            W_recs.append(nn.Parameter(W_rec))
            biases.append(nn.Parameter(torch.zeros(n_rec)))
            if l > 0:
                W_ffs.append(nn.Parameter(torch.randn(n_rec, n_rec) / (n_rec ** 0.5)))

        self.W_recs  = nn.ParameterList(W_recs)   # len = n_layers
        self.W_ffs   = nn.ParameterList(W_ffs)    # len = n_layers - 1
        self.biases  = nn.ParameterList(biases)   # len = n_layers

        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / (n_rec ** 0.5))
        self.b_out = nn.Parameter(torch.zeros(n_out))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def W_rec(self, l: int) -> Tensor:
        """Recurrent weight matrix for layer l (0-indexed)."""
        return self.W_recs[l]

    def W_ff(self, l: int) -> Tensor:
        """Feedforward weight matrix for layer l >= 1 (0-indexed)."""
        return self.W_ffs[l - 1]

    def bias(self, l: int) -> Tensor:
        return self.biases[l]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int, device=None) -> List[Tensor]:
        dev = device or self.W_in.device
        return [torch.zeros(batch_size, self.n_rec, device=dev)
                for _ in range(self.n_layers)]

    def step(self, x: Tensor, hs: List[Tensor]) -> Tuple[List[Tensor], Tensor]:
        """
        Single timestep.
        x   : (B, n_in)
        hs  : list of L tensors (B, n_rec), hidden states at t-1

        Returns new hidden list and output (B, n_out).
        """
        new_hs = []
        for l in range(self.n_layers):
            if l == 0:
                inp = x @ self.W_in.T
            else:
                inp = new_hs[l - 1] @ self.W_ff(l).T
            rec = hs[l] @ self.W_rec(l).T
            h_new = torch.tanh(inp + rec + self.bias(l))
            new_hs.append(h_new)
        o = new_hs[-1] @ self.W_out.T + self.b_out
        return new_hs, o

    def forward(self, inputs: Tensor) -> Tuple[Tensor, List[List[Tensor]]]:
        """
        inputs : (T, B, n_in)

        Returns:
          outputs       : (T, B, n_out)
          all_hiddens   : list of T+1 elements, each a list of L tensors (B, n_rec)
                          all_hiddens[0] is the initial (zero) state
        """
        T, B, _ = inputs.shape
        hs = self.init_hidden(B, device=inputs.device)
        outputs, all_hiddens = [], [hs]
        for t in range(T):
            hs, o = self.step(inputs[t], hs)
            outputs.append(o)
            all_hiddens.append(hs)
        return torch.stack(outputs, dim=0), all_hiddens
