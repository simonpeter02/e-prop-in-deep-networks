"""
Multi-layer LIF recurrent network.

Architecture for L layers:
  u^0_t = alpha·u^0_{t-1} + (1-alpha)·(W_rec^0 @ s^0_{t-1} + W_in @ x_t + b^0) − v_th·s^0_{t-1}
  s^0_t = H(u^0_t − v_th)

  u^l_t = alpha·u^l_{t-1} + (1-alpha)·(W_rec^l @ s^l_{t-1} + W_ff^l @ s^{l-1}_t + b^l) − v_th·s^l_{t-1}
  s^l_t = H(u^l_t − v_th)    for l = 1,...,L-1

  o_t   = W_out @ s^{L-1}_t + b_out

All spike functions use the piecewise-linear SpikeFn surrogate gradient, so BPTT
works via autograd through the full unrolled sequence.

Parameters per layer l:
  W_rec^l : (n_rec, n_rec)     spectral-norm init to 0.9
  W_in    : (n_rec, n_in)      [layer 0 only]
  W_ff^l  : (n_rec, n_rec)     [layers 1+], smaller init scale
  b_recs^l: (n_rec,)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from models.lif_rnn import SpikeFn


class DeepLIFNetwork(nn.Module):
    def __init__(
        self,
        n_in:        int,
        n_rec:       int,
        n_out:       int,
        n_layers:    int   = 2,
        alpha:       float = 0.9,
        v_th:        float = 0.1,
        gamma:       float = 0.3,
        w_in_scale:  float = 1.0,
    ):
        super().__init__()
        assert n_layers >= 1
        self.n_in     = n_in
        self.n_rec    = n_rec
        self.n_out    = n_out
        self.n_layers = n_layers
        self.alpha    = alpha
        self.v_th     = v_th
        self.gamma    = gamma

        # W_in is initialized with a larger scale than standard 1/sqrt(n_in) to
        # compensate for sparse inputs (e.g. SHD has ~1% sparsity across 700 channels,
        # giving drive std ≈ (1-alpha)*sqrt(p*n_in)*scale).  Default w_in_scale=1 keeps
        # the standard formula; callers should increase it for sparse-input tasks.
        self.W_in = nn.Parameter(
            torch.randn(n_rec, n_in) / n_in ** 0.5 * w_in_scale
        )

        W_recs, W_ffs, b_recs = [], [], []
        for l in range(n_layers):
            W_rec = torch.randn(n_rec, n_rec) / n_rec ** 0.5
            with torch.no_grad():
                sr = torch.linalg.eigvals(W_rec).abs().max().item()
                W_rec *= 0.9 / sr
            W_recs.append(nn.Parameter(W_rec))
            b_recs.append(nn.Parameter(torch.zeros(n_rec)))
            if l > 0:
                # Smaller feedforward init to avoid inter-layer explosion
                W_ffs.append(nn.Parameter(
                    torch.randn(n_rec, n_rec) / n_rec ** 0.5 * 0.5
                ))

        self.W_recs = nn.ParameterList(W_recs)   # len = n_layers
        self.W_ffs  = nn.ParameterList(W_ffs)    # len = n_layers - 1
        self.b_recs = nn.ParameterList(b_recs)   # len = n_layers

        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / n_rec ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(n_out))

    # ── Convenience accessors (mirrors DeepRNN interface) ─────────────────────
    def W_rec(self, l: int) -> Tensor:
        return self.W_recs[l]

    def W_ff(self, l: int) -> Tensor:
        """Feedforward weight for layer l >= 1 (connects layer l-1 → layer l)."""
        return self.W_ffs[l - 1]

    def bias(self, l: int) -> Tensor:
        return self.b_recs[l]

    # ── Forward ───────────────────────────────────────────────────────────────
    def init_hidden(self, batch_size: int, device=None) -> Tuple[List[Tensor], List[Tensor]]:
        dev = device or self.W_in.device
        u = [torch.zeros(batch_size, self.n_rec, device=dev) for _ in range(self.n_layers)]
        s = [torch.zeros(batch_size, self.n_rec, device=dev) for _ in range(self.n_layers)]
        return u, s

    def forward(self, inputs: Tensor) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        inputs : (T, B, n_in)

        Returns
        -------
        outputs : (T, B, n_out)
        state   : list of L tuples (u_seq, s_seq), each (T, B, n_rec)
                  u_seq is detached (for e-prop); s_seq is NOT detached (for BPTT).
        """
        T, B, _ = inputs.shape
        dev = inputs.device

        u, s = self.init_hidden(B, device=dev)
        u_lists = [[] for _ in range(self.n_layers)]
        s_lists = [[] for _ in range(self.n_layers)]
        o_list  = []

        for t in range(T):
            x_t   = inputs[t]
            s_new = []

            for l in range(self.n_layers):
                if l == 0:
                    ff_inp = x_t @ self.W_in.T
                else:
                    ff_inp = s_new[l - 1] @ self.W_ff(l).T   # s_new[l-1]: current-step spikes

                u_new = (self.alpha * u[l]
                         + (1 - self.alpha) * (s[l] @ self.W_rec(l).T + ff_inp + self.b_recs[l])
                         - self.v_th * s[l])
                s_l   = SpikeFn.apply(u_new - self.v_th, self.gamma)

                u_lists[l].append(u_new.detach())   # detached for e-prop bookkeeping
                s_lists[l].append(s_l)               # NOT detached — BPTT flows through here
                s_new.append(s_l)
                u[l] = u_new
                s[l] = s_l

            o = s_new[-1] @ self.W_out.T + self.b_out
            o_list.append(o)

        outputs = torch.stack(o_list)                                     # (T, B, n_out)
        state   = [
            (torch.stack(u_lists[l]), torch.stack(s_lists[l]))
            for l in range(self.n_layers)
        ]
        return outputs, state
