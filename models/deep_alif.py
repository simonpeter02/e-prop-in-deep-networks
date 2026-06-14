"""
Multi-layer Adaptive LIF (ALIF) recurrent network.

Architecture for L layers:
  theta^0_t = v_th + beta * a^0_{t-1}
  u^0_t = alpha*u^0_{t-1} + (1-alpha)*(W_rec^0 @ s^0_{t-1} + W_in @ x_t + b^0) - theta^0_t * s^0_{t-1}
  a^0_t = rho * a^0_{t-1} + s^0_{t-1}
  s^0_t = H(u^0_t - theta^0_t)

  theta^l_t = v_th + beta * a^l_{t-1}
  u^l_t = alpha*u^l_{t-1} + (1-alpha)*(W_rec^l @ s^l_{t-1} + W_ff^l @ s^{l-1}_t + b^l) - theta^l_t * s^l_{t-1}
  a^l_t = rho * a^l_{t-1} + s^l_{t-1}
  s^l_t = H(u^l_t - theta^l_t)    for l = 1,...,L-1

  o_t = W_out @ s^{L-1}_t + b_out

The adaptation a^l gives e-prop a slow eligibility trace (decay rho) that provides
temporal credit far beyond the fast membrane trace (decay alpha - v_th*psi ~ 0.87).

For BPTT: a[l] in the running state is NOT detached — gradients flow through theta
into the adaptation chain. Only the stored a_lists entries are detached.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from models.lif_rnn import SpikeFn


class DeepALIFNetwork(nn.Module):
    def __init__(
        self,
        n_in:       int,
        n_rec:      int,
        n_out:      int,
        n_layers:   int   = 2,
        alpha:      float = 0.9,
        rho:        float = 0.98,
        beta:       float = 0.02,
        v_th:       float = 0.1,
        gamma:      float = 0.3,
        w_in_scale: float = 1.0,
    ):
        super().__init__()
        assert n_layers >= 1
        self.n_in     = n_in
        self.n_rec    = n_rec
        self.n_out    = n_out
        self.n_layers = n_layers
        self.alpha    = alpha
        self.rho      = rho
        self.beta     = beta
        self.v_th     = v_th
        self.gamma    = gamma

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
                W_ffs.append(nn.Parameter(
                    torch.randn(n_rec, n_rec) / n_rec ** 0.5 * 0.5
                ))

        self.W_recs = nn.ParameterList(W_recs)
        self.W_ffs  = nn.ParameterList(W_ffs)
        self.b_recs = nn.ParameterList(b_recs)

        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / n_rec ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(n_out))

    def W_rec(self, l: int) -> Tensor:
        return self.W_recs[l]

    def W_ff(self, l: int) -> Tensor:
        return self.W_ffs[l - 1]

    def bias(self, l: int) -> Tensor:
        return self.b_recs[l]

    def init_hidden(self, batch_size: int, device=None):
        dev = device or self.W_in.device
        u = [torch.zeros(batch_size, self.n_rec, device=dev) for _ in range(self.n_layers)]
        s = [torch.zeros(batch_size, self.n_rec, device=dev) for _ in range(self.n_layers)]
        a = [torch.zeros(batch_size, self.n_rec, device=dev) for _ in range(self.n_layers)]
        return u, s, a

    def forward(self, inputs: Tensor) -> Tuple[Tensor, List[Tuple[Tensor, Tensor, Tensor]]]:
        """
        inputs : (T, B, n_in)

        Returns
        -------
        outputs : (T, B, n_out)
        state   : list of L tuples (u_seq, s_seq, a_seq), each (T, B, n_rec)
                  u_seq and a_seq are detached (for e-prop); s_seq is NOT detached (for BPTT).
                  The running a[l] is NOT detached so BPTT gradients flow through theta.
        """
        T, B, _ = inputs.shape
        dev = inputs.device

        u, s, a = self.init_hidden(B, device=dev)
        u_lists = [[] for _ in range(self.n_layers)]
        s_lists = [[] for _ in range(self.n_layers)]
        a_lists = [[] for _ in range(self.n_layers)]
        o_list  = []

        for t in range(T):
            x_t   = inputs[t]
            s_new = []

            for l in range(self.n_layers):
                theta = self.v_th + self.beta * a[l]   # adaptive threshold (B, n_rec)

                if l == 0:
                    ff_inp = x_t @ self.W_in.T
                else:
                    ff_inp = s_new[l - 1] @ self.W_ff(l).T

                u_new = (self.alpha * u[l]
                         + (1 - self.alpha) * (s[l] @ self.W_rec(l).T + ff_inp + self.b_recs[l])
                         - theta * s[l])
                a_new = self.rho * a[l] + s[l]         # a[l] here is s_{t-1} at layer l
                s_l   = SpikeFn.apply(u_new - theta, self.gamma)

                u_lists[l].append(u_new.detach())
                s_lists[l].append(s_l)
                a_lists[l].append(a_new.detach())
                s_new.append(s_l)
                u[l] = u_new
                s[l] = s_l
                a[l] = a_new    # NOT detached — BPTT gradient flows through theta_next

            o = s_new[-1] @ self.W_out.T + self.b_out
            o_list.append(o)

        outputs = torch.stack(o_list)
        state   = [
            (torch.stack(u_lists[l]), torch.stack(s_lists[l]), torch.stack(a_lists[l]))
            for l in range(self.n_layers)
        ]
        return outputs, state
