"""
Single-layer LIF and Adaptive-LIF (ALIF) recurrent networks.

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

import math
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


class LIFHeteroNetwork(nn.Module):
    """
    LIF recurrent network with heterogeneous membrane time constants.

    Each neuron i has its own decay constant alpha_i, spaced log-uniformly
    from alpha_min to alpha_max:

        alpha_i = exp( linspace(log(alpha_min), log(alpha_max), n_rec)[i] )

    This gives membrane time constants ranging from
        tau_min = -1 / log(alpha_min)  to  tau_max = -1 / log(alpha_max)

    Example (alpha_min=0.9, alpha_max=0.999):
        tau_min ≈ 10 steps,   tau_max ≈ 1000 steps

    The e-prop carry is already per-neuron in eprop_lif.py:
        c_t[b,i] = alpha_i − v_th * psi_{t-1}[b,i]
    so the existing learning rule works without modification as long as
    eprop_lif handles a tensor (not scalar) alpha — which it does after
    the corresponding update to compute_eprop_lif_gradients.

    Biologically, neurons in cortex span at least two decades in membrane
    time constant (10–1000 ms).  A uniform-alpha LIF is a special case;
    this class makes the distribution explicit and learnable if desired.
    """

    def __init__(
        self,
        n_in:      int,
        n_rec:     int,
        n_out:     int,
        alpha_min: float = 0.9,
        alpha_max: float = 0.999,
        v_th:      float = 0.1,
        gamma:     float = 0.3,
    ):
        super().__init__()
        self.n_in  = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.v_th  = v_th
        self.gamma = gamma

        log_a = torch.linspace(math.log(alpha_min), math.log(alpha_max), n_rec)
        self.register_buffer('alpha', torch.exp(log_a))   # (n_rec,) — not a learned param

        self.W_in  = nn.Parameter(torch.randn(n_rec, n_in)  / n_in  ** 0.5)
        self.W_rec = nn.Parameter(torch.randn(n_rec, n_rec) / n_rec ** 0.5)
        self.b_rec = nn.Parameter(torch.zeros(n_rec))
        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / n_rec ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(n_out))

        with torch.no_grad():
            sr = torch.linalg.eigvals(self.W_rec).abs().max().item()
            self.W_rec.data *= 0.9 / sr

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tuple]:
        """
        inputs : (T, B, n_in)

        Returns
        -------
        outputs : (T, B, n_out)
        state   : (u_seq, s_seq)  each (T, B, n_rec)
        """
        T, B, _ = inputs.shape
        dev   = inputs.device
        alpha = self.alpha   # (n_rec,) — broadcasts over batch

        u = torch.zeros(B, self.n_rec, device=dev)
        s = torch.zeros(B, self.n_rec, device=dev)

        u_list, s_list, o_list = [], [], []

        for t in range(T):
            x = inputs[t]
            u_new = (alpha * u
                     + (1 - alpha) * (s @ self.W_rec.T + x @ self.W_in.T + self.b_rec)
                     - self.v_th * s)
            s_new = SpikeFn.apply(u_new - self.v_th, self.gamma)
            o     = s_new @ self.W_out.T + self.b_out

            u, s = u_new, s_new
            u_list.append(u_new.detach())
            s_list.append(s_new)
            o_list.append(o)

        outputs = torch.stack(o_list)
        u_seq   = torch.stack(u_list)
        s_seq   = torch.stack(s_list)
        return outputs, (u_seq, s_seq)


class ALIFNetwork(nn.Module):
    """
    Adaptive LIF (ALIF) recurrent network (Bellec et al. 2020).

    Adds a per-neuron slow adaptation variable a_t that raises the spike
    threshold after each spike.  E-prop must track a slow eligibility trace
    for this variable; d=0 misses it entirely, creating a large e-prop > d=0
    gap at delays longer than the LIF membrane horizon (~2-3 steps at carry≈0.6).

    Dynamics
    --------
      a_t  = rho * a_{t-1} + s_{t-1}               # slow adaptation trace
      θ_t  = v_th + beta * a_{t-1}                  # adaptive threshold (uses a at t-1)
      u_t  = alpha * u_{t-1}
             + (1-alpha) * (W_rec @ s_{t-1} + W_in @ x_t + b)
             - θ_{t-1} * s_{t-1}                    # soft reset at adaptive threshold
      s_t  = H(u_t - θ_t)

    Parameters
    ----------
    rho  : float, adaptation decay ∈ (0,1).  rho=0.9 → τ_a≈10 steps;
           rho=0.99 → τ_a≈100 steps.
    beta : float, adaptation strength (threshold rise per accumulated spike).
           beta=0 recovers plain LIF.  beta=0.07 is a typical value from Bellec 2020.
    """

    def __init__(
        self,
        n_in:  int,
        n_rec: int,
        n_out: int,
        alpha: float = 0.9,
        rho:   float = 0.9,
        beta:  float = 0.07,
        v_th:  float = 0.1,
        gamma: float = 0.3,
    ):
        super().__init__()
        self.n_in  = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.alpha = alpha
        self.rho   = rho
        self.beta  = beta
        self.v_th  = v_th
        self.gamma = gamma

        self.W_in  = nn.Parameter(torch.randn(n_rec, n_in)  / n_in  ** 0.5)
        self.W_rec = nn.Parameter(torch.randn(n_rec, n_rec) / n_rec ** 0.5)
        self.b_rec = nn.Parameter(torch.zeros(n_rec))
        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / n_rec ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(n_out))

        with torch.no_grad():
            sr = torch.linalg.eigvals(self.W_rec).abs().max().item()
            self.W_rec.data *= 0.9 / sr

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tuple]:
        """
        inputs : (T, B, n_in)

        Returns
        -------
        outputs : (T, B, n_out)
        state   : (u_seq, s_seq, a_seq)  each (T, B, n_rec)
                  a_seq is the adaptation trace (needed by compute_eprop_alif_gradients)
        """
        T, B, _ = inputs.shape
        dev  = inputs.device
        v_th = self.v_th

        u = torch.zeros(B, self.n_rec, device=dev)
        s = torch.zeros(B, self.n_rec, device=dev)
        a = torch.zeros(B, self.n_rec, device=dev)  # adaptation trace

        u_list, s_list, a_list, o_list = [], [], [], []

        for t in range(T):
            x     = inputs[t]
            theta = v_th + self.beta * a                           # adaptive threshold (B, n_rec)

            u_new = (self.alpha * u
                     + (1 - self.alpha) * (s @ self.W_rec.T + x @ self.W_in.T + self.b_rec)
                     - theta * s)                                  # soft reset at theta
            a_new = self.rho * a + s                               # update adaptation

            s_new = SpikeFn.apply(u_new - theta, self.gamma)       # spike against adaptive θ
            o     = s_new @ self.W_out.T + self.b_out

            u, s, a = u_new, s_new, a_new
            u_list.append(u_new.detach())
            s_list.append(s_new)
            a_list.append(a_new.detach())
            o_list.append(o)

        outputs = torch.stack(o_list)
        u_seq   = torch.stack(u_list)
        s_seq   = torch.stack(s_list)
        a_seq   = torch.stack(a_list)
        return outputs, (u_seq, s_seq, a_seq)
