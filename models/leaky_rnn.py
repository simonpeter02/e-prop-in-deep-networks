"""
Leaky-integrator tanh RNN with optional per-neuron (heterogeneous) time constants.

Update rule:
  h_t = (1 - α) * h_{t-1}  +  α * tanh(W_rec @ h_{t-1} + W_in @ x_t + b_rec)
  o_t = W_out @ h_t + b_out

where α ∈ (0, 1] is the integration rate (related to time constant τ = 1/α).

Diagonal Jacobian (used by e-prop eligibility trace):
  ∂h_t[i]/∂h_{t-1}[i] = (1 - α_i) + α_i * ψ_raw_t[i] * W_rec[i,i]

  For small α (e.g. α=0.1): dominant carry ≈ (1-α) = 0.9 → trace survives
  ~1/(1-α) = 10 steps; e-prop keeps this while d=0 sets carry=0.
  For α=1 (no leak): reduces to VanillaRNN; carry ≈ ψ * W_diag ≈ 0.005 → minimal.

Per-neuron (heterogeneous) alphas
----------------------------------
Pass alpha_min / alpha_max to spread time constants log-uniformly across neurons.
Neurons with small α have long time constants (slow, persistent integration).
Neurons with large α ≈ 1 behave like vanilla tanh (fast, no memory).
α is registered as a non-trainable buffer of shape (n_rec,) in both cases.

Compatibility
-------------
compute_eprop_leaky_gradients (learning_rules/eprop.py) already handles α as
either a scalar float or a (n_rec,) tensor; both broadcast correctly.
BPTT via autograd works automatically since α is a buffer (not a Parameter).
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class LeakyRNN(nn.Module):
    """Leaky-integrator tanh RNN; α may be uniform or per-neuron.

    Parameters
    ----------
    n_in        : input dimension
    n_rec       : number of recurrent units
    n_out       : output dimension
    alpha       : integration rate — scalar float for uniform α, or a pre-built
                  Tensor of shape (n_rec,) for per-neuron α.
                  Ignored when alpha_min and alpha_max are both given.
    alpha_min   : minimum α for log-uniform per-neuron distribution
    alpha_max   : maximum α for log-uniform per-neuron distribution
                  (if both are provided, overrides the alpha argument)
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        alpha: Union[float, Tensor] = 0.1,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ):
        super().__init__()
        self.n_in  = n_in
        self.n_rec = n_rec
        self.n_out = n_out

        # ── Build alpha vector ───────────────────────────────────────────────
        if alpha_min is not None and alpha_max is not None:
            assert 0.0 < alpha_min <= alpha_max <= 1.0
            log_alpha = torch.linspace(math.log(alpha_min), math.log(alpha_max), n_rec)
            alpha_vec = torch.exp(log_alpha)
        elif isinstance(alpha, Tensor):
            assert alpha.shape == (n_rec,), f"alpha Tensor must have shape ({n_rec},)"
            alpha_vec = alpha.float().clone()
        else:
            alpha_f = float(alpha)
            assert 0.0 < alpha_f <= 1.0, "alpha must be in (0, 1]"
            alpha_vec = torch.full((n_rec,), alpha_f)

        # Register as non-trainable buffer so it moves with .to(device) and
        # is included in state_dict (for checkpoint reproducibility), but
        # gradients are never computed through it.
        self.register_buffer("alpha", alpha_vec)   # (n_rec,)

        # ── Weights ──────────────────────────────────────────────────────────
        W_rec = torch.randn(n_rec, n_rec) / (n_rec ** 0.5)
        with torch.no_grad():
            sr = torch.linalg.eigvals(W_rec).abs().max().item()
            W_rec *= 0.9 / sr   # spectral radius ≈ 0.9
        self.W_rec = nn.Parameter(W_rec)

        self.W_in  = nn.Parameter(torch.randn(n_rec, n_in)  / (n_in  ** 0.5))
        self.b_rec = nn.Parameter(torch.zeros(n_rec))
        self.W_out = nn.Parameter(torch.randn(n_out, n_rec) / (n_rec ** 0.5))
        self.b_out = nn.Parameter(torch.zeros(n_out))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device=None) -> Tensor:
        dev = device if device is not None else self.W_rec.device
        return torch.zeros(batch_size, self.n_rec, device=dev)

    # ── Core dynamics ────────────────────────────────────────────────────────

    def step(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """One time step.

        x : (B, n_in)
        h : (B, n_rec)   hidden state at t-1

        Returns new hidden state (B, n_rec) and output (B, n_out).
        """
        pre   = x @ self.W_in.T + h @ self.W_rec.T + self.b_rec
        h_new = (1.0 - self.alpha) * h + self.alpha * torch.tanh(pre)
        o_new = h_new @ self.W_out.T + self.b_out
        return h_new, o_new

    def forward(self, inputs: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Full sequence forward pass.

        inputs : (T, B, n_in)

        Returns
        -------
        outputs : (T, B, n_out)
        hiddens : list of T+1 tensors (B, n_rec) — h[0] is the initial zero state
        """
        T, B, _ = inputs.shape
        h = self.init_hidden(B, device=inputs.device)
        outputs: List[Tensor] = []
        hiddens: List[Tensor] = [h]
        for t in range(T):
            h, o = self.step(inputs[t], h)
            outputs.append(o)
            hiddens.append(h)
        return torch.stack(outputs, dim=0), hiddens
