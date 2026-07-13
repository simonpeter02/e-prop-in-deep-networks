"""
Experiment 1 (feasibility check): single-layer e-prop vs BPTT on cue accumulation.

Backs §1 / Figures 1.1-1.3 of the main results notebook (technical note §2.1).

Question
--------
How well does single-layer e-prop track exact BPTT on a task that requires
holding evidence across a silent delay, and how does that alignment decay as
the delay grows?

Model
-----
A single-layer leaky tanh RNN, held as a plain dict of tensors rather than an
nn.Module so that the e-prop and BPTT gradients can be compared on exactly the
same parameters:

  h_t    = (1 - α) h_{t-1} + α tanh(W_rec h_{t-1} + W_in x_t)
  logits = W_out * mean(h_t over the recall window) + b_out

The readout averages the hidden state over the whole recall window and emits
ONE decision per trial (cross-entropy against the majority-side label). This
differs from models/leaky_rnn.py + learning_rules/eprop.py, which emit a
per-step output and mask the loss; hence the dedicated implementation here.

Learning rules
--------------
eprop_gradients  : symmetric e-prop. The eligibility trace is carried forward
                   with the DIAGONAL local Jacobian only,
                     J_t[i,i] = (1-α) + α ψ_t[i] · W_rec[i,i],
                   where ψ_t = 1 - tanh²(pre_t). The learning signal is the
                   readout error projected back through W_out (symmetric
                   feedback). O(n²) memory, no backward pass through time.

bptt_loss        : exact BPTT via autograd through the unrolled net — the
                   ground truth the e-prop gradient is scored against.

Run:
    python -m experiments.single_layer_cue_accum
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import numpy as np
from torch import Tensor

from tasks.cue_accumulation import (generate_poisson_batch, poisson_sequence_length,
                                    POISSON_N_IN, POISSON_N_OUT)

Params = Dict[str, Tensor]


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    """Task + model + optimiser settings for the single-layer experiment."""
    # task
    n_in:               int   = POISSON_N_IN
    n_out:              int   = POISSON_N_OUT
    n_cues:             int   = 7
    cue_duration:       int   = 15
    inter_cue_interval: int   = 5
    recall_duration:    int   = 25
    f_hi:               float = 0.25
    f_lo:               float = 0.05
    # model
    n_rec:              int   = 100
    alpha:              float = 0.005   # leak rate; τ = 1/(1-α) ≈ 200 steps
    # optimiser (Adam; readout learns faster than the recurrent core)
    lr_rec:             float = 1e-3    # W_in, W_rec
    lr_out:             float = 3e-3    # W_out, b_out
    device:             str   = "cpu"

    @property
    def task_kwargs(self) -> dict:
        return dict(n_cues=self.n_cues, cue_duration=self.cue_duration,
                    inter_cue_interval=self.inter_cue_interval,
                    recall_duration=self.recall_duration,
                    f_hi=self.f_hi, f_lo=self.f_lo,
                    n_in=self.n_in, device=self.device)

    def seq_len(self, delay: int) -> int:
        return poisson_sequence_length(delay, self.n_cues, self.cue_duration,
                                       self.inter_cue_interval, self.recall_duration)

    def batch(self, batch_size: int, delay: int) -> Tuple[Tensor, Tensor]:
        """Draw a batch of trials; see tasks.cue_accumulation."""
        return generate_poisson_batch(batch_size, delay, **self.task_kwargs)


def default_config(device: str = None) -> Config:
    """Config on the best available device (cuda if present)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return Config(device=device)


# ── Model ────────────────────────────────────────────────────────────────────

def init_params(seed: int, cfg: Config) -> Params:
    """Fresh weights, seeded on CPU so a seed gives the same net on any device."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = cfg.device
    return {
        "W_in":  (torch.randn(cfg.n_rec, cfg.n_in,  generator=g) / math.sqrt(cfg.n_in )).to(dev),
        "W_rec": (torch.randn(cfg.n_rec, cfg.n_rec, generator=g) / math.sqrt(cfg.n_rec)).to(dev),
        "W_out": (torch.randn(cfg.n_out, cfg.n_rec, generator=g) / math.sqrt(cfg.n_rec)).to(dev),
        "b_out": torch.zeros(cfg.n_out, device=dev),
    }


def cosine(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two gradient tensors (flattened)."""
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


# ── Learning rule: e-prop ────────────────────────────────────────────────────

@torch.no_grad()
def eprop_gradients(P: Params, x: Tensor, labels: Tensor, delay: int,
                    cfg: Config) -> Tuple[Params, float, float]:
    """Symmetric e-prop gradients for the leaky tanh RNN — no autograd.

    x      : (B, T, n_in)
    labels : (B,)

    Returns (grads, loss, accuracy).
    """
    T, B  = cfg.seq_len(delay), x.shape[0]
    dev   = cfg.device
    alpha = cfg.alpha
    n_rec, n_in, n_out = cfg.n_rec, cfg.n_in, cfg.n_out
    T_recall = cfg.recall_duration

    W_rec_diag = torch.diag(P["W_rec"])            # (n_rec,) — the only recurrent
                                                    # term e-prop keeps

    h       = torch.zeros(B, n_rec,        device=dev)
    eps_in  = torch.zeros(B, n_rec, n_in,  device=dev)   # eligibility traces
    eps_rec = torch.zeros(B, n_rec, n_rec, device=dev)
    E_in    = torch.zeros(B, n_rec, n_in,  device=dev)   # traces summed over the
    E_rec   = torch.zeros(B, n_rec, n_rec, device=dev)   # recall window
    h_sum   = torch.zeros(B, n_rec,        device=dev)

    for t in range(T):
        xt, h_prev = x[:, t], h
        pre = h @ P["W_rec"].t() + xt @ P["W_in"].t()
        h   = (1 - alpha) * h + alpha * torch.tanh(pre)

        # ψ_t (surrogate derivative) and the diagonal local Jacobian
        d_h       = alpha * (1 - torch.tanh(pre) ** 2)
        local_jac = (1 - alpha) + d_h * W_rec_diag.unsqueeze(0)

        eps_in  = local_jac.unsqueeze(2) * eps_in  + d_h.unsqueeze(2) * xt.unsqueeze(1)
        eps_rec = local_jac.unsqueeze(2) * eps_rec + d_h.unsqueeze(2) * h_prev.unsqueeze(1)

        if t >= T - T_recall:
            E_in  += eps_in
            E_rec += eps_rec
            h_sum += h

    h_bar  = h_sum / T_recall                       # readout state (B, n_rec)
    logits = h_bar @ P["W_out"].t() + P["b_out"]

    # Learning signal: softmax error fed back through W_out (symmetric feedback)
    p   = torch.softmax(logits, dim=1)
    oh  = torch.zeros(B, n_out, device=dev)
    oh[torch.arange(B), labels] = 1.0
    err = p - oh                                    # (B, n_out)
    L   = (err @ P["W_out"]) / T_recall             # (B, n_rec)

    grads = {
        "W_in":  (L.unsqueeze(2) * E_in ).sum(0) / B,
        "W_rec": (L.unsqueeze(2) * E_rec).sum(0) / B,
        "W_out": (err.t() @ h_bar) / B,
        "b_out": err.mean(0),
    }
    loss = torch.nn.functional.cross_entropy(logits, labels).item()
    acc  = (logits.argmax(1) == labels).float().mean().item()
    return grads, loss, acc


# ── Learning rule: BPTT (ground truth) ───────────────────────────────────────

def bptt_loss(P: Params, x: Tensor, labels: Tensor, delay: int,
              cfg: Config) -> Tuple[Tensor, float]:
    """Exact BPTT: autograd through the unrolled net. Returns (loss, accuracy).

    The loss is returned as a live tensor — call .backward() on it to populate
    P[k].grad (P must have requires_grad=True).
    """
    T, B  = cfg.seq_len(delay), x.shape[0]
    alpha = cfg.alpha

    h     = torch.zeros(B, cfg.n_rec, device=cfg.device)
    h_sum = torch.zeros(B, cfg.n_rec, device=cfg.device)
    for t in range(T):
        pre = h @ P["W_rec"].t() + x[:, t] @ P["W_in"].t()
        h   = (1 - alpha) * h + alpha * torch.tanh(pre)
        if t >= T - cfg.recall_duration:
            h_sum = h_sum + h

    logits = (h_sum / cfg.recall_duration) @ P["W_out"].t() + P["b_out"]
    loss   = torch.nn.functional.cross_entropy(logits, labels)
    acc    = (logits.argmax(1) == labels).float().mean().item()
    return loss, acc


def bptt_gradients(P: Params, x: Tensor, labels: Tensor, delay: int,
                   cfg: Config) -> Tuple[Params, float, float]:
    """BPTT gradients on a detached copy of P — mirrors eprop_gradients()."""
    Pg = {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}
    loss, acc = bptt_loss(Pg, x, labels, delay, cfg)
    loss.backward()
    return {k: v.grad for k, v in Pg.items()}, loss.item(), acc


# ── Training ─────────────────────────────────────────────────────────────────

RULES = ("eprop", "bptt")


def train(rule: str, seed: int, delay: int, iters: int, batch: int,
          eval_every: int, cfg: Config, eval_batch: int = 200
          ) -> Tuple[List[int], List[float]]:
    """Train one net with `rule` ('eprop' or 'bptt'); Adam, held-out eval.

    Returns (eval_steps, eval_accuracies).
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES} (got {rule!r})")

    torch.manual_seed(seed)                 # fixes both init and the task stream
    P = init_params(seed, cfg)
    for k in P:
        P[k].requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [P["W_in"], P["W_rec"]],  "lr": cfg.lr_rec},
        {"params": [P["W_out"], P["b_out"]], "lr": cfg.lr_out},
    ])

    steps: List[int] = []
    accs:  List[float] = []
    for it in range(iters):
        x, labels = cfg.batch(batch, delay)

        opt.zero_grad()
        if rule == "eprop":
            grads, _, _ = eprop_gradients(P, x, labels, delay, cfg)
            for k in P:
                P[k].grad = grads[k]
        else:
            loss, _ = bptt_loss(P, x, labels, delay, cfg)
            loss.backward()
        opt.step()

        if it % eval_every == 0 or it == iters - 1:
            with torch.no_grad():
                xv, lv = cfg.batch(eval_batch, delay)
                if rule == "eprop":
                    _, _, acc = eprop_gradients(P, xv, lv, delay, cfg)
                else:
                    _, acc = bptt_loss(P, xv, lv, delay, cfg)
            steps.append(it)
            accs.append(acc)

    return steps, accs


def steps_to_threshold(steps: List[int], accs: List[float],
                       thresh: float = 0.90) -> Tuple[int, bool]:
    """First eval step at which held-out accuracy crosses `thresh`.

    Returns (step, reached). If the threshold is never reached, returns the
    final step and reached=False — a censored lower bound, not a real crossing.
    """
    for s, a in zip(steps, accs):
        if a >= thresh:
            return s, True
    return steps[-1], False


# ── Delay sweep: gradient cosine vs BPTT ─────────────────────────────────────

def _mean_ci95(vals) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    mean = vals.mean()
    sem  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return float(mean), float(1.96 * sem)


def delay_sweep_point(delay: int, cfg: Config, n_avg: int = 8, batch: int = 64,
                      seed0: int = 1000) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Cosine(e-prop, BPTT) at one delay, averaged over `n_avg` fresh nets/batches.

    Reports the two ends of the network separately:
      cos_full : input-adjacent params (W_in + W_rec) — the ones whose credit
                 must travel back through time; this is where e-prop approximates.
      cos_out  : output-adjacent params (W_out) — e-prop is exact here, so this
                 acts as a control that should stay ≈ 1 at every delay.

    Returns ((cos_full_mean, ci95), (cos_out_mean, ci95)).
    """
    cos_full, cos_out = [], []
    for s in range(n_avg):
        P = init_params(seed0 + s, cfg)
        x, labels = cfg.batch(batch, delay)

        g_ep, _, _ = eprop_gradients(P, x, labels, delay, cfg)
        g_bp, _, _ = bptt_gradients(P, x, labels, delay, cfg)

        ep_vec = torch.cat([g_ep["W_in"].flatten(), g_ep["W_rec"].flatten()])
        bp_vec = torch.cat([g_bp["W_in"].flatten(), g_bp["W_rec"].flatten()])

        cos_full.append(cosine(ep_vec, bp_vec))
        cos_out.append(cosine(g_ep["W_out"], g_bp["W_out"]))

    return _mean_ci95(cos_full), _mean_ci95(cos_out)


# ── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = default_config()
    print("device:", cfg.device, "| trial length (delay=50):", cfg.seq_len(50))

    # Gradient agreement at init: e-prop vs BPTT, per parameter
    P = init_params(0, cfg)
    x, labels = cfg.batch(64, delay=50)
    g_ep, _, _ = eprop_gradients(P, x, labels, 50, cfg)
    g_bp, _, _ = bptt_gradients(P, x, labels, 50, cfg)

    print("\nGradient cosine vs BPTT (untrained network, delay=50):")
    for k in P:
        print(f"  {k:6s}  e-prop cos = {cosine(g_ep[k], g_bp[k]):+.4f}")
