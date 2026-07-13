"""
Cue accumulation (evidence accumulation) task.

The network observes a stream of brief left/right cue pulses separated by
silence, then must accumulate evidence across a long silent delay and report
which side received the majority of cues.

Two encodings of the same task live here:

  generate_batch()          — dense/analog, n_in = 5, one-hot targets + mask.
                              Time-major (T, B, ·). Used by the depth/alpha
                              sweeps and tests/sanity_checks.py.

  generate_poisson_batch()  — population-coded Bernoulli ("Poisson") spikes,
                              n_in = 40, integer labels, no mask. Batch-major
                              (B, T, ·). This is the encoding of Bellec et al.
                              (2020) and backs §1 of the main results notebook
                              (experiments/single_layer_cue_accum.py).

Both share the decision rule: whichever side had the STRICT majority of cues
wins. Ties (possible when n_cues is even) are broken uniformly at random;
odd n_cues eliminates them. Label 0 = "left wins", 1 = "right wins".


Variant 1 — dense, 5-channel (generate_batch)
---------------------------------------------
Input channels (n_in = 5):
  0 : left cue  (pulse of height 1.0 for cue_duration steps)
  1 : right cue (pulse of height 1.0 for cue_duration steps)
  2 : recall signal (active at the single decision step only)
  3 : noise  (i.i.d. Gaussian, std = noise_level, every step)
  4 : bias   (constant 1.0)

Output channels (n_out = 2):
  0 : "left wins"  (one-hot target at recall step)
  1 : "right wins" (one-hot target at recall step)

Timing layout (each box = cue_duration + inter_cue_interval steps):
  [cue1][cue2]...[cueN][----delay D----][recall]

  cue_window = n_cues * (cue_duration + inter_cue_interval)
  T = cue_window + delay + 1           (+1 is the decision/recall step)


Variant 2 — population-coded spikes (generate_poisson_batch)
------------------------------------------------------------
The n_in channels split into four equal populations of NC = n_in // 4 neurons:

  [0 : NC)      left-cue population   — fires at f_hi while a LEFT cue is on
  [NC : 2NC)    right-cue population  — fires at f_hi while a RIGHT cue is on
  [2NC : 3NC)   recall population     — fires at f_hi during the recall window
  [3NC : 4NC)   background/decoy      — background rate f_lo only

Every channel additionally fires at background rate f_lo at every step, so
"silence" is noisy rather than exactly zero. Spikes are Bernoulli(f) draws
(a Poisson process discretised to one step), hence the name.

Timing layout:
  [cue1][cue2]...[cueN][---- delay D ----][--- recall window ---]

  cue_window = n_cues * (cue_duration + inter_cue_interval)
  T = cue_window + delay + recall_duration

The readout is taken over the whole recall window (not a single step), so no
mask is returned — the caller knows the window is the last recall_duration
steps.

Design motivation (shared by both variants)
-------------------------------------------
The silent delay D is the central feature: the accumulated left/right count
must survive D steps with no external input.  This is only possible if
stored in the slow per-neuron leak state, not in transient firing activity.
D is therefore the primary difficulty knob.

A plain tanh RNN with alpha=1 has no per-neuron memory (state decays to 0
in one step of silence), so the task becomes unsolvable at large D.
A leaky RNN with alpha<1 has per-neuron time constants τ=1/(1-alpha),
enabling the count to survive D >> τ steps.

This creates the e-prop > d=0 wedge: e-prop's eligibility trace carries the
slow diagonal decay (1-alpha) forward in time, while d=0 discards it.

Unit sanity checks (in tests/sanity_checks.py):
  - Shapes: inputs (T,B,5), targets (T,B,2), mask (T,B)
  - Mask sums: mask.sum() == batch_size (one decision step per trial)
  - Label balance: P(label=0) ≈ 0.5 over many trials
  - Frozen-net accuracy: argmax of untrained outputs ≈ 50% chance
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

N_IN  = 5   # left, right, recall, noise, bias
N_OUT = 2   # 0=left wins, 1=right wins

# ── Defaults for the population-coded variant (Bellec et al. 2020 style) ─────
POISSON_N_IN  = 40   # 4 populations x 10 neurons
POISSON_N_OUT = 2    # 0=left wins, 1=right wins


def generate_batch(
    batch_size: int,
    n_cues: int = 5,
    delay: int = 20,
    cue_duration: int = 1,
    inter_cue_interval: int = 5,
    noise_level: float = 0.01,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate a batch of cue-accumulation trials.

    Parameters
    ----------
    batch_size           : number of independent trials
    n_cues               : number of cue pulses per trial (odd → no ties)
    delay                : silent gap between last cue and recall (steps)
    cue_duration         : active duration of each cue pulse (steps)
    inter_cue_interval   : silence after each cue pulse (steps)
    noise_level          : std of Gaussian noise on channel 3
    seed                 : integer seed for full reproducibility (None = random)
    device               : torch device string

    Returns
    -------
    inputs  : (T, B, 5)   float32  — left, right, recall, noise, bias
    targets : (T, B, 2)   float32  — one-hot at recall step only
    mask    : (T, B)       float32  — 1.0 at recall step, 0 elsewhere

    where T = n_cues*(cue_duration + inter_cue_interval) + delay + 1
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    cue_stride  = cue_duration + inter_cue_interval
    cue_window  = n_cues * cue_stride
    T           = cue_window + delay + 1   # last step is recall/decision
    B           = batch_size

    inputs  = torch.zeros(T, B, N_IN,  device=device)
    targets = torch.zeros(T, B, N_OUT, device=device)
    mask    = torch.zeros(T, B,        device=device)

    # Sample cue sides: 0 = left, 1 = right — shape (B, n_cues)
    cue_sides = torch.randint(0, 2, (B, n_cues), generator=gen)

    # Place cue pulses
    for c in range(n_cues):
        t_start = c * cue_stride
        for dt in range(cue_duration):
            t = t_start + dt
            # Channel 0 fires for left cues, channel 1 for right cues
            inputs[t, :, 0] = (cue_sides[:, c] == 0).float()
            inputs[t, :, 1] = (cue_sides[:, c] == 1).float()

    # Recall signal at the single decision step
    t_recall = cue_window + delay
    inputs[t_recall, :, 2] = 1.0

    # Gaussian noise on channel 3 across all timesteps
    if noise_level > 0.0:
        inputs[:, :, 3] = torch.randn(T, B, generator=gen) * noise_level

    # Bias channel always on
    inputs[:, :, 4] = 1.0

    # Labels: majority vote over cue sides
    left_count  = (cue_sides == 0).sum(dim=1).float()   # (B,)
    right_count = (cue_sides == 1).sum(dim=1).float()   # (B,)
    labels = torch.zeros(B, dtype=torch.long)

    # Strict majority
    labels[right_count > left_count] = 1

    # Ties: assign randomly (rare / absent with odd n_cues)
    tied = left_count == right_count
    if tied.any():
        n_tied = int(tied.sum().item())
        labels[tied] = torch.randint(0, 2, (n_tied,), generator=gen)

    # One-hot targets and mask at recall step
    targets[t_recall, torch.arange(B), labels] = 1.0
    mask[t_recall] = 1.0

    # Move to target device (sampling was on CPU for determinism with Generator)
    inputs  = inputs.to(device)
    targets = targets.to(device)
    mask    = mask.to(device)

    return inputs, targets, mask


def task_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correct trials (argmax over n_out) at masked timesteps.

    Parameters
    ----------
    logits  : (T, B, n_out) — network outputs (pre-softmax is fine)
    targets : (T, B, n_out) — one-hot targets
    mask    : (T, B)         — 1.0 at the decision step

    Returns
    -------
    accuracy in [0, 1]
    """
    pred    = logits.argmax(dim=-1)    # (T, B)
    tgt     = targets.argmax(dim=-1)   # (T, B)
    correct = ((pred == tgt) * mask).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0


# ── Metadata helpers ────────────────────────────────────────────────────────

def sequence_length(
    n_cues: int = 5,
    delay: int = 20,
    cue_duration: int = 1,
    inter_cue_interval: int = 5,
) -> int:
    """Return the total sequence length T for the given task parameters."""
    return n_cues * (cue_duration + inter_cue_interval) + delay + 1


# ── Population-coded (Poisson-spike) variant ────────────────────────────────
#
# Used by §1 of the main results notebook via experiments/single_layer_cue_accum.py.
# See the module docstring for the channel layout and timing.

def poisson_sequence_length(
    delay: int = 50,
    n_cues: int = 7,
    cue_duration: int = 15,
    inter_cue_interval: int = 5,
    recall_duration: int = 25,
) -> int:
    """Total trial length T for the population-coded variant.

    T = n_cues * (cue_duration + inter_cue_interval) + delay + recall_duration
    """
    return n_cues * (cue_duration + inter_cue_interval) + delay + recall_duration


def generate_poisson_batch(
    batch_size: int,
    delay: int = 50,
    n_cues: int = 7,
    cue_duration: int = 15,
    inter_cue_interval: int = 5,
    recall_duration: int = 25,
    f_hi: float = 0.25,
    f_lo: float = 0.05,
    n_in: int = POISSON_N_IN,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    """Generate a batch of population-coded cue-accumulation trials.

    Spikes are Bernoulli draws: every channel fires at the background rate
    `f_lo` at every step, and the population coding the currently-active event
    (left cue / right cue / recall) fires at the high rate `f_hi` instead.

    Parameters
    ----------
    batch_size          : number of independent trials
    delay               : silent gap between last cue and the recall window (steps)
    n_cues              : number of cue pulses per trial (odd → no ties)
    cue_duration        : active duration of each cue pulse (steps)
    inter_cue_interval  : silence after each cue pulse (steps)
    recall_duration     : length of the recall window at the end of the trial
    f_hi                : per-step spike probability of an active population
    f_lo                : per-step background spike probability of every channel
    n_in                : number of input channels; must be divisible by 4
    device              : torch device string

    Returns
    -------
    inputs : (B, T, n_in)  float32 — Bernoulli spikes, batch-major
    labels : (B,)           int64   — 0 = "left wins", 1 = "right wins"

    where T = poisson_sequence_length(delay, n_cues, cue_duration,
                                      inter_cue_interval, recall_duration)

    Notes
    -----
    Sampling uses the *global* torch RNG on `device`, so seeding with
    torch.manual_seed(seed) reproduces a batch exactly. This matches how the
    training loops in experiments/single_layer_cue_accum.py seed their runs.
    """
    if n_in % 4 != 0:
        raise ValueError(f"n_in must be divisible by 4 (got {n_in})")

    nc          = n_in // 4                       # neurons per population
    cue_stride  = cue_duration + inter_cue_interval
    T           = poisson_sequence_length(delay, n_cues, cue_duration,
                                          inter_cue_interval, recall_duration)
    B           = batch_size

    # Background firing everywhere, at rate f_lo
    inputs = (torch.rand(B, T, n_in, device=device) < f_lo).float()

    # Cue sides: 0 = left, 1 = right — shape (B, n_cues)
    cue_sides = (torch.rand(B, n_cues, device=device) < 0.5).long()

    # Overwrite the active cue population with the high rate, cue by cue.
    # Trials in the batch see different cue sides, so each side is filled
    # for the subset of trials showing it.
    for c in range(n_cues):
        t0 = c * cue_stride
        for side in (0, 1):
            sel = cue_sides[:, c] == side
            if sel.any():
                inputs[sel, t0:t0 + cue_duration, side * nc:(side + 1) * nc] = (
                    torch.rand(int(sel.sum()), cue_duration, nc, device=device) < f_hi
                ).float()

    # Recall population fires throughout the recall window
    inputs[:, -recall_duration:, 2 * nc:3 * nc] = (
        torch.rand(B, recall_duration, nc, device=device) < f_hi
    ).float()

    # Labels: strict majority of cue sides (1 = right). With odd n_cues there
    # are no ties, so the > comparison is exact.
    right_count = cue_sides.sum(dim=1)
    labels      = (right_count > n_cues // 2).long()

    # Ties (even n_cues only): break uniformly at random
    if n_cues % 2 == 0:
        tied = right_count == n_cues // 2
        if tied.any():
            labels[tied] = (torch.rand(int(tied.sum()), device=device) < 0.5).long()

    return inputs, labels


def poisson_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Fraction of correct trials for the population-coded variant.

    logits : (B, n_out) — one decision per trial (pre-softmax is fine)
    labels : (B,)        — integer class labels
    """
    return (logits.argmax(dim=-1) == labels).float().mean().item()
