"""
Cue accumulation (evidence accumulation) task.

The network observes a stream of brief left/right cue pulses separated by
silence, then must accumulate evidence across a long silent delay and report
which side received the majority of cues at a single decision step.

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

Design motivation
-----------------
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

Decision rule: whichever side had the STRICT majority of cues wins.
Ties (possible when n_cues is even) are broken uniformly at random.
Using odd n_cues (default n_cues=5) eliminates ties.

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
