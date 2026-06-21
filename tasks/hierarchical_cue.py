"""
Hierarchical "classify-then-count" cue task (temporal-motif version).

A compositional variant of cue accumulation designed so that *both* depth and
time credit are genuinely required — making it the decisive test of whether
deep e-prop assigns credit across depth and time **simultaneously**.

Task structure
--------------
Each trial presents n_cues cues separated by silence, then a long silent delay,
then a single decision step.  Each cue is a short *temporal motif* on a single
feature channel, presented over cue_duration steps:

      side 0 ("rising")  : a ramp that increases over the cue window
      side 1 ("falling") : a ramp that decreases over the cue window

Crucially the two motifs have the SAME mean (≈0) and the SAME energy — they
differ only in their temporal ORDER (the sign of the derivative).  The trial
label is the majority cue side over the n_cues cues (odd n_cues ⇒ no ties).

Why this needs depth AND time simultaneously
--------------------------------------------
  * CLASSIFY (depth + within-layer time):  because the motifs are mean-zero,
    no static readout — and in particular no *random* (un-learned) feature
    projection — can separate them; a learner must detect the temporal pattern.
    This forces the LOWER layer to learn a genuine temporal feature extractor
    (its within-layer eligibility trace ϵ^h does the per-cue integration).
    A frozen random lower layer cannot do this, so removing its credit
    (ablate_spatial) breaks the task.
  * COUNT (cross-layer time):  each per-cue classification, produced transiently
    by the lower layer, must be accumulated by the (slow) TOP layer and held
    through the silent delay.  The credit for a lower-layer parameter from an
    early cue must therefore travel UP (depth) and FORWARD (time) through the
    top layer's recurrence — i.e. through the cross-layer temporal trace ϵ^z.
    Removing that carry (ablate_temporal) starves early cues of credit.

Recommended architecture: a 2-layer DeepRNN with a FAST lower layer (transient
motif detector, e.g. α≈0.7) and a SLOW top layer (integrator, e.g. α≈0.1).

Input channels (n_in = 5):
  0 : feature   (rising/falling ramp during a cue + Gaussian noise; 0 otherwise)
  1 : distractor feature (pure Gaussian noise — must be ignored)
  2 : recall signal (active at the single decision step only)
  3 : noise  (i.i.d. Gaussian every step — distractor channel)
  4 : bias   (constant 1.0)

Output channels (n_out = 2):  0 = "rising majority", 1 = "falling majority".
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

N_IN  = 5   # feature, distractor, recall, noise, bias
N_OUT = 2   # 0 = rising-majority, 1 = falling-majority


def generate_batch(
    batch_size: int,
    n_cues: int = 5,
    delay: int = 20,
    cue_duration: int = 5,
    inter_cue_interval: int = 4,
    amp: float = 1.0,
    feature_noise: float = 0.3,
    noise_level: float = 0.01,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate a batch of hierarchical temporal-motif trials.

    Parameters
    ----------
    batch_size         : number of independent trials
    n_cues             : number of cues per trial (odd ⇒ no ties)
    delay              : silent gap between last cue and the decision step
    cue_duration       : steps each motif spans (must be >= 2 for a ramp)
    inter_cue_interval : silent steps after each cue
    amp                : ramp amplitude (peak value of the motif)
    feature_noise      : std of Gaussian noise on the feature channel during a cue
    noise_level        : std of Gaussian noise on the distractor channel (3)
    seed               : integer seed for reproducibility (None = random)
    device             : torch device string

    Returns
    -------
    inputs  : (T, B, 5)
    targets : (T, B, 2)  one-hot at the decision step only
    mask    : (T, B)      1.0 at the decision step, 0 elsewhere

    where T = n_cues*(cue_duration + inter_cue_interval) + delay + 1
    """
    assert cue_duration >= 2, "cue_duration must be >= 2 for a temporal ramp"
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    cue_stride = cue_duration + inter_cue_interval
    cue_window = n_cues * cue_stride
    T = cue_window + delay + 1
    B = batch_size

    inputs  = torch.zeros(T, B, N_IN)
    targets = torch.zeros(T, B, N_OUT)
    mask    = torch.zeros(T, B)

    # Per-cue side: 0 = rising, 1 = falling.  (B, n_cues)
    sides = torch.randint(0, 2, (B, n_cues), generator=gen)
    # Base rising ramp, mean-zero, from -amp to +amp across the cue window.
    base = torch.linspace(-amp, amp, cue_duration)            # (cue_duration,)

    for c in range(n_cues):
        t0 = c * cue_stride
        # direction: +1 for rising (side 0), -1 for falling (side 1)
        direction = (1 - 2 * sides[:, c]).float()             # (B,) ∈ {+1,-1}
        for dt in range(cue_duration):
            t = t0 + dt
            val = direction * base[dt]                        # (B,)
            noise = torch.randn(B, generator=gen) * feature_noise
            inputs[t, :, 0] = val + noise

    # Recall signal at the decision step
    t_recall = cue_window + delay
    inputs[t_recall, :, 2] = 1.0

    # Distractor feature (channel 1) and distractor noise (channel 3) + bias
    inputs[:, :, 1] = torch.randn(T, B, generator=gen) * feature_noise
    if noise_level > 0.0:
        inputs[:, :, 3] = torch.randn(T, B, generator=gen) * noise_level
    inputs[:, :, 4] = 1.0

    # Label = majority side over cues (1 = falling-majority)
    falling = sides.sum(dim=1).float()                        # (B,)
    rising  = float(n_cues) - falling
    labels = torch.zeros(B, dtype=torch.long)
    labels[falling > rising] = 1
    tied = falling == rising
    if tied.any():
        n_tied = int(tied.sum().item())
        labels[tied] = torch.randint(0, 2, (n_tied,), generator=gen)

    targets[t_recall, torch.arange(B), labels] = 1.0
    mask[t_recall] = 1.0

    inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
    return inputs, targets, mask


def task_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correct trials (argmax over n_out) at masked timesteps."""
    pred    = logits.argmax(dim=-1)
    tgt     = targets.argmax(dim=-1)
    correct = ((pred == tgt) * mask).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0


def sequence_length(
    n_cues: int = 5,
    delay: int = 20,
    cue_duration: int = 5,
    inter_cue_interval: int = 4,
) -> int:
    """Total sequence length T for the given task parameters."""
    return n_cues * (cue_duration + inter_cue_interval) + delay + 1
