"""
Distractor-selection cue task (reservoir-resistant variant of the cue task).

A variant of the hierarchical classify-then-count task (tasks/hierarchical_cue.py)
redesigned so that a FROZEN random input-adjacent layer (a reservoir) genuinely
cannot solve it, while a trained one can — and so that concurrent supervision lets
deep e-prop's TEMPORAL ablation still train the lower layer.  This is the task that
separates the two deep-e-prop ablations:  full ≈ ablate_temporal ≫ ablate_spatial.

Why this design
---------------
In a 2-layer net, `ablate_spatial` freezes ONLY layer 0; layer 1 + readout keep
training.  So it is "a trained nonlinear recurrent net reading a random reservoir",
NOT "a linear readout on a reservoir".  A trained top layer can supply any missing
nonlinearity, so a merely-harder feature does not make the frozen layer fail — the
reservoir only fails if it DESTROYS information the top layer cannot recover.  We
force that with **capacity overload + a temporal feature**:

  * One feature channel (channel 0) carries a mean-zero rising/falling motif each
    cue; D-1 other channels carry INDEPENDENT random motifs every cue — dynamic
    distractors.  The task is: report channel 0's motif direction (ignore the rest).
  * With D > n_rec, a random layer 0 cannot retain channel 0's *temporal motif
    shape* (rising vs falling) through its few units while they are also driven by
    D-1 competing dynamic motifs (total memory capacity ≈ #units; Dambre et al.
    2012).  A trained layer 0 simply learns an input weight that selects channel 0
    and suppresses the distractors — which is easy to learn (unlike a context gate).
  * Empirically (n_rec=12, D=32): full/ablate_temporal reach ~0.99 while
    ablate_spatial collapses to ~0.56 ≈ the pure-reservoir (esn) baseline, i.e. the
    trained top layer does NOT rescue the frozen layer 0.

Selection vs routing.  An earlier variant cued WHICH channel was relevant per trial
(context-dependent routing).  That gate proved unlearnable even for `full` at the
overloaded widths needed to break the reservoir, so the relevant channel is fixed
(channel 0); resistance comes from capacity overload, which is both learnable for a
trained layer 0 and genuinely lossy for a random one.

Supervision has TWO heads (so the asymmetry is separable from the time story):
  * Aux head (CONCURRENT): at the end of each cue, output channel 0's motif side.
    The loss fires while the lower layer's eligibility trace ε^h is still alive, so
    `ablate_temporal` (which keeps only instantaneous cross-layer credit) can still
    train layer 0.  This head is the asymmetry signal.
  * Delayed head (COUNTING): after a silent delay, output the majority side over the
    n_cues cues — the top layer must integrate across the delay (keeps the original
    depth-AND-time narrative; cross-layer temporal credit matters a little more here).

Input channels (n_in = D + 2):
  [0      : D  ]  feature channels  (channel 0 relevant; channels 1..D-1 distractors)
  [D          ]  recall signal      (active at the single decision step only)
  [D + 1      ]  bias               (constant 1.0)

Output channels (n_out = 4):
  [0:2]  aux head      0 = relevant cue rising, 1 = relevant cue falling
  [2:4]  delayed head  2 = rising-majority,     3 = falling-majority

Targets are one-hot in the active head's two slots at the supervised steps and zero
elsewhere; train with MSE (mse_error) so the two heads do not share one softmax.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

# Slot layout in the n_out = 4 output vector.
AUX_SLOTS = (0, 2)   # [aux_rising, aux_falling]
DEL_SLOTS = (2, 4)   # [del_rising, del_falling]


def n_in_for(D: int) -> int:
    """Number of input channels for D feature channels (+ recall + bias)."""
    return D + 2


def sequence_length(
    n_cues: int = 5,
    delay: int = 12,
    cue_duration: int = 3,
    inter_cue_interval: int = 2,
) -> int:
    """Total sequence length T for the given task parameters."""
    return n_cues * (cue_duration + inter_cue_interval) + delay + 1


def generate_batch(
    batch_size: int,
    D: int = 32,
    n_cues: int = 5,
    delay: int = 12,
    cue_duration: int = 3,
    inter_cue_interval: int = 2,
    amp: float = 2.0,
    feature_noise: float = 0.15,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate a batch of distractor-selection cue trials.

    Parameters
    ----------
    batch_size         : number of independent trials
    D                  : number of feature channels (the overload knob).  Channel 0
                         is relevant; channels 1..D-1 are dynamic distractors.
    n_cues             : cues per trial (odd ⇒ no ties for the majority head)
    delay              : silent gap between last cue and the decision step
    cue_duration       : steps each motif spans (>= 2 for a ramp)
    inter_cue_interval : silent steps after each cue (>= 1: the aux readout is the
                         first silent step, when the full ramp has been integrated)
    amp                : ramp amplitude
    feature_noise      : std of Gaussian noise added to feature channels during a cue
    seed               : integer seed for reproducibility (None = random)
    device             : torch device string

    Returns
    -------
    inputs  : (T, B, D + 2)
    targets : (T, B, 4)
    mask    : (T, B)   1.0 at every aux step and at the decision step, else 0
    """
    assert cue_duration >= 2, "cue_duration must be >= 2 for a temporal ramp"
    assert inter_cue_interval >= 1, "need >=1 silent step for the concurrent aux readout"
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    cue_stride = cue_duration + inter_cue_interval
    cue_window = n_cues * cue_stride
    T = cue_window + delay + 1
    B = batch_size
    n_in = n_in_for(D)

    inputs  = torch.zeros(T, B, n_in)
    targets = torch.zeros(T, B, 4)
    mask    = torch.zeros(T, B)

    base = torch.linspace(-amp, amp, cue_duration)            # (cue_duration,)
    b_idx = torch.arange(B)

    # Per-cue, per-channel motif side: 0 = rising, 1 = falling.  (B, n_cues, D)
    sides_all = torch.randint(0, 2, (B, n_cues, D), generator=gen)
    rel_side = sides_all[:, :, 0]                             # channel 0 relevant. (B, n_cues)

    for c in range(n_cues):
        t0 = c * cue_stride
        # direction: +1 rising (side 0), -1 falling (side 1).  (B, D)
        direction = (1 - 2 * sides_all[:, c, :]).float()
        for dt in range(cue_duration):
            t = t0 + dt
            val = direction * base[dt]                        # ramp per channel
            val = val + torch.randn(B, D, generator=gen) * feature_noise
            inputs[t, :, 0:D] = val

        # Aux head: concurrent readout at the first silent step after the ramp,
        # when the lower layer has just integrated the full motif.
        t_aux = t0 + cue_duration
        targets[t_aux, b_idx, AUX_SLOTS[0] + rel_side[:, c]] = 1.0
        mask[t_aux] = 1.0

    # Delayed counting head at the decision step.
    t_recall = cue_window + delay
    inputs[t_recall, :, D] = 1.0                              # recall signal
    falling = rel_side.sum(dim=1).float()                     # # falling relevant cues
    rising  = float(n_cues) - falling
    del_label = torch.zeros(B, dtype=torch.long)
    del_label[falling > rising] = 1                           # 1 = falling-majority
    targets[t_recall, b_idx, DEL_SLOTS[0] + del_label] = 1.0
    mask[t_recall] = 1.0

    inputs[:, :, D + 1] = 1.0                                 # bias

    return inputs.to(device), targets.to(device), mask.to(device)


def head_accuracy(logits: Tensor, targets: Tensor, slots: Tuple[int, int]) -> float:
    """Accuracy for one output head over its supervised steps.

    A step is "supervised for this head" iff the target is non-zero in the head's
    slot range (so we don't need a separate per-head mask).  argmax is taken WITHIN
    the head's two slots.

    logits, targets : (T, B, 4)
    slots           : (lo, hi) slot range for the head, e.g. AUX_SLOTS
    """
    lo, hi = slots
    active  = targets[..., lo:hi].abs().sum(-1) > 0           # (T, B) bool
    pred    = logits[..., lo:hi].argmax(dim=-1)
    tgt     = targets[..., lo:hi].argmax(dim=-1)
    correct = ((pred == tgt) & active).sum().item()
    total   = active.sum().item()
    return correct / total if total > 0 else 0.0
