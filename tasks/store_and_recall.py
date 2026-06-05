"""
Store-and-recall (delayed copy) task.

At each trial the network sees a cue pattern, then a blank delay, then a
recall signal.  It must reproduce the cue pattern at the output immediately
after the recall signal.

Input channels
--------------
0 .. n_patterns-1 : one-hot cue (active only at cue_t)
n_patterns        : recall signal (active only at recall_t)
n_patterns+1      : bias (always 1)

Target
------
One-hot index of the stored pattern, active at recall_t and held for
output_duration steps.  MSE or cross-entropy loss over those steps.
"""

import torch
from torch import Tensor
from typing import Tuple


def generate_batch(
    batch_size: int,
    n_patterns: int,
    delay: int,
    cue_duration: int = 1,
    output_duration: int = 1,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (inputs, targets, loss_mask).

    inputs  : (T, B, n_patterns + 2)
    targets : (T, B, n_patterns)   — one-hot
    mask    : (T, B)               — 1 where loss is computed
    """
    T = cue_duration + delay + output_duration
    n_in = n_patterns + 2  # patterns + recall + bias

    inputs = torch.zeros(T, batch_size, n_in, device=device)
    targets = torch.zeros(T, batch_size, n_patterns, device=device)
    mask = torch.zeros(T, batch_size, device=device)

    labels = torch.randint(0, n_patterns, (batch_size,), device=device)

    # cue
    for t in range(cue_duration):
        inputs[t, torch.arange(batch_size), labels] = 1.0

    # recall signal
    recall_t = cue_duration + delay
    inputs[recall_t, :, n_patterns] = 1.0

    # bias
    inputs[:, :, n_patterns + 1] = 1.0

    # target and mask during output window
    for t in range(recall_t, recall_t + output_duration):
        targets[t, torch.arange(batch_size), labels] = 1.0
        mask[t] = 1.0

    return inputs, targets, mask


def task_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correct trials (argmax match) over masked timesteps."""
    pred = logits.argmax(dim=-1)  # (T, B)
    tgt = targets.argmax(dim=-1)   # (T, B)
    correct = ((pred == tgt) * mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0
