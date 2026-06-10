"""
BPTT gradient computation — ground-truth reference.

Uses PyTorch autograd through the full unrolled sequence.
Returns per-parameter gradient tensors as a dict keyed by model.named_parameters()
names — works for VanillaRNN, LeakyRNN, DeepRNN, and any other nn.Module whose
forward() returns (outputs, hiddens).
"""

import torch
from torch import Tensor
from typing import Dict


def compute_bptt_gradients(
    model,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    loss_fn=None,
) -> Dict[str, Tensor]:
    """Run the full sequence forward, compute loss, backprop.

    Works for any nn.Module whose forward() returns (outputs, _) and whose
    trainable parameters are in named_parameters().

    loss_fn : callable(outputs, targets, mask) -> scalar
              defaults to MSE over masked timesteps
    """
    if loss_fn is None:
        loss_fn = _mse_loss

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    outputs, _ = model(inputs)   # (T, B, n_out)
    loss = loss_fn(outputs, targets, mask)
    loss.backward()

    grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    return grads


def _mse_loss(outputs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    # outputs, targets: (T, B, n_out); mask: (T, B)
    sq = ((outputs - targets) ** 2).sum(-1)  # (T, B)
    return (sq * mask).sum() / (mask.sum() * outputs.shape[-1])


def _xent_loss(outputs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    # outputs, targets: (T, B, n_out); mask: (T, B)
    log_p = torch.log_softmax(outputs, dim=-1)   # (T, B, n_out)
    ce    = -(targets * log_p).sum(-1)            # (T, B)
    return (ce * mask).sum() / mask.sum()
