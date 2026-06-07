"""
BPTT gradient computation — ground-truth reference.

Uses PyTorch autograd through the full unrolled sequence.
Returns per-parameter gradient tensors matching e-prop's dictionary format.
"""

import torch
from torch import Tensor
from models.vanilla_rnn import VanillaRNN
from typing import Dict


def compute_bptt_gradients(
    model: VanillaRNN,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    loss_fn=None,
) -> Dict[str, Tensor]:
    """
    Run the full sequence forward via autograd, compute loss, backprop.

    loss_fn : callable(outputs, targets, mask) -> scalar loss
              defaults to MSE over masked timesteps
    """
    if loss_fn is None:
        loss_fn = _mse_loss

    # Zero any existing gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    outputs, _ = model(inputs)  # (T, B, n_out)
    loss = loss_fn(outputs, targets, mask)
    loss.backward()

    grads = {
        'W_rec': model.W_rec.grad.clone(),
        'W_in':  model.W_in.grad.clone(),
        'b_rec': model.b_rec.grad.clone(),
        'W_out': model.W_out.grad.clone(),
        'b_out': model.b_out.grad.clone(),
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
