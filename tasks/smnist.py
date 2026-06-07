"""
Sequential MNIST (sMNIST) and Permuted sMNIST (psMNIST).

Each MNIST image (28x28) is flattened to T=784 pixels presented one per
timestep as a scalar input.  The network must classify the digit (10 classes)
from the final-timestep output only — mask is 1 only at t=783.

psMNIST applies a fixed pixel permutation, removing all local spatial
structure and requiring truly long-range temporal credit assignment.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

T_SMNIST  = 784
N_CLASSES = 10

# Fixed permutation for psMNIST — seeded for reproducibility across runs
_PERM: Tensor = torch.randperm(T_SMNIST,
                                generator=torch.Generator().manual_seed(12345))

_cache: dict = {}


def _load_mnist(train: bool = True):
    key = 'train' if train else 'test'
    if key not in _cache:
        try:
            import torchvision
        except ImportError:
            raise ImportError("torchvision required: pip install torchvision")
        ds   = torchvision.datasets.MNIST(root='/tmp/mnist_data',
                                           train=train, download=True)
        imgs = ds.data.float() / 255.0      # (N, 28, 28) in [0, 1]
        lbls = ds.targets.long()             # (N,)
        _cache[key] = (imgs.reshape(-1, T_SMNIST), lbls)
    return _cache[key]


def generate_batch(
    batch_size: int,
    permuted:   bool = False,
    device:     str  = 'cpu',
    train:      bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample a random mini-batch for (p)sMNIST.

    Returns
    -------
    inputs  : (T, B, 1)     pixel values in [0, 1]
    targets : (T, B, 10)    one-hot class label at final timestep, zeros elsewhere
    mask    : (T, B)         1 at final timestep only
    """
    imgs, lbls = _load_mnist(train=train)
    idx  = torch.randint(0, imgs.shape[0], (batch_size,))
    seqs = imgs[idx]                              # (B, T)

    if permuted:
        seqs = seqs[:, _PERM]

    inputs  = seqs.T.unsqueeze(-1).to(device)     # (T, B, 1)

    targets = torch.zeros(T_SMNIST, batch_size, N_CLASSES, device=device)
    targets[-1] = F.one_hot(lbls[idx], N_CLASSES).float().to(device)

    mask      = torch.zeros(T_SMNIST, batch_size, device=device)
    mask[-1]  = 1.0

    return inputs, targets, mask


def task_accuracy(outputs: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correctly classified digits — argmax of final-step output."""
    pred  = outputs[-1].argmax(-1)                # (B,)
    label = targets[-1].argmax(-1)                # (B,)
    return (pred == label).float().mean().item()
