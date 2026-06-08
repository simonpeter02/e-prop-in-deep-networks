"""
Spiking Heidelberg Digits (SHD) dataset.

Cramer et al. 2020: "The Heidelberg Spiking Data Sets for the Systematic
Evaluation of Spiking Neural Networks." IEEE TNNLS.

70 speakers say digits 0-9 in English and German → 20 classes.
Recorded via a simulated silicon cochlea with 700 frequency channels.
Spike trains span ~1.4 seconds at ~1 ms resolution.

We bin to T=100 timesteps (14 ms / bin) and use all 700 input channels.
The network must classify from spike trains to one of 20 digit classes,
outputting its prediction at the final timestep.
"""

import os
import subprocess
import gzip
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

T_SHD     = 100    # time bins after downsampling
N_IN_SHD  = 700    # cochlea frequency channels
N_CLASSES = 20     # 10 digits × 2 languages (English / German)
MAX_TIME  = 1.4    # seconds — recording duration

_URLS = {
    'train': 'https://zenkelab.org/datasets/shd_train.h5.gz',
    'test':  'https://zenkelab.org/datasets/shd_test.h5.gz',
}
_PATHS = {
    'train': '/tmp/shd_train.h5',
    'test':  '/tmp/shd_test.h5',
}

_cache: dict = {}


def _download_shd(split: str) -> None:
    """Download and decompress the SHD HDF5 file if not already present."""
    path    = _PATHS[split]
    gz_path = path + '.gz'
    if os.path.exists(path):
        return
    print(f"Downloading SHD {split} split from Zenodo (≈100–200 MB) ...")
    subprocess.run(
        ['wget', '-q', '--show-progress', '-O', gz_path, _URLS[split]],
        check=True,
    )
    print("Decompressing ...")
    with gzip.open(gz_path, 'rb') as f_in, open(path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"Saved to {path}")


def _load_shd(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return dense spike arrays for the chosen split.

    Returns
    -------
    spikes : (N, T_SHD, 700)   float32 binary spike tensors
    labels : (N,)               int64 class indices (0–19)
    """
    if split in _cache:
        return _cache[split]

    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for SHD: run  !pip install h5py  in Colab."
        )

    _download_shd(split)

    with h5py.File(_PATHS[split], 'r') as f:
        spike_times = f['spikes']['times'][:]   # ragged array of float arrays
        spike_units = f['spikes']['units'][:]   # ragged array of int arrays
        labels      = f['labels'][:].astype(np.int64)

    N  = len(labels)
    dt = MAX_TIME / T_SHD

    spikes = np.zeros((N, T_SHD, N_IN_SHD), dtype=np.float32)
    for i in range(N):
        t_idx  = np.floor(spike_times[i] / dt).astype(int).clip(0, T_SHD - 1)
        ch_idx = spike_units[i].astype(int).clip(0, N_IN_SHD - 1)
        spikes[i, t_idx, ch_idx] = 1.0

    _cache[split] = (spikes, labels)
    return spikes, labels


def generate_batch(
    batch_size: int,
    device: str = 'cpu',
    train: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample a random mini-batch from SHD.

    Returns
    -------
    inputs  : (T_SHD, B, 700)   binary spike trains
    targets : (T_SHD, B, 20)    one-hot class at final timestep only
    mask    : (T_SHD, B)         1 at final timestep, 0 elsewhere
    """
    split  = 'train' if train else 'test'
    spikes, labels = _load_shd(split)
    N   = spikes.shape[0]
    idx = np.random.randint(0, N, batch_size)

    seqs    = torch.from_numpy(spikes[idx])        # (B, T, 700)
    inputs  = seqs.permute(1, 0, 2).to(device)    # (T, B, 700)

    targets = torch.zeros(T_SHD, batch_size, N_CLASSES, device=device)
    targets[-1] = F.one_hot(
        torch.from_numpy(labels[idx]).long(), N_CLASSES
    ).float().to(device)

    mask        = torch.zeros(T_SHD, batch_size, device=device)
    mask[-1]    = 1.0

    return inputs, targets, mask


def task_accuracy(outputs: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Classification accuracy measured at the final timestep."""
    pred  = outputs[-1].argmax(-1)
    label = targets[-1].argmax(-1)
    return (pred == label).float().mean().item()
