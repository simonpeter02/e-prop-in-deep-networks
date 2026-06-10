"""
Shared utilities for multi-seed experiments and gradient analysis.

Key exports
-----------
run_multi_seed   — run a training/evaluation function over N seeds and
                   return mean ± stderr of collected metrics
cosine_sim_grads — cosine similarity between two gradient dicts
flat_grads       — flatten a gradient dict into a single vector
"""

from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np
import torch
from torch import Tensor


# ── Gradient utilities ────────────────────────────────────────────────────────

def flat_grads(grads: Dict[str, Tensor], keys: Optional[Sequence[str]] = None) -> Tensor:
    """Concatenate gradient tensors into a single 1-D vector.

    Parameters
    ----------
    grads : dict name → Tensor
    keys  : subset of keys to use (default: all, sorted for reproducibility)
    """
    if keys is None:
        keys = sorted(grads.keys())
    return torch.cat([grads[k].flatten() for k in keys if k in grads])


def cosine_sim_grads(
    g1: Dict[str, Tensor],
    g2: Dict[str, Tensor],
    keys: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> float:
    """Cosine similarity between two gradient dicts.

    Returns NaN if either vector has near-zero norm.
    """
    v1 = flat_grads(g1, keys)
    v2 = flat_grads(g2, keys)
    n1, n2 = v1.norm().item(), v2.norm().item()
    if n1 < eps or n2 < eps:
        return float("nan")
    return (v1 @ v2 / (n1 * n2)).item()


def relative_error_grads(
    g_approx: Dict[str, Tensor],
    g_ref: Dict[str, Tensor],
    keys: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> float:
    """||g_approx - g_ref|| / ||g_ref|| (returns nan if g_ref ≈ 0)."""
    va = flat_grads(g_approx, keys)
    vr = flat_grads(g_ref,    keys)
    nr = vr.norm().item()
    if nr < eps:
        return float("nan")
    return ((va - vr).norm() / nr).item()


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_multi_seed(
    fn: Callable[..., Dict[str, Any]],
    seeds: List[int],
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run fn(seed=seed, **kwargs) for each seed and aggregate results.

    fn must return a dict of scalar or list values.
    Aggregated result contains mean and std-err for each key.

    Parameters
    ----------
    fn      : function with signature (seed: int, **kwargs) -> Dict
    seeds   : list of integer seeds to use
    verbose : print per-seed summary
    **kwargs: forwarded to fn

    Returns
    -------
    Dict with keys from fn's output, plus '{key}_mean', '{key}_stderr'
    for numeric keys.  Also includes 'all_results' (list of per-seed dicts).
    """
    all_results: List[Dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = fn(seed=seed, **kwargs)
        all_results.append(result)
        if verbose:
            summary = {k: f"{v:.4f}" if isinstance(v, float) else v
                       for k, v in result.items() if not isinstance(v, list)}
            print(f"  seed={seed}: {summary}")

    # Aggregate
    aggregated: Dict[str, Any] = {"all_results": all_results}
    if all_results:
        for key in all_results[0]:
            vals = [r[key] for r in all_results]
            if all(isinstance(v, (int, float)) for v in vals):
                arr = np.array(vals, dtype=float)
                aggregated[f"{key}_mean"]   = float(arr.mean())
                aggregated[f"{key}_stderr"] = float(arr.std(ddof=1) / np.sqrt(len(arr)))
            elif all(isinstance(v, list) for v in vals):
                # Aggregate lists element-wise (e.g., learning curves)
                try:
                    mat = np.array(vals, dtype=float)   # (n_seeds, T)
                    aggregated[f"{key}_mean"]   = mat.mean(axis=0).tolist()
                    aggregated[f"{key}_stderr"] = (mat.std(axis=0, ddof=1) /
                                                   np.sqrt(len(vals))).tolist()
                except (ValueError, TypeError):
                    pass

    return aggregated


# ── LR heuristic (re-exported from interface for convenience) ─────────────────

def lr_for_config(base_lr: float, depth: int = 1, alpha: float = 1.0) -> float:
    """See learning_rules.interface.lr_for_config for documentation."""
    from learning_rules.interface import lr_for_config as _lfc
    return _lfc(base_lr, depth, alpha)
