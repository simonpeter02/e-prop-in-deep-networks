"""
Common interface for all learning rules.

Wraps the compute_*_gradients functions so the whole rule ladder
(single-layer e-prop, d=0, deep e-prop, deep d=0, deep RTRL, BPTT)
can be swapped from a single config string.

Usage
-----
    rule = make_learning_rule('eprop_leaky')
    grads = rule.compute_gradients(model, inputs, targets, mask)
    # or in one shot:
    grads = rule.update(model, inputs, targets, mask, lr=1e-3)

Learning-rate utilities
-----------------------
    lr = lr_for_config(base_lr=1e-3, depth=2, alpha=0.1)

apply_gradients is also exported for direct use in notebooks.
"""

from typing import Dict, Optional
import torch
from torch import Tensor


# ── Gradient application ─────────────────────────────────────────────────────

def apply_gradients(model, grads: Dict[str, Tensor], lr: float) -> None:
    """SGD step: param.data -= lr * grad for each key in grads."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in grads:
                param.data -= lr * grads[name]


# ── LR heuristic ─────────────────────────────────────────────────────────────

def lr_for_config(
    base_lr: float,
    depth: int = 1,
    alpha: float = 1.0,
) -> float:
    """Heuristic learning-rate scaling for depth and leak rate.

    The eligibility trace window is ~1/(1-alpha) steps (for leaky RNNs).
    Longer traces produce larger accumulated gradients, so we scale LR down
    to keep the effective update magnitude roughly constant.  The depth
    scaling is mild (1/sqrt(depth)) to compensate for gradient signal
    attenuation across layers.

    Parameters
    ----------
    base_lr : the reference learning rate at depth=1, alpha=1 (vanilla tanh)
    depth   : number of recurrent layers
    alpha   : integration rate (1 = vanilla, <1 = leaky)

    Returns
    -------
    Scaled learning rate.
    """
    leak = 1.0 - alpha          # effective leak magnitude (0 for vanilla)
    trace_window = 1.0 / max(alpha, 1e-3)   # ~steps the trace survives
    # Longer windows → larger cumulative gradients → scale LR down
    lr = base_lr / (1.0 + leak * trace_window * 0.3)
    # Depth attenuation
    lr = lr / (depth ** 0.5)
    return float(lr)


# ── Base class ───────────────────────────────────────────────────────────────

class LearningRule:
    """Callable wrapper for a gradient-computation function."""

    name: str = "base"

    def compute_gradients(
        self,
        model,
        inputs: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def update(
        self,
        model,
        inputs: Tensor,
        targets: Tensor,
        mask: Tensor,
        lr: float,
    ) -> Dict[str, Tensor]:
        """Compute gradients and apply SGD step. Returns the grad dict."""
        grads = self.compute_gradients(model, inputs, targets, mask)
        apply_gradients(model, grads, lr)
        return grads


# ── Concrete rules ────────────────────────────────────────────────────────────

class EpropRule(LearningRule):
    """Single-layer e-prop for VanillaRNN (or LeakyRNN via auto-dispatch)."""

    name = "eprop"

    def __init__(self, learning_signal_fn=None, d_zero: bool = False):
        from learning_rules.eprop import mse_error
        self.lsf    = learning_signal_fn or mse_error
        self.d_zero = d_zero

    def compute_gradients(self, model, inputs, targets, mask):
        from learning_rules.eprop import compute_eprop_gradients, compute_eprop_leaky_gradients
        from models.leaky_rnn import LeakyRNN
        try:
            from models.vanilla_rnn import LeakyRNN as _LegacyLeaky
        except ImportError:
            _LegacyLeaky = None

        if isinstance(model, LeakyRNN) or (
            _LegacyLeaky is not None and isinstance(model, _LegacyLeaky)
        ) or hasattr(model, 'alpha'):
            return compute_eprop_leaky_gradients(
                model, inputs, targets, mask, self.lsf, d_zero=self.d_zero
            )
        return compute_eprop_gradients(
            model, inputs, targets, mask, self.lsf, d_zero=self.d_zero
        )


class DeepEpropRule(LearningRule):
    """Deep e-prop for DeepRNN (arbitrary depth)."""

    name = "deep_eprop"

    def __init__(self, learning_signal_fn=None, d_zero: bool = False):
        from learning_rules.deep_eprop import mse_error
        self.lsf    = learning_signal_fn or mse_error
        self.d_zero = d_zero

    def compute_gradients(self, model, inputs, targets, mask):
        from learning_rules.deep_eprop import compute_deep_eprop_gradients
        return compute_deep_eprop_gradients(
            model, inputs, targets, mask, self.lsf, d_zero=self.d_zero
        )


class DeepRTRLRule(LearningRule):
    """Full RTRL for 2-layer DeepRNN (exact, slow — verification only)."""

    name = "deep_rtrl"

    def __init__(self, learning_signal_fn=None):
        from learning_rules.deep_rtrl import mse_error
        self.lsf = learning_signal_fn or mse_error

    def compute_gradients(self, model, inputs, targets, mask):
        from learning_rules.deep_rtrl import compute_deep_rtrl_gradients
        return compute_deep_rtrl_gradients(model, inputs, targets, mask, self.lsf)


class BPTTRule(LearningRule):
    """BPTT via autograd — ground-truth reference."""

    name = "bptt"

    def __init__(self, loss_fn=None):
        self.loss_fn = loss_fn

    def compute_gradients(self, model, inputs, targets, mask):
        from learning_rules.bptt import compute_bptt_gradients
        return compute_bptt_gradients(model, inputs, targets, mask, self.loss_fn)


# ── Factory ───────────────────────────────────────────────────────────────────

_RULE_MAP = {
    "eprop":        (EpropRule,      dict(d_zero=False)),
    "d0":           (EpropRule,      dict(d_zero=True)),
    "deep_eprop":   (DeepEpropRule,  dict(d_zero=False)),
    "deep_d0":      (DeepEpropRule,  dict(d_zero=True)),
    "deep_rtrl":    (DeepRTRLRule,   {}),
    "bptt":         (BPTTRule,       {}),
}


def make_learning_rule(
    name: str,
    learning_signal_fn=None,
    loss_fn=None,
    **kwargs,
) -> LearningRule:
    """Create a LearningRule by name.

    Valid names: 'eprop', 'd0', 'deep_eprop', 'deep_d0', 'deep_rtrl', 'bptt'.

    Parameters
    ----------
    name               : rule identifier (see _RULE_MAP)
    learning_signal_fn : override the default MSE error signal
    loss_fn            : override for bptt rule only
    **kwargs           : passed to the rule constructor
    """
    if name not in _RULE_MAP:
        raise ValueError(f"Unknown rule '{name}'. Valid: {list(_RULE_MAP)}")
    cls, defaults = _RULE_MAP[name]
    kw = {**defaults, **kwargs}
    if name == "bptt":
        return cls(loss_fn=loss_fn or kw.pop("loss_fn", None), **kw)
    if learning_signal_fn is not None:
        kw["learning_signal_fn"] = learning_signal_fn
    return cls(**kw)
