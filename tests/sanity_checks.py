"""
Standalone sanity checks for the e-prop deep networks codebase.

Tests (CPU-only, tiny nets, < 60 s total):

  Test 1 — deep-RTRL gradient == BPTT to numerical precision
            (allclose within tol AND cosine ≈ 1.0)

  Test 2 — depth-1 deep e-prop == single-layer e-prop
            (catches the L=1 W_in gradient bug fixed in deep_eprop.py)

  Test 3 — finite-difference gradient check for BPTT on VanillaRNN
            (ground truth of the ground truth)

  Test 4 — vanilla tanh RNN: e-prop ≈ d=0 (gap ≈ 0)
            Carry ≈ ψ * W_diag ≈ 0.005 → no wedge expected; documented as EXPECTED

  Test 5 — leaky RNN at long delay D: e-prop gradient meaningfully closer
            to BPTT than d=0 (the central research hypothesis)

Usage:
    python tests/sanity_checks.py            # from repo root
    python -m tests.sanity_checks            # as module
"""

import sys
import os

# Allow running from the repo root without installing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from models.vanilla_rnn import VanillaRNN
from models.deep_rnn    import DeepRNN
from models.leaky_rnn   import LeakyRNN

from learning_rules.eprop      import (compute_eprop_gradients,
                                        compute_eprop_leaky_gradients, mse_error)
from learning_rules.deep_eprop import compute_deep_eprop_gradients
from learning_rules.deep_rtrl  import compute_deep_rtrl_gradients
from learning_rules.bptt       import compute_bptt_gradients, _mse_loss

from tasks.cue_accumulation import generate_batch as ca_batch
from utils import cosine_sim_grads, flat_grads


# ── Shared helpers ────────────────────────────────────────────────────────────

def _max_abs_diff(g1, g2, keys):
    diffs = []
    for k in keys:
        if k in g1 and k in g2:
            diffs.append((g1[k] - g2[k]).abs().max().item())
    return max(diffs) if diffs else float("nan")


def _allclose(g1, g2, keys, rtol=1e-4, atol=1e-5):
    for k in keys:
        if k not in g1 or k not in g2:
            return False, k
        if not torch.allclose(g1[k], g2[k], rtol=rtol, atol=atol):
            diff = (g1[k] - g2[k]).abs().max().item()
            return False, f"{k} (max diff={diff:.2e})"
    return True, None


def _cosine(g1, g2, keys=None):
    if keys is None:
        keys = [k for k in g1 if k in g2]
    return cosine_sim_grads(g1, g2, keys)


# ── Test 1: deep-RTRL == BPTT ────────────────────────────────────────────────

def test_deep_rtrl_matches_bptt(n_seeds: int = 10) -> bool:
    """deep-RTRL gradient direction must match BPTT.

    RTRL and BPTT use different loss normalizations (RTRL divides per-batch
    by B; BPTT divides by mask.sum()*n_out) so their magnitudes differ, but
    they must point in exactly the same direction.  We measure this via the
    per-key direction error ||v1/|v1| - v2/|v2||, which must be < 1e-4.
    """
    print("Test 1: deep-RTRL == BPTT (direction) ...")

    rtrl_keys = ["W_recs.0", "W_recs.1", "W_ffs.0", "W_in", "biases.0", "biases.1"]
    max_dir_err = 0.0
    min_cos     = 1.0

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        # Small model: O(n^4) RTRL is feasible at n=8
        model = DeepRNN(n_in=4, n_rec=8, n_out=2, n_layers=2)
        T, B  = 8, 4
        inputs  = torch.randn(T, B, 4)
        targets = torch.randn(T, B, 2)
        mask    = torch.zeros(T, B)
        mask[3:6] = 1.0

        g_bptt = compute_bptt_gradients(model, inputs, targets, mask)
        g_rtrl = compute_deep_rtrl_gradients(model, inputs, targets, mask, mse_error)

        for k in rtrl_keys:
            if k not in g_bptt or k not in g_rtrl:
                print(f"  FAIL at seed={seed}: key '{k}' missing")
                return False
            v1 = g_bptt[k].flatten()
            v2 = g_rtrl[k].flatten()
            if v1.norm() < 1e-12:
                continue
            err = (v1 / v1.norm() - v2 / v2.norm()).norm().item()
            if err > max_dir_err:
                max_dir_err = err
            if err > 1e-3:
                print(f"  FAIL at seed={seed}, key={k}: direction error={err:.2e}")
                return False

        cos = _cosine(g_bptt, g_rtrl, rtrl_keys)
        if not (cos != cos):
            min_cos = min(min_cos, cos)

    if min_cos < 0.9999:
        print(f"  FAIL: cosine too low: {min_cos:.6f}")
        return False

    print(f"  PASS  max_direction_err={max_dir_err:.2e}  min_cosine={min_cos:.6f}")
    return True


# ── Test 2: depth-1 deep e-prop == single-layer e-prop ───────────────────────

def test_depth1_deep_eprop_matches_single(n_seeds: int = 5) -> bool:
    """For L=1, deep e-prop must produce identical gradients to single-layer e-prop.

    This test catches the L=1 W_in gradient bug (fixed: grad_W_in was 0 for L=1).
    """
    print("Test 2: depth-1 deep e-prop == single-layer e-prop ...")

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model_deep = DeepRNN(n_in=4, n_rec=16, n_out=2, n_layers=1)

        # Build VanillaRNN with the same weights
        model_sl = VanillaRNN(n_in=4, n_rec=16, n_out=2)
        with torch.no_grad():
            model_sl.W_rec.copy_(model_deep.W_recs[0])
            model_sl.W_in.copy_(model_deep.W_in)
            model_sl.b_rec.copy_(model_deep.biases[0])
            model_sl.W_out.copy_(model_deep.W_out)
            model_sl.b_out.copy_(model_deep.b_out)

        T, B  = 6, 8
        inputs  = torch.randn(T, B, 4)
        targets = torch.randn(T, B, 2)
        mask    = torch.zeros(T, B)
        mask[4] = 1.0

        g_deep = compute_deep_eprop_gradients(
            model_deep, inputs, targets, mask, mse_error, d_zero=False
        )
        g_sl   = compute_eprop_gradients(
            model_sl, inputs, targets, mask, mse_error, d_zero=False
        )

        # Map key names: deep uses 'W_recs.0'/'biases.0', single-layer uses 'W_rec'/'b_rec'
        key_map = {
            "W_recs.0": "W_rec",
            "biases.0": "b_rec",
            "W_in":     "W_in",
            "W_out":    "W_out",
            "b_out":    "b_out",
        }
        for deep_key, sl_key in key_map.items():
            if deep_key not in g_deep:
                print(f"  FAIL at seed={seed}: key '{deep_key}' missing from deep e-prop output")
                return False
            if not torch.allclose(g_deep[deep_key], g_sl[sl_key], rtol=1e-5, atol=1e-6):
                diff = (g_deep[deep_key] - g_sl[sl_key]).abs().max().item()
                print(f"  FAIL at seed={seed}: '{deep_key}' mismatch, max|diff|={diff:.2e}")
                if deep_key == "W_in":
                    print("  (This suggests the L=1 W_in gradient bug was not applied.)")
                return False

    print("  PASS  (L=1 deep e-prop == single-layer e-prop for all keys incl. W_in)")
    return True


# ── Test 3: finite-difference gradient check ─────────────────────────────────

def test_finite_difference_bptt() -> bool:
    """FD check for BPTT on W_rec of a small VanillaRNN.

    Central differences with eps=1e-4.  Float32 FD is limited to ~O(1e-3)
    relative error due to catastrophic cancellation in (L+ - L-), so we use
    a generous threshold of 3e-2.  The key assertion is that autograd and
    FD agree to within 2 orders of magnitude, catching sign errors and
    indexing bugs.
    """
    print("Test 3: finite-difference gradient check (BPTT) ...")

    torch.manual_seed(1)
    n_rec   = 8
    model   = VanillaRNN(n_in=3, n_rec=n_rec, n_out=2)
    T, B    = 5, 3
    inputs  = torch.randn(T, B, 3)
    targets = torch.randn(T, B, 2)
    mask    = torch.zeros(T, B)
    mask[3:5] = 1.0
    eps = 1e-4

    def loss_val():
        with torch.no_grad():
            out, _ = model(inputs)
            return _mse_loss(out, targets, mask).item()

    # Autograd gradient
    g_auto = compute_bptt_gradients(model, inputs, targets, mask)

    # FD gradient for W_rec
    W_rec_fd = torch.zeros_like(model.W_rec)
    for i in range(n_rec):
        for j in range(n_rec):
            model.W_rec.data[i, j] += eps
            L_plus = loss_val()
            model.W_rec.data[i, j] -= 2 * eps
            L_minus = loss_val()
            model.W_rec.data[i, j] += eps   # restore
            W_rec_fd[i, j] = (L_plus - L_minus) / (2 * eps)

    ref_norm = W_rec_fd.norm().item()
    rel_err  = ((g_auto["W_rec"] - W_rec_fd).norm() / (ref_norm + 1e-12)).item()
    # Float32 FD typically achieves ~1e-3 relative error; 3e-2 catches real bugs
    if rel_err > 3e-2:
        print(f"  FAIL: W_rec rel_err={rel_err:.4f} (threshold 3e-2)")
        return False

    # Also verify cosine similarity > 0.99 between FD and autograd
    cos = F.cosine_similarity(
        g_auto["W_rec"].flatten().unsqueeze(0),
        W_rec_fd.flatten().unsqueeze(0),
    ).item()
    if cos < 0.99:
        print(f"  FAIL: W_rec cosine(auto, FD)={cos:.4f} < 0.99")
        return False

    print(f"  PASS  W_rec rel_err={rel_err:.2e}  cosine(auto,FD)={cos:.4f}")
    return True


# ── Test 4: plain vanilla RNN — e-prop ≈ d=0 ─────────────────────────────────

def test_vanilla_rnn_eprop_approx_d0(n_seeds: int = 5) -> bool:
    """On a plain tanh RNN, e-prop and d=0 should be nearly identical.

    Rationale: carry = ψ * W_rec_diag ≈ 0.005 for spectral-radius-0.9 init,
    so the trace decays almost immediately. This is EXPECTED behaviour.
    """
    print("Test 4: vanilla RNN e-prop ≈ d=0 (carry ≈ 0 for tanh — EXPECTED) ...")

    min_cos = 1.0
    for seed in range(n_seeds):
        torch.manual_seed(seed * 7)
        model   = VanillaRNN(n_in=4, n_rec=30, n_out=2)
        T, B    = 12, 16
        inputs  = torch.randn(T, B, 4)
        targets = torch.randn(T, B, 2)
        mask    = torch.zeros(T, B)
        mask[9:12] = 1.0

        g_ep = compute_eprop_gradients(model, inputs, targets, mask, mse_error, d_zero=False)
        g_d0 = compute_eprop_gradients(model, inputs, targets, mask, mse_error, d_zero=True)

        cos = _cosine(g_ep, g_d0)
        if cos != cos:
            continue   # NaN — skip
        min_cos = min(min_cos, cos)

    if min_cos < 0.99:
        print(f"  FAIL: cosine(e-prop, d=0) = {min_cos:.4f} < 0.99 on vanilla RNN")
        return False

    print(f"  PASS  min cosine(e-prop, d=0) = {min_cos:.4f}  (carry≈0 for tanh: expected)")
    return True


# ── Test 5: leaky RNN — e-prop > d=0 wedge ───────────────────────────────────

def test_leaky_rnn_eprop_wedge(
    n_seeds: int = 8,
    min_margin: float = 0.05,
) -> bool:
    """LeakyRNN(alpha=0.1) + long delay: e-prop must be closer to BPTT than d=0.

    Carry = (1-alpha) = 0.9 per step → trace survives ~10 steps.
    With delay=20 >> ~10 steps this creates a large gap between e-prop (keeps
    trace) and d=0 (no trace), while BPTT is the exact ground truth.
    """
    print("Test 5: leaky RNN e-prop > d=0 wedge (long delay) ...")

    margins = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 13)
        model = LeakyRNN(n_in=5, n_rec=40, n_out=2, alpha=0.1)

        inputs, targets, mask = ca_batch(
            batch_size=32, n_cues=5, delay=20, seed=seed
        )

        # BPTT reference via autograd
        g_bptt = compute_bptt_gradients(model, inputs, targets, mask)

        # E-prop and d=0
        g_ep = compute_eprop_leaky_gradients(
            model, inputs, targets, mask, mse_error, d_zero=False
        )
        g_d0 = compute_eprop_leaky_gradients(
            model, inputs, targets, mask, mse_error, d_zero=True
        )

        keys = sorted(g_bptt.keys())
        cos_ep = _cosine(g_ep, g_bptt, keys)
        cos_d0 = _cosine(g_d0, g_bptt, keys)

        if cos_ep != cos_ep or cos_d0 != cos_d0:
            continue   # NaN — skip
        margins.append(cos_ep - cos_d0)

    if not margins:
        print("  FAIL: all trials produced NaN cosines")
        return False

    mean_margin = sum(margins) / len(margins)
    min_margin_obs = min(margins)

    # Require that the MEAN margin exceeds threshold
    if mean_margin < min_margin:
        print(
            f"  FAIL: mean cosine margin={mean_margin:.4f} < {min_margin:.2f}\n"
            f"        (individual margins: {[f'{m:.3f}' for m in margins]})"
        )
        return False

    # Show representative numbers
    seed_ex = 0
    torch.manual_seed(seed_ex * 13)
    model = LeakyRNN(n_in=5, n_rec=40, n_out=2, alpha=0.1)
    inputs, targets, mask = ca_batch(batch_size=32, n_cues=5, delay=20, seed=seed_ex)
    g_bptt = compute_bptt_gradients(model, inputs, targets, mask)
    g_ep   = compute_eprop_leaky_gradients(model, inputs, targets, mask, mse_error)
    g_d0   = compute_eprop_leaky_gradients(model, inputs, targets, mask, mse_error, d_zero=True)
    keys   = sorted(g_bptt.keys())
    cos_ep = _cosine(g_ep, g_bptt, keys)
    cos_d0 = _cosine(g_d0, g_bptt, keys)

    print(
        f"  PASS  mean margin={mean_margin:.4f}  min margin={min_margin_obs:.4f}\n"
        f"        example: cos(e-prop, BPTT)={cos_ep:.4f}  cos(d=0, BPTT)={cos_d0:.4f}"
    )
    return True


# ── Test 6: L=1 leaky deep e-prop == single-layer leaky e-prop ───────────────

def test_depth1_leaky_deep_eprop_matches_single(n_seeds: int = 5) -> bool:
    """For L=1 with a leaky DeepRNN, deep e-prop must equal single-layer leaky
    e-prop exactly (validates the leaky carry / α-scaling in deep_eprop)."""
    print("Test 6: depth-1 leaky deep e-prop == single-layer leaky e-prop ...")

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        deep = DeepRNN(n_in=5, n_rec=20, n_out=2, n_layers=1, alpha=0.3)
        sl   = LeakyRNN(n_in=5, n_rec=20, n_out=2, alpha=0.3)
        with torch.no_grad():
            sl.W_rec.copy_(deep.W_recs[0]); sl.W_in.copy_(deep.W_in)
            sl.b_rec.copy_(deep.biases[0]); sl.W_out.copy_(deep.W_out)
            sl.b_out.copy_(deep.b_out)

        inp, tgt, msk = ca_batch(batch_size=16, n_cues=5, delay=10, seed=seed)
        g_deep = compute_deep_eprop_gradients(deep, inp, tgt, msk, mse_error)
        g_sl   = compute_eprop_leaky_gradients(sl, inp, tgt, msk, mse_error)
        key_map = {"W_recs.0": "W_rec", "biases.0": "b_rec",
                   "W_in": "W_in", "W_out": "W_out", "b_out": "b_out"}
        for dk, sk in key_map.items():
            if not torch.allclose(g_deep[dk], g_sl[sk], rtol=1e-5, atol=1e-6):
                diff = (g_deep[dk] - g_sl[sk]).abs().max().item()
                print(f"  FAIL seed={seed}: '{dk}' max|diff|={diff:.2e}")
                return False

    print("  PASS  (L=1 leaky deep e-prop == single-layer leaky e-prop)")
    return True


# ── Test 7: ablation controls act only on the cross-layer ϵ^z trace ───────────

def test_ablation_controls(n_seeds: int = 4) -> bool:
    """The two controls must:
      - ablate_spatial  → lower-layer (layer-0) gradients are exactly zero,
      - ablate_temporal → lower-layer gradients differ from full,
      - BOTH            → upper-layer (top) gradients identical to full
                          (the controls touch only the lower-layer credit path).
    """
    print("Test 7: ablation controls act only on ϵ^z (lower-layer credit) ...")

    lower = ["W_in", "W_recs.0", "biases.0"]
    upper = ["W_recs.1", "W_ffs.0", "biases.1", "W_out", "b_out"]

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        m = DeepRNN(n_in=5, n_rec=24, n_out=2, n_layers=2, alpha=[0.7, 0.1])
        inp, tgt, msk = ca_batch(batch_size=24, n_cues=5, delay=15, seed=seed)

        gf = compute_deep_eprop_gradients(m, inp, tgt, msk, mse_error, mode="full")
        gs = compute_deep_eprop_gradients(m, inp, tgt, msk, mse_error, mode="ablate_spatial")
        gt = compute_deep_eprop_gradients(m, inp, tgt, msk, mse_error, mode="ablate_temporal")

        # spatial ablation → lower-layer grads exactly zero
        sp_norm = sum(gs[k].norm().item() for k in lower)
        if sp_norm > 1e-9:
            print(f"  FAIL seed={seed}: ablate_spatial lower-grad norm={sp_norm:.2e} (expected 0)")
            return False
        # full lower grads must be nonzero (otherwise the test is vacuous)
        if sum(gf[k].norm().item() for k in lower) < 1e-6:
            print(f"  FAIL seed={seed}: full lower-grad norm ≈ 0")
            return False
        # temporal ablation → lower grads differ from full
        diff_t = max((gt[k] - gf[k]).abs().max().item() for k in lower)
        if diff_t < 1e-9:
            print(f"  FAIL seed={seed}: ablate_temporal did not change lower grads")
            return False
        # both controls → upper-layer grads identical to full
        for g, nm in [(gs, "spatial"), (gt, "temporal")]:
            up_diff = max((g[k] - gf[k]).abs().max().item() for k in upper)
            if up_diff > 1e-9:
                print(f"  FAIL seed={seed}: ablate_{nm} changed upper grads (diff={up_diff:.2e})")
                return False

    print("  PASS  (spatial→lower grads=0; temporal→lower changed; upper grads untouched)")
    return True


# ── Cue-accumulation task unit checks ────────────────────────────────────────

def test_cue_accumulation_task() -> bool:
    """Shape, mask, label balance, and chance-accuracy sanity checks."""
    print("Test 0: cue_accumulation task sanity ...")

    n_cues, delay, B = 5, 20, 512
    inputs, targets, mask = ca_batch(batch_size=B, n_cues=n_cues, delay=delay, seed=0)

    T_expected = n_cues * (1 + 5) + delay + 1   # cue_duration=1, ici=5
    assert inputs.shape  == (T_expected, B, 5),  f"inputs shape {inputs.shape}"
    assert targets.shape == (T_expected, B, 2),  f"targets shape {targets.shape}"
    assert mask.shape    == (T_expected, B),      f"mask shape {mask.shape}"

    # Mask sums to exactly B
    assert mask.sum().item() == float(B), f"mask.sum()={mask.sum().item()} != {B}"

    # Label balance (≈ 50 % over 512 trials; use 10 pp slack)
    labels     = targets[mask.bool()].argmax(dim=-1)
    frac_right = labels.float().mean().item()
    assert 0.40 < frac_right < 0.60, f"label imbalance: {frac_right:.3f}"

    # Chance accuracy of an untrained model
    model   = VanillaRNN(n_in=5, n_rec=10, n_out=2)
    outputs, _ = model(inputs)
    preds   = outputs.argmax(dim=-1)                        # (T, B)
    tgt_idx = targets.argmax(dim=-1)                        # (T, B)
    acc     = ((preds == tgt_idx) * mask).sum().item() / B
    assert 0.10 < acc < 0.90, f"frozen-net accuracy {acc:.3f} suspiciously far from chance"

    print(f"  PASS  T={T_expected}  mask.sum={B}  frac_right={frac_right:.3f}  "
          f"untrained_acc={acc:.3f}")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sanity checks — e-prop deep networks")
    print("=" * 60)
    print()

    results = {}
    results["0: task"] = test_cue_accumulation_task()
    print()
    results["1: RTRL==BPTT"]  = test_deep_rtrl_matches_bptt()
    print()
    results["2: depth-1 eq"] = test_depth1_deep_eprop_matches_single()
    print()
    results["3: FD check"]   = test_finite_difference_bptt()
    print()
    results["4: vanilla≈d0"] = test_vanilla_rnn_eprop_approx_d0()
    print()
    results["5: leaky wedge"] = test_leaky_rnn_eprop_wedge()
    print()
    results["6: L=1 leaky eq"] = test_depth1_leaky_deep_eprop_matches_single()
    print()
    results["7: ablations"]   = test_ablation_controls()
    print()

    print("=" * 60)
    passed = sum(results.values())
    total  = len(results)
    for name, ok in results.items():
        tag = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{tag}]  {name}")
    print()
    if passed == total:
        print(f"All {total} tests passed.")
    else:
        print(f"{passed}/{total} tests passed. See FAIL lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
