"""
Experiment 2: Deep e-prop vs d=0 vs BPTT — 2-layer vanilla RNN.

Covers minimal viable results #2 and #3:
  (a) Verify deep-RTRL matches BPTT to numerical precision.
  (b) Compare deep e-prop vs d=0 vs BPTT gradient cosine similarity at
      2 layers, sweeping delay length and tracking per-layer alignment.

Run:
    python -m experiments.deep_eprop_comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tasks.store_and_recall import generate_batch, task_accuracy
from models.deep_rnn import DeepRNN
from learning_rules.deep_rtrl   import compute_deep_rtrl_gradients
from learning_rules.deep_rtrl   import mse_error
from learning_rules.deep_eprop  import compute_deep_eprop_gradients
from learning_rules.bptt        import _mse_loss


# ── Configuration ─────────────────────────────────────────────────────────────
SEED         = 42
N_PATTERNS   = 4
N_REC_VERIFY = 10      # small n for deep-RTRL verification (O(n^4) cost)
N_REC_MAIN   = 50      # larger n for the cosine sweep
N_LAYERS     = 2
DELAY_MAIN   = 2
BATCH_SIZE   = 32
DEVICE       = "cpu"

torch.manual_seed(SEED)


# ── BPTT helper for deep network ──────────────────────────────────────────────
def bptt_grads(model: DeepRNN, inputs, targets, mask):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    outputs, _ = model(inputs)
    loss = _mse_loss(outputs, targets, mask)
    loss.backward()
    return {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}


# ── Cosine helper ─────────────────────────────────────────────────────────────
def cos(g1, g2, key):
    v1 = g1[key].flatten()
    v2 = g2[key].flatten()
    if v1.norm() < 1e-12 or v2.norm() < 1e-12:
        return float('nan')
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


# ── Part A: Deep-RTRL vs BPTT verification ───────────────────────────────────
def rtrl_verification(n_reps=20):
    print("=== Part A: Deep-RTRL vs BPTT verification ===")
    n_in = N_PATTERNS + 2
    max_dir_err = 0.0
    worst_key   = None

    for rep in range(n_reps):
        torch.manual_seed(SEED + rep)
        model  = DeepRNN(n_in, N_REC_VERIFY, N_PATTERNS, n_layers=2)
        inputs, targets, mask = generate_batch(8, N_PATTERNS, DELAY_MAIN, 1, 1, DEVICE)

        g_bptt = bptt_grads(model, inputs, targets, mask)
        g_rtrl = compute_deep_rtrl_gradients(model, inputs, targets, mask, mse_error)

        for k in ['W_recs.0', 'W_recs.1', 'W_ffs.0', 'W_in', 'biases.0', 'biases.1']:
            v1, v2 = g_bptt[k].flatten(), g_rtrl[k].flatten()
            if v1.norm() < 1e-12:
                continue
            dir_err = (v1 / v1.norm() - v2 / v2.norm()).norm().item()
            if dir_err > max_dir_err:
                max_dir_err = dir_err
                worst_key   = k

    print(f"  Max direction error over {n_reps} trials: {max_dir_err:.2e}  (worst: {worst_key})")
    if max_dir_err < 1e-4:
        print("  PASS: deep-RTRL matches BPTT to numerical precision ✓")
    else:
        print("  FAIL: deep-RTRL deviates from BPTT — check implementation")
    return max_dir_err


# ── Part B: Gradient cosine vs delay for all three methods ───────────────────
def cosine_sweep(delays, n_trials=30):
    """
    For each delay, compute mean gradient cosine similarity between
    (deep e-prop / d=0 / deep-RTRL) and BPTT on an untrained model.
    Returns dict  method -> dict  layer_group -> list of cosines (one per delay).
    """
    print("\n=== Part B: Gradient cosine similarity vs delay ===")
    n_in = N_PATTERNS + 2

    # Parameter groups (layer 1 = deeper, layer 2 = closer to output)
    layer1_keys = ['W_recs.0', 'W_in', 'biases.0']
    layer2_keys = ['W_recs.1', 'W_ffs.0', 'biases.1']

    results = {
        method: {grp: [] for grp in ['layer1', 'layer2', 'all']}
        for method in ['deep-eprop', 'd=0']
    }

    for delay in delays:
        sims = {m: {g: [] for g in ['layer1', 'layer2', 'all']} for m in ['deep-eprop', 'd=0']}

        for _ in range(n_trials):
            torch.manual_seed(np.random.randint(0, 10000))
            model  = DeepRNN(n_in, N_REC_MAIN, N_PATTERNS, n_layers=2)
            inputs, targets, mask = generate_batch(BATCH_SIZE, N_PATTERNS, delay, 1, 1, DEVICE)

            g_bptt  = bptt_grads(model, inputs, targets, mask)
            g_eprop = compute_deep_eprop_gradients(model, inputs, targets, mask, mse_error, d_zero=False)
            g_d0    = compute_deep_eprop_gradients(model, inputs, targets, mask, mse_error, d_zero=True)

            for method, g in [('deep-eprop', g_eprop), ('d=0', g_d0)]:
                c1 = np.mean([cos(g, g_bptt, k) for k in layer1_keys])
                c2 = np.mean([cos(g, g_bptt, k) for k in layer2_keys])
                ca = np.mean([cos(g, g_bptt, k) for k in layer1_keys + layer2_keys])
                sims[method]['layer1'].append(c1)
                sims[method]['layer2'].append(c2)
                sims[method]['all'].append(ca)

        for method in ['deep-eprop', 'd=0']:
            for grp in ['layer1', 'layer2', 'all']:
                results[method][grp].append(np.mean(sims[method][grp]))

        print(f"  delay={delay:3d}  "
              f"eprop(L1/L2)={results['deep-eprop']['layer1'][-1]:.3f}/{results['deep-eprop']['layer2'][-1]:.3f}  "
              f"d0(L1/L2)={results['d=0']['layer1'][-1]:.3f}/{results['d=0']['layer2'][-1]:.3f}")

    return results


# ── Part C: Learning curves at delay=2 ───────────────────────────────────────
def apply_grads(model: DeepRNN, grads: dict, lr: float):
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in grads:
                p.data -= lr * grads[k]


def learning_curve(label, use_bptt, d_zero, n_steps=1000, lr=1e-3, delay=DELAY_MAIN):
    torch.manual_seed(SEED)
    n_in = N_PATTERNS + 2
    model = DeepRNN(n_in, N_REC_MAIN, N_PATTERNS, n_layers=2)
    accs = []
    for step in range(n_steps):
        inputs, targets, mask = generate_batch(BATCH_SIZE, N_PATTERNS, delay, 1, 1, DEVICE)
        if use_bptt:
            grads = bptt_grads(model, inputs, targets, mask)
        else:
            grads = compute_deep_eprop_gradients(
                model, inputs, targets, mask, mse_error, d_zero=d_zero)
        apply_grads(model, grads, lr)
        if step % 50 == 0:
            with torch.no_grad():
                out, _ = model(inputs)
            accs.append(task_accuracy(out, targets, mask))
            print(f"  [{label}] step {step:4d}  acc={accs[-1]:.3f}")
    return accs


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    np.random.seed(SEED)

    # ── Part A ────────────────────────────────────────────────────────────────
    max_err = rtrl_verification(n_reps=30)

    # ── Part B ────────────────────────────────────────────────────────────────
    delays = [1, 2, 3, 5, 10, 20]
    results = cosine_sweep(delays, n_trials=30)

    colors = {'deep-eprop': 'tab:blue', 'd=0': 'tab:orange'}
    linestyles = {'layer1': '--', 'layer2': '-'}
    markers    = {'layer1': 's', 'layer2': 'o'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax_idx, (method, label) in enumerate([('deep-eprop', 'Deep e-prop'), ('d=0', 'd=0')]):
        ax = axes[ax_idx]
        ax.plot(delays, results[method]['layer1'],
                label='Layer 1 (deeper)', marker='s', linestyle='--', color=colors[method])
        ax.plot(delays, results[method]['layer2'],
                label='Layer 2 (output-adjacent)', marker='o', linestyle='-', color=colors[method])
        ax.set_xlabel('Delay length (steps)')
        ax.set_ylabel('Gradient cosine similarity with BPTT')
        ax.set_title(f'{label} vs BPTT — 2-layer network')
        ax.legend()
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_ylim(-0.1, 1.05)
    fig.suptitle('Deep e-prop and d=0 gradient alignment with BPTT')
    fig.tight_layout()
    fig.savefig('results/deep_cosine_by_layer.png', dpi=150)
    print("\nSaved results/deep_cosine_by_layer.png")

    # Combined comparison (all params) on one plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(delays, results['deep-eprop']['all'], label='Deep e-prop (all params)',
            marker='o', color='tab:blue')
    ax.plot(delays, results['d=0']['all'],        label='d=0 (all params)',
            marker='s', color='tab:orange')
    ax.plot(delays, results['deep-eprop']['layer1'], label='Deep e-prop (layer 1 only)',
            marker='^', linestyle='--', color='tab:blue', alpha=0.6)
    ax.plot(delays, results['d=0']['layer1'],        label='d=0 (layer 1 only)',
            marker='v', linestyle='--', color='tab:orange', alpha=0.6)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Delay length (steps)')
    ax.set_ylabel('Gradient cosine similarity with BPTT')
    ax.set_title('Deep e-prop vs d=0 — gradient alignment, 2-layer RNN')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    fig.savefig('results/deep_cosine_combined.png', dpi=150)
    print("Saved results/deep_cosine_combined.png")

    # ── Part C ────────────────────────────────────────────────────────────────
    print("\n=== Part C: Learning curves (2-layer, delay=2) ===")
    accs_eprop = learning_curve("deep-eprop", False, False)
    accs_d0    = learning_curve("d=0",        False, True)
    accs_bptt  = learning_curve("BPTT",       True,  False)

    steps = list(range(0, 1000, 50))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, accs_eprop, label='Deep e-prop', marker='o', markersize=3)
    ax.plot(steps, accs_d0,    label='d=0',         marker='s', markersize=3)
    ax.plot(steps, accs_bptt,  label='BPTT',        marker='^', markersize=3)
    ax.axhline(1.0 / N_PATTERNS, color='gray', linestyle='--', label='chance')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Store-and-recall (delay={DELAY_MAIN}, 2-layer RNN)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/deep_learning_curves.png', dpi=150)
    print("Saved results/deep_learning_curves.png")
