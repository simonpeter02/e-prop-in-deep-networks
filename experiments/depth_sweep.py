"""
Experiment 3: Depth sweep L=1..5 — deep e-prop vs d=0 vs BPTT.

Asks: How does credit assignment quality degrade with depth?
  - Gradient cosine similarity vs layer depth L (delays 2, 5, 10)
  - Learning curves for L=1,2,3,4,5 at delay=2

Run:
    python -m experiments.depth_sweep
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
from learning_rules.deep_eprop import compute_deep_eprop_gradients, mse_error
from learning_rules.bptt import _mse_loss


SEED       = 42
N_PATTERNS = 4
N_REC      = 50
BATCH_SIZE = 32
DEVICE     = "cpu"
DEPTHS     = [1, 2, 3, 4, 5]
DELAYS     = [2, 5, 10]
N_TRIALS   = 20   # gradient cosine trials per (depth, delay)
N_STEPS    = 800  # training steps for learning curves
LR         = 1e-3
EVAL_EVERY = 40

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────
def bptt_grads(model, inputs, targets, mask):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    outputs, _ = model(inputs)
    _mse_loss(outputs, targets, mask).backward()
    return {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}


def all_keys(model):
    return list(model.state_dict().keys())


def cosine_all_params(g_approx, g_bptt, model):
    """Average cosine similarity over all shared parameters."""
    sims = []
    for k in all_keys(model):
        if k not in g_approx or k not in g_bptt:
            continue
        v1 = g_approx[k].flatten()
        v2 = g_bptt[k].flatten()
        if v1.norm() < 1e-12 or v2.norm() < 1e-12:
            continue
        sims.append(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
    return float(np.mean(sims)) if sims else float('nan')


def apply_grads(model, grads, lr):
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in grads:
                p.data -= lr * grads[k]


# ── Part A: Cosine similarity vs depth ───────────────────────────────────────
def cosine_vs_depth():
    print("=== Part A: Gradient cosine vs depth ===")
    n_in = N_PATTERNS + 2
    results = {
        method: {delay: [] for delay in DELAYS}
        for method in ['deep-eprop', 'd=0']
    }

    for L in DEPTHS:
        for delay in DELAYS:
            sims_e, sims_d = [], []
            for trial in range(N_TRIALS):
                torch.manual_seed(SEED + trial * 100 + L * 10 + delay)
                model = DeepRNN(n_in, N_REC, N_PATTERNS, n_layers=L)
                inputs, targets, mask = generate_batch(
                    BATCH_SIZE, N_PATTERNS, delay, 1, 1, DEVICE)

                g_bptt  = bptt_grads(model, inputs, targets, mask)
                g_eprop = compute_deep_eprop_gradients(
                    model, inputs, targets, mask, mse_error, d_zero=False)
                g_d0    = compute_deep_eprop_gradients(
                    model, inputs, targets, mask, mse_error, d_zero=True)

                sims_e.append(cosine_all_params(g_eprop, g_bptt, model))
                sims_d.append(cosine_all_params(g_d0,   g_bptt, model))

            results['deep-eprop'][delay].append(np.nanmean(sims_e))
            results['d=0'][delay].append(np.nanmean(sims_d))

        print(f"  L={L}  " + "  ".join(
            f"d={d}: eprop={results['deep-eprop'][d][-1]:.3f} d0={results['d=0'][d][-1]:.3f}"
            for d in DELAYS))

    return results


# ── Part B: Learning curves at delay=2 ───────────────────────────────────────
def learning_curves_by_depth():
    print("\n=== Part B: Learning curves by depth (delay=2) ===")
    n_in = N_PATTERNS + 2
    delay = 2
    curves = {}   # (method, L) -> list of accuracies

    for method, d_zero, use_bptt in [
        ('deep-eprop', False, False),
        ('d=0',        True,  False),
        ('BPTT',       False, True),
    ]:
        for L in DEPTHS:
            torch.manual_seed(SEED)
            model = DeepRNN(n_in, N_REC, N_PATTERNS, n_layers=L)
            accs = []
            for step in range(N_STEPS):
                inputs, targets, mask = generate_batch(
                    BATCH_SIZE, N_PATTERNS, delay, 1, 1, DEVICE)
                if use_bptt:
                    grads = bptt_grads(model, inputs, targets, mask)
                else:
                    grads = compute_deep_eprop_gradients(
                        model, inputs, targets, mask, mse_error, d_zero=d_zero)
                apply_grads(model, grads, LR)
                if step % EVAL_EVERY == 0:
                    with torch.no_grad():
                        out, _ = model(inputs)
                    accs.append(task_accuracy(out, targets, mask))
            curves[(method, L)] = accs
            print(f"  [{method}, L={L}] final acc={accs[-1]:.3f}")

    return curves


# ── Plotting helpers ──────────────────────────────────────────────────────────
def save_fig(fig, name):
    os.makedirs("results", exist_ok=True)
    fig.savefig(f"results/{name}.pdf")
    fig.savefig(f"results/{name}.svg")
    print(f"Saved results/{name}.pdf / .svg")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # ── Part A ────────────────────────────────────────────────────────────────
    cosine_results = cosine_vs_depth()

    # Plot 1: cosine vs depth, one line per delay, two panels (e-prop / d=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DELAYS)))

    for ax, method, title in zip(axes,
                                  ['deep-eprop', 'd=0'],
                                  ['Deep e-prop', 'd=0']):
        for delay, c in zip(DELAYS, colors):
            ax.plot(DEPTHS, cosine_results[method][delay],
                    marker='o', color=c, label=f'delay={delay}')
        ax.set_xlabel('Number of layers L')
        ax.set_ylabel('Gradient cosine similarity (vs BPTT)')
        ax.set_title(f'{title}')
        ax.set_xticks(DEPTHS)
        ax.axhline(0, color='gray', linestyle=':')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle('Gradient alignment vs network depth — e-prop approximations')
    fig.tight_layout()
    save_fig(fig, 'depth_cosine_by_delay')
    plt.close(fig)

    # Plot 2: cosine vs depth, all delays and methods on one plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {'deep-eprop': 'o', 'd=0': 's'}
    ls_map  = {'deep-eprop': '-', 'd=0': '--'}
    for method in ['deep-eprop', 'd=0']:
        for delay, c in zip(DELAYS, colors):
            ax.plot(DEPTHS, cosine_results[method][delay],
                    marker=markers[method], linestyle=ls_map[method], color=c,
                    label=f'{method}, d={delay}', alpha=0.85)
    ax.set_xlabel('Number of layers L')
    ax.set_ylabel('Gradient cosine similarity (vs BPTT)')
    ax.set_title('Gradient alignment vs depth — deep e-prop and d=0')
    ax.set_xticks(DEPTHS)
    ax.axhline(0, color='gray', linestyle=':')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    save_fig(fig, 'depth_cosine_combined')
    plt.close(fig)

    # ── Part B ────────────────────────────────────────────────────────────────
    curves = learning_curves_by_depth()

    steps = list(range(0, N_STEPS, EVAL_EVERY))

    # Plot 3: learning curves per method, one line per depth
    method_labels = {'deep-eprop': 'Deep e-prop', 'd=0': 'd=0', 'BPTT': 'BPTT'}
    depth_colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(DEPTHS)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, method in zip(axes, ['deep-eprop', 'd=0', 'BPTT']):
        for L, c in zip(DEPTHS, depth_colors):
            ax.plot(steps, curves[(method, L)],
                    color=c, label=f'L={L}', marker='o', markersize=2.5)
        ax.axhline(1.0 / N_PATTERNS, color='gray', linestyle='--', label='chance')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Accuracy')
        ax.set_title(method_labels[method])
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle('Learning curves by depth — store-and-recall (delay=2)')
    fig.tight_layout()
    save_fig(fig, 'depth_learning_curves')
    plt.close(fig)

    # Plot 4: final accuracy vs depth for all methods
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, marker in [('deep-eprop', 'o'), ('d=0', 's'), ('BPTT', '^')]:
        finals = [curves[(method, L)][-1] for L in DEPTHS]
        ax.plot(DEPTHS, finals, marker=marker, label=method_labels[method])
    ax.axhline(1.0 / N_PATTERNS, color='gray', linestyle='--', label='chance')
    ax.set_xlabel('Number of layers L')
    ax.set_ylabel('Final accuracy (after training)')
    ax.set_title('Final accuracy vs depth — store-and-recall (delay=2)')
    ax.set_xticks(DEPTHS)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    save_fig(fig, 'depth_final_accuracy')
    plt.close(fig)
