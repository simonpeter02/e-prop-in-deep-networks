"""
Experiment 1: Single-layer e-prop on the store-and-recall task.

Reproduces the standard e-prop result (Bellec et al. 2020):
  - trains a vanilla tanh RNN with e-prop
  - plots learning curves: e-prop vs d=0 vs BPTT
  - plots gradient cosine similarity between e-prop / d=0 and BPTT
    as a function of delay length

Run:
    python -m experiments.single_layer_eprop
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.store_and_recall import generate_batch, task_accuracy
from models.vanilla_rnn import VanillaRNN
from learning_rules.eprop import compute_eprop_gradients, mse_error, xent_error
from learning_rules.bptt import compute_bptt_gradients


# ── Hyperparameters ──────────────────────────────────────────────────────────
SEED         = 42
N_PATTERNS   = 4
N_REC        = 100
DELAY        = 2           # short delay: all methods should converge
CUE_DUR      = 1
OUT_DUR      = 1
BATCH_SIZE   = 32
N_STEPS      = 1000        # training iterations
LR_EPROP     = 1e-3
LR_BPTT      = 1e-3
EVAL_EVERY   = 50
DEVICE       = "cpu"

torch.manual_seed(SEED)


# ── Cosine similarity between two gradient dicts ──────────────────────────────
def grad_cosine(g1: dict, g2: dict, keys=('W_rec', 'W_in', 'b_rec')) -> float:
    v1 = torch.cat([g1[k].flatten() for k in keys])
    v2 = torch.cat([g2[k].flatten() for k in keys])
    return torch.nn.functional.cosine_similarity(v1.unsqueeze(0),
                                                  v2.unsqueeze(0)).item()


# ── Apply gradient dict to model with SGD ────────────────────────────────────
def apply_grads(model: VanillaRNN, grads: dict, lr: float):
    with torch.no_grad():
        model.W_rec.data -= lr * grads['W_rec']
        model.W_in.data  -= lr * grads['W_in']
        model.b_rec.data -= lr * grads['b_rec']
        model.W_out.data -= lr * grads['W_out']
        model.b_out.data -= lr * grads['b_out']


# ── Training loop ─────────────────────────────────────────────────────────────
def train(label: str, use_bptt: bool = False, d_zero: bool = False, lr: float = LR_EPROP):
    torch.manual_seed(SEED)
    n_in = N_PATTERNS + 2
    model = VanillaRNN(n_in, N_REC, N_PATTERNS)

    accs = []

    for step in range(N_STEPS):
        inputs, targets, mask = generate_batch(
            BATCH_SIZE, N_PATTERNS, DELAY, CUE_DUR, OUT_DUR, DEVICE
        )

        if use_bptt:
            grads = compute_bptt_gradients(model, inputs, targets, mask)
        else:
            grads = compute_eprop_gradients(
                model, inputs, targets, mask,
                learning_signal_fn=mse_error,
                d_zero=d_zero,
            )

        apply_grads(model, grads, lr)

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                outputs, _ = model(inputs)
            acc = task_accuracy(outputs, targets, mask)
            accs.append(acc)
            print(f"[{label}] step {step:4d}  acc={acc:.3f}")

    return accs


# ── Gradient cosine similarity vs delay ──────────────────────────────────────
def cosine_vs_delay(delays, n_trials=50):
    """
    For each delay value, average grad cosine similarity between
    e-prop (and d=0) vs BPTT over n_trials random batches on an
    untrained model.
    """
    n_in = N_PATTERNS + 2
    cos_eprop = []
    cos_d0    = []

    for delay in delays:
        torch.manual_seed(SEED)
        model = VanillaRNN(n_in, N_REC, N_PATTERNS)

        sims_e, sims_d = [], []
        for _ in range(n_trials):
            inputs, targets, mask_ = generate_batch(
                BATCH_SIZE, N_PATTERNS, delay, CUE_DUR, OUT_DUR, DEVICE
            )
            g_bptt  = compute_bptt_gradients(model, inputs, targets, mask_)
            g_eprop = compute_eprop_gradients(model, inputs, targets, mask_, mse_error, d_zero=False)
            g_d0    = compute_eprop_gradients(model, inputs, targets, mask_, mse_error, d_zero=True)

            sims_e.append(grad_cosine(g_eprop, g_bptt))
            sims_d.append(grad_cosine(g_d0,    g_bptt))

        cos_eprop.append(np.mean(sims_e))
        cos_d0.append(np.mean(sims_d))
        print(f"  delay={delay:3d}  cos(e-prop,bptt)={cos_eprop[-1]:.4f}  cos(d=0,bptt)={cos_d0[-1]:.4f}")

    return cos_eprop, cos_d0


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("=== Training runs ===")
    accs_eprop = train("e-prop",  use_bptt=False, d_zero=False)
    accs_d0    = train("d=0",     use_bptt=False, d_zero=True)
    accs_bptt  = train("BPTT",    use_bptt=True,  lr=LR_BPTT)

    steps = list(range(0, N_STEPS, EVAL_EVERY))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, accs_eprop, label="e-prop",  marker="o", markersize=3)
    ax.plot(steps, accs_d0,    label="d=0",     marker="s", markersize=3)
    ax.plot(steps, accs_bptt,  label="BPTT",    marker="^", markersize=3)
    ax.axhline(1.0 / N_PATTERNS, color="gray", linestyle="--", label="chance")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Store-and-recall  (delay={DELAY}, single layer)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp1_tanh_store_recall_delay2_learning_curves.pdf")
    fig.savefig("results/exp1_tanh_store_recall_delay2_learning_curves.svg")
    print("Saved results/learning_curves.pdf / .svg")

    print("\n=== Gradient cosine similarity vs delay ===")
    delays = [1, 2, 3, 5, 10, 20, 50]
    cos_e, cos_d = cosine_vs_delay(delays)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(delays, cos_e, label="e-prop vs BPTT", marker="o")
    ax.plot(delays, cos_d, label="d=0 vs BPTT",    marker="s")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Delay length (steps)")
    ax.set_ylabel("Gradient cosine similarity")
    ax.set_title("Gradient alignment with BPTT — single layer, untrained model")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp1_tanh_store_recall_cosine_vs_delay.pdf")
    fig.savefig("results/exp1_tanh_store_recall_cosine_vs_delay.svg")
    print("Saved results/cosine_vs_delay.pdf / .svg")

    print("\n=== Final accuracy by delay (500 training steps) ===")
    delay_sweep = [1, 2, 3, 5, 10, 20]
    n_in = N_PATTERNS + 2
    results = {m: [] for m in ["e-prop", "d=0", "BPTT"]}
    for dl in delay_sweep:
        for label, bptt_flag, dz in [("e-prop", False, False), ("d=0", False, True), ("BPTT", True, False)]:
            torch.manual_seed(SEED)
            model = VanillaRNN(n_in, N_REC, N_PATTERNS)
            for _ in range(500):
                inp, tgt, msk = generate_batch(BATCH_SIZE, N_PATTERNS, dl, CUE_DUR, OUT_DUR, DEVICE)
                if bptt_flag:
                    g = compute_bptt_gradients(model, inp, tgt, msk)
                else:
                    g = compute_eprop_gradients(model, inp, tgt, msk, mse_error, d_zero=dz)
                apply_grads(model, g, LR_EPROP)
            with torch.no_grad():
                out, _ = model(inp)
            results[label].append(task_accuracy(out, tgt, msk))
        print(f"  delay={dl:3d}  eprop={results['e-prop'][-1]:.2f}  d0={results['d=0'][-1]:.2f}  bptt={results['BPTT'][-1]:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, marker in [("e-prop", "o"), ("d=0", "s"), ("BPTT", "^")]:
        ax.plot(delay_sweep, results[label], label=label, marker=marker)
    ax.axhline(1.0 / N_PATTERNS, color="gray", linestyle="--", label="chance")
    ax.set_xlabel("Delay length")
    ax.set_ylabel("Final accuracy (after 500 steps)")
    ax.set_title("Store-and-recall accuracy vs delay — single layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp1_tanh_store_recall_accuracy_vs_delay.pdf")
    fig.savefig("results/exp1_tanh_store_recall_accuracy_vs_delay.svg")
    print("Saved results/accuracy_vs_delay.pdf / .svg")
