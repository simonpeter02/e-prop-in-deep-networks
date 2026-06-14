"""
Experiment 5: Deep LIF network on Spiking Heidelberg Digits (SHD).

Demonstrates the BPTT > e-prop > d=0 ordering on a real spiking dataset:
  - BPTT : full temporal + spatial credit via autograd (achieves 65-80% accuracy)
  - e-prop: per-layer temporal carry (LIF carry ≈ 0.6, ~15 step horizon) + spatial
  - d=0  : no temporal carry; only instantaneous eligibility terms

Two plots are produced:
  1. Training curves (accuracy vs step) for all three methods
  2. Gradient cosine similarity vs temporal extent T (sweep over truncated SHD)

Run:
    python -m experiments.exp5_shd_lif

SHD will be downloaded automatically to /tmp/ on first run (~150 MB).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.shd import generate_batch, task_accuracy, T_SHD, N_IN_SHD, N_CLASSES
from models.deep_lif import DeepLIFNetwork
from learning_rules.deep_eprop_lif import compute_deep_eprop_lif_gradients, xent_error
from learning_rules.bptt import compute_bptt_gradients, _xent_loss

# ── Hyperparameters ───────────────────────────────────────────────────────────

SEED       = 42
N_REC      = 256
N_LAYERS   = 2
ALPHA      = 0.9      # membrane decay; carry ≈ alpha - v_th*gamma ≈ 0.87; horizon ~8 steps
V_TH       = 0.1
GAMMA      = 0.3

# SHD inputs are binary spike trains across 700 cochlear channels with ~1% sparsity.
# Standard 1/sqrt(n_in) init gives drive std ≈ (1-alpha)*sqrt(p*n_in)/sqrt(n_in) = 0.004,
# far below v_th=0.1.  Scale W_in so that drive std ≈ v_th (~16% firing rate in Layer 0).
# Derivation: (1-alpha)*sqrt(p*n_in)*scale ≈ v_th → scale ≈ v_th / (0.1*sqrt(7)) ≈ 3.8;
# use 5.0 for lower actual SHD density.
W_IN_SCALE = 5.0

# Layer 1+ also need scaled feedforward weights.  After Layer 0 fires at ~5%,
# the W_ff drive to Layer 1 needs (1-alpha)*sqrt(p0*n_rec)*W_ff_std ≈ v_th.
# With p0≈0.05, n_rec=256, W_ff_init_std≈0.031 → need ≈8x scaling.
W_FF_SCALE = 8.0

BATCH_SIZE = 64
N_STEPS    = 3000
EVAL_EVERY = 100
LR         = 1e-3
GRAD_CLIP  = 1.0
DEVICE     = "cpu"

# Cosine sweep: truncate SHD to first T_use bins to vary temporal difficulty
T_SWEEP          = [10, 20, 50, 100]
N_COSINE_TRIALS  = 50   # random batches per T_use value

# Keys used for cosine similarity (hidden weights only — W_out is immediate, same for all)
HIDDEN_KEYS = (
    ['W_in']
    + [f'W_recs.{l}' for l in range(N_LAYERS)]
    + [f'W_ffs.{l}'  for l in range(N_LAYERS - 1)]
    + [f'b_recs.{l}' for l in range(N_LAYERS)]
)

torch.manual_seed(SEED)


# ── Utilities ─────────────────────────────────────────────────────────────────

def grad_cosine(g1: dict, g2: dict, keys=None) -> float:
    """Cosine similarity between two gradient dicts over the given keys."""
    if keys is None:
        keys = HIDDEN_KEYS
    v1 = torch.cat([g1[k].flatten() for k in keys])
    v2 = torch.cat([g2[k].flatten() for k in keys])
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def make_model() -> DeepLIFNetwork:
    m = DeepLIFNetwork(
        n_in=N_IN_SHD, n_rec=N_REC, n_out=N_CLASSES,
        n_layers=N_LAYERS, alpha=ALPHA, v_th=V_TH, gamma=GAMMA,
        w_in_scale=W_IN_SCALE,
    ).to(DEVICE)
    with torch.no_grad():
        for W_ff in m.W_ffs:
            W_ff.data *= W_FF_SCALE
    return m


def apply_eprop_grads(model: DeepLIFNetwork, grads: dict, optimizer: torch.optim.Optimizer):
    """Set param.grad from the manually-computed gradient dict, then step Adam."""
    for name, param in model.named_parameters():
        if name in grads:
            param.grad = grads[name].clone()
        else:
            param.grad = None
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad()


# ── Training loop (one method) ────────────────────────────────────────────────

def train(label: str, use_bptt: bool = False, d_zero: bool = False) -> dict:
    torch.manual_seed(SEED)
    model  = make_model()
    optim  = torch.optim.Adam(model.parameters(), lr=LR)

    accs_train = []
    accs_test  = []
    steps_log  = []

    for step in range(N_STEPS):
        inputs, targets, mask = generate_batch(BATCH_SIZE, device=DEVICE, train=True)

        if use_bptt:
            # BPTT via autograd
            optim.zero_grad()
            outputs, _ = model(inputs)
            loss = _xent_loss(outputs, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
        else:
            grads = compute_deep_eprop_lif_gradients(
                model, inputs, targets, mask,
                learning_signal_fn=xent_error,
                d_zero=d_zero,
            )
            apply_eprop_grads(model, grads, optim)

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                outputs, _ = model(inputs)
            acc_tr = task_accuracy(outputs, targets, mask)

            t_in, t_tgt, t_msk = generate_batch(BATCH_SIZE, device=DEVICE, train=False)
            with torch.no_grad():
                t_out, _ = model(t_in)
            acc_te = task_accuracy(t_out, t_tgt, t_msk)

            accs_train.append(acc_tr)
            accs_test.append(acc_te)
            steps_log.append(step)
            print(f"[{label}] step {step:4d}  train={acc_tr:.3f}  test={acc_te:.3f}")

    return {'steps': steps_log, 'train': accs_train, 'test': accs_test}


# ── Gradient cosine vs temporal extent ────────────────────────────────────────

def cosine_vs_T(t_sweep=T_SWEEP, n_trials=N_COSINE_TRIALS):
    """
    For each T_use, truncate SHD to the first T_use time bins and compute
    gradient cosine similarity between (e-prop, BPTT) and (d=0, BPTT)
    on an untrained model.
    """
    torch.manual_seed(SEED)
    model = make_model()

    cos_eprop = []
    cos_d0    = []

    for T_use in t_sweep:
        sims_e, sims_d = [], []
        for _ in range(n_trials):
            inp, tgt, msk = generate_batch(BATCH_SIZE, device=DEVICE, train=True)
            # Truncate to first T_use bins, but keep the class labels at the
            # FINAL step of the truncated sequence (tgt[-1] holds the true labels;
            # intermediate bins in tgt are all-zero).
            inp_trunc = inp[:T_use]
            tgt_trunc = torch.zeros(T_use, BATCH_SIZE, N_CLASSES, device=DEVICE)
            tgt_trunc[-1] = tgt[-1]   # class label always at final step
            msk_trunc = torch.zeros(T_use, BATCH_SIZE, device=DEVICE)
            msk_trunc[-1] = 1.0

            g_bptt  = compute_bptt_gradients(model, inp_trunc, tgt_trunc, msk_trunc, _xent_loss)
            g_eprop = compute_deep_eprop_lif_gradients(
                model, inp_trunc, tgt_trunc, msk_trunc, xent_error, d_zero=False)
            g_d0    = compute_deep_eprop_lif_gradients(
                model, inp_trunc, tgt_trunc, msk_trunc, xent_error, d_zero=True)

            sims_e.append(grad_cosine(g_eprop, g_bptt))
            sims_d.append(grad_cosine(g_d0,    g_bptt))

        cos_eprop.append(float(np.mean(sims_e)))
        cos_d0.append(float(np.mean(sims_d)))
        print(f"  T_use={T_use:3d}  cos(e-prop,bptt)={cos_eprop[-1]:.4f}"
              f"  cos(d=0,bptt)={cos_d0[-1]:.4f}")

    return cos_eprop, cos_d0


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("=== Training: BPTT ===")
    res_bptt  = train("BPTT",   use_bptt=True)

    print("\n=== Training: e-prop ===")
    res_eprop = train("e-prop", use_bptt=False, d_zero=False)

    print("\n=== Training: d=0 ===")
    res_d0    = train("d=0",    use_bptt=False, d_zero=True)

    steps = res_bptt['steps']

    # ── Plot 1: Training curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for res, label, marker in [
        (res_bptt,  "BPTT",   "^"),
        (res_eprop, "e-prop", "o"),
        (res_d0,    "d=0",    "s"),
    ]:
        ax.plot(steps, res['test'], label=label, marker=marker, markersize=3)

    ax.axhline(1.0 / N_CLASSES, color="gray", linestyle="--", label="chance (1/20)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"SHD — deep LIF ({N_LAYERS}×{N_REC})  α={ALPHA}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp5_shd_lif_learning_curves.pdf")
    fig.savefig("results/exp5_shd_lif_learning_curves.svg")
    plt.close(fig)
    print("\nSaved results/exp5_shd_lif_learning_curves.pdf/.svg")

    # ── Plot 2: Gradient cosine vs temporal extent ────────────────────────────
    print("\n=== Gradient cosine similarity vs T_use ===")
    cos_e, cos_d = cosine_vs_T()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_SWEEP, cos_e, label="e-prop vs BPTT", marker="o")
    ax.plot(T_SWEEP, cos_d, label="d=0 vs BPTT",    marker="s")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Sequence length T (SHD bins used)")
    ax.set_ylabel("Gradient cosine similarity")
    ax.set_title(f"Gradient alignment with BPTT — untrained deep LIF ({N_LAYERS}×{N_REC})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp5_shd_lif_cosine_vs_T.pdf")
    fig.savefig("results/exp5_shd_lif_cosine_vs_T.svg")
    plt.close(fig)
    print("Saved results/exp5_shd_lif_cosine_vs_T.pdf/.svg")
