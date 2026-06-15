"""
Experiment 5: Deep ALIF network on Spiking Heidelberg Digits (SHD).

ALIF neurons carry two eligibility traces:
  - Fast trace (decay carry ≈ alpha - v_th*psi ≈ 0.87): ~8-step horizon
  - Slow trace (decay rho = 0.98):                      ~50-step horizon

This gives three clearly separated gradient estimators:
  BPTT   : full 100-step credit via autograd through adaptation chain
  e-prop : ~50-step horizon via slow ALIF trace + ~8-step fast trace
  d=0    : only instantaneous eligibility (no temporal carry whatsoever)

Expected: BPTT >> e-prop > d=0 on a T=100 classification task.

Run:
    python -m experiments.exp5_shd_alif

SHD is auto-downloaded to /tmp/ on first run (~150 MB).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.shd                      import generate_batch, task_accuracy, T_SHD, N_IN_SHD, N_CLASSES
from models.deep_alif               import DeepALIFNetwork
from learning_rules.deep_eprop_alif import compute_deep_eprop_alif_gradients, xent_error
from learning_rules.bptt            import compute_bptt_gradients, _xent_loss

# ── Hyperparameters ────────────────────────────────────────────────────────────

SEED       = 42
N_REC      = 256
N_LAYERS   = 2
ALPHA      = 0.9    # membrane: fast trace carry ≈ 0.87, 1/e horizon ~8 steps
RHO        = 0.98   # adaptation: slow trace 1/e horizon ~50 steps ≈ T/2
# beta must scale with (1-rho) since a_t = rho*a_{t-1} + s_{t-1} (no (1-rho) factor):
# at 5% firing, a_ss = 0.05/0.02 = 2.5 → threshold shift = 0.02*2.5 = 0.05 = v_th/2
BETA       = 0.02
V_TH       = 0.1
GAMMA      = 0.3

# SHD inputs are ~1% sparse across 700 channels; scale W_in/W_ff to get
# 3–20% firing rate at init (same reasoning as deep_lif.py)
W_IN_SCALE = 5.0
W_FF_SCALE = 8.0

BATCH_SIZE = 64
N_STEPS    = 3000
EVAL_EVERY = 100
LR         = 1e-3
GRAD_CLIP  = 1.0
DEVICE     = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

T_SWEEP         = [10, 20, 50, 100]
N_COSINE_TRIALS = 50

HIDDEN_KEYS = (
    ['W_in']
    + [f'W_recs.{l}' for l in range(N_LAYERS)]
    + [f'W_ffs.{l}'  for l in range(N_LAYERS - 1)]
    + [f'b_recs.{l}' for l in range(N_LAYERS)]
)

torch.manual_seed(SEED)


def grad_cosine(g1: dict, g2: dict, keys=None) -> float:
    if keys is None:
        keys = HIDDEN_KEYS
    v1 = torch.cat([g1[k].flatten() for k in keys])
    v2 = torch.cat([g2[k].flatten() for k in keys])
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def make_model() -> DeepALIFNetwork:
    m = DeepALIFNetwork(
        n_in=N_IN_SHD, n_rec=N_REC, n_out=N_CLASSES,
        n_layers=N_LAYERS, alpha=ALPHA, rho=RHO, beta=BETA,
        v_th=V_TH, gamma=GAMMA, w_in_scale=W_IN_SCALE,
    ).to(DEVICE)
    with torch.no_grad():
        for W_ff in m.W_ffs:
            W_ff.data *= W_FF_SCALE
    return m


def check_firing_rates(model: DeepALIFNetwork, n_batches: int = 5):
    inp = (torch.rand(T_SHD, n_batches * BATCH_SIZE, N_IN_SHD) < 0.01).float().to(DEVICE)
    with torch.no_grad():
        _, state = model(inp)
    for l, (_, s_seq, _) in enumerate(state):
        rate = s_seq.mean().item()
        dead = (s_seq.sum(0).sum(0) == 0).float().mean().item()
        print(f"  Layer {l}: firing rate = {rate:.4f}  dead = {dead:.1%}")


def apply_eprop_grads(model, grads, optimizer):
    for name, p in model.named_parameters():
        p.grad = grads.get(name, torch.zeros_like(p))
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad()


def train(label: str, use_bptt: bool = False, d_zero: bool = False) -> dict:
    torch.manual_seed(SEED)
    model = make_model()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    accs_train, accs_test, steps_log = [], [], []

    for step in range(N_STEPS):
        inputs, targets, mask = generate_batch(BATCH_SIZE, device=DEVICE, train=True)

        if use_bptt:
            optim.zero_grad()
            outputs, _ = model(inputs)
            _xent_loss(outputs, targets, mask).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
        else:
            grads = compute_deep_eprop_alif_gradients(
                model, inputs, targets, mask, xent_error, d_zero=d_zero)
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


def cosine_vs_T(t_sweep=T_SWEEP, n_trials=N_COSINE_TRIALS):
    torch.manual_seed(SEED)
    model = make_model()
    cos_eprop, cos_d0 = [], []

    for T_use in t_sweep:
        sims_e, sims_d = [], []
        for _ in range(n_trials):
            inp, tgt, _ = generate_batch(BATCH_SIZE, device=DEVICE, train=True)
            inp_tr = inp[:T_use]
            tgt_tr = torch.zeros(T_use, BATCH_SIZE, N_CLASSES, device=DEVICE)
            tgt_tr[-1] = tgt[-1]
            msk_tr = torch.zeros(T_use, BATCH_SIZE, device=DEVICE)
            msk_tr[-1] = 1.0

            g_bptt  = compute_bptt_gradients(model, inp_tr, tgt_tr, msk_tr, _xent_loss)
            g_eprop = compute_deep_eprop_alif_gradients(
                model, inp_tr, tgt_tr, msk_tr, xent_error, d_zero=False)
            g_d0    = compute_deep_eprop_alif_gradients(
                model, inp_tr, tgt_tr, msk_tr, xent_error, d_zero=True)

            sims_e.append(grad_cosine(g_eprop, g_bptt))
            sims_d.append(grad_cosine(g_d0,    g_bptt))

        cos_eprop.append(float(np.mean(sims_e)))
        cos_d0.append(float(np.mean(sims_d)))
        print(f"  T={T_use:3d}  cos(e-prop,BPTT)={cos_eprop[-1]:.4f}"
              f"  cos(d=0,BPTT)={cos_d0[-1]:.4f}")

    return cos_eprop, cos_d0


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("=== Firing-rate sanity check at init ===")
    torch.manual_seed(SEED)
    check_firing_rates(make_model())

    print("\n=== Training: BPTT ===")
    res_bptt  = train("BPTT",   use_bptt=True)

    print("\n=== Training: e-prop ===")
    res_eprop = train("e-prop", use_bptt=False, d_zero=False)

    print("\n=== Training: d=0 ===")
    res_d0    = train("d=0",    use_bptt=False, d_zero=True)

    steps = res_bptt['steps']

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
    ax.set_title(f"SHD — deep ALIF ({N_LAYERS}×{N_REC})  α={ALPHA}  ρ={RHO}  β={BETA}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp5_shd_alif_learning_curves.pdf")
    fig.savefig("results/exp5_shd_alif_learning_curves.svg")
    plt.close(fig)
    print("\nSaved results/exp5_shd_alif_learning_curves.pdf/.svg")

    print("\n=== Gradient cosine similarity vs T_use ===")
    cos_e, cos_d = cosine_vs_T()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_SWEEP, cos_e, label="e-prop vs BPTT", marker="o")
    ax.plot(T_SWEEP, cos_d, label="d=0 vs BPTT",    marker="s")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Sequence length T (SHD bins used)")
    ax.set_ylabel("Gradient cosine similarity")
    ax.set_title(f"Gradient alignment — untrained deep ALIF ({N_LAYERS}×{N_REC})  ρ={RHO}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp5_shd_alif_cosine_vs_T.pdf")
    fig.savefig("results/exp5_shd_alif_cosine_vs_T.svg")
    plt.close(fig)
    print("Saved results/exp5_shd_alif_cosine_vs_T.pdf/.svg")
