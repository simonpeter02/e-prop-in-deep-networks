"""
Deep e-prop: credit assignment across time AND depth, simultaneously.

Main experiment for the claim that e-prop assigns credit correctly across both
the temporal (within-layer recurrence) and the depth (cross-layer) dimensions at
the same time — the deep generalisation of Bellec et al. (2020) derived by
Millidge (2025).

Architecture: 2-layer leaky DeepRNN with a FAST lower layer (transient feature
extractor, α=0.5) and a SLOW top layer (integrator, α=0.05).  Task: hierarchical
classify-then-count of mean-zero temporal motifs (tasks/hierarchical_cue.py),
which requires the lower layer to learn a genuine temporal feature (depth) whose
per-cue output must be accumulated by the top layer across a delay (time).
Mean-zero motifs ⇒ a frozen/random lower layer (reservoir) cannot fake the
feature, so removing lower-layer credit actually hurts.

Methods (all share the SAME forward model; only the gradient differs):
  bptt              — exact ground truth (autograd through time + depth)
  deep_eprop (full) — Millidge deep e-prop:  ϵ^z = (∂z/∂h)ϵ^h + (∂z/∂z_{t-1})ϵ^z_{t-1}
  ablate_spatial    — control: ∂z/∂h = 0    → removes DEPTH credit (lower grads → 0)
  ablate_temporal   — control: ∂z/∂z_{t-1}=0 → removes cross-layer TIME credit

Parts:
  E1  per-layer gradient cosine vs BPTT + cross-temporal credit fraction, vs delay
  E2  learning curves: BPTT ≥ full > both controls
  E3  delay sweep: final accuracy vs delay for each method

Runs on GPU (Colab CUDA), Apple MPS, or CPU — auto-detected.  Seeds are trained
in parallel with a THREAD pool: PyTorch releases the GIL during tensor ops, so
threads parallelise on CPU and share the CUDA context cleanly on GPU (no
process-pool / spawn / fork-vs-CUDA hangs).

Run:
    python -u -m experiments.deep_credit_time_depth          # all
    python -u -m experiments.deep_credit_time_depth e1       # one part
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from models.deep_rnn import DeepRNN
from learning_rules.deep_eprop import compute_deep_eprop_gradients, xent_error
from learning_rules.bptt import compute_bptt_gradients, _xent_loss
from learning_rules.interface import apply_gradients
from tasks.hierarchical_cue import generate_batch as hier_batch, task_accuracy
from utils import cosine_sim_grads, flat_grads


# ─────────────────────────── device ───────────────────────────
def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = os.environ.get("DEVICE") or _pick_device()   # override e.g. DEVICE=cpu
# One intra-op thread per process so the (fork) seed-pool gets one core each
# instead of oversubscribing (7 workers x 4 threads on 8 cores). Negligible cost
# for the sequential/GPU paths.
torch.set_num_threads(1)


# ─────────────────────────── config ───────────────────────────
SEED        = 0
N_REC       = 32
ALPHA       = [0.5, 0.05]      # [fast lower extractor, slow top integrator]
N_CUES      = 3
DELAY_MAIN  = 12               # delay for E1/E2
DELAYS      = [6, 12, 20, 32]  # for E1 cosine-vs-delay
BATCH       = 128 if DEVICE == "cuda" else 48
LR          = 5e-2
N_STEPS     = 1500             # training steps for E2 learning curves
EVAL_EVERY  = 100
N_SEEDS_COS = 12               # seeds for gradient-cosine averages (E1)
N_SEEDS_LC  = 3                # seeds for learning curves (E2)
# Parallel seeds across PROCESSES (fork) on CPU — this Python-loop-heavy code is
# GIL-bound, so threads don't help; separate processes give real multi-core
# speedup. On GPU we run sequentially (a process pool + CUDA is unsafe, and the
# GPU is the bottleneck anyway). Fork (Colab/Linux default; also on macOS) avoids
# the spawn pitfall of re-importing/re-running __main__ in each worker.
USE_POOL    = (DEVICE == "cpu") and (os.environ.get("NO_POOL") != "1") \
    and ("fork" in mp.get_all_start_methods())
N_WORKERS   = max(1, min(8, (os.cpu_count() or 4) - 1))

# E3 (delay sweep) trains a fresh net per delay.  Delays kept where BPTT learns.
E3_DELAYS   = [4, 8, 12, 16]
E3_SEEDS    = 3
E3_STEPS    = 1500

TASK_KW = dict(cue_duration=3, inter_cue_interval=2, amp=2.0, feature_noise=0.15)

LOWER = ["W_in", "W_recs.0", "biases.0"]    # lower-layer (layer-0) params
UPPER = ["W_recs.1", "W_ffs.0", "biases.1"] # top-layer (layer-1) recurrent params
METHODS_LC = [("bptt", "BPTT"), ("full", "deep e-prop (full)"),
              ("ablate_temporal", "ablate temporal"), ("ablate_spatial", "ablate spatial")]
COLORS = {"bptt": "k", "full": "C0", "ablate_temporal": "C3", "ablate_spatial": "C1"}

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


# ─────────────────────────── helpers ───────────────────────────
def new_model(seed):
    torch.manual_seed(seed)
    return DeepRNN(5, N_REC, 2, n_layers=2, alpha=ALPHA).to(DEVICE)


def _map(worker, items):
    """Map a top-level worker over items; seeds in parallel via a fork process
    pool on CPU, sequential on GPU. Falls back to sequential on any pool error."""
    if not USE_POOL or N_WORKERS <= 1 or len(items) <= 1:
        return [worker(x) for x in items]
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items)), mp_context=ctx) as ex:
            return list(ex.map(worker, items))
    except Exception as e:                      # pragma: no cover
        print(f"  [process pool failed ({type(e).__name__}: {e}); running sequentially]", flush=True)
        return [worker(x) for x in items]


def batch(B, delay, seed):
    return hier_batch(B, n_cues=N_CUES, delay=delay, seed=seed, device=DEVICE, **TASK_KW)


def grads_all(model, inp, tgt, msk):
    gb = compute_bptt_gradients(model, inp, tgt, msk, _xent_loss)
    gf = compute_deep_eprop_gradients(model, inp, tgt, msk, xent_error, mode="full")
    gs = compute_deep_eprop_gradients(model, inp, tgt, msk, xent_error, mode="ablate_spatial")
    gt = compute_deep_eprop_gradients(model, inp, tgt, msk, xent_error, mode="ablate_temporal")
    return gb, gf, gs, gt


def relmag(gf, gt, keys):
    a = flat_grads(gf, keys); b = flat_grads(gt, keys)
    return (a - b).norm().item() / (a.norm().item() + 1e-12)


def evaluate(model, delay, n=256):
    accs = []
    for e in range(4):
        inp, tgt, msk = batch(n, delay, 90000 + e)
        with torch.no_grad():
            out, _ = model(inp)
        accs.append(task_accuracy(out, tgt, msk))
    return float(np.mean(accs))


def train_one(method, seed, delay, n_steps, record=True):
    m = new_model(seed)
    curve = []
    for s in range(n_steps + 1):
        if record and s % EVAL_EVERY == 0:
            curve.append(evaluate(m, delay))
        if s == n_steps:
            break
        inp, tgt, msk = batch(BATCH, delay, 10000 + s)
        if method == "bptt":
            g = compute_bptt_gradients(m, inp, tgt, msk, _xent_loss)
        else:
            g = compute_deep_eprop_gradients(m, inp, tgt, msk, xent_error, mode=method)
        apply_gradients(m, g, LR)
    return m, curve


# ── top-level workers (picklable for the process pool) ────────────────────────
def _e1_job(args):
    d, s = args
    m = new_model(1000 + s)
    inp, tgt, msk = batch(BATCH, d, 5000 + s)
    gb, gf, gs, gt = grads_all(m, inp, tgt, msk)
    return (cosine_sim_grads(gf, gb, LOWER), cosine_sim_grads(gf, gb, UPPER),
            cosine_sim_grads(gt, gb, LOWER), cosine_sim_grads(gt, gb, UPPER),
            relmag(gf, gt, LOWER))


def _lc_job(args):
    method, seed = args
    return (method, seed, train_one(method, seed, DELAY_MAIN, N_STEPS, record=True)[1])


def _e3_job(args):
    method, seed, d = args
    m, _ = train_one(method, seed, d, E3_STEPS, record=False)
    return (method, d, evaluate(m, d))


# ───────────────── E1: per-layer cosine + cross-temporal share vs delay ────────
def e1_gradient_credit():
    print(f"=== E1: per-layer gradient cosine vs BPTT (+ cross-temporal share) [{DEVICE}] ===", flush=True)
    out = {}
    for d in DELAYS:
        res = np.array(_map(_e1_job, [(d, s) for s in range(N_SEEDS_COS)]), dtype=float)
        out[d] = dict(zip(["full_low", "full_up", "temp_low", "temp_up", "xtemp_share"],
                          np.nanmean(res, axis=0).tolist()))
        print(f"  D={d:3d}  full(low={out[d]['full_low']:.3f}, up={out[d]['full_up']:.3f})  "
              f"temporal(low={out[d]['temp_low']:.3f})  spatial(low=0)  "
              f"xtemp_share={out[d]['xtemp_share']:.3f}", flush=True)

    json.dump(out, open(f"{RESULTS}/e1_gradient_credit.json", "w"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(DELAYS, [out[d]["full_low"] for d in DELAYS], "o-", color="C0", label="full · lower layer")
    ax.plot(DELAYS, [out[d]["full_up"] for d in DELAYS], "o--", color="C0", label="full · top layer")
    ax.plot(DELAYS, [out[d]["temp_low"] for d in DELAYS], "s-", color="C3", label="ablate_temporal · lower")
    ax.plot(DELAYS, [0]*len(DELAYS), "x", color="C1", label="ablate_spatial · lower (=0)")
    ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("delay D"); ax.set_ylabel("gradient cosine vs BPTT")
    ax.set_title("Per-layer credit alignment"); ax.legend(fontsize=8); ax.set_ylim(-0.1, 1.05)

    ax = axes[1]
    ax.plot(DELAYS, [out[d]["xtemp_share"] for d in DELAYS], "o-", color="C2")
    ax.set_xlabel("delay D")
    ax.set_ylabel("‖full − ablate_temporal‖ / ‖full‖  (lower layer)")
    ax.set_title("Share of lower-layer credit carried by\ncross-layer temporal trace ϵ^z")
    ax.set_ylim(0, 1.05)
    fig.suptitle("E1 — deep e-prop assigns credit across depth AND time")
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e1_gradient_credit.{ext}")
    plt.close(fig)
    print(f"  saved {RESULTS}/e1_gradient_credit.[pdf,svg,json]", flush=True)
    return out


# ───────────────── E2: learning curves (seeds in parallel) ─────────────────────
def e2_learning_curves():
    print(f"=== E2: learning curves (hierarchical, {DEVICE}, pool={USE_POOL} x{N_WORKERS}) ===", flush=True)
    steps = list(range(0, N_STEPS + 1, EVAL_EVERY))
    jobs = [(meth, s) for meth, _ in METHODS_LC for s in range(N_SEEDS_LC)]
    t0 = time.time()
    results = _map(_lc_job, jobs)
    acc = {meth: {} for meth, _ in METHODS_LC}
    for method, seed, curve in results:
        acc[method][seed] = curve
    curves = {}
    for method, label in METHODS_LC:
        mat = np.array([acc[method][s] for s in range(N_SEEDS_LC)])
        curves[method] = dict(mean=mat.mean(0).tolist(), std=mat.std(0).tolist())
        print(f"  {label:22s} final={mat.mean(0)[-1]:.3f}±{mat.std(0)[-1]:.3f}", flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

    json.dump(dict(steps=steps, curves=curves), open(f"{RESULTS}/e2_learning_curves.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method, label in METHODS_LC:
        mu = np.array(curves[method]["mean"]); sd = np.array(curves[method]["std"])
        ax.plot(steps, mu, "-o", ms=3, color=COLORS[method], label=label)
        ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[method], alpha=0.15)
    ax.axhline(0.5, color="gray", ls="--", label="chance")
    ax.set_xlabel("training step"); ax.set_ylabel("accuracy")
    ax.set_title(f"Learning the hierarchical task (D={DELAY_MAIN}, α={ALPHA})")
    ax.legend(fontsize=8); ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e2_learning_curves.{ext}")
    plt.close(fig)
    print(f"  saved {RESULTS}/e2_learning_curves.[pdf,svg,json]", flush=True)
    return curves


# ───────────────── E3: final accuracy vs delay (seeds in parallel) ─────────────
def e3_delay_sweep():
    print(f"=== E3: final accuracy vs delay ({DEVICE}, pool={USE_POOL} x{N_WORKERS}) ===", flush=True)
    jobs = [(meth, s, d) for meth, _ in METHODS_LC for s in range(E3_SEEDS) for d in E3_DELAYS]
    t0 = time.time()
    out = _map(_e3_job, jobs)
    bucket = {(meth, d): [] for meth, _ in METHODS_LC for d in E3_DELAYS}
    for method, d, a in out:
        bucket[(method, d)].append(a)
    res = {meth: {d: [float(np.mean(bucket[(meth, d)])), float(np.std(bucket[(meth, d)]))]
                  for d in E3_DELAYS} for meth, _ in METHODS_LC}
    for d in E3_DELAYS:
        print(f"  D={d:3d}  " + "  ".join(
            f"{lab.split()[0]}={res[meth][d][0]:.3f}" for meth, lab in METHODS_LC), flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

    json.dump(res, open(f"{RESULTS}/e3_delay_sweep.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, label in METHODS_LC:
        mu = [res[method][d][0] for d in E3_DELAYS]; sd = [res[method][d][1] for d in E3_DELAYS]
        ax.errorbar(E3_DELAYS, mu, yerr=sd, marker="o", color=COLORS[method], label=label, capsize=3)
    ax.axhline(0.5, color="gray", ls="--", label="chance")
    ax.set_xlabel("delay D"); ax.set_ylabel("final accuracy")
    ax.set_title("Accuracy vs delay — credit must cross more time"); ax.legend(fontsize=8)
    ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e3_delay_sweep.{ext}")
    plt.close(fig)
    print(f"  saved {RESULTS}/e3_delay_sweep.[pdf,svg,json]", flush=True)
    return res


if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "e1"):
        e1_gradient_credit()
    if which in ("all", "e2"):
        c = e2_learning_curves()
        fin = {m: c[m]["mean"][-1] for m, _ in METHODS_LC}
        print("\n--- E2 ORDERING ---", flush=True)
        print(f"  BPTT={fin['bptt']:.3f}  full={fin['full']:.3f}  "
              f"ablate_temporal={fin['ablate_temporal']:.3f}  ablate_spatial={fin['ablate_spatial']:.3f}", flush=True)
        ok = fin["bptt"] + 0.02 >= fin["full"] > max(fin["ablate_temporal"], fin["ablate_spatial"])
        print(f"  BPTT >= full > both controls: {'YES' if ok else 'NO'}", flush=True)
    if which in ("all", "e3"):
        e3_delay_sweep()
