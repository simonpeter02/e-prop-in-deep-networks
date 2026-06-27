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
LR          = 3.5e-2           # lower than BPTT-stable 5e-2: smooths e-prop overshoot
N_STEPS     = 2000             # training steps for E2 learning curves
EVAL_EVERY  = 100
N_SEEDS_COS = 16               # seeds for gradient-cosine averages (E1)
N_SEEDS_LC  = 6                # seeds for learning curves (E2)
EVAL_N      = 512              # samples per eval batch (×EVAL_REPS) — de-noises curves
EVAL_REPS   = 4
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
    """Map a top-level worker over items; seeds in parallel via a SPAWN process
    pool on CPU, sequential on GPU. Spawn (not fork) because PyTorch autograd is
    incompatible with fork. Falls back to sequential on any pool error."""
    if not USE_POOL or N_WORKERS <= 1 or len(items) <= 1:
        return [worker(x) for x in items]
    try:
        ctx = mp.get_context("spawn")
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


def evaluate(model, delay, n=EVAL_N, reps=EVAL_REPS):
    accs = []
    for e in range(reps):
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
        # Per-seed data stream (not shared across seeds) so hard batches don't hit
        # every seed at the same step → seed-averaged curves are smooth.
        inp, tgt, msk = batch(BATCH, delay, 10_000 + seed * 1_000_000 + s)
        if method == "bptt":
            g = compute_bptt_gradients(m, inp, tgt, msk, _xent_loss)
        else:
            g = compute_deep_eprop_gradients(m, inp, tgt, msk, xent_error, mode=method)
        apply_gradients(m, g, LR)
    return m, curve


# ── top-level workers (picklable for the process pool) ────────────────────────
E1_KEYS = ["full_low", "full_up", "temp_low", "temp_up", "spat_low", "spat_up", "xtemp_share"]


def _e1_job(args):
    d, s = args
    m = new_model(1000 + s)
    inp, tgt, msk = batch(BATCH, d, 5000 + s)
    gb, gf, gs, gt = grads_all(m, inp, tgt, msk)
    return (cosine_sim_grads(gf, gb, LOWER), cosine_sim_grads(gf, gb, UPPER),
            cosine_sim_grads(gt, gb, LOWER), cosine_sim_grads(gt, gb, UPPER),
            cosine_sim_grads(gs, gb, LOWER),   # nan: ablate_spatial zeros lower grads
            cosine_sim_grads(gs, gb, UPPER),   # == full top (controls don't touch top)
            relmag(gf, gt, LOWER))


def _lc_job(args):
    method, seed = args
    return (method, seed, train_one(method, seed, DELAY_MAIN, N_STEPS, record=True)[1])


def _e3_job(args):
    method, seed, d = args
    m, _ = train_one(method, seed, d, E3_STEPS, record=False)
    return (method, d, evaluate(m, d))


# ───────────────── E1: per-layer cosine + cross-temporal share vs delay ────────
def _stats(res):
    """Per-series mean and standard error (nan-aware) over seeds (rows).
    All-nan columns (e.g. ablate_spatial lower cosine, which is exactly 0/undefined)
    return nan without noisy warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(res, axis=0)
        n = np.sum(~np.isnan(res), axis=0)
        sd = np.nanstd(res, axis=0, ddof=1)
    sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
    return mean, sem


def e1_gradient_credit():
    print(f"=== E1: per-layer gradient cosine vs BPTT (+ cross-temporal share) [{DEVICE}] ===", flush=True)
    mean, sem = {}, {}
    for d in DELAYS:
        res = np.array(_map(_e1_job, [(d, s) for s in range(N_SEEDS_COS)]), dtype=float)  # (seeds, 7)
        m, e = _stats(res)
        mean[d] = dict(zip(E1_KEYS, m.tolist()))
        sem[d] = dict(zip(E1_KEYS, e.tolist()))
        print(f"  D={d:3d}  full(low={mean[d]['full_low']:.3f}±{sem[d]['full_low']:.3f}, "
              f"up={mean[d]['full_up']:.3f})  temporal(low={mean[d]['temp_low']:.3f}±{sem[d]['temp_low']:.3f})  "
              f"spatial(low=0)  xtemp_share={mean[d]['xtemp_share']:.3f}", flush=True)

    json.dump({"delays": DELAYS, "mean": mean, "sem": sem, "n_seeds": N_SEEDS_COS},
              open(f"{RESULTS}/e1_gradient_credit.json", "w"), indent=2)

    def series(stat, key):   # nan→0 (ablate_spatial lower is exactly zero)
        return np.nan_to_num(np.array([stat[d][key] for d in DELAYS]))

    # ── Figure 1: per-layer cosine vs delay, with stderr bands ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    # ordered top-down to match the lines in the plot; full·output-adjacent gets a
    # dotted line + hollow markers so it can't be confused with full·input-adjacent
    for key, color, ls, lab, mfc in [
            ("full_up",  "C0", ":",  "full · output-adjacent layer", "none"),
            ("full_low", "C0", "-",  "full · input-adjacent layer",  "C0"),
            ("temp_low", "C3", "-",  "ablate_temporal · input-adjacent", "C3"),
            ("spat_low", "C1", "-",  "ablate_spatial · input-adjacent (=0)", "C1")]:
        mu, er = series(mean, key), series(sem, key)
        ax.plot(DELAYS, mu, marker="o", color=color, ls=ls, label=lab, markerfacecolor=mfc)
        ax.fill_between(DELAYS, mu - er, mu + er, color=color, alpha=0.15)
    ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("delay D"); ax.set_ylabel("gradient cosine vs BPTT")
    ax.legend(fontsize=8); ax.set_ylim(-0.1, 1.05)

    ax = axes[1]
    mu, er = series(mean, "xtemp_share"), series(sem, "xtemp_share")
    ax.plot(DELAYS, mu, "o-", color="C2")
    ax.fill_between(DELAYS, mu - er, mu + er, color="C2", alpha=0.15)
    ax.set_xlabel("delay D")
    ax.set_ylabel("‖full − ablate_temporal‖ / ‖full‖  (input-adjacent layer)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e1_gradient_credit.{ext}")
    plt.close(fig)

    # ── Figure 2: summary bars at the main delay (reuses the same cosines) ─────
    dbar = DELAY_MAIN if DELAY_MAIN in mean else DELAYS[len(DELAYS) // 2]
    methods = [("full", "C0"), ("ablate_temporal", "C3"), ("ablate_spatial", "C1")]
    key_for = {("full", "low"): "full_low", ("full", "top"): "full_up",
               ("ablate_temporal", "low"): "temp_low", ("ablate_temporal", "top"): "temp_up",
               ("ablate_spatial", "low"): "spat_low", ("ablate_spatial", "top"): "spat_up"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)                      # 0=lower layer, 1=top layer
    w = 0.26
    for i, (meth, color) in enumerate(methods):
        mus = [np.nan_to_num(mean[dbar][key_for[(meth, lyr)]]) for lyr in ("low", "top")]
        ers = [np.nan_to_num(sem[dbar][key_for[(meth, lyr)]]) for lyr in ("low", "top")]
        ax.bar(x + (i - 1) * w, mus, w, yerr=ers, capsize=4, color=color,
               label=meth.replace("ablate_", "ablate "))
    ax.set_xticks(x); ax.set_xticklabels(["input-adjacent layer", "output-adjacent layer"])
    ax.set_ylabel("gradient cosine vs BPTT"); ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="gray", ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e1_credit_summary.{ext}")
    plt.close(fig)
    print(f"  saved {RESULTS}/e1_gradient_credit.[pdf,svg,json] + e1_credit_summary.[pdf,svg]", flush=True)
    return mean


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
        n = mat.shape[0]
        sem = ((mat.std(0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros(mat.shape[1])).tolist()
        curves[method] = dict(mean=mat.mean(0).tolist(), std=mat.std(0).tolist(), sem=sem)
        print(f"  {label:22s} final={mat.mean(0)[-1]:.3f}±{sem[-1]:.3f} (SEM)", flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

    json.dump(dict(steps=steps, curves=curves, n_seeds=N_SEEDS_LC),
              open(f"{RESULTS}/e2_learning_curves.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method, label in METHODS_LC:
        mu = np.array(curves[method]["mean"]); er = np.array(curves[method]["sem"])
        ax.plot(steps, mu, "-o", ms=3, color=COLORS[method], label=label)
        ax.fill_between(steps, mu - er, mu + er, color=COLORS[method], alpha=0.15)
    ax.axhline(0.5, color="gray", ls="--", label="chance")
    ax.set_xlabel("training step"); ax.set_ylabel("accuracy")
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
    t0 = time.time()
    bucket = {(meth, d): [] for meth, _ in METHODS_LC for d in E3_DELAYS}
    # One _map per delay (smaller batches) — robust with the spawn pool.
    for d in E3_DELAYS:
        jobs = [(meth, s, d) for meth, _ in METHODS_LC for s in range(E3_SEEDS)]
        for method, dd, a in _map(_e3_job, jobs):
            bucket[(method, dd)].append(a)
        print(f"  D={d:3d} done [{time.time()-t0:.0f}s]", flush=True)
    def _ms(v):
        v = np.array(v, dtype=float)
        return [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]   # mean, SEM
    res = {meth: {d: _ms(bucket[(meth, d)]) for d in E3_DELAYS} for meth, _ in METHODS_LC}
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
    ax.legend(fontsize=8)
    ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e3_delay_sweep.{ext}")
    plt.close(fig)
    print(f"  saved {RESULTS}/e3_delay_sweep.[pdf,svg,json]", flush=True)
    return res


# ───────────────── Significance testing + power analysis ──────────────────────
STATS_EVAL_N    = 2048      # held-out trials per eval batch for the per-seed endpoint
STATS_EVAL_REPS = 2         # → 4096 held-out trials per seed (precise final accuracy)
METHODS_ALL     = ["full", "ablate_temporal", "ablate_spatial", "bptt"]
STATS_COMPS     = [("ablate_temporal", "full vs ablate_temporal"),
                   ("ablate_spatial",  "full vs ablate_spatial"),
                   ("bptt",            "full vs BPTT")]


def _final_acc_job(args):
    """Train one (method, seed) to completion; return final accuracy on a large
    fixed held-out test set (the per-seed endpoint for the paired tests)."""
    method, seed, delay = args
    m, _ = train_one(method, seed, delay, N_STEPS, record=False)
    return (method, seed, evaluate(m, delay, n=STATS_EVAL_N, reps=STATS_EVAL_REPS))


def collect_final_accs(seeds, delay=DELAY_MAIN):
    # Chunk per method (smaller _map calls) — the spawn pool stalls on very large
    # single batches, so keep each _map to ~len(seeds) jobs (same fix as E3).
    acc = {meth: {} for meth in METHODS_ALL}
    for meth in METHODS_ALL:
        for method, seed, a in _map(_final_acc_job, [(meth, s, delay) for s in seeds]):
            acc[method][seed] = a
        print(f"  [{meth} done: {len(acc[meth])}/{len(seeds)} seeds]", flush=True)
    return acc


def power_analysis(pilot_seeds=None):
    from experiments.stats import paired_report, power_curve, smallest_n_for_power
    pilot_seeds = pilot_seeds if pilot_seeds is not None else list(range(10))
    print(f"=== Power analysis: pilot {len(pilot_seeds)} seeds, D={DELAY_MAIN} [{DEVICE}] ===", flush=True)
    acc = collect_final_accs(pilot_seeds)
    full = np.array([acc["full"][s] for s in pilot_seeds])
    cols = []
    print("  pilot effect sizes (paired, full − method):", flush=True)
    for m, label in STATS_COMPS:
        b = np.array([acc[m][s] for s in pilot_seeds]); cols.append(full - b)
        r = paired_report(full, b)
        print(f"    {label:24s} meanΔ={r['mean_diff']:+.3f}  dz={r['cohen_dz']:+.2f}  perm p={r['p_perm']:.4f}", flush=True)
    D = np.column_stack(cols)
    ns = list(range(4, 21))
    powers = power_curve(D, ns, alpha=0.05, n_sim=2000, n_perm=4096, seed=0)
    nstar = smallest_n_for_power(powers, 0.90)
    print("  power vs n (per comparison, Holm-corrected, two-sided α=0.05):", flush=True)
    for n in ns:
        print(f"    n={n:2d}: {[round(x, 2) for x in powers[n]]}", flush=True)
    print(f"  → n* (≥0.90 power on all comparisons) = {nstar}", flush=True)

    json.dump({"pilot_seeds": pilot_seeds, "powers": powers, "nstar": nstar,
               "pilot_acc": {m: {int(s): acc[m][s] for s in pilot_seeds} for m in acc}},
              open(f"{RESULTS}/e2_power_curve.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, (_, label) in enumerate(STATS_COMPS):
        ax.plot(ns, [powers[n][j] for n in ns], "-o", ms=3, label=label)
    ax.axhline(0.9, color="gray", ls="--", label="0.90 target")
    if nstar:
        ax.axvline(nstar, color="k", ls=":", label=f"n* = {nstar}")
    ax.set_xlabel("number of seeds n"); ax.set_ylabel("power (Holm, two-sided α=0.05)")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e2_power_curve.{ext}")
    plt.close(fig)
    print(f"  saved {RESULTS}/e2_power_curve.[pdf,svg,json]", flush=True)
    return nstar


def e2_significance(nstar, seed_start=100):
    from experiments.stats import paired_report, holm
    seeds = list(range(seed_start, seed_start + nstar))
    print(f"=== Significance: {nstar} fresh seeds {seeds[0]}–{seeds[-1]}, D={DELAY_MAIN} [{DEVICE}] ===", flush=True)
    acc = collect_final_accs(seeds)
    full = np.array([acc["full"][s] for s in seeds])
    reports = [(label, paired_report(full, np.array([acc[m][s] for s in seeds])))
               for m, label in STATS_COMPS]
    padj = holm([r["p_perm"] for _, r in reports])

    print(f"  {'comparison':24s} {'meanΔ':>8} {'dz':>7} {'perm p':>8} {'Holm p':>8}  sig", flush=True)
    rows = []
    for (label, r), pa in zip(reports, padj):
        sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "ns"
        print(f"  {label:24s} {r['mean_diff']:+8.3f} {r['cohen_dz']:+7.2f} "
              f"{r['p_perm']:8.4f} {pa:8.4f}  {sig}", flush=True)
        rows.append({"comparison": label, **r, "p_holm": float(pa), "sig": sig})

    means = {m: float(np.mean([acc[m][s] for s in seeds])) for m in METHODS_ALL}
    sems = {m: float(np.std([acc[m][s] for s in seeds], ddof=1) / np.sqrt(nstar)) for m in METHODS_ALL}
    json.dump({"seeds": seeds, "n": nstar, "final_acc_mean": means, "final_acc_sem": sems,
               "per_seed": {m: {int(s): acc[m][s] for s in seeds} for m in METHODS_ALL},
               "comparisons": rows},
              open(f"{RESULTS}/e2_significance.json", "w"), indent=2)

    # Final-accuracy bar chart with SEM + significance brackets (full vs controls).
    order = ["bptt", "full", "ablate_temporal", "ablate_spatial"]
    short = {"bptt": "BPTT", "full": "full", "ablate_temporal": "ablate\ntemporal",
             "ablate_spatial": "ablate\nspatial"}
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs = np.arange(len(order))
    ax.bar(xs, [means[m] for m in order], yerr=[sems[m] for m in order], capsize=4,
           color=[COLORS[m] for m in order])
    ax.set_xticks(xs); ax.set_xticklabels([short[m] for m in order])
    ax.set_ylabel("final accuracy (held-out)"); ax.axhline(0.5, color="gray", ls="--")
    ax.set_ylim(0.45, 1.22)
    fx = order.index("full")
    y0 = max(means.values()) + 0.05
    for k, (m, _, pa) in enumerate([("ablate_temporal", reports[0][1], padj[0]),
                                    ("ablate_spatial", reports[1][1], padj[1])]):
        x2 = order.index(m); y = y0 + k * 0.06
        ax.plot([fx, fx, x2, x2], [y - 0.01, y, y, y - 0.01], color="k", lw=1)
        s = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "ns"
        ax.text((fx + x2) / 2, y + 0.004, s, ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/e2_significance.{ext}")
    plt.close(fig)
    print(f"  saved {RESULTS}/e2_significance.[pdf,svg,json]", flush=True)
    return rows


if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "power":
        power_analysis()
    if which == "stats":
        nstar = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        e2_significance(nstar)
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
