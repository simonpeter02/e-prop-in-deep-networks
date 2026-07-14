"""
Experiment 2 — is the lower-layer gradient CUE-AGNOSTIC under ablate_temporal?

Story (which term of the cross-layer trace carries which credit):
    eps^z = (dz/dh)·eps^h  +  (dz/dz_{t-1})·eps^z_{t-1}
             └ spatial seed ┘   └── temporal carry ──┘
              = DEPTH credit        = cross-layer TIME credit

  * ablate_spatial  → lower-layer gradient is exactly 0 (no depth path).
  * ablate_temporal → lower-layer gradient is nonzero but CUE-AGNOSTIC: it only
    reflects the decision-step (recall) transform, not the cue evidence.

We show this by DECODING per-trial cue variables from the per-trial gradient.

Confound: the per-trial lower gradient is  delta · eps^z, and the readout error
delta = softmax(o) - y flips sign with the BINARY label. So the binary label is
trivially decodable even from a cue-agnostic gradient (it is just the sign). We
therefore also decode a readout-sign-independent variable — the cue MARGIN
(unanimous 3-0  vs  split 2-1, i.e. rising-count in {0,3} vs {1,2}) — which
delta's sign cannot encode. Cue-margin decodability is the true test of cue credit.

Run:
    python -u -m experiments.exp2_cue_decoding            # full
    QUICK=1 python -u -m experiments.exp2_cue_decoding    # fast smoke test
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learning_rules.deep_eprop import compute_deep_eprop_gradients, xent_error
from utils import flat_grads
from experiments.deep_credit_time_depth import (
    new_model, batch, N_CUES, TASK_KW, DELAY_MAIN, LOWER, UPPER, RESULTS, DEVICE,
)

QUICK      = os.environ.get("QUICK") == "1"
N_TRIALS   = 150 if QUICK else 400      # trials per (seed, mode) batch
N_SEEDS    = 2   if QUICK else 5
DELAY      = DELAY_MAIN                  # 12
K_FOLD     = 5
N_PERM     = 30  if QUICK else 100
MODES      = ["full", "ablate_temporal", "ablate_spatial"]
COLORS     = {"full": "C0", "ablate_temporal": "C3", "ablate_spatial": "C1"}
CUE_DUR    = TASK_KW["cue_duration"]        # 3
CUE_STRIDE = CUE_DUR + TASK_KW["inter_cue_interval"]   # 5


# ── cue variables recovered from the input (channel 0 ramp direction) ─────────
def cue_variables(inp):
    """Per-trial (binary label, cue margin, rising-count) from the feature channel.

    For each cue window the deterministic ramp is direction*linspace(-amp,amp);
    rising (side 0) increases, falling (side 1) decreases across the window. With
    amp=2 >> feature_noise=0.15 the slope sign recovers the side robustly.

    Returns
    -------
    label  : (N,) int   1 = falling-majority (rising-count < 1.5), else 0
    margin : (N,) int   1 = unanimous (count in {0,3}), 0 = split (count in {1,2})
    count  : (N,) int   number of rising cues, in {0..N_CUES} (the cue "type")
    """
    x0 = inp[:, :, 0]                                   # (T, B)
    count_rising = torch.zeros(inp.shape[1], device=inp.device)
    for c in range(N_CUES):
        t0 = c * CUE_STRIDE
        rising = x0[t0 + CUE_DUR - 1] > x0[t0]          # end > start
        count_rising += rising.float()
    count = count_rising.cpu().numpy().astype(int)      # (N,) in {0..N_CUES}
    label  = (count < N_CUES / 2.0).astype(int)         # falling-majority = 1
    margin = np.isin(count, [0, N_CUES]).astype(int)    # unanimous = 1
    return label, margin, count


# ── per-trial gradients (B=1) reusing the existing rule ───────────────────────
def per_trial_grads(model, inp, tgt, msk, mode):
    """Return (X_lower, X_upper): (N, D) arrays of per-trial flattened grads."""
    xl, xu = [], []
    B = inp.shape[1]
    for i in range(B):
        g = compute_deep_eprop_gradients(
            model, inp[:, i:i+1], tgt[:, i:i+1], msk[:, i:i+1], xent_error, mode=mode)
        xl.append(flat_grads(g, LOWER).cpu().numpy())
        xu.append(flat_grads(g, UPPER).cpu().numpy())
    return np.asarray(xl), np.asarray(xu)


# ── dependency-light linear decoder (sklearn is not installed) ────────────────
def _stratified_folds(y, k, rng):
    """List of k test-index arrays, each stratified by the classes in y."""
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, ix in enumerate(idx):
            folds[j % k].append(ix)
    return [np.array(sorted(f)) for f in folds]


def _balanced_acc(y_true, y_pred):
    accs = []
    for cls in np.unique(y_true):
        m = y_true == cls
        accs.append((y_pred[m] == cls).mean())
    return float(np.mean(accs))


def _ridge_predict(Xtr, ytr_pm, Xte, reg=1.0):
    """Linear-kernel ridge classifier (dual form, N<<D). ytr_pm in {-1,+1}.
    Returns +-1 predictions on Xte."""
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32)
    Xte = torch.as_tensor(Xte, dtype=torch.float32)
    ytr = torch.as_tensor(ytr_pm, dtype=torch.float32)
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T                                     # (n, n) linear kernel
    lam = reg * (K.diagonal().mean() + 1e-8)
    alpha = torch.linalg.solve(K + lam * torch.eye(n), ytr)
    f = (Xte @ Xtr.T) @ alpha                           # (n_te,)
    return torch.sign(f).cpu().numpy()


def decode_cv(X, y, k=K_FOLD, seed=0):
    """Cross-validated balanced accuracy of a linear classifier predicting y
    from X, with a label-shuffle permutation chance estimate.

    Returns dict(acc, chance_mean, chance_hi)."""
    y = np.asarray(y).astype(int)
    # degenerate feature matrix (e.g. ablate_spatial lower grad ≡ 0) → chance.
    if X.size == 0 or np.abs(X).max() < 1e-20 or len(np.unique(y)) < 2:
        base = 0.5
        return dict(acc=base, chance_mean=base, chance_hi=base)

    def _cv(yv, rng):
        folds = _stratified_folds(yv, k, rng)
        preds = np.zeros(len(yv)); truth = np.zeros(len(yv))
        for f in range(k):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(k) if j != f])
            mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd < 1e-8] = 1.0
            Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
            ypm = np.where(yv[tr] == yv[tr].max(), 1.0, -1.0)
            p = _ridge_predict(Xtr, ypm, Xte)
            preds[te] = np.where(p > 0, yv[tr].max(), yv[tr].min())
            truth[te] = yv[te]
        return _balanced_acc(truth, preds)

    rng = np.random.default_rng(seed)
    acc = _cv(y, rng)
    perm = np.array([_cv(rng.permutation(y), rng) for _ in range(N_PERM)])
    return dict(acc=acc, chance_mean=float(perm.mean()),
                chance_hi=float(np.percentile(perm, 95)))


# ── PCA geometry of the per-trial lower-layer gradient ─────────────────────────
def pca_2d(X):
    """Project (N, D) onto its top-2 principal components.

    Returns (proj (N, 2), explained_variance_ratio (2,)), or None if X is
    degenerate (e.g. ablate_spatial's lower gradient is identically zero)."""
    if X.size == 0 or np.abs(X).max() < 1e-20:
        return None
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    var = S ** 2
    return proj, (var[:2] / var.sum())


def geometry_figure(geom_X, label0, count0, geom_acc):
    """PCA scatter of the seed-0 lower-layer per-trial gradient, per mode,
    colored by the cue TYPE — the number of rising cues (0..N_CUES) — the visual
    counterpart to the decode-accuracy bars above.

    geom_acc maps mode -> seed-0 lower margin-decode accuracy (reused from run()'s
    decode loop; not recomputed here)."""
    count0 = np.asarray(count0).astype(int)
    cmap = plt.cm.viridis                                 # sequential: 0 -> N_CUES rising
    ccolors = {c: cmap(c / N_CUES) for c in range(N_CUES + 1)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    data_out = {}
    for ax, m in zip(axes, MODES):
        X = geom_X[m]
        out = pca_2d(X)
        mode_lab = m.replace("ablate_", "ablate ")
        if out is None:
            ax.text(0.5, 0.5, "gradient ≡ 0\n(ablate_spatial removes\nlower-layer credit)",
                     ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(mode_lab, fontsize=10)
            data_out[m] = None
        else:
            proj, var = out
            acc = geom_acc[m]
            for c in range(N_CUES + 1):
                sel = count0 == c
                if not sel.any():
                    continue
                ax.scatter(proj[sel, 0], proj[sel, 1], s=14, alpha=0.7,
                           color=ccolors[c], label=f"{c} rising")
            ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)")
            ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
            ax.set_title(f"{mode_lab}\ndecode acc (margin) = {acc:.2f}", fontsize=10)
            data_out[m] = dict(proj=proj.tolist(), explained_variance_ratio=var.tolist(),
                                margin_decode_acc=acc)
    axes[0].legend(fontsize=8, loc="best", title="rising cues")
    fig.suptitle("Geometry of the per-trial lower-layer gradient (PCA, seed 0), colored by "
                 "cue type (rising-cue count): the temporal carry makes it cue-specific", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/exp2_gradient_geometry.{ext}")
    plt.close(fig)

    json.dump({"seed": 0, "modes": MODES, "count": count0.tolist(),
               "label": np.asarray(label0).astype(int).tolist(), "per_mode": data_out},
              open(f"{RESULTS}/exp2_gradient_geometry.json", "w"), indent=2)
    print(f"saved {RESULTS}/exp2_gradient_geometry.[pdf,svg,json]", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    print(f"=== Exp5 cue decoding [{DEVICE}] "
          f"seeds={N_SEEDS} trials={N_TRIALS} delay={DELAY} quick={QUICK} ===", flush=True)
    # results[layer][mode][target] = list over seeds of dict(acc, chance_mean, chance_hi)
    layers = {"lower": LOWER, "upper": UPPER}
    res = {lyr: {m: {"label": [], "margin": []} for m in MODES} for lyr in layers}
    geom_X, label0, count0 = {}, None, None  # seed-0 lower-layer grads, for the PCA figure

    for s in range(N_SEEDS):
        model = new_model(1000 + s)
        inp, tgt, msk = batch(N_TRIALS, DELAY, 5000 + s)
        label, margin, count = cue_variables(inp)
        print(f"  seed {s}: unanimous frac={margin.mean():.2f} "
              f"falling-maj frac={label.mean():.2f}", flush=True)
        for m in MODES:
            Xl, Xu = per_trial_grads(model, inp, tgt, msk, m)
            for lyr, X in (("lower", Xl), ("upper", Xu)):
                res[lyr][m]["label"].append(decode_cv(X, label, seed=s))
                res[lyr][m]["margin"].append(decode_cv(X, margin, seed=s))
            if s == 0:
                geom_X[m] = Xl
        if s == 0:
            label0, count0 = label, count
        print(f"  seed {s} done", flush=True)

    # aggregate mean ± sem over seeds
    def agg(lst, key):
        v = np.array([d[key] for d in lst])
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)

    summary = {}
    print("\n--- decoding accuracy (mean ± sem over seeds; chance in []) ---", flush=True)
    for lyr in layers:
        summary[lyr] = {}
        print(f"[{lyr}-layer / {'input-adjacent' if lyr=='lower' else 'output-adjacent'}]", flush=True)
        for m in MODES:
            summary[lyr][m] = {}
            for tgt_name in ("label", "margin"):
                mu, se = agg(res[lyr][m][tgt_name], "acc")
                ch, _ = agg(res[lyr][m][tgt_name], "chance_mean")
                chi, _ = agg(res[lyr][m][tgt_name], "chance_hi")
                summary[lyr][m][tgt_name] = dict(acc=mu, sem=se, chance=ch, chance_hi=chi)
                print(f"    {m:16s} {tgt_name:6s}: {mu:.3f}±{se:.3f}  [chance {ch:.3f}, 95% {chi:.3f}]",
                      flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    json.dump({"seeds": N_SEEDS, "trials": N_TRIALS, "delay": DELAY, "summary": summary},
              open(f"{RESULTS}/exp2_cue_decoding.json", "w"), indent=2)

    # ── figure: 2 panels (lower / upper), grouped bars over {label, margin} ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    targets = [("label", "binary label\n(readout-sign leak)"), ("margin", "cue margin\n(unanimous vs split)")]
    for ax, lyr, title in [(axes[0], "lower", "input-adjacent (lower) layer"),
                           (axes[1], "upper", "output-adjacent (top) layer")]:
        x = np.arange(len(targets)); w = 0.26
        for i, m in enumerate(MODES):
            mus = [summary[lyr][m][t]["acc"] for t, _ in targets]
            ers = [summary[lyr][m][t]["sem"] for t, _ in targets]
            ax.bar(x + (i - 1) * w, mus, w, yerr=ers, capsize=4, color=COLORS[m],
                   label=m.replace("ablate_", "ablate "))
        # chance band (use the label target's permutation 95% as a representative line)
        chance_hi = np.mean([summary[lyr][m][t]["chance_hi"] for m in MODES for t, _ in targets])
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.axhline(chance_hi, color="gray", ls=":", lw=1, label="perm. 95%")
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in targets])
        ax.set_title(title); ax.set_ylim(0.4, 1.02)
        if lyr == "lower":
            ax.set_ylabel("cross-validated decode accuracy")
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Cue information in the per-trial gradient: the temporal carry makes "
                 "lower-layer credit cue-specific", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{RESULTS}/exp2_cue_decoding.{ext}")
    plt.close(fig)
    print(f"\nsaved {RESULTS}/exp2_cue_decoding.[pdf,svg,json]", flush=True)

    # seed-0 lower margin-decode accuracies already computed in the loop above
    geom_acc = {m: res["lower"][m]["margin"][0]["acc"] for m in MODES}
    geometry_figure(geom_X, label0, count0, geom_acc)
    return summary


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    run()
