"""
Paired significance tests + simulation-based power analysis for the deep-eprop
learning-curve comparisons (full deep e-prop vs the ablated controls / BPTT).

Why paired: for a given seed every method shares the same initial weights AND the
same training-data stream — only the gradient rule differs — so the natural unit
is the per-seed difference d_i = acc(method_a)_i - acc(method_b)_i.

Headline test: exact paired SIGN-FLIP PERMUTATION test (two-sided). Under H0 the
per-seed differences are symmetric about 0, so each d_i's sign is ±1 with prob
1/2; we compare the observed |mean(d)| to its distribution over all 2^n sign
flips. Note the floor: two-sided p >= 2 / 2^n regardless of effect size.

Also reported: paired t-test (95% CI + Cohen's dz) and Wilcoxon signed-rank
(robustness). Family of comparisons corrected with Holm-Bonferroni.

numpy + scipy.stats only.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


# ─────────────────────────── sign-flip permutation ───────────────────────────
def _exact_sign_matrix(n: int) -> np.ndarray:
    """All 2^n sign vectors in {+1,-1}, shape (2^n, n) int8."""
    idx = np.arange(2 ** n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
    return (1 - 2 * bits).astype(np.int8)


def sign_flip_perm_test(diffs, two_sided: bool = True,
                        max_exact: int = 20, n_mc: int = 200_000,
                        seed: int = 0) -> float:
    """Exact (n <= max_exact) or Monte-Carlo paired sign-flip permutation p-value.

    Statistic = mean(diffs). Drops NaNs. Returns 1.0 for n==0.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = d.mean()
    tol = 1e-12
    if n <= max_exact:
        signs = _exact_sign_matrix(n)                 # (2^n, n)
        means = (signs @ d) / n                        # (2^n,)
        if two_sided:
            p = np.mean(np.abs(means) >= abs(obs) - tol)
        else:
            p = np.mean(means >= obs - tol)
        return float(p)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([1.0, -1.0]), size=(n_mc, n))
    means = (signs @ d) / n
    if two_sided:
        hits = np.sum(np.abs(means) >= abs(obs) - tol)
    else:
        hits = np.sum(means >= obs - tol)
    return float((hits + 1) / (n_mc + 1))             # +1: never report p=0


# ─────────────────────────── paired report ───────────────────────────────────
def paired_report(a, b, alpha: float = 0.05) -> dict:
    """Paired comparison of two methods' per-seed values a, b (a - b).

    Returns mean diff, t-based (1-alpha) CI, Cohen's dz, and three p-values:
    permutation (headline), paired t, Wilcoxon.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1)) if n > 1 else float("nan")
    dz = mean / sd if sd > 0 else float("inf") * (np.sign(mean) or 1.0)
    p_perm = sign_flip_perm_test(d, two_sided=True)
    p_t = float(stats.ttest_rel(a, b).pvalue) if n > 1 else float("nan")
    try:
        p_w = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:                                 # e.g. all differences zero
        p_w = float("nan")
    return dict(n=n, mean_diff=mean, sd_diff=sd,
                ci_low=mean - tcrit * se if n > 1 else float("nan"),
                ci_high=mean + tcrit * se if n > 1 else float("nan"),
                cohen_dz=float(dz), p_perm=p_perm, p_t=p_t, p_wilcoxon=p_w)


# ─────────────────────────── Holm-Bonferroni ─────────────────────────────────
def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ─────────────────────── cluster permutation (over time) ─────────────────────
def _t_per_column(D: np.ndarray) -> np.ndarray:
    """One-sample t-statistic of each column of D (n_seeds, T) across seeds."""
    n = D.shape[0]
    mean = D.mean(0)
    sd = D.std(0, ddof=1)
    se = sd / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, mean / se, 0.0)
    return t


def _clusters_from_t(t: np.ndarray, tcrit: float):
    """Contiguous same-sign runs with |t| > tcrit. Returns [(i0, i1, sign, mass)]."""
    T = len(t)
    sig = np.abs(t) > tcrit
    out, i = [], 0
    while i < T:
        if sig[i]:
            s = np.sign(t[i]); j = i
            while j + 1 < T and sig[j + 1] and np.sign(t[j + 1]) == s:
                j += 1
            out.append((i, j, int(s), float(t[i:j + 1].sum())))
            i = j + 1
        else:
            i += 1
    return out


def cluster_perm_test(D, alpha_point: float = 0.05, max_exact: int = 20,
                      n_mc: int = 20_000, seed: int = 0) -> list:
    """Cluster-based sign-flip permutation test over a time axis (paired design).

    D : (n_seeds, T) per-seed paired differences on a COMMON time grid (no NaN).
    Pointwise one-sample t across seeds; contiguous same-sign supra-threshold runs
    (|t| > t_crit at alpha_point) are candidate clusters with mass = sum of t over the
    run. Null = distribution of the MAX |cluster mass| under sign-flips of whole seed
    rows (exact 2^n enumeration when n <= max_exact, else Monte-Carlo). This controls
    the family-wise error rate across time.

    Returns the OBSERVED clusters as [{t0, t1, sign, mass, p_cluster}] (t0/t1 are
    inclusive column indices); empty list if no supra-threshold cluster exists.
    """
    D = np.asarray(D, float)
    if D.ndim != 2 or D.shape[0] < 2 or D.shape[1] == 0:
        return []
    n, _ = D.shape
    tcrit = float(stats.t.ppf(1 - alpha_point / 2, n - 1))
    obs = _clusters_from_t(_t_per_column(D), tcrit)
    if not obs:
        return []
    if n <= max_exact:
        signs = _exact_sign_matrix(n).astype(float)        # (2^n, n)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([1.0, -1.0]), size=(n_mc, n))
    null_max = np.empty(len(signs))
    for k in range(len(signs)):
        cl = _clusters_from_t(_t_per_column(signs[k][:, None] * D), tcrit)
        null_max[k] = max((abs(m) for *_, m in cl), default=0.0)
    res = []
    for (i0, i1, s, mass) in obs:
        p = float(np.sum(null_max >= abs(mass) - 1e-12) / len(signs))
        res.append(dict(t0=i0, t1=i1, sign=s, mass=mass, p_cluster=p))
    return res


# ─────────────────────────── power analysis ──────────────────────────────────
def _holm_rows(P: np.ndarray) -> np.ndarray:
    """Vectorised Holm over rows of P (n_sim, m)."""
    m = P.shape[1]
    order = np.argsort(P, axis=1)
    Psort = np.take_along_axis(P, order, axis=1)
    factors = (m - np.arange(m))[None, :]
    adj = np.clip(np.maximum.accumulate(Psort * factors, axis=1), 0.0, 1.0)
    out = np.empty_like(adj)
    np.put_along_axis(out, order, adj, axis=1)
    return out


def power_curve(D: np.ndarray, ns, alpha: float = 0.05, n_sim: int = 2000,
                n_perm: int = 4096, seed: int = 0) -> dict:
    """Simulation power vs n for a family of paired comparisons.

    D : (n_pilot, n_comp) per-seed pilot differences (one column per comparison).
        Bootstrap-resamples SEEDS (rows) jointly so cross-comparison correlation
        (all comparisons share `full`) is preserved; runs the actual sign-flip
        permutation test (Monte-Carlo, n_perm flips) per comparison; Holm-corrects
        across comparisons per simulated dataset; reports rejection rate.

    Returns {n: power_array(n_comp)} — power = P(reject H0 after Holm) per comparison.
    """
    D = np.asarray(D, float)
    n_pilot, n_comp = D.shape
    rng = np.random.default_rng(seed)
    flips = rng.choice(np.array([1.0, -1.0]), size=(n_perm, max(ns)))  # reused
    out = {}
    for n in ns:
        S = flips[:, :n]                                       # (n_perm, n)
        rej = np.zeros(n_comp)
        for _ in range(n_sim):
            rows = rng.integers(0, n_pilot, size=n)            # bootstrap seeds
            samp = D[rows]                                      # (n, n_comp)
            obs = np.abs(samp.mean(axis=0))                    # (n_comp,)
            permmeans = np.abs((S @ samp) / n)                # (n_perm, n_comp)
            pvals = (np.sum(permmeans >= obs[None, :] - 1e-12, axis=0) + 1) / (n_perm + 1)
            padj = holm(pvals)
            rej += (padj < alpha)
        out[int(n)] = (rej / n_sim).tolist()
    return out


def smallest_n_for_power(powers: dict, target: float = 0.90) -> int | None:
    """Smallest n whose power meets `target` for ALL comparisons."""
    for n in sorted(powers):
        if all(p >= target for p in powers[n]):
            return n
    return None
