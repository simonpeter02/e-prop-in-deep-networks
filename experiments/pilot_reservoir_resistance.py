"""
Pilot: is the routed cue task reservoir-resistant, and do the spatial / temporal
e-prop ablations actually diverge?

We want, in some config:   full ≈ ablate_temporal  ≫  ablate_spatial ≈ esn_linear
  * full / ablate_temporal high  → the lower layer learns the routed feature, and
    cross-layer TEMPORAL credit is dispensable (concurrent aux supervision).
  * ablate_spatial low           → a frozen random layer 0 (reservoir) cannot
    route/retain the feature (capacity overload + context routing).
  * ablate_spatial ≈ esn_linear  → the trained TOP layer is NOT rescuing it
    (the decisive diagnostic; if spatial ≫ esn_linear the top reconstructs the
    feature and the task is not reservoir-resistant in this architecture).

The asymmetry signal is the AUX head (did layer 0 learn the feature). The DELAYED
head is reported too (top-layer counting across the delay; the time narrative).

Conditions (all share the same forward DeepRNN; only which params learn differs):
  full            — deep e-prop, both layers train
  ablate_temporal — deep e-prop, cross-layer temporal carry off
  ablate_spatial  — deep e-prop, spatial seed off → layer 0 frozen at random init
  esn_linear      — only the readout trains; BOTH recurrent layers stay random
                    (pure deep-ESN baseline; readout grad from e-prop is exact)

Run:
    python -u experiments/pilot_reservoir_resistance.py sanity   # quick learnability check (C1, full)
    python -u experiments/pilot_reservoir_resistance.py all      # full sweep C1..C4
    python -u experiments/pilot_reservoir_resistance.py c2       # one config
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.deep_rnn import DeepRNN
from learning_rules.deep_eprop import compute_deep_eprop_gradients, mse_error
from tasks.routed_cue import (
    generate_batch, head_accuracy, n_in_for, sequence_length, AUX_SLOTS, DEL_SLOTS,
)


# ─────────────────────────── config ───────────────────────────
def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE   = _pick_device()
ALPHA    = [0.3, 0.05]      # layer 0 spans the 3-step cue; layer 1 integrates the delay
N_CUES   = 5
DELAY    = 12
BATCH    = 64
LR       = 3e-3
N_STEPS  = 1500
N_SEEDS  = 2
EVAL_N   = 512
GRAD_CLIP = 1.0
TASK_KW  = dict(n_cues=N_CUES, delay=DELAY, cue_duration=3, inter_cue_interval=2,
                amp=2.0, feature_noise=0.15)

CONDITIONS = ["full", "ablate_temporal", "ablate_spatial", "esn_linear"]

# (name, n_rec, D).  D > n_rec ⇒ capacity overload (the asymmetry); D <= n_rec ⇒
# no overload (control: a random reservoir preserves channel 0, so spatial succeeds).
SWEEP = {
    "c1": ("overload (headline)",  12, 32),  # clean: full≈temporal≫spatial≈esn
    "c2": ("heavy overload",       16, 48),  # spatial creeps up but gap remains
    "c3": ("mild overload",        16, 16),  # intermediate
    "c4": ("no overload (control)", 24, 4),  # spatial should also succeed
}


def _mode_for(cond: str) -> str:
    if cond == "ablate_temporal":
        return "ablate_temporal"
    if cond == "ablate_spatial":
        return "ablate_spatial"
    return "full"   # full and esn_linear both use the full (exact) gradient


def _clip(grads, max_norm):
    if max_norm is None:
        return grads
    total = 0.0
    for g in grads.values():
        total += float((g * g).sum())
    total = total ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for k in grads:
            grads[k] = grads[k] * scale
    return grads


@torch.no_grad()
def evaluate(model, D, seed):
    inp, tgt, msk = generate_batch(EVAL_N, D=D, seed=seed, device=DEVICE, **TASK_KW)
    logits, _ = model(inp)
    return (head_accuracy(logits, tgt, AUX_SLOTS),
            head_accuracy(logits, tgt, DEL_SLOTS))


def train_condition(cond, n_rec, D, seed, n_steps, verbose=False):
    torch.manual_seed(seed)
    model = DeepRNN(n_in=n_in_for(D), n_rec=n_rec, n_out=4,
                    n_layers=2, alpha=ALPHA).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mode = _mode_for(cond)
    for s in range(n_steps):
        inp, tgt, msk = generate_batch(BATCH, D=D, seed=1_000_000 * (seed + 1) + s,
                                       device=DEVICE, **TASK_KW)
        grads = compute_deep_eprop_gradients(model, inp, tgt, msk, mse_error, mode=mode)
        if cond == "esn_linear":
            grads = {k: v for k, v in grads.items() if k in ("W_out", "b_out")}
        grads = _clip(grads, GRAD_CLIP)
        for name, p in model.named_parameters():
            p.grad = grads[name].detach() if name in grads else None
        opt.step()
        if verbose and (s % 400 == 0 or s == n_steps - 1):
            aux, dly = evaluate(model, D, seed=7_000_000 + seed)
            print(f"      [{cond:16s}] step {s:4d}  aux={aux:.3f}  del={dly:.3f}", flush=True)
    return model


def run_config(key, n_steps=N_STEPS, n_seeds=N_SEEDS, verbose=False):
    label, n_rec, D = SWEEP[key]
    T = sequence_length(n_cues=N_CUES, delay=DELAY,
                        cue_duration=TASK_KW["cue_duration"],
                        inter_cue_interval=TASK_KW["inter_cue_interval"])
    print(f"\n=== {key.upper()} ({label}): n_rec={n_rec}, D={D}, "
          f"n_in={n_in_for(D)}, T={T}, chance=0.500 [{DEVICE}] ===", flush=True)
    out = {}
    for cond in CONDITIONS:
        aux_s, del_s = [], []
        for seed in range(n_seeds):
            t0 = time.time()
            m = train_condition(cond, n_rec, D, seed, n_steps, verbose=verbose)
            aux, dly = evaluate(m, D, seed=9_000_000 + seed)
            aux_s.append(aux); del_s.append(dly)
            print(f"  {cond:16s}  seed {seed}  aux={aux:.3f}  del={dly:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out[cond] = dict(
            aux_mean=sum(aux_s) / len(aux_s), del_mean=sum(del_s) / len(del_s),
            aux=aux_s, del_=del_s)
    return dict(label=label, n_rec=n_rec, D=D, conditions=out)


def print_table(results):
    print("\n" + "=" * 72)
    print("PILOT SUMMARY — aux-head accuracy (asymmetry signal) [delayed-head]")
    print("=" * 72)
    head = f"{'config':24s}" + "".join(f"{c.split('_')[0][:8]:>10s}" for c in CONDITIONS)
    print(head)
    for key, res in results.items():
        row = f"{key+' '+res['label']:24s}"
        for c in CONDITIONS:
            row += f"{res['conditions'][c]['aux_mean']:>10.3f}"
        print(row)
    print("-" * 72)
    print("Decision rule: want full≈temporal aux ≥ .85, spatial aux ≤ .65,")
    print("and spatial ≈ esn_linear (top NOT rescuing). Delayed-head means:")
    for key, res in results.items():
        cs = res["conditions"]
        print(f"  {key}: " + ", ".join(f"{c.split('_')[0][:4]}={cs[c]['del_mean']:.2f}"
                                       for c in CONDITIONS))


def main():
    arg = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    os.makedirs("results", exist_ok=True)

    if arg == "sanity":
        print("SANITY: C1, full only, 500 steps (learnability + code check)")
        m = train_condition("full", *SWEEP["c1"][1:], seed=0, n_steps=500, verbose=True)
        aux, dly = evaluate(m, SWEEP["c1"][2], seed=123)
        print(f"\nSanity result: aux={aux:.3f}  del={dly:.3f}  (chance=0.5)")
        return

    keys = list(SWEEP) if arg == "all" else [arg]
    results = {k: run_config(k, verbose=(len(keys) == 1)) for k in keys}
    print_table(results)
    path = "results/pilot_reservoir_resistance.json"
    with open(path, "w") as f:
        json.dump(dict(device=DEVICE, alpha=ALPHA, lr=LR, n_steps=N_STEPS,
                       n_seeds=N_SEEDS, task=TASK_KW, results=results), f, indent=2)
    print(f"\nsaved → {path}")


if __name__ == "__main__":
    main()
