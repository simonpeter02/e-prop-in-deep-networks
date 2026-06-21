# Deep E-prop: Testing Online Credit Assignment in Deep Recurrent Networks

**NeuroAI & ML 4 Neuro — Sommersemester 2026**

Group: Simon Peter, Yannick Säckl, Ruchit Kumar Patel

## Project Overview

This project tests whether e-prop's eligibility-trace approximation actually matters when extended to deep networks (Millidge 2025), or whether the simpler d=0 (immediate derivative) baseline suffices.

It also contains the **main result**: a demonstration that deep e-prop performs
credit assignment across **time and depth simultaneously**, using a non-spiking
leaky tanh `DeepRNN` and the two controls derived from Millidge's top-layer trace
`ϵ^z = (∂z/∂h)·ϵ^h + (∂z/∂z_{t-1})·ϵ^z_{t-1}`:

- **ablate_spatial** — set `∂z/∂h = 0`: removes the **depth** credit path
  (lower-layer gradients become exactly zero).
- **ablate_temporal** — set `∂z/∂z_{t-1} = 0`: removes the **cross-layer
  temporal** credit path (lower-layer credit can no longer cross time at the top).

On a hierarchical "classify-then-count" task of mean-zero temporal motifs
(`tasks/hierarchical_cue.py`), where the lower layer must learn a genuine temporal
feature (depth) whose per-cue output is accumulated by the top layer over a delay
(time), the result is **BPTT ≥ full deep e-prop > both controls**, and at the
gradient level full deep e-prop tracks BPTT for both layers while ~90% of the
lower-layer credit flows through the cross-layer temporal trace `ϵ^z`.

Why **leaky** (not vanilla) tanh: a vanilla tanh RNN's e-prop temporal carry is
the diagonal `ψ·W_ii ≈ 0.005` (negligible), so e-prop collapses onto d=0 and there
is no temporal-credit effect to measure; a leaky unit adds a `(1-α)` diagonal
carry that e-prop captures exactly (memory horizon `τ ≈ 1/(1-α)`).

Run it: `python -u -m experiments.deep_credit_time_depth` (E1 gradient credit,
E2 learning curves, E3 delay sweep → figures + JSON in `results/`).

## Repository Structure

```
tasks/           # Benchmark tasks (store-and-recall, evidence accumulation)
models/          # RNN model definitions
learning_rules/  # E-prop, deep e-prop, d=0, BPTT, deep-RTRL
experiments/     # Experiment scripts
results/         # Output figures and metrics
```

## Key References

- Bellec et al. (2020) — E-prop: Biologically plausible learning in RNNs
- Millidge (2025) — Deep E-prop
- Shalev-Merin (2026) — d=0 baseline / RTRL equivalences
- Zucchet et al. — Instantaneous spatial backprop

## Tasks

1. [x] Single-layer e-prop on store-and-recall (reproduce standard result)
2. [x] Deep-RTRL correctness check (match BPTT to numerical precision)
3. [x] Deep e-prop vs d=0 vs BPTT at 2 layers — gradient cosine plots
4. [x] Depth sweep (1–3 layers), delay-length sweep
5. [x] **Time-and-depth credit assignment** (leaky `DeepRNN`, hierarchical
       classify-then-count, full vs ablate_spatial vs ablate_temporal vs BPTT) —
       `experiments/deep_credit_time_depth.py`
