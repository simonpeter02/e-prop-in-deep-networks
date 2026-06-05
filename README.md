# Deep E-prop: Testing Online Credit Assignment in Deep Recurrent Networks

**NeuroAI & ML 4 Neuro — Sommersemester 2026**

Group: Simon Peter, Yannick Säckl, Ruchit Kumar Patel

## Project Overview

This project tests whether e-prop's eligibility-trace approximation actually matters when extended to deep networks (Millidge 2025), or whether the simpler d=0 (immediate derivative) baseline suffices.

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
2. [ ] Deep-RTRL correctness check (match BPTT to numerical precision)
3. [ ] Deep e-prop vs d=0 vs BPTT at 2 layers — gradient cosine plots
4. [ ] Depth sweep (1–3 layers), delay-length sweep
