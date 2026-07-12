# Technical Note — Deep E-prop: Credit Assignment Across Time and Depth

**NeuroAI & ML 4 Neuro — Sommersemester 2026**
**Authors:** Simon Peter, Yannick Säckl, Ruchit Kumar Patel

> This note summarizes the method, main results, and limitations. The full mathematical derivation, with every equation and its numerical
> verification against the code, can be found in [`docs/mathematical_note.md`](docs/mathematical_note.md).

---

## 1. Motivation and question

E-prop (Bellec et al. 2020) is a forward, biologically plausible alternative to
backprop-through-time (BPTT): each synapse maintains a local **eligibility trace** that,
multiplied by a top-down **learning signal**, approximates the true loss gradient. It works
well in *single-layer* recurrent networks. Millidge (2025) extends it to **deep** recurrent
networks, where a lower-layer synapse's credit must reach the readout by travelling both
**up through the layers** (a *spatial/depth* path) and **forward through time** in the
upper layers' recurrence (a *cross-layer temporal* path).

**We ask:** does deep e-prop actually carry credit along *both* paths, and how much does each path matter?

### Hypotheses

**H1 — Feasibility.** The deep e-prop recursion performs meaningful credit assignment across
depth: its parameter gradients are positively aligned with the exact BPTT gradient at every
layer, and a network trained with it learns a task that genuinely requires routing credit
through the lower recurrent layer.
- **Prediction:** cos(g_deep-eprop, g_BPTT) > 0 at every layer, held-out accuracy improves with
  training, and this holds specifically on a task a *frozen* lower layer cannot solve on its own.
- **Null:** the lower-layer gradient is uncorrelated with the true BPTT gradient, or a network
  trained with it does no better than the degenerate floor where both cross-layer traces are
  ablated (a random reservoir with only the readout trained).

**H2 — Attribution (time vs. depth).** The exact BPTT gradient is a sum over paths through the
time × depth lattice; deep e-prop's cross-layer trace
`ε^z = (∂z/∂h)·ε^h + (∂z/∂z_{t−1})·ε^z_{t−1}` carries two additive components — a **spatial**
term (∂z/∂h, the depth path) and a **cross-layer temporal** term (∂z/∂z_{t−1}, the time path).
Zeroing each term in isolation (`ablate_spatial`, `ablate_temporal`) should isolate its
contribution to the lower layer's credit.
- **Prediction:** `ablate_spatial` removes the only route into the lower-layer gradient, so it
  should collapse to *exactly* zero; `ablate_temporal` should leave a small residual gradient
  from the current time step alone, with lower cosine to BPTT than the full rule. Because the
  task requires holding information across a silent delay, the majority of the lower layer's
  credit *magnitude* should be attributable to the temporal term rather than the spatial one.
- **Null:** the two ablations leave the lower-layer gradient materially unchanged from full
  (i.e. the traces are not functionally distinguishable), or the temporal term instead accounts
  for only a minor share of the credit.

## 2. Method

1) Feasibility check --> 1-layer non-spiking reproduction of Bellec et al as sanity check

**Model.** A two-layer leaky-integrator RNN (`models/deep_rnn.py`):
`hˡ_t = (1−αˡ)·hˡ_{t−1} + αˡ·tanh(aˡ_t)`, with per-layer rates α = [0.5, 0.05] (fast lower,
slow top; top memory horizon τ ≈ 20 steps), n_rec = 32, linear readout from the top layer.
The leak is essential: it gives e-prop a real temporal carry to capture — in a vanilla
tanh RNN that carry is ≈ 0.005 and e-prop collapses onto the memoryless d=0 baseline.

**Task — hierarchical "classify-then-count"** (`tasks/hierarchical_cue.py`). Each trial
shows several short temporal **motifs** (mean-zero *rising* vs *falling* ramps — identical
mean and energy, differing only in the sign of their time-derivative), separated by silence,
then a long silent **delay**, then one **decision** step asking for the majority motif class.
By construction:
- *Classify (depth):* mean-zero motifs force the **lower** layer to learn a genuine temporal
  feature detector — a frozen random layer cannot fake it.
- *Count (time):* the top layer must accumulate per-motif classifications and hold them
  across the delay, so credit for an early motif must cross both depth and time.

**Learning rules compared** (all share the same forward model; only the gradient differs):

| Rule | What it does |
|---|---|
| **BPTT** | exact autograd — ground truth |
| **full deep e-prop** | full cross-layer trace `ε^z` (spatial seed + temporal carry) |
| **ablate_spatial** | set ∂z/∂h = 0 → removes the **depth** path |
| **ablate_temporal** | set ∂z/∂z_{t−1} = 0 → removes the **cross-layer temporal** path |
| **readout-only reservoir** | freeze both recurrent layers, train only the linear readout |

**Evaluation.** (E1) per-parameter **gradient cosine** to BPTT, and the fraction of
lower-layer credit carried by the temporal path; (E2) **learning curves** to convergence;
(E3) a **delay sweep**. Uncertainty is reported as SEM across seeds; the headline comparison
uses a paired sign-flip **permutation test** with Holm–Bonferroni correction (`experiments/stats.py`),
with the seed count (n = 8) chosen by a simulation **power analysis**.

## 3. Main results

**Result 1 — deep e-prop tracks BPTT for both layers, mostly via the temporal path.**
Full deep e-prop matches BPTT gradients for both layers (lower cosine ≈ 0.65–0.77, top
≈ 0.88–0.95), and ≈ **91–95%** of the lower-layer credit magnitude flows through the
cross-layer temporal trace `ε^z` across delays 4–32 — exactly the "count across the delay"
path the task is built to require.

![Figure 2.2 — Per-layer gradient cosine to BPTT and cross-temporal credit share vs delay.](results/main_results/exp2.2_gradient_credit.png)
*Figure 2.2 — Per-layer gradient cosine to BPTT and cross-temporal credit share vs delay. (`results/main_results/exp2.2_gradient_credit.{svg,pdf,png}`, `notebooks/main_results.ipynb` §2.2)*

**Result 2 — the two ablations behave exactly as the credit-path picture predicts.**
`ablate_spatial` zeroes the lower-layer gradient *exactly* (the depth path is the only
injection into `ε^z`); `ablate_temporal` leaves only a small, cue-agnostic decision-step
gradient (≈ 6–12% of full, cosine to BPTT ≈ 0.6). Both leave the **top layer and readout
gradients bit-for-bit identical to full**, because the ablations act only on the lower-layer
cross-trace.

![Figure 2.4 — Lower- vs top-layer gradient cosine at D=12 for full and both ablations.](results/main_results/exp2.4_credit_summary.png)
*Figure 2.4 — Lower- vs top-layer gradient cosine at D=12 for full and both ablations. (`results/main_results/exp2.4_credit_summary.{svg,pdf,png}`, `notebooks/main_results.ipynb` §2.4)*

**Result 3 — the credit-quality difference shows up in learning.**
At D = 12, final accuracy orders as **BPTT ≥ full > both controls** under matched SGD; under
Adam (which normalizes magnitude) the difference appears as convergence *speed*.

![Figure 2.1 — Learning curves at D=12 (mean ± SEM across seeds).](results/main_results/exp2.1_learning_curves.png)
*Figure 2.1 — Learning curves at D=12 (mean ± SEM across seeds). (`results/main_results/exp2.1_learning_curves.{svg,pdf,png}`, `notebooks/main_results.ipynb` §2.1)*

**Result 4 — reservoir control locates the floor.**
Freezing both layers (random reservoir + trained readout) reaches ≈ 0.75 — above chance
(0.5) but well below the trainable rules (≈ 1.0).

![Figure 2.6 — Random-reservoir floor vs trainable rules.](results/main_results/exp2.6_reservoir_control.png)
*Figure 2.6 — Random-reservoir floor vs trainable rules. (`results/main_results/exp2.6_reservoir_control.{svg,pdf,png}`, `notebooks/main_results.ipynb` §2.6)*

## 4. Limitations

- **"Depth is required" is weaker than it looks.** Under Adam, a trained top layer reading a
  *random* lower layer nearly solves the task (`ablate_spatial` ≈ 0.996 final accuracy),
  far above the frozen-both-layers reservoir floor (≈ 0.75). So zeroing all lower-layer
  credit barely hurts *final* accuracy on this task — the top recurrent layer reconstructs
  the temporal feature itself. A task where lower-layer credit is *indispensable* would need
  to prevent this (capacity/routing constraints; cf. `tasks/routed_cue.py`,
  `experiments/pilot_reservoir_resistance.py`). The reservoir control is exactly the
  instrument that exposes this caveat.
- **Diagonal approximation.** E-prop's only approximation is replacing the recurrent
  Jacobian with its diagonal; the residual cosine gap to BPTT (top ≈ 0.93, lower ≈ 0.72)
  grows with recurrence strength and delay, as expected. When the approximation is made
  exact (zero recurrence), deep e-prop matches BPTT to float32 precision (≈ 1e-9).
- **Scope.** Results are on a small (n_rec = 32), 2-layer, non-spiking leaky network on one
  synthetic task family; spiking (LIF/ALIF) and larger/real datasets (e.g. SHD) are present
  in the repo only as exploratory scaffolding, not validated results.
- **Statistics.** The power analysis fixes n = 8 seeds; the permutation test's resolution is
  floor-limited under Holm correction, so effect *significance* is established but effect
  *sizes* come from a modest seed count.

## 5. Reproducing the figures

| Figure | File in `results/` | Command |
|---|---|---|
| Fig 1.1 - learning curves single layer | `` | `` |
| Fig 1.2 - single layer speed threshold | `` | `` |
| Fig 1.3 - single layer delay sweep | `` | `` |
| Fig 2.1 - learning curves | `main_results/exp2.1_learning_curves.{svg,pdf}` | `notebooks/main_results.ipynb` §2.1 |
| Fig 2.2 - gradient credit | `main_results/exp2.2_gradient_credit.{svg,pdf}` | `notebooks/main_results.ipynb` §2.2 |
| Fig 2.3 - speed threshold | `main_results/exp2.3_speed_threshold.{svg,pdf}` | `notebooks/main_results.ipynb` §2.3 |
| Fig 2.4 - credit summary | `main_results/exp2.4_credit_summary.{svg,pdf}` | `notebooks/main_results.ipynb` §2.4 |
| Fig 2.5 - cue decoding | `main_results/exp2.5_cue_decoding.{svg,pdf}` | `notebooks/main_results.ipynb` §2.5 |
| Fig 2.6 - reservoir control | `main_results/exp2.6_reservoir_control.{svg,pdf}` | `notebooks/main_results.ipynb` §2.6 |
