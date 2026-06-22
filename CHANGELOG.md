# Changelog

All notable changes to this project are documented here.
Format: `## [Date] — description`.
Math changes are flagged with **[MATH CHANGE]**.
Bug fixes are flagged with **[BUG FIX]**.

---

## [2026-06-21] Time-and-depth credit assignment

Main result: deep e-prop assigns credit across **time and depth simultaneously**.

### New / changed

**[MATH CHANGE] `models/deep_rnn.py` — leaky integration**
`DeepRNN` gains a per-layer integration rate `alpha` (scalar or length-`n_layers`),
stored as a non-trainable buffer. Step becomes
`h^l_t = (1-α_l) h^l_{t-1} + α_l tanh(a^l_t)`. `alpha=1.0` (default) reproduces the
original vanilla tanh exactly, so all prior experiments are unchanged.

**[MATH CHANGE] `learning_rules/deep_eprop.py` — leaky carry + two ϵ^z controls**
- Leaky-aware: temporal carry `c^l = (1-α_l) + α_l ψ_raw W_rec_diag`, every
  instantaneous derivative scaled by the drive `α_l ψ_raw`, feedforward Jacobian
  `α_l ψ_raw ⊙ W_ff`. Uses `ψ_raw = 1 - tanh(a)^2` (not `1 - h^2`). At `α=1`
  identical to the previous vanilla deep e-prop.
- New `mode` argument acting on the cross-layer (ϵ^z) trace only:
  `'ablate_spatial'` (∂z/∂h=0 → removes depth credit; lower-layer grads → 0) and
  `'ablate_temporal'` (∂z/∂z_{t-1}=0 → removes cross-layer temporal credit). The
  within-layer self-traces ϵ^h are always kept intact.

**`learning_rules/interface.py`** — new rules `deep_ablate_spatial`,
`deep_ablate_temporal`; `DeepEpropRule` takes `mode`.

**`tasks/hierarchical_cue.py`** (new) — hierarchical classify-then-count of
mean-zero rising/falling temporal motifs. Mean-zero ⇒ a frozen/random lower layer
(reservoir) cannot fake the feature, so lower-layer credit genuinely matters.

**`experiments/deep_credit_time_depth.py`** (new) — E1 per-layer gradient cosine
vs BPTT + cross-temporal credit share vs delay; E2 learning curves; E3 delay
sweep. E2/E3 training parallelised across processes (deep e-prop is latency-bound
on many small ops; multiprocessing — not threads/GPU — is the effective speedup).

**`tests/sanity_checks.py`** — Test 6 (L=1 leaky deep e-prop == single-layer leaky
e-prop, exact) and Test 7 (ablations: spatial→lower grads exactly 0, temporal→lower
changed, upper-layer grads untouched by both). All 8 tests pass.

### Beef-up (same day)
- More seeds: gradient cosine 12→16, learning curves 3→6.
- Uncertainty bands switched from std to **SEM** (uncertainty of the mean) everywhere
  (E1 cosine curves now have bands; E2/E3 too) → tighter, statistically meaningful.
- New **credit summary** figure (`e1_credit_summary` / `exp5_credit_summary`): grouped
  bars at D=12 of lower- vs top-layer cosine for full/ablate_temporal/ablate_spatial —
  shows spatial→lower≈0, temporal→lower degraded, top-layer identical across methods.
- Learning curves de-noised: per-seed training data streams (seed in the batch seed) so
  hard batches don't hit all seeds at the same step, plus larger eval (4×512).
- Parallelism fixed: seeds run across a **spawn** process pool (fork is incompatible with
  PyTorch autograd — it crashed); GPU runs sequentially. `DEVICE=cpu` env forces CPU.

### Results (2-layer leaky DeepRNN, α=[0.5, 0.05], n_rec=32, hierarchical task)
- E1: full deep e-prop tracks BPTT for BOTH layers (lower cos 0.65–0.77, top
  0.88–0.95); `ablate_temporal` lower drops to ~0.61–0.66; `ablate_spatial`
  lower = 0; ~93–95% of lower-layer credit magnitude flows through ϵ^z.
- E2 (D=12): BPTT = 1.00 ≥ full = 0.88 ± 0.02 > ablate_temporal = 0.75 ≈
  ablate_spatial = 0.74.

---

## [2026-06-10] Phase 1 implementation

### Bug fixes

**[BUG FIX] `learning_rules/deep_eprop.py` — L=1 missing W_in gradient**
When `compute_deep_eprop_gradients` is called with a single-layer model
(`DeepRNN(n_layers=1)`), the gradient for `W_in` was never accumulated and
remained zero.  Root cause: gradient accumulation for `W_in` was placed
exclusively inside the "lower layers" cross-trace loop
(`for l_src in range(L-1)`), which is empty for `L=1`.  The self-trace
`eps_self_in` correctly tracked `∂h^0/∂W_in` but was never contracted with the
learning signal.

Fix: added a guarded clause after the top-layer self-trace block:

```python
if Lt == 0:          # layer 0 is both top AND the input layer
    grad_W_in += einsum('bi,bij->ij', delta, eps_self_in) / B
```

This does not affect `L >= 2` (where `Lt >= 1`, so the condition is false).
Detected by `tests/sanity_checks.py` Test 2 ("depth-1 deep e-prop == single-layer e-prop").

---

### New files

- **`tasks/cue_accumulation.py`** — Evidence accumulation task.  Over a cue
  window the network sees brief left/right pulses, then a silent delay `D`, then
  a single recall step.  Majority side wins.  Same interface as
  `store_and_recall.py`: `generate_batch(...)` → `(inputs, targets, mask)` and
  `task_accuracy(...)`.  `n_in=5` (left, right, recall, noise, bias),
  `n_out=2`.  The delay `D` is the primary difficulty knob: running evidence
  must survive `D` silent steps, which is only possible if stored in the
  per-neuron slow state.

- **`models/leaky_rnn.py`** — Promoted from `models/vanilla_rnn.py::LeakyRNN`.
  New version stores `alpha` as a registered buffer (shape `(n_rec,)`), supports
  optional per-neuron log-uniform alphas via `alpha_min` / `alpha_max`
  constructor arguments.  Compatible with `compute_eprop_leaky_gradients` (which
  already broadcasts over `(n_rec,)` alpha tensors).

- **`learning_rules/interface.py`** — `LearningRule` base class with
  `compute_gradients` / `update` methods, `apply_gradients` utility, and
  `make_learning_rule` factory for swappable rules from a single config string.
  Per-condition LR heuristic `lr_for_config(base_lr, depth, alpha)`.

- **`utils.py`** — `run_multi_seed` for aggregated multi-seed experiments with
  mean ± standard-error; `cosine_similarity_grads` helper.

- **`tests/__init__.py`** — Empty package init.

- **`tests/sanity_checks.py`** — Standalone fast correctness suite (6 tests,
  CPU-only, tiny nets, < 60 s total):
  0. Cue accumulation task: shapes, mask.sum()==B, label balance, frozen-net accuracy
  1. deep-RTRL gradient == BPTT to numerical precision (allclose + cosine ≈ 1)
  2. depth-1 deep e-prop == single-layer e-prop (catches the L=1 W_in bug)
  3. Finite-difference check of BPTT gradients on a tiny VanillaRNN
  4. Vanilla RNN: e-prop ≈ d=0 (carry ≈ 0 for tanh; documented as EXPECTED)
  5. Leaky RNN at long delay D: e-prop gradient meaningfully closer to BPTT
     than d=0 (the key hypothesis wedge)

---

### Modified files

- **`learning_rules/deep_eprop.py`** — Applied L=1 W_in gradient fix (see Bug
  fixes above).

- **`learning_rules/bptt.py`** — `compute_bptt_gradients` now uses
  `model.named_parameters()` instead of hard-coded attribute access, making it
  model-agnostic (works for `VanillaRNN`, `LeakyRNN`, `DeepRNN`).  Return keys
  now match whatever the model's named parameters are (backwards compatible for
  `VanillaRNN`).

- **`models/__init__.py`** — Exports `VanillaRNN`, `LeakyRNN` (from
  `leaky_rnn.py`), `DeepRNN`, `LIFNetwork`, `ALIFNetwork`, `LIFHeteroNetwork`.

- **`learning_rules/__init__.py`** — Exports all gradient-computation functions
  and the `LearningRule` interface.

- **`tasks/__init__.py`** — Exports `generate_batch`, `task_accuracy` from both
  tasks.

- **`requirements.txt`** — Added `tqdm` (optional, for progress bars in
  `run_multi_seed`).

---

### Notebook (`deep_eprop_colab.ipynb`)

**Sections removed from default run** (too slow for one Colab CPU session):
- Section 7: sMNIST / psMNIST at T=784 (cells moved to commented-out appendix)
- Section 8: Spiking LIF networks (OUT of main notebook per spec; code intact)
- Section 6/Exp 6: ALIF networks (same)
- Section 9: Spiking Heidelberg Digits (SHD) — removed entirely
- Section 10: Heterogeneous LIF (OUT of main notebook)

**Changes to kept sections:**
- Depth sweep: `DEPTHS_SWEEP` capped at `[1, 2, 3]` (was `[1, 2, 3, 4, 5]`)
- Master config block added near top (all tunable knobs in one cell, clearly
  marked for easy scale-up)
- Every section now saves figures/metrics to disk immediately on completion so a
  timeout never loses finished results
- Imports updated: `LeakyRNN` imported from `models.leaky_rnn`

**New section added:**
- Exp 5: Cue Accumulation + Leaky RNN — gradient cosine vs delay D and learning
  curves; primary demonstration of the e-prop > d=0 wedge on the new task

**Appendix (commented-out):**
- sMNIST / psMNIST cells appended after all figures are saved, left commented
  out and clearly labelled as opt-in

---

---

## [2026-06-11] Exp 4 learning curves redesign

### Motivation

Exp 4's original single-panel learning curve (D=5, single delay, MSE loss) did not
demonstrate a clear e-prop vs d=0 difference.  Investigation revealed two issues:

1. **LR too small** (`LR_LK = 1e-3`): the relative W_out update was ~1e-4/step;
   neither method converged within 1000 steps.

2. **Wrong loss**: MSE for a 4-class task under-penalises confident wrong predictions;
   cross-entropy is better calibrated.

### Changes to `deep_eprop_colab.ipynb`

**Master config (cell 6)**:
- `LR_LK`: `1e-3` → `5e-3`
- `N_STEPS_LK`: `1000` → `2000`

**Exp 4 setup (cell 28)**:
- `BATCH_LK`: `BATCH_DEFAULT` → `256` (explicit; larger batch → more stable gradient estimates)

**Exp 4 learning curves (cell 31) — complete redesign**:
- **Two-panel figure**: D=5 (left, control) vs D=20 (right, critical)
  - D=5 < τ=10: e-prop ≈ d=0 ≈ BPTT — all converge; trace has minimal impact
  - D=20 = 2τ: d=0 never reliably reaches full accuracy; e-prop and BPTT converge
- **Fixed evaluation batch** (1024 samples, seeded, not the training mini-batch):
  removes the per-step noise that made the original curves look like chance
- **Gradient clipping** (global max_norm=1.0) applied uniformly to both methods:
  stabilises SGD without changing gradient directions; clip is methodologically fair
- **Smooth overlay**: 5-point centred moving average drawn on top of raw dots
- **Loss**: `xent_error` / `_xent_loss` throughout (was `sl_mse`)

**Exp 5 learning curves (cell 36)**:
- Changed from D=`DELAY_LC_CA` (20) to D=50 (hardcoded as `DELAY_LC_CA_SHOW`)
- Updated methodological comment: "qualitative illustration" at D≈τ for the
  cue accumulation task; gradient cosine (cell 34) remains the primary result

---

### Deferred (TODO stubs only)

- Classify-then-count compositional variant of cue_accumulation
  (`# TODO: compositional cue task — each cue is a pattern to classify first`)
- Spiking ALIF version of cue_accumulation
  (`# TODO: ALIF version of cue_accumulation task`)
