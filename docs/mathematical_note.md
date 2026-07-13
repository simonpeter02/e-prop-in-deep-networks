# Mathematics of the Model, Task, and Learning Modes

**Credit assignment across time *and* depth, simultaneously.**

This document states every equation behind Experiment 2 (the hierarchical
"classify-then-count" task on a 2-layer leaky `DeepRNN`), and then works out, for
each learning mode  (full deep e-prop, the two ablations, and the readout-only
reservoir control *exactly which parameter receives which gradient and what that
gradient tensor looks like* (nonzero, structurally zero, exact, or approximate).
It closes with the role of the **leaky** architecture and how each control would
behave in a **non-leaky** network.

All statements about which gradients are zero/nonzero were verified numerically
against this repository's code (`learning_rules/deep_eprop.py`,
`models/deep_rnn.py`, `tasks/hierarchical_cue.py`); the key checks are quoted
inline.

---

## 0. Notation

| symbol | meaning |
|---|---|
| `L = 2` | number of recurrent layers |
| `l ∈ {0, 1}` | layer index; **layer 0** = lower / *input-adjacent*, **layer 1** = top / *output-adjacent* |
| `n` | units per layer (`N_REC = 32`) |
| `T` | sequence length; `t = 1 … T` |
| `B` | batch size |
| `x_t ∈ ℝ⁵` | input at time `t` |
| `h^l_t ∈ ℝⁿ` | hidden state of layer `l` |
| `a^l_t ∈ ℝⁿ` | pre-activation (drive) of layer `l` |
| `α_l ∈ (0,1]` | integration rate of layer `l` (`α₀ = 0.5` fast, `α₁ = 0.05` slow) |
| `W_rec^l` | recurrent weights of layer `l`, `(n,n)` |
| `W_in` | input weights (layer 0 only), `(n,5)` |
| `W_ff^1` | feed-forward weights `0 → 1`, `(n,n)` |
| `b^l` | bias of layer `l` |
| `W_out, b_out` | readout, `(2,n)` and `(2,)` |
| `⊙` | elementwise product |
| `diag(M)` | vector of the diagonal of `M` |

Throughout, "top" ≡ layer 1 ≡ `z`, so the top hidden state is `z_t ≡ h^1_t`.

---

## 1. The task (`tasks/hierarchical_cue.py`)

Each trial presents `n_cues` cues (default `N_CUES = 3`, odd ⇒ no ties), each a
short **temporal motif** on feature channel 0, separated by silence, then a long
silent **delay**, then one **decision** step.

**Cue motif.** A cue of side `s ∈ {0,1}` spans `cue_duration` steps. The base
ramp is mean-zero,

```
base = linspace(−amp, +amp, cue_duration)              # e.g. [−2, 0, +2]
```

and the presented feature is

```
x_{t, 0} = direction · base[dt] + η,   η ~ 𝒩(0, feature_noise²)
direction = +1  (side 0, "rising")      or   −1  (side 1, "falling").
```

The two motifs (rising vs falling) have **the same mean (≈0)** and **the same
energy** — they differ only in the *sign of the temporal derivative*. This is the
crux: no static/instantaneous readout of channel 0 can separate the classes; a
learner must extract a **temporal** feature.

**Other channels** (`N_IN = 5`): 1 = distractor feature (pure noise), 2 = recall
signal (`= 1` at the single decision step), 3 = i.i.d. noise, 4 = constant bias.

**Label.** Majority side over the `n_cues` cues:

```
label = 1 (falling-majority)  if   Σ_c s_c > n_cues/2 ,   else 0.
```

**Targets / mask.** One-hot target and `mask_t = 1` **only** at the decision step
`t_recall = n_cues·(cue_duration + inter_cue_interval) + delay`; `0` everywhere
else. Total length `T = n_cues·(cue_duration + inter_cue_interval) + delay + 1`.

**Why this needs depth *and* time simultaneously.**
- *Classify (depth + within-layer time):* mean-zero motifs force the **lower
  layer** to learn a genuine temporal feature detector; a frozen random layer 0
  cannot fake it.
- *Count (cross-layer time):* each transient per-cue classification produced by
  layer 0 must be accumulated by the **slow top layer** and held across the silent
  delay. Credit for a layer-0 parameter from an *early* cue must travel **up**
  (depth) and **forward in time** through the top layer's recurrence.

---

## 2. The model — leaky `DeepRNN` (`models/deep_rnn.py`)

### 2.1 Forward dynamics

For `l = 0`:
```
a^0_t = W_rec^0 h^0_{t−1} + W_in x_t + b^0
```
For `l = 1`:
```
a^1_t = W_rec^1 h^1_{t−1} + W_ff^1 h^0_t + b^1
```
Leaky-integrator update (both layers):
```
h^l_t = (1 − α_l) · h^l_{t−1}  +  α_l · tanh(a^l_t)                        (2.1)
```
Readout (linear, from the top layer):
```
o_t = W_out h^1_t + b_out                                                  (2.2)
```

With `α_l = 1` this reduces to a **vanilla tanh** RNN, `h^l_t = tanh(a^l_t)`. The
`α < 1` term `(1−α_l) h^l_{t−1}` is the **leaky diagonal carry** that gives each
unit a memory horizon `τ_l ≈ 1/(1−α_l)` (top layer: `τ₁ ≈ 20` steps). Weights are
initialised with recurrent spectral radius scaled to `0.9`; `α` is a
non-trainable buffer.

### 2.2 Loss

Cross-entropy at the masked (decision) step:
```
𝓛 = −(1 / Σ_t m_t) · Σ_t m_t Σ_k y_{t,k} log softmax(o_t)_k .              (2.3)
```
Since `Σ_t m_t = B` (one decision step per trial), this averages over the batch.

---

## 3. Ground truth: BPTT (`learning_rules/bptt.py`)

BPTT is exact autograd through the unrolled (2.1)–(2.3). It is the reference every
approximate rule is measured against by per-parameter gradient cosine. Nothing
about it is approximate; it defines "correct."

---

## 4. E-prop building blocks

E-prop factorises each weight gradient into a **local eligibility trace** (forward,
parameter-local) times a **learning signal** (the readout error projected back):

```
d𝓛 / dθ  =  Σ_t  L_t  ·  ε^θ_t .                                           (4.1)
```

### 4.1 The learning signal

The output error is
```
e_t = softmax(o_t) − y_t        (masked)              # xent_error(), (B,2)   (4.2)
```
Projected to the **top hidden layer** it becomes the learning signal
```
δ_t = W_outᵀ e_t     ∈ ℝⁿ         # code: err_out @ W_out                  (4.3)
```
Because `m_t = 0` except at the decision step, **`δ_t` is nonzero only at
`t = t_recall`.** Every hidden-parameter gradient is therefore a single dot
product of `δ_{t_recall}` with that parameter's eligibility trace *at the decision
step*. This one fact explains most of the ablation behaviour below.

### 4.2 Instantaneous factors

Let `ψ^l_t = tanh'(a^l_t) = 1 − tanh(a^l_t)²`. From (2.1):

- **Drive** (sensitivity of `h^l_t` to its own pre-activation):
  ```
  ∂h^l_t / ∂a^l_t = α_l ψ^l_t   =:  drive^l_t .                            (4.4)
  ```
- **Diagonal temporal carry** (diagonal of the recurrent Jacobian):
  ```
  ∂h^l_t / ∂h^l_{t−1}  =  (1−α_l) I + α_l diag(ψ^l_t) W_rec^l
  c^l_t := diag(…)     =  (1−α_l) + drive^l_t ⊙ diag(W_rec^l) .            (4.5)
  ```

**The single e-prop approximation.** When propagating traces *through time*, the
full recurrent Jacobian (4.5, left) is replaced by its **diagonal** `c^l_t`
(4.5, right) — the off-diagonal recurrent couplings are dropped. Cross-layer
(feed-forward) Jacobians are kept in full. This is the *only* approximation; see
§8 for the exactness check.

### 4.3 Self (within-layer) eligibility traces `ε^h`

Each within-layer trace is a leaky accumulation: an instantaneous "seed" plus the
diagonal carry of the previous trace. For layer `l`:

```
ε^{rec,l}_{t,ij} = drive^l_{t,i} · h^l_{t−1,j}  +  c^l_{t,i} · ε^{rec,l}_{t−1,ij}   (W_rec^l)
ε^{b,l}_{t,i}    = drive^l_{t,i}                +  c^l_{t,i} · ε^{b,l}_{t−1,i}      (b^l)
ε^{in}_{t,ij}    = drive^0_{t,i} · x_{t,j}      +  c^0_{t,i} · ε^{in}_{t−1,ij}      (W_in, l=0)
ε^{ff,1}_{t,ij}  = drive^1_{t,i} · h^0_{t,j}    +  c^1_{t,i} · ε^{ff,1}_{t−1,ij}    (W_ff^1, l=1)
```
(The `d_zero` legacy baseline drops the carry term entirely, keeping only the
seed.) **These self-traces are always computed with full e-prop — the ablations
never touch them.**

### 4.4 Cross-layer (hierarchical) trace `ε^z`

Lower-layer (layer-0) parameters influence the **top** hidden state `z = h^1`
only through the feed-forward path. Deep e-prop (Millidge 2025) tracks

```
ε^z_t = ∂h^1_t / ∂θ      for θ ∈ {W_rec^0, W_in, b^0}
```

with the recursion (diagonal approx on the top recurrence):

```
ε^z_t  =  (∂z_t/∂h^0_t) · ε^h_t        +      (∂z_t/∂z_{t−1}) · ε^z_{t−1}    (4.6)
          └──── spatial seed ────┘             └──── temporal carry ────┘
```

with the two Jacobians

```
spatial:   (∂z_t/∂h^0_t)_{pq} = drive^1_{t,p} · W_ff^1_{pq}  =:  J^ff_{t,pq}
temporal:  (∂z_t/∂z_{t−1})    ≈ diag(c^1_t)      (same diagonal approx as (4.5))
```

Concretely, for `θ = W_rec^0_{ij}` the cross-trace is the 4-index tensor

```
ε^{z,rec}_{t,p,i,j} = Σ_q J^ff_{t,pq} · ε^{rec,0}_{t,q,i,j}   +   c^1_{t,p} · ε^{z,rec}_{t−1,p,i,j}   (4.7)
```
and analogously `ε^{z,in}`, `ε^{z,b}`. (For `L > 2` there are also non-adjacent
cross-traces chained through intermediate layers; with `L = 2` only the adjacent
term above exists.)

### 4.5 Gradient accumulation

**Top layer** (its own params) — via self-traces, contracting the learning signal:
```
d𝓛/dW_rec^1_{ij} = Σ_t δ_{t,i} · ε^{rec,1}_{t,ij}
d𝓛/dW_ff^1_{ij}  = Σ_t δ_{t,i} · ε^{ff,1}_{t,ij}
d𝓛/db^1_i        = Σ_t δ_{t,i} · ε^{b,1}_{t,i}                              (4.8)
```

**Lower layer** — via cross-traces (the learning signal lives at the top, so it
contracts the top index `p` of `ε^z`):
```
d𝓛/dW_rec^0_{ij} = Σ_t Σ_p δ_{t,p} · ε^{z,rec}_{t,p,i,j}
d𝓛/dW_in_{ij}    = Σ_t Σ_p δ_{t,p} · ε^{z,in}_{t,p,i,j}
d𝓛/db^0_i        = Σ_t Σ_p δ_{t,p} · ε^{z,b}_{t,p,i}                        (4.9)
```

**Readout** — exact (no trace, no approximation):
```
d𝓛/dW_out = (1/B) Σ_t e_t h^1_tᵀ ,     d𝓛/db_out = (1/B) Σ_t e_t .          (4.10)
```

---

## 5. The learning modes

All modes share the **same forward model**; only the gradient rule differs. The
ablations act **only on the cross-layer trace `ε^z`** (4.6) — the two terms of
that recursion — and never on the self-traces (4.3) or the readout (4.10).

```
mode              spatial seed (∂z/∂h)·ε^h     temporal carry (∂z/∂z_{t−1})·ε^z_{t−1}
────────────────  ─────────────────────────    ──────────────────────────────────────
full              ON                            ON
ablate_spatial    OFF (set ∂z/∂h = 0)           ON
ablate_temporal   ON                            OFF (set ∂z/∂z_{t−1} = 0)
readout-only      —  (no hidden grads at all; only W_out, b_out are trained)  —
```

`readout-only` is not an `ε^z` variant: it computes the **exact** full gradient
and then keeps only `{W_out, b_out}`, zeroing every recurrent/feed-forward/input
gradient. Both recurrent layers stay frozen at their random init — a genuine
**2-layer random reservoir with a trained linear readout** (a deep ESN).

---

## 6. What gradient each parameter receives, per mode

The table gives the gradient tensor for every parameter group.
"= full" means bit-for-bit identical to the full-e-prop gradient; "0" means the
tensor is **exactly** zero.

| parameter group | full | ablate_spatial | ablate_temporal | readout-only |
|---|---|---|---|---|
| `W_out, b_out` (readout) | exact (= BPTT) | **= full** | **= full** | **= full** (exact) |
| `W_rec^1, W_ff^1, b^1` (top) | e-prop (self-trace) | **= full** | **= full** | **0** |
| `W_in, W_rec^0, b^0` (lower) | e-prop (cross-trace) | **0** | small, cue-agnostic | **0** |

Two consequences are worth stating explicitly because they are the usual sources
of surprise:

### 6.1 The top layer and readout are identical across full / ablate_spatial / ablate_temporal

The controls modify `ε^z` (which carries *lower*-layer credit). The top layer's
own gradient (4.8) uses its **self**-traces `ε^{rec,1}, ε^{ff,1}, ε^{b,1}`, which
are never ablated; the readout gradient (4.10) is a pure input-output outer
product. So **ablating temporal or spatial credit leaves the entire top layer and
readout gradient unchanged.** Verified:

```
                 LOWER-norm     UPPER-norm
bptt             4.561e-01      4.303e-01
full             3.134e-01      3.961e-01
ablate_spatial   0.000000e+00   3.961e-01     ← UPPER identical to full
ablate_temporal  2.280e-02      3.961e-01     ← UPPER identical to full
```

### 6.2 `ablate_spatial` gives *exactly zero* lower-layer gradients; `ablate_temporal` gives *small, nonzero* ones

**`ablate_spatial`.** Setting `∂z/∂h = 0` removes the only injection of the lower
self-trace into `ε^z`. With the seed gone, (4.7) reduces to
`ε^z_t = c^1_t ⊙ ε^z_{t−1}`; starting from `ε^z_0 = 0` it stays **identically
zero** for all `t`. Hence (4.9) gives `d𝓛/dW_rec^0 = d𝓛/dW_in = d𝓛/db^0 = 0`
exactly. Per-key check:

```
W_in       full=9.730e-02   ablate_spatial=0.000e+00
W_recs.0   full=2.818e-01   ablate_spatial=0.000e+00
biases.0   full=9.685e-02   ablate_spatial=0.000e+00
```

So if you expect *"this ablation should zero the lower layer,"* that expectation
is correct **for `ablate_spatial`** — and the code delivers it.

**`ablate_temporal`.** Setting `∂z/∂z_{t−1} = 0` removes the accumulation, leaving
`ε^z_t = J^ff_t · ε^h_{0,t}` (same-timestep injection only). Because `δ_t` is
nonzero **only at the decision step** (4.3), the lower-layer gradient collapses to

```
d𝓛/dW_rec^0_{ij} = Σ_p δ_{t_rec,p} · [ J^ff_{t_rec} · ε^{rec,0}_{t_rec} ]_{p,i,j} .
```

This is **not zero**, but it is **not cue credit either.** It is the credit for how
layer-0 parameters shape layer-0's response *at the decision step itself* — and at
that step the network is driven by the **recall input** (channel 2). So the
surviving gradient measures the instantaneous input→layer0→layer1 transformation
at recall, projected up; it is essentially blind to the cues. Numerically it is
tiny and poorly aligned with the true gradient:

- magnitude ≈ **6–12 %** of the full lower-layer gradient (≈14× smaller at
  `D=12`);
- cosine to BPTT ≈ **0.6** (vs ≈0.72 for full) — a different direction, not a
  scaled-down true gradient.

**This small nonzero gradient is correct, not a bug.** If you were expecting
`ablate_temporal` to zero a layer, that is the misconception: `ablate_temporal`
keeps the *depth* (same-timestep) path, so the lower layer still receives the
decision-step transformation credit. Only `ablate_spatial` structurally zeroes the
lower layer.

### 6.3 Cross-temporal credit share

Define the fraction of lower-layer credit that flows through the top layer's
temporal carry:

```
share = ‖ g_full − g_ablate_temporal ‖ / ‖ g_full ‖     (lower-layer params).
```

With the leaky top (`α₁ = 0.05`) this is **≈ 0.91–0.95** across delays 4–32:
almost all lower-layer credit is carried *forward in time through the top layer*,
exactly the "count across the delay" path the task is built to require.

---

## 7. Learning behaviour (why the modes differ in training)

Because the ablations distort mainly the *magnitude* of the lower-layer gradient
(ablate_temporal) or zero it (ablate_spatial), the picture under different
optimisers is:

- **SGD (shared LR):** the magnitude deficit shows up directly as a lower/slower
  final accuracy → `BPTT ≥ full > ablate_temporal > ablate_spatial`.
- **Adam (per-synapse normalisation):** magnitude is neutralised, so the credit
  *quality* difference shows up as convergence **speed** (`full` fastest, controls
  slower); trainable-hidden methods reach the same plateau (this is what
  notebook 5.2/5.3 measure and significance-test).
- **readout-only reservoir:** both layers frozen; the ceiling is whatever a random
  leaky reservoir + linear readout can achieve — on this task ≈ **0.75** (above
  chance 0.5, well below the ~1.0 the trainable rules reach). This is the key
  diagnostic in §7.1.

Measured converged accuracy at `D = 12` under Adam (2 seeds, 1500 steps):

```
full             ≈ 1.00
ablate_temporal    0.995
ablate_spatial     0.996        ← lower layer FROZEN (grad = 0), yet ~perfect
readout-only       0.755        ← both layers frozen (random reservoir)
chance             0.500
```

### 7.1 What the readout-only reservoir tells us about the ablations

The reservoir is the natural floor for the ablations:

- `ablate_spatial` freezes layer 0 (grad = 0) **but still trains layer 1**;
- `readout-only` freezes **both** layers.

So `ablate_spatial − readout-only` isolates *"what does training the top layer buy
you when the bottom is frozen?"* The measured answer is striking:
`ablate_spatial ≈ 0.996 ≫ readout-only ≈ 0.755`. **The trained top layer, reading a
*random* lower layer, very nearly solves the task on its own.** In other words, on
this task under Adam, zeroing all lower-layer credit (`ablate_spatial`) barely
hurts final accuracy, because the top layer learns to extract the temporal motif
directly from the random layer-0 projection.

This is an important — and easily surprising — caveat to the "depth is required"
framing: the mean-zero motif does defeat a *linear readout of a random reservoir*
(0.755), but it does **not** defeat a *trained recurrent top layer over a random
reservoir* (0.996). If the goal is a task where lower-layer credit is genuinely
indispensable (so that `ablate_spatial` collapses toward the reservoir floor), the
top layer must be prevented from reconstructing the feature itself — e.g. by
overloading its capacity or routing (cf. `tasks/routed_cue.py` and
`experiments/pilot_reservoir_resistance.py`). The reservoir control is exactly the
instrument that exposes this, which is why it is added as a third control here.

---

## 8. Correctness check (only approximation is the diagonal)

Setting `W_rec^0 = W_rec^1 = 0` makes the recurrent Jacobian (4.5) *exactly*
diagonal (leak only, no off-diagonal term), so the sole e-prop approximation
vanishes and full deep e-prop must equal BPTT to numerical precision. It does
(float32, max abs difference per parameter):

```
W_ffs.0  4.2e-09   W_in     4.0e-09   W_out    4.7e-09
W_recs.0 4.0e-09   W_recs.1 2.1e-09   b_out    7.5e-09
biases.0 5.1e-09   biases.1 7.5e-09
```

With nonzero recurrence the cosine to BPTT is `1.00` (readout), `≈0.93` (top),
`≈0.72` (lower) — the residual gap is entirely the dropped off-diagonal recurrent
coupling, and it grows with recurrence strength and delay, as expected for
e-prop's diagonal approximation. **The deep e-prop implementation, including the
leaky carry and the cross-layer trace, is correct.**

---

## 9. Impact of the leaky architecture (and the non-leaky counterfactual)

The leak enters through the diagonal carry (4.5): `c^l = (1−α_l) + drive^l ⊙
diag(W_rec^l)`.

- **Leaky (`α₁ = 0.05`):** `c^1 ≈ 0.95` — a strong per-step temporal carry, memory
  horizon `τ₁ ≈ 20` steps.
- **Vanilla (`α = 1`):** `c = drive ⊙ diag(W_rec) = ψ · diag(W_rec)`. With
  `diag(W_rec) = O(1/√n)` and `ψ < 1`, this is `≈ 0.005` — **negligible**. The
  eligibility trace then *resets every step*, and e-prop collapses onto the
  memoryless `d = 0` baseline.

This is why the whole experiment uses a leaky RNN: it is what gives e-prop a real
temporal carry to capture, and it is what makes evidence accumulation over a silent
delay both **solvable** and a **meaningful** test of temporal credit assignment.

**How each control would behave in a non-leaky network:**

- **full.** The top carry `c^1 → ~0`, so the temporal term of `ε^z` (4.6)
  vanishes and `full ≈ ablate_temporal`. Measured cross-temporal share collapses
  from ≈0.91 (leaky) to **≈0.12** (non-leaky top). E-prop can no longer carry
  lower-layer credit across the delay — it degenerates to `d = 0`, and the
  count-across-delay task becomes effectively unlearnable by e-prop. *The leaky
  architecture is a precondition for full deep e-prop to work here at all.*

- **ablate_temporal.** Its *form* is unchanged (it already has no temporal carry),
  but its *relation to full* changes: since `full ≈ ablate_temporal` in a
  non-leaky net, the ablation would produce **almost no measurable effect**. The
  control loses its discriminating power — there is no temporal-credit gap left to
  ablate. (Its small lower-layer gradient, being the decision-step depth path
  §6.2, persists regardless of the leak.)

- **ablate_spatial.** **Unaffected.** The lower-layer gradient is zero
  *structurally* (the spatial seed is removed, so `ε^z ≡ 0` whatever the carry).
  This control behaves identically leaky or non-leaky.

- **readout-only reservoir.** **Architecture-sensitive, in the opposite
  direction.** A leaky top with `α₁ = 0.05` is a low-pass integrator (`τ₁ ≈ 20`):
  even a *random* leaky top retains a decaying running sum of its inputs across the
  delay, giving a linear readout something cue-correlated to read (≈0.75 here). A
  non-leaky random reservoir (spectral radius 0.9) has much shorter, more chaotic
  memory and would retain less across a 12–20 step delay, so **readout-only would
  score lower** (nearer chance). The leak therefore *raises* the reservoir floor —
  worth keeping in mind when arguing "the lower layer must learn the feature," because
  a strong leaky reservoir makes that floor higher and the argument correspondingly
  harder.

The lower layer's own rate `α₀ = 0.5` (`τ₀ ≈ 2`) is deliberately fast: long enough
for its self-trace to integrate the 3-step ramp *within* a cue (so it can detect
the motif), but short enough that it does **not** itself hold the running count —
that job is left to the slow top integrator, which is exactly the depth/time split
the task is designed to probe.

---

## 10. Summary

1. The deep e-prop implementation is **correct**: it matches BPTT to float32
   precision when the diagonal approximation is exact, and its only error
   otherwise is the (expected) dropped off-diagonal recurrent coupling.
2. `ablate_spatial` gives **exactly zero** lower-layer gradients (depth path
   removed); `ablate_temporal` gives a **small, nonzero, cue-agnostic**
   lower-layer gradient (the decision-step depth path survives) — this is correct,
   not a leak.
3. The **top layer and readout receive identical gradients** under full and both
   ablations, because the controls act only on the cross-layer trace `ε^z`.
4. The **readout-only reservoir** freezes both layers and reaches ≈0.75. Comparing
   it to `ablate_spatial` (≈0.996, lower layer frozen but top trained) shows the
   trained top layer nearly solves the task over a *random* lower layer — so on
   this task under Adam the depth ablation barely hurts accuracy, an important
   caveat to the "depth is required" framing.
5. The **leaky** architecture is essential: it gives e-prop a real temporal carry.
   In a non-leaky net `full ≈ ablate_temporal` (temporal control loses its power),
   `ablate_spatial` is unchanged, and the reservoir floor drops.
