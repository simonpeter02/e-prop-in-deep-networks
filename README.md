# Deep E-prop: Credit Assignment in Deep Recurrent Networks

**NeuroAI & ML 4 Neuro - S2026**

**Group:** Simon Peter, Yannick Säckl, Ruchit Kumar Patel

---

## 1. Project summary

**Title:** *Deep E-prop: testing online credit assignment across time and depth in deep recurrent networks.*

**Scientific question.** E-prop (Bellec et al. 2020) replaces backprop-through-time with a
forward, biologically plausible eligibility trace. In a **deep** recurrent network a
lower-layer synapse's credit must reach the readout by travelling both **up through the
layers** (a *spatial/depth* path) and **forward through time** in the upper layers'
recurrence (a *cross-layer temporal* path). We ask:
1. **Feasibility (Q1):** does the deep extension of e-prop (Millidge 2025) actually carry
   *meaningful*, BPTT-aligned credit along both paths in a deep recurrent network?
2. **Attribution (Q2):** which dimension of the cross-layer trace contributes how much —
   time vs. depth?

**Hypotheses.**

- **H1 (feasibility).** Deep e-prop's parameter gradients are positively aligned with the
  exact BPTT gradient at *every* layer, and a network trained with it learns a task that
  routes credit through the lower recurrent layer.
  *Prediction:* cos(g_deep-eprop, g_BPTT) > 0 at every layer; held-out accuracy improves with
  training. *Null:* the lower-layer gradient is uncorrelated with BPTT, or trained networks
  do no better than the floor where both cross-layer traces are ablated.
- **H2 (time vs. depth).** The cross-layer trace
  `ε^z = (∂z/∂h)·ε^h + (∂z/∂z_{t−1})·ε^z_{t−1}` splits into a **spatial** term (∂z/∂h, depth
  path) and a **cross-layer temporal** term (∂z/∂z_{t−1}, time path). Zeroing each in
  isolation (`ablate_spatial`, `ablate_temporal`) isolates its contribution.
  *Prediction:* `ablate_spatial` removes the only route into the lower-layer gradient, so it
  collapses to *exactly* zero; `ablate_temporal` retains most of the raw cosine to BPTT but
  the surviving lower-layer gradient becomes cue-agnostic (spatial carry makes gradients
  *travel*, temporal carry makes them *meaningful*). *Null:* the two ablations leave the
  lower-layer gradient materially unchanged from full.

**Approach.**

*Experiment 1.* We implement a non-spiking version of e-prop using leaky-tanh RNNs and
validate the implementation by reproducing Bellec et al. (2020)'s single-layer results on
their cue-accumulation task.

*Experiment 2.* We extend the setup to multiple layers and compare **full deep e-prop**
against two targeted ablations (`ablate_spatial`, `ablate_temporal`), a **random-reservoir**
control, and **BPTT** as ground truth, on a hierarchical "classify-then-count" task designed
so that solving it *requires* both depth and temporal credit.

---

## 2. Repository structure

```
e-prop-in-deep-networks/
├── README.md                     # this file
├── technical_note.md             # method, main results, limitations (grader-facing summary)
├── requirements.txt              # dependencies
├── utils.py                      # gradient helpers (flat_grads, cosine_sim_grads)
│
├── tasks/                        # benchmark tasks
│   ├── cue_accumulation.py       #   evidence accumulation — dense variant + population-coded
│   │                             #   variant; used by Experiment 1 (feasibility check)
│   ├── hierarchical_cue.py       #   main task (Experiment 2): classify-then-count motifs
│   └── routed_cue.py             #   harder routed variant (referenced in Limitations/future work)
│
├── models/                       # RNN definitions
│   ├── vanilla_rnn.py, leaky_rnn.py
│   └── deep_rnn.py               #   leaky DeepRNN used for the main result
│
├── learning_rules/               # gradient rules (shared interface)
│   ├── bptt.py                   #   ground truth
│   ├── eprop.py                  #   single-layer e-prop
│   ├── deep_eprop.py             #   deep e-prop + the two ablations
│   ├── deep_rtrl.py              #   exact online reference (RTRL)
│   └── interface.py              #   make_learning_rule() factory
│
├── experiments/                  # CLI pipeline (see §3)
│   ├── deep_credit_time_depth.py #   full multi-seed study: E1/E2/E3 + power/significance
│   ├── stats.py                  #   paired tests / permutation / power helpers
│   ├── single_layer_cue_accum.py #   Experiment 1: single-layer e-prop vs BPTT on cue accumulation
│   ├── exp2_cue_decoding.py      #   cue-decoding analysis (writes results/exp2_cue_decoding.json)
│   └── pilot_reservoir_resistance.py   #   reservoir-resistance pilot (future work)
│
├── notebooks/
│   └── main_results.ipynb        # PRIMARY: Part A computes the results JSON, Part B replots every
│                                 #          figure in the technical note (runs locally and on Colab)
│
├── docs/
│   └── mathematical_note.md      # full derivation behind the main result + numerical verification
│
├── results/                      # committed metrics the notebook (Part B) replots from
│   ├── exp2_learning_curves.json, exp2_gradient_credit.json,
│   │   exp2_cue_decoding.json, exp2_reservoir_control.json
│   └── main_results/             # figures shown in the technical note (exp1.*, exp2.*) + the
│                                 #   cue-decoding ladder, as .png/.svg/.pdf
└── tests/
    └── sanity_checks.py          # correctness suite (10 tests, CPU, < 5 s)
```

---

## 3. How to run

**Python:** 3.10+ (developed and tested on 3.12).

**Install**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Reproduce the figures (recommended — `notebooks/main_results.ipynb`).**
This notebook is the single source for every figure in [`technical_note.md`](technical_note.md).
It is split into two parts:

- **Part A — Computation** (cells commented out by default). Trains the networks and computes the
  gradient statistics, writing the metrics JSON into `results/` (e.g. `results/exp2_*.json` and the
  single-layer `results/single_layer_cue_accum_*.json`). Slower; GPU recommended.
- **Part B — Analysis & plots** (run this). Loads the committed `results/*.json` and *replots in
  seconds with no training*, writing figures to `results/main_results/` as `.png/.svg/.pdf`.

For a quick check, run **Part B** top-to-bottom against the committed JSON. To regenerate the numbers
from scratch, uncomment and run **Part A** first, then **Part B**. The notebook self-detects GPU/CPU
and clones the repo in its setup cell, so it also runs as-is on Colab. (The single-layer Experiment 1
JSON is not committed; Part B prints a "skipped" message for Figs 1.1–1.3 until Part A §1 is run.)

| Notebook section (Part B) | Figures |
|---|---|
| **1** Single-layer e-prop (Experiment 1) | `exp1.1`–`exp1.3` |
| **2.1** Learning curves | `exp2.1_learning_curves` |
| **2.2** Gradient credit vs delay + cross-temporal share | `exp2.2_gradient_credit` |
| **2.3** Convergence-speed significance | `exp2.3_speed_threshold`, `exp2.3b_speed_intervals` |
| **2.4** Credit summary (D=12) | `exp2.4_credit_summary` |
| **2.5** Cue decoding (spatial travels / temporal is meaningful) | `exp2.5_cue_decoding`, `exp2.5_cue_decoding_ladder` |
| **2.6** Readout-only reservoir control | `exp2.6_reservoir_control`, `exp2.6b_reservoir_curves` |

**Under the hood (full multi-seed study + CLI).** The notebook's Part A runs a compact version of the
study. A separate, fuller multi-seed pipeline — including the power and paired-significance analyses
behind the convergence-speed claims — lives in `experiments.deep_credit_time_depth`. It writes its
own `results/e1_*.json`, `e2_*.json`, `e3_*.json` outputs (independent of the notebook's
`exp2_*.json`):

```bash
python -u -m experiments.deep_credit_time_depth        # all parts (E1+E2+E3)
python -u -m experiments.deep_credit_time_depth e1     # gradient credit vs BPTT
python -u -m experiments.deep_credit_time_depth e2     # learning curves
python -u -m experiments.deep_credit_time_depth e3     # delay sweep
python -u -m experiments.deep_credit_time_depth power  # power analysis (choose n*)
python -u -m experiments.deep_credit_time_depth stats 8 # paired significance tests, n=8
```

Set `DEVICE=cpu` to force CPU; E2/E3 parallelise seeds across a process pool automatically.

**Supporting scripts**

```bash
python -m experiments.single_layer_cue_accum          # Experiment 1: single-layer e-prop vs BPTT
python -m experiments.exp2_cue_decoding               # cue-decoding analysis (writes exp2_cue_decoding.json)
python experiments/pilot_reservoir_resistance.py all  # reservoir-resistance pilot (future work)
```

**Correctness suite**

```bash
python -m tests.sanity_checks        # 10 tests, CPU-only, ~2 s
```

**Expected runtime** _(approximate; confirm on your hardware — the full study is fastest on GPU):_

| Command | Approx. runtime |
|---|---|
| `tests.sanity_checks` | ~2 s (CPU) |
| `notebooks/main_results.ipynb` Part B (replot only) | seconds (CPU) |
| `notebooks/main_results.ipynb` Part A (recompute) | minutes (GPU recommended) |
| `deep_credit_time_depth all` | minutes–tens of minutes (multi-seed; GPU recommended) |

---

## 4. Author contributions

- **Simon Peter** — co-conceptualized both experiments; implemented the hierarchical
  classify-then-count task and the two credit-path ablations (`ablate_spatial`,
  `ablate_temporal`) for **Experiment 2**.
- **Yannick Säckl** — co-conceptualized both experiments; implemented the random-reservoir
  control checks for **Experiment 2** (in `notebooks/main_results.ipynb` §2.6).
- **Ruchit Kumar Patel** — implemented the single-layer e-prop **cue-accumulation**
  reproduction of Bellec et al. 2020 (**Experiment 1**; `experiments/single_layer_cue_accum.py`
  and §1 of `notebooks/main_results.ipynb`).

All authors contributed to project planning, the design of the final presentation, and the
organization of the code repository.

---

## 5. Documentation of LLM usage

We used **Claude Opus 4.8** and **Fable 5** to assist in producing the code in this
repository — specifically the implementation of the training loops and the plotting code,
as well as for repository organization and documentation. All generated code, results, and
derivations were reviewed and are understood by the authors, who remain responsible for the
work in this project.

---

## Main result (Experiment 2)

Deep e-prop assigns credit across **time and depth simultaneously**. On the hierarchical
classify-then-count task (mean-zero rising/falling temporal motifs that a frozen random
lower layer cannot fake), using a 2-layer leaky `DeepRNN` (α = [0.5, 0.05], n_rec = 32):

- **Gradient level (2.2):** full deep e-prop tracks BPTT for **both** layers
  (lower/input-adjacent cosine ≈ 0.73–0.79, top/output-adjacent ≈ 0.92–0.96), positive at
  every layer and delay (**Q1: yes**); ≈ **93%** of lower-layer credit flows through the
  cross-layer **temporal** trace `ε^z` (**Q2: temporal dominates**).
- **Ablations (2.4/2.5):** `ablate_spatial` (remove ∂z/∂h) zeroes lower-layer gradients
  *exactly*; `ablate_temporal` (remove ∂z/∂z_{t−1}) keeps most of the raw cosine but its
  lower-layer gradient goes **cue-agnostic** (cue-margin decode ≈ 0.68 vs ≈ 0.84 for full).
  Both leave the **top layer and readout gradients unchanged**. Spatial carry makes gradients
  *travel*; temporal carry makes them *meaningful*.
- **Learning (2.1/2.3, D=12):** under Adam all trainable rules reach ≈ 1.0, so credit quality
  shows up as **convergence speed** — steps-to-90% order BPTT < full < ablate_temporal ≈
  ablate_spatial (full significantly faster than either ablation; the two ablations `ns`).
- **Caveat (2.6):** a frozen random lower layer with a trained top reaches ≈ 100%
  (indistinguishable from full, perm p ≈ 1.0), so **depth is used but not *required*** on this
  task — an honest limit on the "depth is required" framing (see `technical_note.md`).

Full derivation and numerical verification: [`docs/mathematical_note.md`](docs/mathematical_note.md).

Why **leaky** (not vanilla) tanh: a vanilla tanh RNN's e-prop temporal carry is negligible
(`ψ·W_ii ≈ 0.005`), so e-prop collapses onto the memoryless d=0 baseline; a leaky unit adds
a `(1−α)` diagonal carry that e-prop captures exactly (memory horizon `τ ≈ 1/(1−α)`).

## Key references

- Bellec et al. (2020) — E-prop: biologically plausible learning in recurrent SNNs
- Millidge (2025) — Deep E-prop
- Shalev-Merin (2026) — d=0 baseline / RTRL equivalences
- Zucchet et al. — Instantaneous spatial backprop
