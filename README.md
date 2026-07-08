# Deep E-prop: Credit Assignment Across Time *and* Depth in Recurrent Networks

**NeuroAI & ML 4 Neuro — Sommersemester 2026**

**Group:** Simon Peter, Yannick Säckl, Ruchit Kumar Patel

---

## 1. Project summary

**Title:** *Deep E-prop — testing online credit assignment across time and depth in deep recurrent networks.*

**Scientific question.** E-prop (Bellec et al. 2020) replaces backprop-through-time with a
forward, biologically plausible eligibility trace. Does its trace approximation still carry
useful credit when the network is made **deep**, where a lower-layer synapse must receive
credit that has to travel both *up through layers* (depth) **and** *forward through time*
(the top layer's recurrence)?

**Hypothesis.** Deep e-prop (Millidge 2025) assigns credit across time and depth
simultaneously; removing either path (the depth path or the cross-layer temporal path)
measurably degrades lower-layer learning, and the temporal path carries most of the
lower-layer credit on a task that requires accumulating a learned feature over a delay.

**Approach.** We build a leaky-integrator deep RNN and compare **full deep e-prop** against
two targeted ablations (`ablate_spatial`, `ablate_temporal`), a **random-reservoir** control,
and **BPTT** as ground truth, on a hierarchical "classify-then-count" task designed so that
solving it *requires* both depth and temporal credit.

---

## 2. Repository structure

```
e-prop-in-deep-networks/
├── README.md                     # this file
├── technical_note.md             # method, main results, limitations (grader-facing summary)
├── CHANGELOG.md                  # dated log of every math/code change
├── requirements.txt              # dependencies (versions pinned in iteration 2)
│
├── tasks/                        # benchmark tasks
│   ├── store_and_recall.py       #   single-layer reproduction task
│   ├── cue_accumulation.py       #   evidence-accumulation task
│   ├── hierarchical_cue.py       #   ★ main task: classify-then-count temporal motifs
│   └── routed_cue.py, shd.py, smnist.py
│
├── models/                       # RNN definitions
│   ├── vanilla_rnn.py, leaky_rnn.py
│   ├── deep_rnn.py               #   ★ leaky DeepRNN used for the main result
│   └── lif_rnn.py, deep_lif.py, deep_alif.py   # spiking variants (exploratory)
│
├── learning_rules/               # gradient rules (shared interface)
│   ├── bptt.py                   #   ground truth
│   ├── eprop.py, eprop_lif.py    #   single-layer e-prop
│   ├── deep_eprop.py             #   ★ deep e-prop + the two ablations
│   ├── deep_rtrl.py              #   exact online reference (RTRL)
│   └── interface.py              #   make_learning_rule() factory
│
├── experiments/                  # runnable scripts (see §3)
│   ├── single_layer_eprop.py     #   single-layer store-and-recall reproduction
│   ├── deep_eprop_comparison.py  #   2-layer deep e-prop vs d=0 vs BPTT
│   ├── depth_sweep.py            #   1–3 layer sweep
│   ├── deep_credit_time_depth.py #   ★ MAIN RESULT (Experiment 2): E1/E2/E3 + stats
│   └── exp5_*.py, pilot_*.py     #   spiking / reservoir-resistance explorations
│
├── notebooks/
│   ├── deep_eprop_colab.ipynb              # end-to-end Colab run
│   └── time_depth_detailed_results.ipynb   # ★ detailed main-result notebook + reservoir checks
│
├── figures/                      # figure-generation scripts + schematic diagrams
├── docs/
│   └── experiment5_mathematics.md          # full derivation behind the main result
├── results/                      # generated figures (.pdf/.svg) + metrics (.json)
└── tests/
    └── sanity_checks.py          # correctness suite (9 tests, CPU, < 60 s)
```

> ⚠️ **Not yet in the repo:** Ruchit's single-layer **cue-accumulation** reproduction of
> Bellec et al. 2020 (**Experiment 1**, see §4) is not committed yet and will be added.
> The `experiments/single_layer_eprop.py` file present today is the single-layer
> **store-and-recall** reproduction, which is a *different* task.

---

## 3. How to run

**Python:** 3.10+ _(exact version + pinned dependency versions to be fixed in iteration 2)._

**Install**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Reproduce the main result (Experiment 2 — hierarchical time-and-depth).**
All figures/metrics are written to `results/`. Internally the main result is named `exp5`
(historical numbering); the runnable module is `deep_credit_time_depth`:

```bash
python -u -m experiments.deep_credit_time_depth        # all parts (E1+E2+E3)
python -u -m experiments.deep_credit_time_depth e1     # gradient credit vs BPTT
python -u -m experiments.deep_credit_time_depth e2     # learning curves
python -u -m experiments.deep_credit_time_depth e3     # delay sweep
python -u -m experiments.deep_credit_time_depth power  # power analysis (choose n*)
python -u -m experiments.deep_credit_time_depth stats 8 # paired significance tests, n=8
```

Set `DEVICE=cpu` to force CPU; E2/E3 parallelise seeds across a process pool automatically.

**Supporting experiments**

```bash
python -m experiments.single_layer_eprop      # single-layer store-and-recall reproduction
python -m experiments.deep_eprop_comparison   # 2-layer deep e-prop vs d=0 vs BPTT
python -m experiments.depth_sweep             # 1–3 layer sweep
python experiments/pilot_reservoir_resistance.py all   # reservoir-resistance pilot
```

**Notebooks.** `notebooks/deep_eprop_colab.ipynb` runs the pipeline end-to-end;
`notebooks/time_depth_detailed_results.ipynb` contains the detailed main-result analysis
and the reservoir control.

**Correctness suite**

```bash
python -m tests.sanity_checks        # 9 tests, CPU-only, < 60 s
```

**Expected runtime** _(placeholder — confirm on your hardware in iteration 2):_

| Command | Approx. runtime |
|---|---|
| `tests.sanity_checks` | < 1 min (CPU) |
| `deep_credit_time_depth e1` | ~TODO |
| `deep_credit_time_depth e2` | ~TODO |
| `deep_credit_time_depth all` | ~TODO |

---

## 4. Author contributions

- **Simon Peter** — co-conceptualized both experiments; implemented the hierarchical
  classify-then-count task and the two credit-path ablations (`ablate_spatial`,
  `ablate_temporal`) for **Experiment 2**.
- **Yannick Säckl** — co-conceptualized both experiments; implemented the random-reservoir
  control checks for **Experiment 2** (in `notebooks/time_depth_detailed_results.ipynb`).
- **Ruchit Kumar Patel** — implemented the single-layer e-prop **cue-accumulation**
  reproduction of Bellec et al. 2020 (**Experiment 1**). _Code pending — not yet added to
  the repository (see §2)._

All authors contributed to project planning, the design of the final presentation, and the
organization of the code repository.

---

## 5. Documentation of LLM usage

We used **Claude Opus 4.8** and **Fable 5** to assist in producing the code in this
repository — specifically the implementation of the training loops and the plotting code —
as well as for repository organization and documentation. All generated code, results, and
derivations were reviewed and are understood by the authors, who remain responsible for the
work in this project.

---

## Main result (Experiment 2)

Deep e-prop assigns credit across **time and depth simultaneously**. On the hierarchical
classify-then-count task (mean-zero rising/falling temporal motifs that a frozen random
lower layer cannot fake), using a 2-layer leaky `DeepRNN` (α = [0.5, 0.05], n_rec = 32):

- **Gradient level (E1):** full deep e-prop tracks BPTT for **both** layers (lower cosine
  ≈ 0.65–0.77, top ≈ 0.88–0.95); ≈ **91–95%** of lower-layer credit flows through the
  cross-layer **temporal** trace `ε^z`.
- **Ablations:** `ablate_spatial` (remove ∂z/∂h) zeroes lower-layer gradients *exactly*;
  `ablate_temporal` (remove ∂z/∂z_{t−1}) leaves only a small, cue-agnostic decision-step
  gradient. Both leave the **top layer and readout gradients unchanged**.
- **Learning (E2, D=12):** BPTT ≥ full > both controls.
- **Caveat:** under Adam, a trained top layer reading a *random* lower layer nearly solves
  the task (`ablate_spatial` ≈ 0.996), well above the frozen-both-layers reservoir floor
  (≈ 0.75) — an honest limit on the "depth is required" framing (see `technical_note.md`).

Full derivation and numerical verification: [`docs/experiment5_mathematics.md`](docs/experiment5_mathematics.md).

Why **leaky** (not vanilla) tanh: a vanilla tanh RNN's e-prop temporal carry is negligible
(`ψ·W_ii ≈ 0.005`), so e-prop collapses onto the memoryless d=0 baseline; a leaky unit adds
a `(1−α)` diagonal carry that e-prop captures exactly (memory horizon `τ ≈ 1/(1−α)`).

## Key references

- Bellec et al. (2020) — E-prop: biologically plausible learning in recurrent SNNs
- Millidge (2025) — Deep E-prop
- Shalev-Merin (2026) — d=0 baseline / RTRL equivalences
- Zucchet et al. — Instantaneous spatial backprop
