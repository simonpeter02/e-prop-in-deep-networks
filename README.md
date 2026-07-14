# Deep E-prop: Credit Assignment in Deep Recurrent Networks

**NeuroAI & ML 4 Neuro - S2026**

**Group:** Simon Peter, Yannick Säckl, Ruchit Kumar Patel

---

## 1. Project summary

**Title:** *Deep E-prop: online credit assignment across time and depth in deep recurrent networks.*

This project tests whether the deep extension of e-prop (Millidge 2025) carries meaningful, BPTT-aligned credit through *both* the temporal and cross-depth paths of a stacked recurrent network, and how much each path contributes. We hypothesise that deep e-prop's gradients are positively aligned with exact BPTT at every layer (feasibility), and that the cross-layer trace splits into a spatial term (∂z/∂h) that lets lower-layer gradients *travel* and a cross-layer temporal term (∂z/∂z_{t−1}) that makes them *meaningful*. We implement a non-spiking, leaky-tanh e-prop, validate it against Bellec et al. (2020)'s single-layer cue-accumulation task, then compare full deep e-prop against targeted trace ablations (`ablate_spatial`, `ablate_temporal`), a random-reservoir control, and a BPTT ground truth. The benchmark is a hierarchical "classify-then-count" task designed so that solving it requires credit assignment through both depth and time.

---

## 2. Repository structure

```
e-prop-in-deep-networks/
├── README.md                     # this file
├── technical_note.md             # method, main results, limitations (grader-facing summary)
├── requirements.txt              # dependencies
├── utils.py                      # multi-seed runner + gradient cosine / flattening helpers
│
├── tasks/                        # benchmark tasks
│   ├── cue_accumulation.py       #   evidence accumulation — dense 5-channel variant, plus the
│   │                             #   population-coded (Poisson) variant used by Experiment 1
│   ├── hierarchical_cue.py       #   main task: classify-then-count temporal motifs (Experiment 2)
│   └── routed_cue.py             #   distractor-selection variant used by the reservoir pilot
│
├── models/                       # RNN definitions (all leaky-tanh, non-spiking)
│   ├── vanilla_rnn.py            #   VanillaRNN + LeakyRNN (single layer)
│   ├── leaky_rnn.py              #   LeakyRNN with per-neuron time constants
│   └── deep_rnn.py               #   DeepRNN — stacked leaky layers, used for the main result
│
├── learning_rules/               # gradient rules (shared interface)
│   ├── bptt.py                   #   ground truth (autograd through time + depth)
│   ├── eprop.py                  #   single-layer e-prop (vanilla + leaky)
│   ├── deep_eprop.py             #   deep e-prop + the two ablations (ablate_spatial/_temporal)
│   ├── deep_rtrl.py              #   exact online reference (RTRL)
│   └── interface.py              #   make_learning_rule() factory
│
├── experiments/                  # runnable scripts (see §3)
│   ├── single_layer_cue_accum.py #   FEASIBILITY CHECK (Experiment 1): single-layer e-prop
│   │                             #   vs BPTT on cue accumulation (Figs 1.1–1.3)
│   ├── deep_credit_time_depth.py #   MAIN RESULT (Experiment 2): E1 gradient credit,
│   │                             #   E2 learning curves, E3 delay sweep, power + significance
│   ├── exp2_cue_decoding.py      #   is the lower-layer gradient cue-agnostic? (Fig 2.5)
│   ├── pilot_reservoir_resistance.py  # reservoir-resistance pilot on routed_cue
│   └── stats.py                  #   paired sign-flip permutation / t / Wilcoxon + Holm, power
│
├── notebooks/
│   └── main_results.ipynb        # PRIMARY: reproduces every figure in the technical note
│
├── results/                      # committed metrics JSON that the notebook replots from
│   ├── exp1_{learning_curves,cosine}.json              #   Experiment 1
│   ├── exp2_{learning_curves,gradient_credit,cue_decoding,reservoir_control}.json  # Experiment 2
│   └── main_results/             #   figures cited by the technical note (exp1.x / exp2.x, .png/.svg/.pdf)
│
└── tests/
    └── sanity_checks.py          # correctness suite (10 tests, CPU, < 60 s)
```
---

## 3. How to run

**Python:** 3.10+ (we used Python 3.12 for development and testing)

**Install**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

All scripts and the notebook auto-detect the device (CUDA → Apple MPS → CPU); force one with
`DEVICE=cpu`. Everything below is run from the repository root.

### 3 Reproduce the figures `notebooks/main_results.ipynb` (recommended)

This notebook is the single source for every figure in [`technical_note.md`](technical_note.md).
It has two parts:

- **Part A: Computation (commented out).** Repeats all training / gradient computation from
  scratch and **overwrites** the `results/*.json` that Part B reads. To verify the numbers,
  uncomment its cells and run them before Part B; this is slow and a GPU is recommended.
- **Part B: Analysis & plots (run this).** Loads the committed `results/*.json`, runs the
  significance tests, and writes the figures to `results/main_results/` as `exp1.x_*` / `exp2.x_*`.
  No training and no GPU needed. Run *only* Part B ("Setup" cell onwards) to reproduce the figures
  as published.

The final cell ("Save generated figures & results to git") is optional: it stages, and by default
commits, whatever the run produced. Set `DO_COMMIT = False` to leave your working tree untouched.

---

## 4. Author contributions

- **Simon Peter:** co-conceptualized both experiments; implemented the hierarchical
  classify-then-count task and the two credit-path ablations (`ablate_spatial`,
  `ablate_temporal`) for **Experiment 2**.
- **Yannick Säckl:** co-conceptualized both experiments; implemented the random-reservoir
  control checks for **Experiment 2** and contributed to the credit-path ablations (in `notebooks/main_results.ipynb`).
- **Ruchit Kumar Patel:** implemented the single-layer e-prop **cue-accumulation**
  reproduction of Bellec et al. (2020) (**Experiment 1**).

All authors contributed to project planning, the design of the final presentation, and the
organization of the code repository.

---

## 5. Documentation of LLM usage

We used **Claude Opus 4.8** and **Fable 5** to assist in producing the code in this
repository — specifically the implementation of the training loops and the plotting code,
as well as for repository organization and documentation. All generated code, results, and
derivations were reviewed and are understood by the authors, who remain responsible for the
work in this project.