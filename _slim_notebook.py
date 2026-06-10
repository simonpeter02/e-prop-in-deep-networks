"""
Notebook slimming script.  Run once from the repo root:
    python _slim_notebook.py
Overwrites deep_eprop_colab.ipynb in-place.
"""
import json, re

with open("deep_eprop_colab.ipynb") as f:
    nb = json.load(f)
orig = nb["cells"]


def src(text):
    """String → notebook source list."""
    lines = text.split("\n")
    out = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def code(text, execution_count=None):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src(text),
    }


def keep(i):
    """Keep original cell i, clearing outputs."""
    cell = dict(orig[i])
    cell = {k: v for k, v in cell.items()}
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def modify(i, new_text):
    """Return original cell i with replaced source."""
    cell = keep(i)
    cell["source"] = src(new_text)
    return cell


# ── Modified cell sources ─────────────────────────────────────────────────────

IMPORTS_SRC = """\
from tasks.store_and_recall  import generate_batch, task_accuracy
from tasks.cue_accumulation  import generate_batch as ca_batch, task_accuracy as ca_acc
from models.vanilla_rnn      import VanillaRNN
from models.leaky_rnn        import LeakyRNN          # promoted module; supports per-neuron alpha
from models.deep_rnn         import DeepRNN
from models.lif_rnn          import LIFNetwork, ALIFNetwork
from learning_rules.eprop         import compute_eprop_gradients, mse_error as sl_mse
from learning_rules.eprop         import compute_eprop_leaky_gradients
from learning_rules.bptt          import compute_bptt_gradients, _mse_loss
from learning_rules.deep_eprop    import compute_deep_eprop_gradients, mse_error
from learning_rules.deep_rtrl     import compute_deep_rtrl_gradients
from learning_rules.interface     import apply_gradients
from learning_rules.bptt          import _xent_loss

SEED       = 42
N_PATTERNS = 4
torch.manual_seed(SEED)
np.random.seed(SEED)

print('Imports OK')"""

# Master config block (inserted after imports+helpers)
MASTER_CFG_SRC = """\
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MASTER CONFIG — all tunable knobs in one place                         ║
# ║  Defaults are the MINIMUM that shows clean method separation on CPU.     ║
# ║  Values marked ↑ can be increased for publication-quality results.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Exp 1: single-layer e-prop (store-and-recall) ───────────────────────────
N_REC_SL    = 100
DELAY_SL    = 2
N_STEPS     = 1000      # ↑
LR          = 1e-3
EVAL_EVERY  = 50
delays_sl   = [1, 2, 3, 5, 10, 20, 50]
n_trials_sl = 30        # ↑

# ── Exp 2A: deep-RTRL verification ──────────────────────────────────────────
N_REC_RTRL  = 10
N_REPS      = 30        # ↑

# ── Exp 2B: 2-layer deep e-prop ─────────────────────────────────────────────
N_REC_2L    = 50
delays_2l_grid = [1, 2, 3, 5, 10, 20]
n_trials    = 30        # ↑
N_STEPS_2L  = 1000      # ↑
LR_2L       = 1e-3
EVAL_2L     = 50
delay_2l    = 2

# ── Exp 3: depth sweep ──────────────────────────────────────────────────────
# Cap at 3 for Colab CPU; change to [1,2,3,4,5] for pub-quality ↑
DEPTHS_SWEEP = [1, 2, 3]
DELAYS_SWEEP = [2, 5, 10]
N_REC_DS     = 50
N_TRIALS_DS  = 15       # ↑
N_STEPS_DS   = 600      # ↑
LR_DS        = 1e-3
EVAL_DS      = 30

# ── Exp 4: leaky RNN alpha sweep ─────────────────────────────────────────────
N_REC_LK     = 100
ALPHAS_LK    = [0.05, 0.1, 0.2, 0.5, 1.0]
DELAY_LK     = 5
N_TRIALS_LK  = 30       # ↑
ALPHA_LC     = 0.1      # alpha for learning curves
N_STEPS_LK   = 1000    # ↑
LR_LK        = 1e-3
EVAL_LK      = 50

# ── Exp 5: cue accumulation + leaky RNN ─────────────────────────────────────
N_REC_CA     = 80       # ↑
ALPHA_CA     = 0.1      # strong leak: τ ≈ 11 steps
N_CUES_CA    = 5        # odd → no ties
CUE_DUR_CA   = 1
ICI_CA       = 5        # inter-cue silence (steps)
DELAYS_CA    = [5, 10, 20, 30, 50]
N_TRIALS_CA  = 20       # ↑
N_STEPS_CA   = 800      # ↑
LR_CA        = 3e-4     # scaled for strong leak
EVAL_CA      = 40
DELAY_LC_CA  = 20       # delay for learning curves

# ── Batch sizes (set from device; see cell above) ────────────────────────────
BATCH_SL  = BATCH_DEFAULT
BATCH_2L  = BATCH_DEFAULT
BATCH_DS  = BATCH_DEFAULT
BATCH_LK  = BATCH_DEFAULT
BATCH_CA  = BATCH_DEFAULT

# ── Input sizes ──────────────────────────────────────────────────────────────
n_in     = N_PATTERNS + 2   # store-and-recall: patterns + recall + bias
n_in_lk  = N_PATTERNS + 2   # leaky RNN (same task)
n_in_ca  = 5                 # cue-accum: left, right, recall, noise, bias

torch.manual_seed(SEED)
np.random.seed(SEED)
print("Master config loaded.")"""

# Cell 22 modified (depth sweep config -- override DEPTHS_SWEEP from master config)
DEPTHS_SWEEP_SRC = """\
# Depth sweep config — taken from MASTER CONFIG block above.
# DEPTHS_SWEEP, DELAYS_SWEEP, N_REC_DS, etc. already set.
# This cell is kept for readability; variables are already defined.
print(f"Depth sweep: depths={DEPTHS_SWEEP}, delays={DELAYS_SWEEP}, "
      f"n_rec={N_REC_DS}, {N_TRIALS_DS} trials, {N_STEPS_DS} steps")"""

# Append JSON save to cell 23 (depth sweep cosine computation)
def patch_cell_23_add_save(orig_src):
    addition = """
# ── Incremental save ─────────────────────────────────────────────────────────
import json as _json, os
os.makedirs("results", exist_ok=True)
with open("results/exp3_depth_sweep_cosine.json", "w") as _f:
    _json.dump({"depths": DEPTHS_SWEEP, "delays": DELAYS_SWEEP,
                "cosines": ds_cosine}, _f, default=float)
print("Saved exp3 cosine metrics.")"""
    return orig_src + addition


# Append JSON save to cell 29 (leaky alpha sweep cosine)
def patch_cell_29_add_save(orig_src):
    addition = """
# ── Incremental save ─────────────────────────────────────────────────────────
import json as _json, os
os.makedirs("results", exist_ok=True)
with open("results/exp4_leaky_cosine_vs_alpha.json", "w") as _f:
    _json.dump({"alphas": ALPHAS_LK, "cos_ep": cos_lk_ep, "cos_d0": cos_lk_d0}, _f,
               default=float)
print("Saved exp4 cosine metrics.")"""
    return orig_src + addition


# Leaky RNN learning curves cell (31) — patch to use new LeakyRNN and master config
LEAKY_LC_SRC = """\
# Learning curves at alpha=0.1: e-prop should converge faster than d=0
# (N_STEPS_LK, LR_LK, EVAL_LK, ALPHA_LC set in MASTER CONFIG)
delay_lk_lc = 5

curves_lk = {}
for label, d_zero, use_bptt in [('e-prop', False, False),
                                  ('d=0',   True,  False),
                                  ('BPTT',  False, True )]:
    torch.manual_seed(SEED)
    model = LeakyRNN(n_in_lk, N_REC_LK, N_PATTERNS, alpha=ALPHA_LC).to(DEVICE)
    accs  = []
    for step in range(N_STEPS_LK):
        inp, tgt, msk = generate_batch(BATCH_LK, N_PATTERNS, delay_lk_lc, 1, 1, DEVICE)
        if use_bptt:
            grads = bptt_grads_deep(model, inp, tgt, msk)
        else:
            grads = compute_eprop_leaky_gradients(model, inp, tgt, msk, sl_mse, d_zero=d_zero)
        apply_grads_sl(model, grads, LR_LK)
        if step % EVAL_LK == 0:
            with torch.no_grad():
                out, _ = model(inp)
            accs.append(task_accuracy(out, tgt, msk))
    curves_lk[label] = accs
    print(f"  [{label}] final acc={accs[-1]:.3f}")

steps_lk = list(range(0, N_STEPS_LK, EVAL_LK))
fig, ax = plt.subplots(figsize=(7, 4))
colors_lk = {'e-prop': 'tab:blue', 'd=0': 'tab:orange', 'BPTT': 'tab:green'}
markers_lk = {'e-prop': 'o', 'd=0': 's', 'BPTT': '^'}
for label in ['e-prop', 'd=0', 'BPTT']:
    ax.plot(steps_lk, curves_lk[label], marker=markers_lk[label], ms=4,
            label=label, color=colors_lk[label])
ax.axhline(0.5, color='gray', ls='--', alpha=0.4, label='chance')
ax.set_xlabel('Training step', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title(f'Leaky RNN Learning Curves (α={ALPHA_LC}, D={delay_lk_lc})')
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()
save_fig(fig, 'exp4_leaky_learning_curves')
plt.show()

import json as _json
with open('results/exp4_leaky_curves.json', 'w') as _f:
    _json.dump({'steps': steps_lk, 'curves': curves_lk}, _f)
print("Saved exp4 learning curves.")"""


# ── New Experiment 5 cells ────────────────────────────────────────────────────

EXP5_MD = """\
---
## 5. Experiment 5 — Cue Accumulation with Leaky Integrator RNN

**Question:** Does e-prop's temporal eligibility trace enable credit assignment
over a silent delay on a multi-cue accumulation task?

**Task:** A stream of `n_cues` brief left/right pulses is presented, then a silent
delay of `D` steps with no input.  The network must report which side had more
cues at a single recall step.  The count can only survive the silence if stored in
the slow per-neuron leak state.

**Why this matters:** Unlike store-and-recall (one binary cue, trivial memory
for any RNN), the accumulator must keep a running count.  This is only possible
with per-neuron leak (alpha << 1).  The e-prop eligibility trace carries the
diagonal decay `(1-alpha)` forward; d=0 discards it.

**Prediction:** cos(e-prop, BPTT) - cos(d=0, BPTT) should grow with D
and be largest for small alpha (strong leak / long time constant)."""

EXP5_CONFIG = """\
# Config already set in MASTER CONFIG (N_REC_CA, ALPHA_CA, etc.)
print(f"Exp 5 setup:")
print(f"  LeakyRNN: n_rec={N_REC_CA}, alpha={ALPHA_CA} (tau≈{1/(1-ALPHA_CA):.1f} steps)")
print(f"  Task: n_cues={N_CUES_CA}, cue_dur={CUE_DUR_CA}, ici={ICI_CA}")
print(f"  Delays: {DELAYS_CA}")
T_max = N_CUES_CA*(CUE_DUR_CA+ICI_CA) + max(DELAYS_CA) + 1
print(f"  T at max delay: {T_max} steps")"""

EXP5_COSINE_SRC = """\
# Gradient cosine vs delay D: e-prop vs d=0 vs BPTT
import time
print(f"Gradient cosine vs delay (alpha={ALPHA_CA}, {N_TRIALS_CA} trials/delay) ...")

cos_ca = {'eprop': [], 'd0': []}
for delay in DELAYS_CA:
    t0 = time.time()
    se, sd = [], []
    for trial in range(N_TRIALS_CA):
        torch.manual_seed(SEED + trial * 7)
        model = LeakyRNN(n_in_ca, N_REC_CA, 2, alpha=ALPHA_CA).to(DEVICE)
        inp, tgt, msk = ca_batch(BATCH_CA, N_CUES_CA, delay,
                                  CUE_DUR_CA, ICI_CA, seed=SEED+trial)
        inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)

        g_b  = bptt_grads_deep(model, inp, tgt, msk)
        g_ep = compute_eprop_leaky_gradients(model, inp, tgt, msk, sl_mse, d_zero=False)
        g_d0 = compute_eprop_leaky_gradients(model, inp, tgt, msk, sl_mse, d_zero=True)

        all_k = list(g_b.keys())
        ce = cosine_keys(g_ep, g_b, all_k)
        cd = cosine_keys(g_d0, g_b, all_k)
        if not (np.isnan(ce) or np.isnan(cd)):
            se.append(ce); sd.append(cd)

    cos_ca['eprop'].append(float(np.mean(se)) if se else float('nan'))
    cos_ca['d0'].append(float(np.mean(sd)) if sd else float('nan'))
    print(f"  D={delay:3d}: e-prop={cos_ca['eprop'][-1]:.3f}  "
          f"d=0={cos_ca['d0'][-1]:.3f}  [{time.time()-t0:.1f}s]")

import json as _json
with open('results/exp5_cue_accum_cosine.json', 'w') as _f:
    _json.dump({'delays': DELAYS_CA, 'cos_eprop': cos_ca['eprop'],
                'cos_d0': cos_ca['d0']}, _f)
print("Saved exp5 cosine metrics.")"""

EXP5_COSINE_PLOT = """\
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(DELAYS_CA, cos_ca['eprop'], 'o-', color='tab:blue',   label='e-prop vs BPTT')
ax.plot(DELAYS_CA, cos_ca['d0'],   's-', color='tab:orange',  label='d=0 vs BPTT')
ax.fill_between(DELAYS_CA,
                [e - d for e, d in zip(cos_ca['eprop'], cos_ca['d0'])],
                [0]*len(DELAYS_CA), alpha=0.1, color='tab:blue', label='wedge')
ax.set_xlabel('Delay D (steps)', fontsize=12)
ax.set_ylabel('Cosine similarity with BPTT', fontsize=12)
ax.set_title(f'Cue Accumulation: Gradient Alignment vs Delay\\n'
             f'(LeakyRNN α={ALPHA_CA}, n={N_REC_CA}, {N_CUES_CA} cues)')
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(0.5, color='gray', ls='--', alpha=0.4)
fig.tight_layout()
save_fig(fig, 'exp5_cue_accum_cosine_vs_delay')
plt.show()"""

EXP5_LC_SRC = """\
# Learning curves on cue accumulation at fixed delay D=DELAY_LC_CA
print(f"Training LeakyRNN (alpha={ALPHA_CA}) on cue accumulation (D={DELAY_LC_CA}) ...")

curves_ca = {}
for label, d_zero, use_bptt in [('e-prop', False, False),
                                  ('d=0',   True,  False),
                                  ('BPTT',  False, True )]:
    torch.manual_seed(SEED)
    model = LeakyRNN(n_in_ca, N_REC_CA, 2, alpha=ALPHA_CA).to(DEVICE)
    accs  = []
    for step in range(N_STEPS_CA):
        inp, tgt, msk = ca_batch(BATCH_CA, N_CUES_CA, DELAY_LC_CA,
                                  CUE_DUR_CA, ICI_CA, seed=SEED+step)
        inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
        if use_bptt:
            grads = bptt_grads_deep(model, inp, tgt, msk)
        else:
            grads = compute_eprop_leaky_gradients(model, inp, tgt, msk,
                                                   sl_mse, d_zero=d_zero)
        apply_grads_sl(model, grads, LR_CA)
        if step % EVAL_CA == 0:
            with torch.no_grad():
                out, _ = model(inp)
            accs.append(ca_acc(out, tgt, msk))
    curves_ca[label] = accs
    print(f"  [{label}] final acc={accs[-1]:.3f}")

steps_ca = list(range(0, N_STEPS_CA, EVAL_CA))
fig, ax = plt.subplots(figsize=(7, 4))
col_ca  = {'e-prop': 'tab:blue', 'd=0': 'tab:orange', 'BPTT': 'tab:green'}
mark_ca = {'e-prop': 'o', 'd=0': 's', 'BPTT': '^'}
for label in ['e-prop', 'd=0', 'BPTT']:
    ax.plot(steps_ca, curves_ca[label], marker=mark_ca[label], ms=4,
            label=label, color=col_ca[label])
ax.axhline(0.5, color='gray', ls='--', alpha=0.4, label='chance')
ax.set_xlabel('Training step', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title(f'Cue Accumulation Learning Curves\\n'
             f'(LeakyRNN α={ALPHA_CA}, D={DELAY_LC_CA}, {N_CUES_CA} cues)')
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()
save_fig(fig, 'exp5_cue_accum_learning_curves')
plt.show()

import json as _json
with open('results/exp5_cue_accum_curves.json', 'w') as _f:
    _json.dump({'steps': steps_ca, 'curves': curves_ca,
                'delay': DELAY_LC_CA, 'alpha': ALPHA_CA}, _f)
print("Saved exp5 learning curves.")"""


# ── Commented-out sMNIST appendix ────────────────────────────────────────────

SMNIST_MD = """\
---
## Appendix — Sequential MNIST (commented out)

The cells below contain the sMNIST / psMNIST experiments from the original
notebook.  They are **not run by default** because T=784 steps through
online rules takes 60+ min on Colab CPU.

To run them, uncomment the cells and execute.  They use the `DeepRNN(n_layers=1)`
model on the 10-class MNIST classification problem.

Note: these cells require `torchvision` for MNIST download (auto-installed by pip)."""

# Collect original sMNIST source strings (cells 33-37)
smnist_cells_commented = []
for ci in range(33, 38):
    orig_src = "".join(orig[ci]["source"])
    # Prefix every line with '#' to comment it out
    commented = "\n".join("# " + line for line in orig_src.split("\n"))
    smnist_cells_commented.append(code(commented))


# ── Build new cell list ───────────────────────────────────────────────────────

new_cells = []

# Preamble (unchanged)
for i in range(5):                  # 0-4: title, setup MD, clone, pip, device
    new_cells.append(keep(i))

# Updated imports
new_cells.append(modify(5, IMPORTS_SRC))

# Master config (new)
new_cells.append(code(MASTER_CFG_SRC))

# Helpers: save_fig, bptt_grads_deep, cosine_keys, apply_grads_deep (cell 6)
new_cells.append(keep(6))

# Background & theory (cells 7-9)
for i in range(7, 10):
    new_cells.append(keep(i))

# Exp 1: single-layer e-prop (cells 10-14)
# Cell 11 used to set N_REC_SL etc. — these are now in master config, keep for
# backward compat but it just re-assigns from master config
new_cells.append(keep(10))          # Exp 1 MD
# Skip cell 11 (config now in master config block)
new_cells.append(keep(12))          # training helpers
new_cells.append(keep(13))          # training curves plot (already saves)
new_cells.append(keep(14))          # cosine+accuracy vs delay (already saves)

# Exp 2A: RTRL verification (cells 15-16)
new_cells.append(keep(15))
new_cells.append(keep(16))

# Exp 2B: 2-layer (cells 17-20)
# Cell 18 sets N_REC_2L etc. — keep it (doesn't conflict with master config keys)
for i in range(17, 21):
    new_cells.append(keep(i))

# Exp 3: depth sweep (cells 21-26)
new_cells.append(keep(21))          # Exp 3 MD
new_cells.append(code(DEPTHS_SWEEP_SRC))   # slimmed config cell (replaces cell 22)
# Cell 23: cosine computation + add JSON save
c23_src = "".join(orig[23]["source"])
new_cells.append(code(patch_cell_23_add_save(c23_src)))
for i in range(24, 27):             # cosine plot, learning curves, final plot
    new_cells.append(keep(i))

# Exp 4: leaky RNN alpha sweep (cells 27-31)
# Cell 28 sets N_REC_LK etc.
new_cells.append(keep(27))          # Exp 4 MD
new_cells.append(keep(28))          # config (N_REC_LK etc.)
# Cell 29: cosine vs alpha + add JSON save
c29_src = "".join(orig[29]["source"])
new_cells.append(code(patch_cell_29_add_save(c29_src)))
new_cells.append(keep(30))          # cosine plot (already saves)
# Cell 31: learning curves — replace with updated version using LeakyRNN module
new_cells.append(code(LEAKY_LC_SRC))

# ── NEW: Experiment 5 — Cue Accumulation + Leaky RNN ─────────────────────────
new_cells.append(md(EXP5_MD))
new_cells.append(code(EXP5_CONFIG))
new_cells.append(code(EXP5_COSINE_SRC))
new_cells.append(code(EXP5_COSINE_PLOT))
new_cells.append(code(EXP5_LC_SRC))

# Summary and Git push (cells 61-63)
new_cells.append(keep(61))
new_cells.append(keep(62))
new_cells.append(keep(63))

# ── Appendix: commented-out sMNIST ───────────────────────────────────────────
new_cells.append(md(SMNIST_MD))
for c in smnist_cells_commented:
    new_cells.append(c)

# ── Write ─────────────────────────────────────────────────────────────────────
nb["cells"] = new_cells
with open("deep_eprop_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Done. New notebook: {len(new_cells)} cells (was {len(orig)})")
