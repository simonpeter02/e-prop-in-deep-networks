"""
Explanatory figures for the deep e-prop project.

Generates and saves the following diagrams as PDF and SVG:
  1. single_layer_rnn       — architecture of a 1-layer tanh RNN
  2. deep_rnn               — stacked L-layer RNN with feedforward connections
  3. eprop_trace_timeline   — eligibility trace update over time
  4. deep_eprop_propagation — cross-layer trace in a 3-layer network
  5. store_and_recall_task  — task structure (cue / delay / output windows)
  6. credit_assignment_paths — BPTT vs e-prop vs d=0 credit paths

Run:
    python -m figures.architecture_diagrams
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Arc

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches='tight')
    print(f"Saved {OUT}/{name}.pdf / .svg")
    plt.close(fig)


# ── Shared style ──────────────────────────────────────────────────────────────
NODE_KW  = dict(ha='center', va='center', fontsize=10, fontweight='bold')
ARROW_KW = dict(arrowstyle='->', color='#333333', lw=1.5,
                connectionstyle='arc3,rad=0.')
REC_ARROW_KW = dict(arrowstyle='->', color='tab:blue', lw=2.0,
                    connectionstyle='arc3,rad=0.4')


def node(ax, x, y, label, r=0.35, color='#DDEEFF', ec='#335577', **kw):
    circ = Circle((x, y), r, color=color, ec=ec, zorder=3, lw=1.5)
    ax.add_patch(circ)
    ax.text(x, y, label, zorder=4, **{**NODE_KW, **kw})
    return (x, y)


def arrow(ax, src, dst, color='#333333', rad=0.0, lw=1.5, **kw):
    ax.annotate('', xy=dst, xytext=src,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2, **kw)


def box(ax, cx, cy, w, h, label, color='#EEFFEE', ec='#336633', fs=9):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle='round,pad=0.05',
                          facecolor=color, edgecolor=ec, lw=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs, zorder=4)


# ── 1. Single-layer RNN architecture ─────────────────────────────────────────
def fig_single_layer():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_aspect('equal')

    # Nodes
    xin, yin = 0.5, 1.0
    xh,  yh  = 3.5, 1.0
    xo,  yo  = 6.5, 1.0

    node(ax, xin, yin, r'$x_t$',   color='#FFEECC', ec='#886600')
    node(ax, xh,  yh,  r'$h_t$',   color='#DDEEFF', ec='#335577')
    node(ax, xo,  yo,  r'$o_t$',   color='#EEFFEE', ec='#336633')

    # W_in arrow
    arrow(ax, (xin + 0.35, yin), (xh - 0.35, yh), color='#333333')
    ax.text((xin+xh)/2, yh+0.25, r'$W_{in}$', ha='center', fontsize=9, color='#333333')

    # W_out arrow
    arrow(ax, (xh + 0.35, yh), (xo - 0.35, yo), color='#336633')
    ax.text((xh+xo)/2, yh+0.25, r'$W_{out}$', ha='center', fontsize=9, color='#336633')

    # W_rec self-loop
    arc = Arc((xh, yh+0.55), 0.9, 0.7, angle=0, theta1=30, theta2=150, color='tab:blue', lw=2)
    ax.add_patch(arc)
    arrow(ax, (xh - 0.42, yh + 0.7), (xh - 0.38, yh + 0.36), color='tab:blue', rad=0.0)
    ax.text(xh, yh + 1.15, r'$W_{rec}$', ha='center', fontsize=9, color='tab:blue')

    # Time arrow
    ax.annotate('', xy=(xh + 0.35, yh - 0.5), xytext=(xh - 0.35, yh - 0.5),
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1.5))
    ax.text(xh, yh - 0.75, r'$h_{t-1} \to h_t$', ha='center', fontsize=8, color='tab:blue')

    # Activation note
    ax.text(xh, yh - 1.05, r'$h_t = \tanh(W_{rec}\,h_{t-1} + W_{in}\,x_t + b)$',
            ha='center', fontsize=9, style='italic')

    ax.set_title('Single-layer vanilla RNN', fontsize=12, fontweight='bold', pad=8)
    return fig


# ── 2. L-layer deep RNN ───────────────────────────────────────────────────────
def fig_deep_rnn(L=3):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 4.5)
    ax.axis('off')

    colors = ['#FFEECC', '#DDEEFF', '#EEE0FF', '#FFE0E0']
    ecs    = ['#886600', '#335577', '#553388', '#883333']

    x_left = 0.5
    x_gap  = 3.0
    y_base = 1.5

    xs = [x_left + l * x_gap for l in range(L)]
    x_out = xs[-1] + x_gap

    # Input
    node(ax, x_left - x_gap, y_base, r'$x_t$', color='#FFEECC', ec='#886600')

    layer_nodes = []
    for l in range(L):
        c   = colors[l % len(colors)]
        ec  = ecs[l % len(ecs)]
        lbl = rf'$h^{{{l+1}}}_t$'
        n   = node(ax, xs[l], y_base, lbl, color=c, ec=ec)
        layer_nodes.append(n)

    # Output
    node(ax, x_out, y_base, r'$o_t$', color='#EEFFEE', ec='#336633')

    # W_in
    arrow(ax, (x_left - x_gap + 0.35, y_base), (xs[0] - 0.35, y_base))
    ax.text((x_left - x_gap + xs[0])/2, y_base + 0.28, r'$W_{in}$', ha='center', fontsize=8)

    # W_ff feedforward
    for l in range(1, L):
        arrow(ax, (xs[l-1] + 0.35, y_base), (xs[l] - 0.35, y_base), color='#333333')
        ax.text((xs[l-1]+xs[l])/2, y_base + 0.28, rf'$W^{{{l+1}}}_{{\!ff}}$',
                ha='center', fontsize=8)

    # W_out
    arrow(ax, (xs[-1] + 0.35, y_base), (x_out - 0.35, y_base), color='#336633')
    ax.text((xs[-1]+x_out)/2, y_base + 0.28, r'$W_{out}$', ha='center', fontsize=8, color='#336633')

    # W_rec self-loops
    for l in range(L):
        c  = ecs[l % len(ecs)]
        xc = xs[l]
        arc = Arc((xc, y_base + 0.55), 0.9, 0.7, angle=0, theta1=30, theta2=150, color=c, lw=2)
        ax.add_patch(arc)
        arrow(ax, (xc - 0.42, y_base + 0.7), (xc - 0.38, y_base + 0.36), color=c)
        ax.text(xc, y_base + 1.25, rf'$W^{{{l+1}}}_{{\!rec}}$', ha='center', fontsize=8, color=c)

    # Layer labels
    for l in range(L):
        ax.text(xs[l], -0.5, f'Layer {l+1}', ha='center', fontsize=8, color='gray')

    ax.text(5.0, -0.9,
            r'$h^l_t = \tanh(W^l_{rec}\,h^l_{t-1} + W^l_{ff}\,h^{l-1}_t + b^l)$',
            ha='center', fontsize=9, style='italic')

    ax.set_title(f'{L}-layer deep RNN', fontsize=12, fontweight='bold', pad=8)
    return fig


# ── 3. Eligibility trace timeline ────────────────────────────────────────────
def fig_eprop_trace():
    T   = 6
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    t_vals = np.arange(T)
    psi    = np.array([0.8, 0.5, 0.9, 0.3, 0.7, 0.6])
    h_prev = np.array([0.2, 0.6, 0.1, 0.8, 0.4, 0.3])
    wii    = 0.15   # typical diagonal W_rec[i,i]

    # E-prop trace (with carry)
    eps_eprop = np.zeros(T)
    eps_d0    = np.zeros(T)
    for t in range(T):
        carry = psi[t] * wii
        eps_eprop[t] = psi[t] * h_prev[t] + (carry * eps_eprop[t-1] if t > 0 else 0)
        eps_d0[t]    = psi[t] * h_prev[t]

    ax = axes[0]
    ax.bar(t_vals - 0.18, eps_eprop, 0.35, label='e-prop (carry ≠ 0)', color='tab:blue', alpha=0.8)
    ax.bar(t_vals + 0.18, eps_d0,    0.35, label='d=0 (no carry)',      color='tab:orange', alpha=0.8)
    ax.set_ylabel(r'$\varepsilon_t[i,j]$', fontsize=10)
    ax.set_title('Eligibility trace magnitude — e-prop vs d=0', fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, color='gray', lw=0.5)

    ax2 = axes[1]
    ax2.stem(t_vals, psi, linefmt='C2-', markerfmt='C2o', basefmt=' ', label=r'$\psi_t$')
    ax2.stem(t_vals, h_prev, linefmt='C3-', markerfmt='C3s', basefmt=' ', label=r'$h_{t-1}$')
    ax2.set_xlabel('Timestep t', fontsize=10)
    ax2.set_ylabel('Activation', fontsize=10)
    ax2.set_title(r'Inputs to trace: $\psi_t = 1 - h_t^2$ and $h_{t-1}$', fontsize=11)
    ax2.legend(fontsize=9)

    # Annotation: trace formula
    axes[0].annotate(
        r'$\varepsilon_t[i,j] = \psi_t[i]\,h_{t-1}[j] + \psi_t[i]\,W_{rec}[i,i]\,\varepsilon_{t-1}[i,j]$'
        '\n(carry factor = diagonal of recurrent Jacobian)',
        xy=(0.5, 0.92), xycoords='axes fraction', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='goldenrod'))

    fig.tight_layout()
    return fig


# ── 4. Deep e-prop cross-layer propagation ────────────────────────────────────
def fig_deep_eprop_propagation():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    # 3 layers: x positions for t-1 and t
    t1_xs = [1.0, 3.5, 6.0]
    t2_xs = [2.5, 5.0, 7.5]
    ys    = [1.0, 2.5, 4.0]

    layer_names = ['Layer 1', 'Layer 2', 'Layer 3']
    colors_l    = ['#FFEECC', '#DDEEFF', '#EEE0FF']
    ecs_l       = ['#886600', '#335577', '#553388']

    for col, xs, t_lbl in [(t1_xs, ys, 't-1'), (t2_xs, ys, 't')]:
        for l, (x, y) in enumerate(zip(col, xs)):
            c  = colors_l[l]
            ec = ecs_l[l]
            lbl = rf'$h^{l+1}_{{\!\!{t_lbl}}}$'
            node(ax, x, y, lbl, r=0.32, color=c, ec=ec, fontsize=8)

    # Time axis labels
    ax.text(t1_xs[0], -0.2, 't-1', ha='center', fontsize=9, color='gray', style='italic')
    ax.text(t2_xs[0], -0.2, 't',   ha='center', fontsize=9, color='gray', style='italic')

    # Layer labels
    for l, y in enumerate(ys):
        ax.text(-0.3, y, layer_names[l], ha='right', va='center', fontsize=8, color='gray')

    # Recurrent arrows (t-1 → t same layer)
    for l, (x1, x2, y) in enumerate(zip(t1_xs, t2_xs, ys)):
        arrow(ax, (x1 + 0.32, y), (x2 - 0.32, y), color=ecs_l[l], lw=2)

    # Feedforward arrows (t, layer l → layer l+1)
    for l in range(len(ys)-1):
        arrow(ax, (t2_xs[l], ys[l]+0.32), (t2_xs[l+1], ys[l+1]-0.32), color='#444444')

    # Highlight: cross-layer trace ε^{3←1} path
    # Bold dashed arrow from h^1_t trace → h^3_t
    ax.annotate('', xy=(t2_xs[2], ys[2] - 0.32),
                xytext=(t2_xs[0], ys[0] + 0.32),
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=2.5,
                                connectionstyle='arc3,rad=-0.35',
                                linestyle='dashed'))
    ax.text(9.2, 2.5,
            r'$\varepsilon^{3\!\leftarrow\!1}_{t}$' + '\n(cross-layer\ntrace)',
            ha='center', fontsize=9, color='tab:red',
            bbox=dict(boxstyle='round', fc='#FFEEEE', ec='tab:red'))

    # Annotation boxes
    ax.text(4.5, 5.1,
            r'$\varepsilon^{3\leftarrow1}_t = c_3\,\varepsilon^{3\leftarrow1}_{t-1} + J^{32}_t\,\varepsilon^{2\leftarrow1}_t$',
            ha='center', fontsize=9, color='#222222',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='goldenrod'))

    ax.set_title('Deep e-prop: cross-layer eligibility trace in a 3-layer network',
                 fontsize=11, fontweight='bold')
    return fig


# ── 5. Store-and-recall task ──────────────────────────────────────────────────
def fig_store_and_recall():
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1.5, 1.5]})

    n_patterns = 4
    cue_dur    = 1
    delay      = 5
    out_dur    = 1
    T          = cue_dur + delay + out_dur

    t = np.arange(T)

    # Input: cue (one-hot pattern 2 of 4) + recall signal
    inp = np.zeros((T, n_patterns + 2))
    inp[0, 1] = 1.0   # cue: pattern index 1
    inp[cue_dur + delay, n_patterns] = 1.0  # recall signal
    inp[:, n_patterns + 1] = 1.0            # bias

    # Target
    tgt = np.zeros((T, n_patterns))
    tgt[cue_dur + delay, 1] = 1.0

    # Mask
    mask = np.zeros(T)
    mask[cue_dur + delay] = 1.0

    ax0 = axes[0]
    im = ax0.imshow(inp.T, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                    extent=[-0.5, T-0.5, -0.5, n_patterns+2-0.5])
    ax0.set_ylabel('Input channel', fontsize=9)
    ax0.set_yticks(range(n_patterns + 2))
    ax0.set_yticklabels([f'P{i}' for i in range(n_patterns)] + ['Recall', 'Bias'],
                         fontsize=7)
    fig.colorbar(im, ax=ax0, fraction=0.03)

    ax1 = axes[1]
    im2 = ax1.imshow(tgt.T, aspect='auto', cmap='Greens', vmin=0, vmax=1,
                     extent=[-0.5, T-0.5, -0.5, n_patterns-0.5])
    ax1.set_ylabel('Target channel', fontsize=9)
    ax1.set_yticks(range(n_patterns))
    ax1.set_yticklabels([f'P{i}' for i in range(n_patterns)], fontsize=7)
    fig.colorbar(im2, ax=ax1, fraction=0.03)

    ax2 = axes[2]
    ax2.bar(t, mask, color='tab:orange', alpha=0.7, label='Loss mask')
    ax2.set_ylabel('Loss\nmask', fontsize=9)
    ax2.set_xlabel('Timestep', fontsize=9)
    ax2.set_ylim(0, 1.3)
    ax2.set_yticks([0, 1])

    # Phase labels
    for ax in axes[:2]:
        ax.axvline(cue_dur - 0.5, color='gray', lw=1, linestyle='--')
        ax.axvline(cue_dur + delay - 0.5, color='gray', lw=1, linestyle='--')

    y_top = 1.15
    for ax in [axes[0]]:
        ax.text(cue_dur/2 - 0.5, n_patterns+1.5, 'CUE', ha='center', fontsize=8,
                color='steelblue', fontweight='bold')
        ax.text(cue_dur + delay/2 - 0.5, n_patterns+1.5, f'DELAY ({delay} steps)',
                ha='center', fontsize=8, color='gray')
        ax.text(cue_dur + delay, n_patterns+1.5, 'OUT', ha='center', fontsize=8,
                color='darkgreen', fontweight='bold')

    fig.suptitle('Store-and-recall (delayed copy) task', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ── 6. Credit assignment paths ────────────────────────────────────────────────
def fig_credit_assignment():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    titles  = ['BPTT', 'E-prop', 'd=0']
    methods = ['bptt', 'eprop', 'd0']

    T_show = 5
    for ax, title, method in zip(axes, titles, methods):
        ax.set_xlim(-0.5, T_show + 0.5)
        ax.set_ylim(-0.2, 2.5)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Draw hidden state nodes h_t for t = 0..T_show
        for t in range(T_show + 1):
            color = '#DDEEFF' if t < T_show else '#AADDCC'
            node(ax, t, 1.0, rf'$h_{t}$', r=0.28, color=color, ec='#335577', fontsize=7)

        # Forward arrows
        for t in range(T_show):
            arrow(ax, (t + 0.28, 1.0), (t + 0.72, 1.0), color='#888888', lw=1.0)

        # Output/loss node at final step
        node(ax, T_show, 2.0, r'$\mathcal{L}$', r=0.28, color='#FFEECC', ec='#AA6600', fontsize=9)
        arrow(ax, (T_show, 1.28), (T_show, 1.72), color='#AA6600', lw=1.5)

        # Credit paths
        if method == 'bptt':
            # Full backward pass: all connections
            for t in range(T_show - 1, -1, -1):
                arrow(ax, (t + 0.72, 0.8), (t + 0.28, 0.8),
                      color='tab:red', lw=2.0)
            ax.text(T_show/2, 0.2,
                    'Full back-propagation\nthrough time (O(T·n²))',
                    ha='center', fontsize=8, color='tab:red')

        elif method == 'eprop':
            # Forward trace from t=0, learning signal only at output
            for t in range(T_show):
                c = max(0.2, 0.9 - t * 0.12)
                ax.annotate('', xy=(t + 0.72, 1.25), xytext=(t + 0.28, 1.25),
                            arrowprops=dict(arrowstyle='->', color='tab:blue',
                                            lw=1.0 + t*0.2, alpha=c))
            # Diagonal carry highlight
            for t in range(1, T_show):
                ax.text(t, 1.48, '×W[i,i]', ha='center', fontsize=5.5, color='tab:blue')
            ax.text(T_show/2, 0.2,
                    'Eligibility trace (forward)\n× learning signal (diagonal carry)',
                    ha='center', fontsize=8, color='tab:blue')

        else:  # d=0
            # Single-step only, no carry
            for t in range(T_show):
                ax.annotate('', xy=(t + 0.5, 1.3), xytext=(t + 0.5, 1.5),
                            arrowprops=dict(arrowstyle='->', color='tab:orange',
                                            lw=1.5))
            ax.text(T_show/2, 0.2,
                    'No temporal carry\n(instantaneous only)',
                    ha='center', fontsize=8, color='tab:orange')

    fig.suptitle('Credit assignment mechanisms: BPTT vs e-prop vs d=0',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    figs = [
        (fig_single_layer(),           'single_layer_rnn'),
        (fig_deep_rnn(L=3),            'deep_rnn_3layer'),
        (fig_eprop_trace(),            'eprop_trace_timeline'),
        (fig_deep_eprop_propagation(), 'deep_eprop_propagation'),
        (fig_store_and_recall(),       'store_and_recall_task'),
        (fig_credit_assignment(),      'credit_assignment_paths'),
    ]
    for fig, name in figs:
        save(fig, name)

    print(f"\nAll {len(figs)} figures saved to {OUT}/")
