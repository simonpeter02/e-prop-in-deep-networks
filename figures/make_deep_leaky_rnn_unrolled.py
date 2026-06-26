#!/usr/bin/env python3
"""Generate an editable SVG of a deep, leaky RNN unrolled in time.

Layers l_1, l_2, ..., l_L (arbitrary depth, shown with a vertical ellipsis),
each leaky-recurrent (horizontal arrows) and unrolled over time steps t = 1,2,3
(superscript). The defining leaky feature is the diagonal temporal carry: the
within-layer recurrence retains a fraction (1-alpha) of the previous state,
  h^l_t = (1-alpha) * h^l_{t-1} + alpha * tanh(W_rec^l h^l_{t-1} + ... ),
so each horizontal (recurrence) arrow is annotated with the leak factor 1-alpha.

Output: clean SVG with real <text>/<line>/<polygon> nodes so it can be dropped
into PowerPoint and "Convert to Shape" for full editing.
"""

# ---- palette / style -------------------------------------------------------
TXT   = "#3f3f3f"   # node glyph color
ARROW = "#7c7c7c"   # arrow / line color
FONT  = "'Helvetica Neue', Helvetica, Arial, sans-serif"

W, H = 780, 610

# ---- layout ----------------------------------------------------------------
COLS = {1: 140, 2: 390, 3: 640}          # time steps t = 1,2,3 (x positions)
ROW  = dict(x=560, l1=468, l2=380, dots=300, lL=220, y=130, loss=50)

PADV = 24   # vertical clearance around a glyph
PADH = 38   # horizontal clearance around a glyph

svg = []
def add(s): svg.append(s)

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="{FONT}">')
add('  <defs>')
add(f'    <marker id="ah" markerWidth="9" markerHeight="9" refX="7.5" refY="3" '
    f'orient="auto" markerUnits="userSpaceOnUse">')
add(f'      <path d="M0,0 L8,3 L0,6 Z" fill="{ARROW}"/>')
add('    </marker>')
add('  </defs>')

def vline(x, y0, y1):
    """vertical arrow from y0 (low/start) up to y1 (high/end, arrowhead)."""
    add(f'  <line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="{ARROW}" '
        f'stroke-width="1.6" marker-end="url(#ah)"/>')

def hline(x0, x1, y, label=None):
    add(f'  <line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{ARROW}" '
        f'stroke-width="1.6" marker-end="url(#ah)"/>')
    if label:  # leak factor above the recurrence arrow
        add(f'  <text x="{(x0+x1)/2}" y="{y-13}" font-size="16" fill="{TXT}" '
            f'text-anchor="middle" dominant-baseline="central">{label}</text>')

def node(x, y, content, size=30):
    add(f'  <text x="{x}" y="{y}" font-size="{size}" fill="{TXT}" '
        f'text-anchor="middle" dominant-baseline="central">{content}</text>')

def sub(n):  return f'<tspan dy="9" font-size="16">{n}</tspan>'
def sup(n):  return f'<tspan dy="-15" font-size="16">{n}</tspan>'
def subsup(layer, t):  # layer subscript + time superscript: ell_layer^(t)
    return f'&#8467;<tspan dy="9" font-size="16">{layer}</tspan>' \
           f'<tspan dy="-15" font-size="16">({t})</tspan>'

# ---- arrows ----------------------------------------------------------------
# vertical feed-forward arrows in every time column
for t, cx in COLS.items():
    vline(cx, ROW['x']  - PADV - 2, ROW['l1']   + PADV)        # x   -> l_1
    vline(cx, ROW['l1'] - PADV,     ROW['l2']   + PADV)        # l_1 -> l_2
    vline(cx, ROW['l2'] - PADV,     ROW['dots'] + 18)          # l_2 -> (deeper)
    vline(cx, ROW['dots'] - 18,     ROW['lL']   + PADV)        # (deeper) -> l_L
    # theta label on the input weights (x -> l_1), matching the source figure
    add(f'  <text x="{cx+15}" y="{(ROW["x"]+ROW["l1"])/2}" font-size="22" '
        f'fill="{TXT}" text-anchor="start" dominant-baseline="central">'
        f'&#952;</text>')

# horizontal leaky-recurrent arrows on each hidden layer row,
# annotated with the leak (state-retention) factor 1 - alpha
LEAK = '1&#8722;&#945;'   # "1-alpha"
for rkey in ('l1', 'l2', 'lL'):
    y = ROW[rkey]
    hline(COLS[1] + PADH, COLS[2] - PADH - 2, y, label=LEAK)
    hline(COLS[2] + PADH, COLS[3] - PADH - 2, y, label=LEAK)

# read-out + loss in the final time column
cx = COLS[3]
vline(cx, ROW['lL'] - PADV, ROW['y']    + PADV - 2)            # l_L -> y
vline(cx, ROW['y']  - PADV, ROW['loss'] + PADV)               # y   -> L (loss)

# ---- vertical ellipsis (arbitrary depth) -----------------------------------
for cx in COLS.values():
    for dy in (-8, 0, 8):
        add(f'  <circle cx="{cx}" cy="{ROW["dots"]+dy}" r="2.6" fill="{ARROW}"/>')

# ---- nodes -----------------------------------------------------------------
for t, cx in COLS.items():
    node(cx, ROW['x'],  f'x<tspan dy="-13" font-size="16">({t})</tspan>')
    node(cx, ROW['l1'], subsup(1, t))
    node(cx, ROW['l2'], subsup(2, t))
    node(cx, ROW['lL'], subsup('L', t))

node(COLS[3], ROW['y'],    'y')
node(COLS[3], ROW['loss'], '&#8466;')   # script L (loss)

add('</svg>')

out = "figures/deep_leaky_rnn_unrolled.svg"
with open(out, "w") as f:
    f.write("\n".join(svg) + "\n")
print("wrote", out)
