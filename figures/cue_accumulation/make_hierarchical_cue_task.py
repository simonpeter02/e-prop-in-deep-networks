#!/usr/bin/env python3
"""Hierarchical cue-accumulation task figure (signal-trace style), editable.

Depicts the core of tasks/hierarchical_cue.py and why it is a harder, more
meaningful test for *deep* RNNs than plain cue accumulation:

  1. CUE TYPE  — each cue is a temporal MOTIF (a rising or falling ramp) with the
     same mean and energy; identity lives in the temporal ORDER, not on a channel.
  2. CLASSIFY (depth) — a fast lower layer must learn to read each motif.
  3. COUNT (time)     — a slow top layer accumulates the per-cue calls across the
     silent delay and reports the majority.

Emits hierarchical_cue_task.svg (clean editable nodes) and .pptx (native shapes).
"""

from figure_canvas import (render_both, TXT, AXIS, LEFT, RIGHT, DELAY, DELAYE,
                           RECALL, L1FILL, L1EDGE, L2FILL, L2EDGE)

W, H = 980, 580
RISE = LEFT     # rising motif accent (cool blue)
FALL = RIGHT    # falling motif accent (warm orange)


def ramp(c, x0, x1, ybase, amp, rising, color, width=2.6):
    """A rising/falling ramp motif drawn on baseline ybase (y grows downward)."""
    if rising:                                   # -amp -> +amp  (down then up)
        pts = [(x0, ybase), (x0, ybase + amp), (x1, ybase - amp), (x1, ybase)]
    else:                                        # +amp -> -amp
        pts = [(x0, ybase), (x0, ybase - amp), (x1, ybase + amp), (x1, ybase)]
    c.polyline(pts, color=color, width=width)


def build(c):
    # ── title ───────────────────────────────────────────────────────────────
    c.text(W / 2, 30, "Hierarchical cue accumulation", size=23, weight="bold")
    c.text(W / 2, 55,
           "Classify each temporal motif (needs depth), then count across the "
           "delay (needs time)", size=13, color="#666666", italic=True)

    # ── SECTION 1: cue dictionary — make the cue TYPE unmistakable ───────────
    dx, dy, dw, dh = 70, 80, 470, 172
    c.rect(dx, dy, dw, dh, fill="#FBFBFD", edge="#cfcfcf", width=1.2, radius=10)
    c.text(dx + dw / 2, dy + 24, "Each cue is a temporal motif", size=15, weight="bold")

    by = dy + 80                       # mini-plot baseline
    amp = 23
    # rising mini
    rx = dx + 120
    c.line(rx - 42, by, rx + 42, by, color=AXIS, width=1.0, dash="3,3")
    ramp(c, rx - 32, rx + 32, by, amp, True, RISE)
    c.text(rx, by + 44, "rising  ↗", size=13, color=RISE, weight="bold")
    # falling mini
    fx = dx + 330
    c.line(fx - 42, by, fx + 42, by, color=AXIS, width=1.0, dash="3,3")
    ramp(c, fx - 32, fx + 32, by, amp, False, FALL)
    c.text(fx, by + 44, "falling  ↘", size=13, color=FALL, weight="bold")
    c.text(dx + dw / 2, dy + dh - 16,
           "same mean (≈0) & energy — differ only in temporal order (slope sign)",
           size=11, color="#777777", italic=True)

    # ── SECTION 1b: contrast with plain cue accumulation ─────────────────────
    nx, nw = 580, 330
    c.text(nx, dy + 18, "Why this is harder for deep RNNs", size=13, weight="bold",
           anchor="start")
    notes = [
        "The side is NOT handed over on a",
        "dedicated channel (as in plain cue",
        "accumulation). It is hidden in the",
        "temporal shape — so a single layer,",
        "or a random/un-learned projection,",
        "cannot simply read it off.",
    ]
    for i, ln in enumerate(notes):
        c.text(nx, dy + 44 + i * 19, ln, size=12, color="#555555", anchor="start")

    # ── SECTION 2: feature-channel trace (the actual stimulus over time) ─────
    yF = 300
    x0 = 175
    motifs = [("R", True), ("F", False), ("R", True)]    # ↗ ↘ ↗ → rising majority
    cw, stride = 54, 118
    starts = [x0 + 30 + i * stride for i in range(len(motifs))]
    last = starts[-1] + cw
    delay_x0, delay_x1 = last + 22, last + 22 + 175
    recall_x = delay_x1 + 30

    c.text(x0 - 12, yF, "Feature", size=12, color=TXT, anchor="end", weight="bold")
    c.text(x0 - 12, yF + 15, "channel", size=12, color=TXT, anchor="end", weight="bold")
    c.line(x0, yF, delay_x1, yF, color=AXIS, width=1.1)        # baseline

    for (_, rising), s in zip(motifs, starts):
        ramp(c, s, s + cw, yF, 26, rising, RISE if rising else FALL)

    # delay band
    c.rect(delay_x0, yF - 34, delay_x1 - delay_x0, 68, fill=DELAY, edge=DELAYE,
           width=1.0)
    c.text((delay_x0 + delay_x1) / 2, yF - 14, "delay", size=12, color="#888888",
           weight="bold")
    # recall
    c.line(recall_x, yF - 40, recall_x, yF + 40, color=RECALL, width=1.6, dash="3,3")
    c.text(recall_x, yF - 50, "recall", size=12, color=RECALL, weight="bold")
    # time axis
    c.arrow(x0, yF + 52, delay_x1 + 6, yF + 52, color=AXIS, width=1.3)
    c.text(delay_x1 + 12, yF + 52, "time", size=12, color="#888888", anchor="start")

    # ── SECTION 3: two-layer processing pipeline ─────────────────────────────
    py, ph = 420, 86
    l1x, l2x, dcx = 70, 380, 700
    bw = 250 - 0
    bw1, bw2, bwd = 270, 270, 210

    # arrow from the feature trace down into Layer 1
    c.arrow((l1x + bw1 / 2), yF + 60, (l1x + bw1 / 2), py - 4, color=AXIS, width=1.4)

    # Layer 1 — fast extractor
    c.rect(l1x, py, bw1, ph, fill=L1FILL, edge=L1EDGE, width=1.6, radius=10)
    c.text(l1x + bw1 / 2, py + 22, "Layer 1 — fast  (α ≈ 0.7)", size=13.5, weight="bold")
    c.text(l1x + bw1 / 2, py + 44, "classify each motif:  ↗ / ↘", size=12.5)
    c.text(l1x + bw1 / 2, py + 65, "temporal feature extractor", size=11,
           color="#666666", italic=True)

    # Layer 2 — slow integrator
    c.rect(l2x, py, bw2, ph, fill=L2FILL, edge=L2EDGE, width=1.6, radius=10)
    c.text(l2x + bw2 / 2, py + 22, "Layer 2 — slow  (α ≈ 0.1)", size=13.5, weight="bold")
    c.text(l2x + bw2 / 2, py + 44, "count + hold through delay", size=12.5)
    c.text(l2x + bw2 / 2, py + 65, "integrator", size=11, color="#666666", italic=True)

    # Decide
    c.rect(dcx, py, bwd, ph, fill="#FFFFFF", edge=TXT, width=1.6, radius=10)
    c.text(dcx + bwd / 2, py + 26, "Decide", size=13.5, weight="bold")
    c.text(dcx + bwd / 2, py + 52, "rising-majority", size=13, color=RECALL,
           weight="bold")
    c.text(dcx + bwd / 2, py + 70, "(2 vs 1)", size=11, color=RECALL)

    # connecting arrows
    c.arrow(l1x + bw1 + 4, py + ph / 2, l2x - 4, py + ph / 2, color=AXIS, width=1.6)
    c.arrow(l2x + bw2 + 4, py + ph / 2, dcx - 4, py + ph / 2, color=AXIS, width=1.6)

    # closing caption — the deep-credit point
    c.text(W / 2, py + ph + 34,
           "Credit for an early cue must travel UP (depth) and FORWARD (time) — "
           "the decisive test for deep e-prop.",
           size=12.5, color="#555555", italic=True)


if __name__ == "__main__":
    render_both(build, W, H, "hierarchical_cue_task")
