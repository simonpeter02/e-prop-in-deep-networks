#!/usr/bin/env python3
"""Cue-accumulation task figure (signal-trace style), slide-ready & editable.

Depicts the core of tasks/cue_accumulation.py: brief left/right cue pulses are
presented over time, evidence is accumulated across a long silent delay, and at a
single recall step the network reports which side received the majority of cues.

Emits cue_accumulation_task.svg (clean editable nodes) and .pptx (native shapes).
"""

from figure_canvas import (render_both, TXT, AXIS, LEFT, RIGHT, DELAY, DELAYE,
                           RECALL)

W, H = 960, 360

# ── layout (points; top-left origin, y down) ────────────────────────────────
X0      = 165          # traces start (after channel labels)
STRIDE  = 60           # spacing between cue pulses
PW      = 16           # pulse width
PH      = 42           # pulse height
Y_L     = 138          # Left-cue baseline
Y_R     = 214          # Right-cue baseline
Y_AXIS  = 258          # time axis

# cue sequence: L R L L R  → 3 left vs 2 right → LEFT wins
CUES    = ["L", "R", "L", "L", "R"]
CX      = [X0 + 20 + i * STRIDE for i in range(len(CUES))]
LAST    = CX[-1] + PW / 2

DELAY_X0 = LAST + 24
DELAY_X1 = DELAY_X0 + 210
RECALL_X = DELAY_X1 + 34
BOX_X    = RECALL_X + 30
BOX_W    = 196
BOX_Y    = 112
BOX_H    = 64


def pulse(c, cx, y, color):
    """A square pulse rising from baseline y at center cx."""
    x0, x1 = cx - PW / 2, cx + PW / 2
    c.polyline([(x0 - 6, y), (x0, y), (x0, y - PH), (x1, y - PH),
                (x1, y), (x1 + 6, y)], color=color, width=2.4)


def build(c):
    # title + one-line task summary
    c.text(W / 2, 30, "Cue accumulation", size=23, weight="bold")
    c.text(W / 2, 56,
           "Brief left / right pulses  →  accumulate across a silent delay  "
           "→  report the majority side",
           size=13, color="#666666", italic=True)

    # channel baselines
    c.line(X0, Y_L, DELAY_X1, Y_L, color=AXIS, width=1.2)
    c.line(X0, Y_R, DELAY_X1, Y_R, color=AXIS, width=1.2)
    # channel labels
    c.text(X0 - 14, Y_L, "Left cue", size=13, color=LEFT, anchor="end", weight="bold")
    c.text(X0 - 14, Y_R, "Right cue", size=13, color=RIGHT, anchor="end", weight="bold")

    # delay band (shaded silent region)
    c.rect(DELAY_X0, Y_L - PH - 6, DELAY_X1 - DELAY_X0, (Y_R + 12) - (Y_L - PH - 6),
           fill=DELAY, edge=DELAYE, width=1.0)
    c.text((DELAY_X0 + DELAY_X1) / 2, Y_L - PH + 12, "delay", size=13,
           color="#888888", weight="bold")
    c.text((DELAY_X0 + DELAY_X1) / 2, Y_L - PH + 30, "(silent gap)", size=11,
           color="#999999", italic=True)

    # cue pulses + per-cue "+1" tally marks (convey accumulation)
    nL = nR = 0
    for side, cx in zip(CUES, CX):
        if side == "L":
            pulse(c, cx, Y_L, LEFT)
            nL += 1
            c.text(cx, Y_L - PH - 11, "+1", size=10, color=LEFT)
        else:
            pulse(c, cx, Y_R, RIGHT)
            nR += 1
            c.text(cx, Y_R + 16, "+1", size=10, color=RIGHT)

    # time axis arrow
    c.arrow(X0, Y_AXIS, DELAY_X1 + 6, Y_AXIS, color=AXIS, width=1.4)
    c.text(DELAY_X1 + 12, Y_AXIS, "time", size=12, color="#888888", anchor="start")

    # recall marker
    c.line(RECALL_X, Y_L - PH - 6, RECALL_X, Y_AXIS, color=RECALL, width=1.6,
           dash="3,3")
    c.text(RECALL_X, Y_L - PH - 18, "recall", size=12, color=RECALL, weight="bold")

    # arrow recall -> decision box
    c.arrow(RECALL_X + 4, BOX_Y + BOX_H / 2, BOX_X - 4, BOX_Y + BOX_H / 2,
            color=AXIS, width=1.5)

    # decision box
    c.rect(BOX_X, BOX_Y, BOX_W, BOX_H, fill="#FFFFFF", edge=TXT, width=1.6, radius=10)
    c.text(BOX_X + BOX_W / 2, BOX_Y + 24, "Which side had", size=13, weight="bold")
    c.text(BOX_X + BOX_W / 2, BOX_Y + 43, "more cues?", size=13, weight="bold")

    # verdict
    c.text(BOX_X + BOX_W / 2, BOX_Y + BOX_H + 24,
           f"→  LEFT wins  ({nL} vs {nR})", size=14, color=RECALL, weight="bold")


if __name__ == "__main__":
    render_both(build, W, H, "cue_accumulation_task")
