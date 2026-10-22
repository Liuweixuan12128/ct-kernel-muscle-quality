"""Compose the Scientific Reports figures from rendered parts as vector PDF, plus a 600-dpi PNG.
Called by figures/srep_figures.py with a layout JSON: {out, width_mm, gap_mm, figures: {name: [[pdf, clip_mm|null], ...]}}.
Clips are in mm from the top-left corner of the source page. Parts are stacked top to bottom: blank rows above and
below each clip are trimmed (to MARGIN_MM of the first and last inked row), and consecutive parts are gap_mm apart.
Needs PyMuPDF (r45 environment)."""
import json
import os
import sys

import fitz
import numpy as np

MM = 72 / 25.4
MARGIN_MM = 0.6
spec = json.load(open(sys.argv[1]))
W, GAP = spec["width_mm"], spec["gap_mm"]


def ink_rows(page, clip, dpi=200, thr=242):
    """Top and bottom of the inked rows inside clip (points)."""
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY, alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.stride)[:, :pix.w]
    rows = np.flatnonzero((a < thr).any(axis=1))
    if not len(rows):
        return clip.y0, clip.y1
    s = clip.height / pix.h
    return clip.y0 + rows[0] * s, clip.y0 + (rows[-1] + 1) * s


for name, parts in spec["figures"].items():
    placed, y = [], MARGIN_MM * MM
    for p, clip in parts:
        src = fitz.open(p)
        c = fitz.Rect(*(v * MM for v in clip)) if clip else src[0].rect
        t0, t1 = ink_rows(src[0], c)
        c = fitz.Rect(c.x0, max(c.y0, t0 - MARGIN_MM * MM), c.x1, min(c.y1, t1 + MARGIN_MM * MM))
        placed.append((src, c, y))
        y += c.height + GAP * MM
    H = y - GAP * MM + MARGIN_MM * MM
    Wf = min(W * MM, max(c.x1 for _, c, _ in placed))
    doc = fitz.open()
    page = doc.new_page(width=Wf, height=H)
    for src, c, top in placed:
        page.show_pdf_page(fitz.Rect(c.x0, top, c.x1, top + c.height), src, 0, clip=c)
    out = os.path.join(spec["out"], name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc.save(out + ".pdf", garbage=3, deflate=True)
    page.get_pixmap(dpi=600).save(out + ".png")
    print(f"{name}: {Wf / MM:.0f} x {H / MM:.1f} mm")
