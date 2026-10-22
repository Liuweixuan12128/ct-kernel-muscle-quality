# ER revision: copy of figures/r1005_figS1_c2c.py with labels aligned to the ER terminology. Output: figures/er_supp/
"""Extended Data Fig. 2 | An independent deep-learning tissue model (Comp2Comp multilevel nnU-Net) at T12.

a  Example T12 sections: our segmentation, Comp2Comp on the native image, Comp2Comp on the noise-harmonised image;
   a typical soft-kernel CT and a thin sharp-kernel CT.
b  Dice overlap with our masks by tissue, input image and reconstruction kernel (median and IQR).
c  Agreement with our measurements, ICC(2,1) with 95% bootstrap CI, native vs harmonised input.
d  Scan-rescan ICC on the 49 pairs: our pipeline vs Comp2Comp (native, harmonised input).
e  Overall survival, aged >= 65 y: HR per SD from the same Cox model, our measures vs Comp2Comp measures.
Inputs: results/c2c/*.csv (stats/c2c_agreement.py), results/public_pooled/cox_per_sd*.csv, data/c2c_t12/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
import style  # noqa: E402
style.OUT = os.path.join(style.ROOT, "figures", "er_supp")
from fig_cohort import forest  # noqa: E402
import c2c_t12  # noqa: E402

FIG = "FigureS1"
W, H = 183.0, 128.0
R = lambda *p: os.path.join(style.ROOT, *p)
EXAMPLES = [("tcga_cptac", "TCGA-38-4629", "Soft kernel, 5 mm"), ("public_rg", "AMC-038", "Sharp kernel, thin slices")]
TCOL = {"muscle": style.TISSUE["muscle"], "imat": style.TISSUE["imat"], "sat": style.TISSUE["sat"], "vat": style.TISSUE["ift"]}
MEAS = ["Muscle area (cm²)", "Muscle radiodensity (HU)", "Intramuscular fat (%)", "Subcutaneous fat (cm²)",
        "Visceral fat (cm²)"]
MKEY = {"Muscle area (cm²)": "muscle", "Muscle radiodensity (HU)": "muscle", "Intramuscular fat (%)": "imat",
        "Subcutaneous fat (cm²)": "sat", "Visceral fat (cm²)": "vat"}
SHORT = {"Muscle area (cm²)": "Muscle area", "Muscle radiodensity (HU)": "Muscle radiodensity",
         "Intramuscular fat (%)": "Intramuscular fat %", "Subcutaneous fat (cm²)": "Subcutaneous fat",
         "Visceral fat (cm²)": "Visceral fat"}
GREY = style.MUTED


def ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def rgb_overlay(img, masks, alpha=0.62):
    g = np.clip((img + 160) / 400, 0, 1)
    disp = lambda a: np.fliplr(np.rot90(a))
    rgb = np.repeat(disp(g)[..., None], 3, 2)
    for k in ["sat", "vat", "muscle", "imat"]:
        m = disp(masks[k].astype(bool))
        c = np.array([int(TCOL[k][i:i + 2], 16) / 255 for i in (1, 3, 5)])
        a = min(alpha, 0.42) if k in ("sat", "vat", "imat") else alpha   # fat drawn translucent
        rgb[m] = (1 - a) * rgb[m] + a * c
    return rgb, disp(masks["body"].astype(bool)) if "body" in masks else None


def panel_a(fig):
    axes = []
    rows = []
    for r, (coh, cid, lab) in enumerate(EXAMPLES):
        wd = R("data", "c2c_t12", coh)
        ours = np.load(os.path.join(wd, f"{cid}_ours.npz"))
        views = [("Ours", ours["harm"], {"muscle": ours["muscle"], "imat": ours["imat"], "sat": ours["sat"],
                                         "vat": ours["ift"], "body": ours["body"]})]
        for v, t in [("native", "Comp2Comp, native"), ("harm", "Comp2Comp, fixed filter")]:
            row, ch, _ = c2c_t12.metrics_one(coh, cid, wd, v)
            ch = dict(ch, body=ours["body"])
            views.append((t, ours["raw"] if v == "native" else ours["harm"], ch))
            rows.append(dict(case=cid, input=v, **{k: round(val, 3) for k, val in row.items() if k.startswith("dice")}))
        body = np.fliplr(np.rot90(ours["body"].astype(bool)))
        ys, xs = np.nonzero(body)
        pad = 6
        cr = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        for c, (t, img, masks) in enumerate(views):
            ax = ax_mm(fig, 4 + c * 31.5, 92 - r * 29, 30, 26)
            rgb, _ = rgb_overlay(img.astype(np.float32), masks)
            ax.imshow(rgb[cr], interpolation="lanczos", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(t, fontsize=6, pad=2)
            if c == 0:
                ax.text(-0.04, 0.5, lab, transform=ax.transAxes, rotation=90, fontsize=5.5, ha="right", va="center")
            axes.append(ax)
    return axes, pd.DataFrame(rows)


def panel_b(ax):
    dc = pd.read_csv(R("results", "c2c", "dice.csv"))
    tissues = [("muscle", "Muscle"), ("sat", "Subcutaneous fat"), ("vat", "Visceral fat"), ("imat", "Intramuscular fat")]
    y0 = 0
    ticks, labs = [], []
    for k, lab in tissues:
        for j, kt in enumerate(["soft", "sharp"]):
            yy = y0 - j * 0.9
            for inp, off, fc in [("native", 0.18, "white"), ("harm", -0.18, TCOL[k])]:
                r = dc[(dc.tissue == k) & (dc.kernel == kt) & (dc.input == inp)].iloc[0]
                ax.plot([r.q1, r.q3], [yy + off] * 2, color=style.INK2, lw=0.7)
                ax.scatter([r["median"]], [yy + off], s=13, facecolor=fc, edgecolor=TCOL[k], lw=0.8, zorder=3)
            ticks.append(yy); labs.append(f"{lab}, {kt}")
        y0 -= 2.3
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=5.3)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Dice with our masks (median, IQR)", fontsize=6)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.5, mfc="white", mec=style.INK, label="Native input"),
                       Line2D([0], [0], marker="o", ls="", ms=3.5, mfc=style.INK2, mec=style.INK2, label="Fixed-filter input")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.3, handletextpad=0.2, columnspacing=1.0)
    return dc


def icc_dots(ax, df, series, title):
    y = {m: i for i, m in enumerate(MEAS[::-1])}
    ax.axvspan(0.9, 1.0, color=style.WASH, lw=0, zorder=0)
    for key, off, fc, ec in series:
        s = df[key(df)].set_index("measure")
        for m in MEAS:
            r = s.loc[m]
            yy = y[m] + off
            ax.plot([r.icc_lo, r.icc_hi], [yy, yy], color=style.INK2 if fc != "white" else GREY, lw=0.6)
            ax.scatter([r.icc], [yy], s=12, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
    ax.set_yticks(list(y.values())); ax.set_yticklabels([SHORT[m] for m in y], fontsize=5.5)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
    ax.set_title(title, fontsize=6, pad=2)


def main():
    style.setup()
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    a_axes, sa = panel_a(fig)
    b = ax_mm(fig, 130, 66, 50, 48)
    sb = panel_b(b)
    ag = pd.read_csv(R("results", "c2c", "agreement.csv"))
    ag = ag[ag.kernel == "all"]
    c = ax_mm(fig, 28, 9, 30, 34)
    icc_dots(c, ag, [(lambda d: d["input"] == "native", 0.17, "white", style.INK),
                     (lambda d: d["input"] == "harm", -0.17, style.INK2, style.INK2)], "Agreement with ours")
    c.set_xlabel("ICC (95% CI)", fontsize=6)
    rep = pd.read_csv(R("results", "c2c", "reproducibility.csv"))
    d = ax_mm(fig, 80, 9, 30, 34)
    icc_dots(d, rep, [(lambda x: x.model == "ours", 0.26, style.INK, style.INK),
                      (lambda x: (x.model == "Comp2Comp") & (x["input"] == "native"), 0.0, "white", style.INK),
                      (lambda x: (x.model == "Comp2Comp") & (x["input"] == "harm"), -0.26, style.INK2, style.INK2)],
             "Repeat CTs (49 pairs)")
    d.set_yticklabels([])
    d.set_xlabel("ICC (95% CI)", fontsize=6)
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.5, mfc=style.INK, mec=style.INK, label="Ours"),
                        Line2D([0], [0], marker="o", ls="", ms=3.5, mfc="white", mec=style.INK,
                               label="Comp2Comp, native input"),
                        Line2D([0], [0], marker="o", ls="", ms=3.5, mfc=style.INK2, mec=style.INK2,
                               label="Comp2Comp, fixed-filter input")],
               loc="center", bbox_to_anchor=(70 / W, 50 / H), ncol=3, fontsize=5.4, handletextpad=0.2,
               columnspacing=1.4)
    # e: survival
    ours = pd.read_csv(R("results", "public_pooled", "cox_per_sd.csv"))
    ours = ours[ours.population == "age >= 65"].set_index("exposure")
    cc = pd.read_csv(R("results", "public_pooled", "cox_per_sd_c2c.csv"))
    cc = cc[cc.population == "age >= 65"].set_index("exposure")
    rows = [("Muscle radiodensity, ours (T5–T12)", ours.loc["smd3d"], style.INK),
            ("Muscle radiodensity, Comp2Comp native", cc.loc["c2c_muscle_hu"], "white"),
            ("Muscle radiodensity, Comp2Comp fixed filter", cc.loc["c2c_muscle_hu_h"], style.INK2),
            ("Intramuscular fat %, ours", ours.loc["imat_pct"], style.INK),
            ("Intramuscular fat %, Comp2Comp native", cc.loc["c2c_imat_pct"], "white"),
            ("Intramuscular fat %, Comp2Comp fixed filter", cc.loc["c2c_imat_pct_h"], style.INK2)]
    e = ax_mm(fig, 152, 9, 28, 34)
    yv = np.array([5.6, 4.6, 3.6, 2.0, 1.0, 0.0])
    e.axvline(1, color=style.HAIR, lw=0.6, zorder=0)
    src_e = []
    for yi, (lab, r, fc) in zip(yv, rows):
        e.plot([r.lo, r.hi], [yi, yi], color=style.INK2, lw=0.7)
        e.scatter([r.HR], [yi], s=14, marker="s", facecolor=fc, edgecolor=style.INK, lw=0.7, zorder=3)
        src_e.append(dict(row=lab, HR=r.HR, lo=r.lo, hi=r.hi, p=r.p))
    e.set_xscale("log"); e.set_xlim(0.7, 2.2); e.set_xticks([0.75, 1, 1.5, 2])
    e.xaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:g}")); e.xaxis.set_minor_formatter(plt.NullFormatter())
    e.set_yticks(yv); e.set_yticklabels([r[0] for r in rows], fontsize=5.2)
    e.tick_params(axis="y", length=0); e.spines["left"].set_visible(False)
    e.set_ylim(-0.6, 6.2)
    e.set_xlabel("HR per SD (95% CI)", fontsize=6)
    e.set_title("Pooled, aged ≥65 y (195/66)\noriginal covariate set", fontsize=5.6, pad=3)
    for t, x, y_ in [("a", 1, 127), ("b", 110, 127), ("c", 1, 55), ("d", 62, 55), ("e", 112, 55)]:
        style.panel_label(None, t, x / W, y_ / H, fig=fig)
    for df, p in [(sa, "a_dice_examples"), (sb, "b_dice"), (ag, "c_agreement"), (rep, "d_reproducibility"),
                  (pd.DataFrame(src_e), "e_survival")]:
        style.source_data(df, FIG, p)
    print(style.save(fig, FIG))


if __name__ == "__main__":
    main()
