"""Figures for the European Radiology version. Data: results/er/*, results/r1005/*. Output: figures/er/.

Figure1  study flow and the three processing strategies
Figure2  kernel effect in reconstructions of one acquisition (one shared mask)
Figure3  noise accounts for the kernel effect (paired noise-convolution decomposition)
Figure4  filter strength: agreement versus scale shift and small-structure loss; mask-boundary voxels
Figure5  survival associations under each processing strategy

Usage: env/bin/python figures/er_figures.py [1 2 3 4 5]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
import style  # noqa: E402

style.OUT = os.path.join(style.ROOT, "figures", "er")
R = lambda *p: os.path.join(style.ROOT, *p)
ER = lambda f: R("results", "er", f)
W = 183.0
C_NAT, C_FIX, C_ADA = style.MUTED, style.INK, style.SERIES[0]
KCOL = {"sharp": style.SERIES[1], "soft": style.SERIES[0]}
C_SOFTSHIFT = style.SERIES[1]  # Figure 4a soft-kernel shift; the Sci Rep figures use the soft-kernel colour
PHANTOM_COL = {2: style.SERIES[1], 3: style.INK2, 5: style.INK}  # Figure 4b inclusion diameters
C_ERO = style.SERIES[2]  # Figure 5 eroded mask
REPEAT_NOTE = "49 separate repeat CTs (supplement)"  # where the repeat examinations are shown (Sci Rep: Fig. 6)
MEAS = [("imat_pct", "Intramuscular fat", "percentage points"), ("smd_window_hu", "Muscle-window density", "HU"),
        ("smd_all_hu", "Whole-mask density", "HU"), ("sma_cm2", "Muscle area", "cm²")]
EST = [("native", "Native", C_NAT), ("fixed matching", "Both filtered\n(fixed)", C_FIX),
       ("sharp-only, pairwise target", "Sharp only\n(to soft noise)", C_ADA)]


def ax_mm(fig, H, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def label(fig, H, x, y, t):
    style.panel_label(None, t, x / W, y / H, fig=fig)


def plain(ax, axis, ticks):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: style.fnum(v, "{:g}")))
    a.set_minor_locator(NullLocator())


def src(df, fig, p):
    style.source_data(df, fig, p)


# =========================================================== Figure 1: flow
def box(ax, x, y, w, h, title, lines, fc="#f4f3ee", bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="none"))
    ax.text(x + 1.6, y + h - 1.6, title, fontsize=5.8, fontweight="bold", va="top", color=style.INK)
    for i, l in enumerate(lines):
        ax.text(x + 1.6, y + h - 5.4 - i * 3.3, l, fontsize=5.2, va="top", color=style.INK2 if i else style.INK)


def arrow(ax, x0, y0, x1, y1, txt=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=style.INK2,
                                                                   mutation_scale=6))
    if txt:
        ax.text(x0 + 1.2, (y0 + y1) / 2, txt, fontsize=4.8, va="center", color=style.INK2)


def fig1():
    style.setup()
    H = 112.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    pw = pd.read_csv(ER("pairs_perpair.csv"))
    nl = pw[pw.dataset == "NLST"]
    vend = nl.vendor.value_counts()
    # column A: NLST pairs
    ax.text(3, H - 3, "a  Same-acquisition reconstruction pairs", fontsize=7, fontweight="bold", va="top")
    box(ax, 3, 82, 56, 18, "NLST, last screening CT before diagnosis", ["616 analysed CTs (soft kernel preferred)", "Same study searched for the other kernel class"])
    arrow(ax, 31, 82, 31, 74.5, "351 with a kernel-discordant partner")
    meta = pd.read_csv(ER("pairs_meta.csv"))
    n_meas = int((meta.dataset == "NLST").sum())
    box(ax, 3, 57, 56, 17, f"{n_meas} pairs measurable at T12", [f"{351 - n_meas} excluded: no complete T12 measurement in the",
                                                                   "segmented series (23) or T12 outside the partner (4)",
                                                                   "Mask mapped to the partner by world coordinates"])
    arrow(ax, 31, 57, 31, 49.5, f"{n_meas - len(nl)} excluded: <95% of the mask area inside\nthe partner's reconstructed field of view")
    box(ax, 3, 28, 56, 21.5, f"{len(nl)} NLST pairs analysed", [
        f"GE {vend.get('GE', 0)} · Siemens {vend.get('SIEMENS', 0)} · Toshiba {vend.get('TOSHIBA', 0)} · Philips {vend.get('PHILIPS', 0)}",
        f"{int(nl.equal_thickness.sum())} with equal slice thickness", "Same frame of reference, acquisition",
        "number, tube voltage and current", "(Philips: acquisition number not recorded)"], fc="#e7eef8")
    box(ax, 63, 57, 52, 43, "NSCLC-Radiogenomics staging CT", [
        "202 measured staging CTs", "26 alternative reconstructions", "   (DICOM-verified, same acquisition)", "21 measurable at T12",
        "14 kernel-discordant pairs", "", REPEAT_NOTE])
    box(ax, 63, 28, 52, 24, "14 RG pairs analysed", ["Siemens 11 · GE 3", "4 with equal slice thickness", "Slice timing matched (median Δt ≤0.36 s)"], fc="#e7eef8")
    arrow(ax, 89, 57, 89, 52)
    # column B: clinical
    ax.text(121, H - 3, "b  Survival analysis", fontsize=7, fontweight="bold", va="top")
    box(ax, 121, 82, 59, 18, "Four surgical cohorts: 312 measured", ["Excluded: 1 missing age, sex or follow-up;",
                                                                     "37 incomplete T5–T12 coverage; 274 patients remain",
                                                                     "NSCLC-Radiogenomics 176 · TCGA/CPTAC 98"])
    arrow(ax, 150, 82, 150, 74.5, "RG, CT 0–180 d before surgery (173);\n13 excluded by the date check")
    box(ax, 121, 57, 59, 17, "160 NSCLC-Radiogenomics patients", ["all ages; 54 deaths"])
    arrow(ax, 150, 57, 150, 49.5, "aged ≥65 years")
    box(ax, 121, 28, 59, 21.5, "121 patients, 44 deaths (primary)", ["Sensitivity: 130 interval-only", "Pooled older cohort: 195, 66 deaths", "Measurements under each processing", "strategy below"], fc="#e7eef8")
    # processing strategies
    ax.text(3, 22, "c  Processing strategies (anatomical masks fixed)", fontsize=7, fontweight="bold", va="top")
    strat = [("Native", "Thresholds applied to the images as reconstructed", C_NAT),
             ("Both filtered (fixed)", "5-mm slab average + in-plane Gaussian σ 0.75 mm,\napplied to every CT", C_FIX),
             ("Sharp only (adaptive)", "Soft-kernel CT unchanged; sharp-kernel CT smoothed\nuntil its muscle noise equals the soft-kernel level", C_ADA)]
    for i, (t, d, c) in enumerate(strat):
        x = 3 + i * 60
        ax.add_patch(FancyBboxPatch((x, 3), 57, 15, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=c, lw=0.9))
        ax.text(x + 1.6, 16.4, t, fontsize=5.8, fontweight="bold", va="top", color=c if c != C_NAT else style.INK)
        ax.text(x + 1.6, 12.6, d, fontsize=5.0, va="top", color=style.INK2, linespacing=1.3)
    src(nl[["pid", "vendor", "equal_thickness"]], "Figure1", "a_nlst_pairs")
    return style.save(fig, "Figure1")


# =========================================================== Figure 2: paired kernel effect
def example_images(axs, pid="120289"):
    from bodycomp_metrics import Case, vertebra_info
    import fig1_design_pipeline as F
    from scipy import ndimage as ndi
    sel = pd.read_csv(R("data", "nlst", "nlst_selected_series_last.csv"))
    rnd = int(sel[sel.pid.astype(str) == pid]["round"].iloc[0])
    name = f"{pid}_T{rnd}"
    ca = Case(R("data", "nlst", "nifti", f"{name}.nii.gz"), R("data", "seg", "nlst", name))
    cb = Case(R("data", "nlst", "pairs", "nifti", f"{pid}_alt.nii.gz"), R("data", "seg", "nlst", name))
    z = int(round(vertebra_info(ca.segs["total"], "T12")[0]))
    pw = pd.read_csv(ER("pairs_perpair.csv"))
    r = pw[(pw.dataset == "NLST") & (pw.pid.astype(str) == pid)].iloc[0]
    s = r.sigma_pair
    mus = F.disp(ca.muscle[:, :, z])
    imgs = [(ca.raw[:, :, z], f"Soft ({r.kernel_soft}), native"), (cb.raw[:, :, z], f"Sharp ({r.kernel_sharp}), native"),
            (ndi.gaussian_filter(cb.raw[:, :, z], sigma=s / ca.zooms[0]), f"Sharp, smoothed to soft noise")]
    esm = F.disp(ca.esm[:, :, z])
    ys, xs = np.nonzero(esm)
    cy, cx = int(ys.mean()), int(xs.mean())
    hh, hw = 70, 110                                   # zoom on the paraspinal muscles
    cr = (slice(max(cy - hh, 0), cy + hh), slice(max(cx - hw, 0), cx + hw))
    rows = []
    for ax, (im, t) in zip(axs, imgs):
        d = F.disp(im)
        fat = (d >= -190) & (d <= -30) & mus
        rgb = F.paint(F.window(d), [(mus & ~fat, style.TISSUE["muscle"])], alpha=0.35)
        rgb[fat] = 0.1 * rgb[fat] + 0.9 * F.hex_rgb("#ffb000")
        ax.imshow(rgb[cr], interpolation="nearest"); F.image_ax(ax)
        imat = 100 * fat.sum() / (fat.sum() + ((d >= -29) & (d <= 150) & mus).sum())
        ax.set_title(t, fontsize=5.6, pad=2)
        ax.text(0.03, 0.04, f"Intramuscular fat {imat:.1f}%", transform=ax.transAxes, fontsize=5.2, color="white",
                bbox=dict(boxstyle="square,pad=0.2", fc="black", ec="none", alpha=0.7))
        rows.append(dict(image=t, imat_pct=imat))
    return pd.DataFrame(rows)


def fig2():
    style.setup()
    H = 128.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ia = [ax_mm(fig, H, 3 + i * 60, 88, 58, 32) for i in range(3)]
    sa = example_images(ia)
    fig.text(3 / W, 85 / H, "One NLST acquisition, two reconstructions, one mask: red, muscle-window voxels; orange, fat-window voxels",
             fontsize=5.2, color=style.INK2)
    # legend-like key
    S = pd.read_csv(ER("pairs_summary.csv"))
    P = pd.read_csv(ER("pairs_perpair.csv"))
    rows = []
    for i, (m, ttl, unit) in enumerate(MEAS):
        ax = ax_mm(fig, H, 14 + i * 43, 14, 33, 58)
        for ds, mk, off in [("NLST", "o", -0.12), ("RG", "D", 0.12)]:
            d0 = P[P.dataset == ds]
            vals = {"native": d0[f"{m}_sharp_native"] - d0[f"{m}_soft_native"],
                    "fixed matching": d0[f"{m}_sharp_fixed"] - d0[f"{m}_soft_fixed"],
                    "sharp-only, pairwise target": d0[f"{m}_sharp_adapt_pair"] - d0[f"{m}_soft_native"]}
            for j, (k, lab, col) in enumerate(EST):
                v = vals[k].dropna()
                if ds == "NLST":
                    parts = ax.violinplot(v, positions=[j + off], widths=0.42, showextrema=False)
                    for b in parts["bodies"]:
                        b.set_facecolor(col); b.set_alpha(0.25); b.set_edgecolor("none")
                else:
                    ax.scatter(np.full(len(v), j + off) + np.random.default_rng(1).uniform(-0.04, 0.04, len(v)), v, s=3,
                               color=col, alpha=0.6, lw=0)
                s = S[(S.dataset == ds) & (S.stratum == "all") & (S.measure == m) & (S.estimand == k)].iloc[0]
                ax.errorbar(j + off, s.hl, yerr=[[s.hl - s.lo], [s.hi - s.hl]], fmt=mk, ms=3, color=col,
                            mfc=col if ds == "NLST" else "white", capsize=0, lw=0.9, zorder=4)
                rows.append(dict(measure=m, dataset=ds, estimand=k, n=int(s.n), hl=s.hl, lo=s.lo, hi=s.hi))
        ax.axhline(0, color=style.INK2, lw=0.5, zorder=0)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in EST], fontsize=4.8)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(ttl, fontsize=6, pad=3)
        ax.set_ylabel(f"Sharp − soft ({unit})", fontsize=5.6, labelpad=1)
        allv = pd.concat([P[f"{m}_sharp_native"] - P[f"{m}_soft_native"], P[f"{m}_sharp_fixed"] - P[f"{m}_soft_fixed"],
                          P[f"{m}_sharp_adapt_pair"] - P[f"{m}_soft_native"]]).dropna()
        lo_, hi_ = min(allv.min(), 0), max(allv.max(), 0)
        ax.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_))
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.2, color=style.INK, label=f"NLST, {int(P[P.dataset=='NLST'].shape[0])} pairs (violin; Hodges–Lehmann estimate, 95% CI)"),
                        Line2D([0], [0], marker="D", ls="", ms=3.2, mfc="white", mec=style.INK, label=f"NSCLC-Radiogenomics, {int(P[P.dataset=='RG'].shape[0])} pairs")],
               loc="lower left", bbox_to_anchor=(14 / W, 74 / H), ncol=2, fontsize=5, frameon=False, handletextpad=0.3)
    for t, x, y in [("a", 1, 126), ("b", 1, 80)]:
        label(fig, H, x, y, t)
    src(sa, "Figure2", "a_example"); src(pd.DataFrame(rows), "Figure2", "b_summary")
    src(P, "Figure2", "b_pairs")
    return style.save(fig, "Figure2")


# =========================================================== Figure 3: decomposition
def fig3():
    from scipy.stats import pearsonr
    style.setup()
    H = 62.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    D = pd.read_csv(ER("pairs_decomposition.csv"))
    P = pd.read_csv(ER("pairs_perpair.csv"))
    # a: muscle noise soft vs sharp
    a = ax_mm(fig, H, 12, 12, 36, 40)
    for ds, mk in [("NLST", "o"), ("RG", "D")]:
        d = P[P.dataset == ds]
        a.scatter(d.noise_soft, d.noise_sharp, s=4 if ds == "NLST" else 9, marker=mk, facecolor="none" if ds == "RG" else style.INK,
                  edgecolor=style.INK, lw=0.5, alpha=0.5 if ds == "NLST" else 1)
    a.plot([5, 400], [5, 400], color=style.HAIR, lw=0.6)
    a.set_xscale("log"); a.set_yscale("log"); plain(a, "x", (10, 30, 100)); plain(a, "y", (10, 30, 100, 300))
    a.set_xlim(8, 200); a.set_ylim(8, 400)
    a.set_xlabel("Muscle noise, soft (HU)", fontsize=6); a.set_ylabel("Muscle noise, sharp (HU)", fontsize=6)
    a.set_title("Same acquisition", fontsize=6, pad=3)
    # b, c: predicted vs observed differences
    rows = []
    for k, (col_obs, col_pred, col_soft, ttl, unit, lim) in enumerate([
            ("imat_sharp", "imat_pred", "imat_soft", "Intramuscular fat", "percentage points", (-5, 60)),
            ("win_sharp", "win_pred", "win_soft", "Muscle-window mean", "HU", (-5, 45))]):
        ax = ax_mm(fig, H, 66 + k * 48, 12, 36, 40)
        for ds, mk in [("NLST", "o"), ("RG", "D")]:
            d = D[D.dataset == ds]
            x, y = d[col_pred] - d[col_soft], d[col_obs] - d[col_soft]
            ax.scatter(x, y, s=4 if ds == "NLST" else 9, marker=mk, facecolor="none" if ds == "RG" else style.INK,
                       edgecolor=style.INK, lw=0.5, alpha=0.5 if ds == "NLST" else 1)
            rows.append(dict(measure=ttl, dataset=ds, n=len(d), r=pearsonr(x, y)[0],
                             frac_explained_median=float(((x) / (y)).median())))
        ax.plot(lim, lim, color=style.HAIR, lw=0.6)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        rn = rows[-2]
        ax.text(0.03, 0.97, f"NLST: $\\mathit{{r}}$ = {rn['r']:.2f}\nmedian explained {100 * rn['frac_explained_median']:.0f}%",
                transform=ax.transAxes, fontsize=5, va="top")
        ax.set_xlabel(f"Predicted from soft image + noise ({unit})", fontsize=5.6)
        ax.set_ylabel(f"Observed sharp − soft ({unit})", fontsize=5.6)
        ax.set_title(ttl, fontsize=6, pad=3)
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=style.INK, label="NLST pairs"),
                        Line2D([0], [0], marker="D", ls="", ms=3, mfc="white", mec=style.INK, label="NSCLC-Radiogenomics pairs")],
               loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=2, fontsize=5, frameon=False)
    for t, x, y in [("a", 1, 58), ("b", 54, 58), ("c", 102, 58)]:
        label(fig, H, x, y, t)
    src(D, "Figure3", "pairs"); src(pd.DataFrame(rows), "Figure3", "summary")
    return style.save(fig, "Figure3")


# =========================================================== Figure 4: trade-off
def fig4():
    style.setup()
    H = 62.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    sw = pd.read_csv(ER("pairs_sigma_sweep.csv"))
    S = pd.read_csv(ER("pairs_summary.csv"))
    a = ax_mm(fig, H, 12, 12, 44, 40)
    for ds, mk, ls in [("NLST", "o", "-"), ("RG", "D", (0, (2, 1.5)))]:
        s = sw[(sw.dataset == ds) & (sw.measure == "imat_pct")].sort_values("sigma")
        a.fill_between(s.sigma, s.kernel_lo, s.kernel_hi, color=style.INK, alpha=0.12, lw=0)
        a.plot(s.sigma, s.kernel_diff, color=style.INK, ls=ls, marker=mk, ms=2.5, lw=0.9, mfc="white" if ds == "RG" else style.INK)
        a.fill_between(s.sigma, s.soft_lo, s.soft_hi, color=C_SOFTSHIFT, alpha=0.15, lw=0)
        a.plot(s.sigma, s.soft_shift, color=C_SOFTSHIFT, ls=ls, marker=mk, ms=2.5, lw=0.9, mfc="white" if ds == "RG" else C_SOFTSHIFT)
        ad = S[(S.dataset == ds) & (S.stratum == "all") & (S.measure == "imat_pct") & (S.estimand == "sharp-only, pairwise target")].iloc[0]
        a.errorbar(1.62 + (0.06 if ds == "RG" else 0), ad.hl, yerr=[[ad.hl - ad.lo], [ad.hi - ad.hl]], fmt=mk, ms=3, color=C_ADA,
                   mfc=C_ADA if ds == "NLST" else "white", capsize=0, lw=0.9)
    a.axhline(0, color=style.HAIR, lw=0.5)
    a.set_xlim(-0.1, 1.75); a.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]); a.set_xticklabels(["0", "", "0.5", "0.75", "1.0", "", "1.5"])
    a.text(1.65, 1.2, "sharp\nonly", fontsize=4.6, ha="center", va="bottom", color=C_ADA)
    a.set_ylim(-8, 16)
    a.set_xlabel("Gaussian SD after 5-mm slab (mm)", fontsize=6)
    a.set_ylabel("Intramuscular fat (percentage points)", fontsize=6)
    a.legend(handles=[Line2D([0], [0], color=style.INK, lw=0.9, label="Sharp − soft, both filtered"),
                      Line2D([0], [0], color=C_SOFTSHIFT, lw=0.9, label="Soft filtered − soft native"),
                      Line2D([0], [0], color=style.INK2, lw=0.9, ls="-", marker="o", ms=2.5, label="NLST"),
                      Line2D([0], [0], color=style.INK2, lw=0.9, ls=(0, (2, 1.5)), marker="D", ms=2.5, mfc="white", label="RG")],
             loc="upper right", fontsize=4.6, handlelength=1.8)
    # b: phantom recovery
    ph = pd.read_csv(ER("phantom_sweep.csv"))
    b = ax_mm(fig, H, 72, 12, 44, 40)
    p = ph[(ph.noise_sd == 25)].sort_values("sigma")
    for d, col in PHANTOM_COL.items():
        b.fill_between(p.sigma, p[f"recovery_{d}mm_mean"] - p[f"recovery_{d}mm_ci"], p[f"recovery_{d}mm_mean"] + p[f"recovery_{d}mm_ci"],
                       color=col, alpha=0.15, lw=0)
        b.plot(p.sigma, p[f"recovery_{d}mm_mean"], color=col, marker="o", ms=2.3, lw=0.9, label=f"{d}-mm fat inclusion")
    b.set_ylim(-0.03, 1.05); b.set_xlim(-0.1, 1.6)
    b.set_xlabel("Gaussian SD after 5-mm slab (mm)", fontsize=6); b.set_ylabel("Recovered fraction (phantom)", fontsize=6)
    b.legend(loc="lower left", fontsize=4.8)
    # c: eroded mask
    e = pd.read_csv(ER("erode_public_rg.csv"))
    sp = pd.read_csv(R("results", "audit", "splits.csv"))
    e = e[e.case_id.isin(sp[sp.split == "development"].case_id)]
    c = ax_mm(fig, H, 136, 12, 42, 40)
    rows = []
    for cond, col, lab in [("native", C_NAT, "Native"), ("matched", C_FIX, "Both filtered (fixed)")]:
        g = e[e.condition == cond].groupby("erode_px").imat_pct
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        c.errorbar(med.index + (0.06 if cond == "matched" else -0.06), med, yerr=[med - q1, q3 - med], fmt="o-", ms=3,
                   color=col, lw=0.9, capsize=0, label=lab)
        for k in med.index:
            rows.append(dict(condition=cond, erode_px=k, median=med[k], q1=q1[k], q3=q3[k]))
    c.set_xticks([0, 1, 2]); c.set_xticklabels(["0", "1\n(0.8 mm)", "2\n(1.5 mm)"], fontsize=5.2)
    c.set_xlabel("Mask erosion (pixels)", fontsize=6); c.set_ylabel("T12 intramuscular fat (%)", fontsize=6)
    c.set_ylim(0, 30); c.legend(loc="upper right", fontsize=4.8)
    c.set_title("107 RG development CTs, median (IQR)", fontsize=5.6, pad=3)
    for t, x, y in [("a", 1, 58), ("b", 61, 58), ("c", 124, 58)]:
        label(fig, H, x, y, t)
    src(sw, "Figure4", "a_sweep"); src(ph, "Figure4", "b_phantom"); src(pd.DataFrame(rows), "Figure4", "c_erosion")
    return style.save(fig, "Figure4")


# =========================================================== Figure 5: survival
def fig5():
    style.setup()
    H = 70.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    r = pd.read_csv(ER("survival_processing.csv"))
    procs = [("native", "Native", C_NAT, "o"), ("fixed", "Both\nfiltered", C_FIX, "o"), ("adaptive", "Sharp\nonly", C_ADA, "o"),
             ("fixed_eroded", "Eroded\nmask", C_ERO, "s")]
    exps = [("imat", "Intramuscular fat, higher"), ("window", "Muscle-window density, lower"),
            ("whole", "Whole-mask density, lower"), ("area", "Muscle area, lower")]
    pops = [("primary_121", "Primary (121/44)", "o", True), ("primary_sharp", "Sharp-kernel CTs (74/25)", "^", True),
            ("all_ages_160", "All ages (160/54)", "o", False)]
    for i, (e, ttl) in enumerate(exps):
        ax = ax_mm(fig, H, 13 + i * 43.5, 13, 36, 44)
        for j, (pk, plab, mk, filled) in enumerate(pops):
            for k, (pr, plb, col, _) in enumerate(procs):
                s = r[(r.population == pk) & (r.exposure == e) & (r.processing == pr)]
                if s.empty:
                    continue
                s = s.iloc[0]
                x = k + (j - 1) * 0.22
                ax.errorbar(x, s.HR, yerr=[[s.HR - s.lo], [s.hi - s.HR]], fmt=mk, ms=3, color=col, capsize=0,
                            mfc=col if filled else "white", lw=0.8)
        ax.axhline(1, color=style.HAIR, lw=0.6, zorder=0)
        ax.set_yscale("log"); plain(ax, "y", (0.5, 1, 2, 4)); ax.set_ylim(0.45, 4.5)
        n = 4 if e == "imat" else 3
        ax.set_xticks(range(n)); ax.set_xticklabels([p[1] for p in procs][:n], fontsize=4.8)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_title(ttl, fontsize=6, pad=3)
        if i == 0:
            ax.set_ylabel("HR per SD (95% CI)", fontsize=6)
    fig.legend(handles=[Line2D([0], [0], marker=mk, ls="", ms=3.2, color=style.INK, mfc=style.INK if f else "white", label=l)
                        for _, l, mk, f in pops], loc="upper left", bbox_to_anchor=(13 / W, 0.99), ncol=3, fontsize=5, frameon=False)
    src(r, "Figure5", "hr")
    return style.save(fig, "Figure5")



# =========================================================== graphical abstract parts
def figGA():
    """Parts of the graphical abstract: image strip and a dot chart (assembled in build_er.py)."""
    style.setup()
    fig = plt.figure(figsize=(150 * style.MM, 50 * style.MM))
    axs = [fig.add_axes([0.005 + i * 0.333, 0.02, 0.323, 0.86]) for i in range(3)]
    example_images(axs)
    for a, t in zip(axs, ["Soft kernel", "Sharp kernel, same acquisition", "Sharp, smoothed to soft noise"]):
        a.set_title(t, fontsize=8, pad=3)
    fig.savefig(os.path.join(style.OUT, "GA_images.png"), dpi=300)
    plt.close(fig)
    S = pd.read_csv(ER("pairs_summary.csv"))
    g = lambda est, meas="imat_pct": S[(S.dataset == "NLST") & (S.stratum == "all") & (S.measure == meas) & (S.estimand == est)].iloc[0]
    fig = plt.figure(figsize=(80 * style.MM, 60 * style.MM))
    ax = fig.add_axes([0.42, 0.2, 0.55, 0.72])
    items = [("Native", g("native"), C_NAT), ("Both filtered", g("fixed matching"), C_FIX),
             ("Sharp only", g("sharp-only, pairwise target"), C_ADA)]
    for i, (lab, r, col) in enumerate(items):
        y = 2 - i
        ax.errorbar(r.hl, y, xerr=[[r.hl - r.lo], [r.hi - r.hl]], fmt="o", ms=6, color=col, capsize=0, lw=1.4)
        ax.text(max(r.hi, 0) + 0.8, y + 0.18, f"{r.hl:+.1f}".replace("-", "−"), fontsize=8, color=col, va="bottom")
    ax.axvline(0, color=style.HAIR, lw=0.8)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([i[0] for i in items], fontsize=8)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(-4, 21); ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel(f"Sharp − soft intramuscular fat\n(percentage points; {int(g('native').n)} pairs)", fontsize=7.5)
    fig.savefig(os.path.join(style.OUT, "GA_chart.png"), dpi=300)
    plt.close(fig)
    return os.path.join(style.OUT, "GA_images.png")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["1", "2", "3", "4", "5", "GA"]:
        print(globals()[f"fig{w}"]())
