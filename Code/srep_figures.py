"""Main figures for the Scientific Reports version: seven multi-panel figures. Output: figures/srep/.

Each figure is composed (vector PDF, figures/srep_compose.py, PyMuPDF in the r45 environment) from parts rendered here
into figures/srep/parts/:
Figure1  a-c participant flow and processing strategies (fig1_flow, plain CONSORT-style boxes and a ruled table; same
         counts as er_figures.fig1) + d measurement steps (fig1_pipeline) + e-g former Supp. Fig. S1 c-e
Figure2  a-b kernel effect (er_figures.fig2) + c-f per-pair soft vs sharp values (fig2_pairs, new)
Figure3  a-d vendor residuals and unpaired clinical CTs (fig3, new)
Figure4  a-c paired noise decomposition (er_figures.fig3) + d-i cohort-level noise analysis (former Supp. Fig. S2)
Figure5  a-c filter strength (er_figures.fig4) + d-f phantom by processing step and slice offset (former Supp. Fig. S8)
Figure6  a-e Comp2Comp (figures/er_supp/FigureS1) + f-h separate repeat examinations (former Supp. Fig. S3 c-e)
Figure7  a HRs by processing strategy (er_figures.fig5) + b-d Kaplan-Meier, exposure set, processing steps (former S4)
Supplementary Figs: S1 = former S3 a-b; S2, S3, S4 = figures/er_supp/FigureS2, FigureS3, FigureS6.
Former supplementary figures are re-rendered with new panel letters (style.panel_label remapped); content unchanged.

One colour code across all seven figures (the ER figures keep their own; patched in render_parts):
processing strategy  native grey, both filtered (fixed) black, sharp only blue, eroded mask brown;
kernel class         sharp orange, soft petrol (not blue, which means sharp-only filtering);
processing steps     native grey, 5-mm slab light blue, Gaussian blue, slab + Gaussian black (= the fixed filter).
Parts are stacked with their blank margins trimmed (srep_compose.py), so the figures carry no empty bands.
No text below 5 pt (text_floor), as the Nature portfolio artwork guide asks.

Data: results/er/*, results/r1005/*, results/c2c/*; cohort_kernel_percase.csv from stats/srep_cohort_kernel_percase.R.
Usage: env/bin/python figures/srep_figures.py
"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import er_figures as E  # noqa: E402
import er_supp_figures as S  # noqa: E402
import style  # noqa: E402

style.OUT = os.path.join(style.ROOT, "figures", "srep")
W, ER, ax_mm, label, plain = E.W, E.ER, E.ax_mm, E.label, E.plain
VENDORS = [("GE", "GE"), ("SIEMENS", "Siemens"), ("TOSHIBA", "Toshiba"), ("PHILIPS", "Philips")]
# vendor colours avoid the kernel and strategy colours; markers differ as well (GE vs Siemens is weak under protanopia)
VCOL = {"GE": "#4a3aa7", "SIEMENS": "#c0398a", "TOSHIBA": "#d9a400", "PHILIPS": "#7a5230"}
VMK = {"GE": "o", "SIEMENS": "s", "TOSHIBA": "^", "PHILIPS": "v"}
KC = {"sharp": style.SERIES[1], "soft": "#006d7c"}  # kernel class; CVD-checked against the strategy colours
C_ERO = "#9a6a2f"
SREP = os.path.join(style.ROOT, "figures", "srep")
PARTS = os.path.join(SREP, "parts")
COMPOSE_PY = "/users/yc4229/.conda/envs/r45/bin/python"
FIGS = os.path.join(style.ROOT, "figures")
# former supplementary figures re-rendered with new panel letters:
# key -> (function, letter map, figure height in mm, letters moved down by mm to sit just above their panel)
RELABEL = {"S1": (S.fig1, dict(zip("bcde", "defg")), 143.5, {}),
           "S2": (S.fig2, dict(zip("abcdef", "defghi")), 122.0, dict(a=5, b=5, c=5)),
           "S3": (S.fig3, dict(zip("cde", "fgh")), 118.0, {}),
           "S4": (S.fig4, dict(zip("abc", "bcd")), 112.0, {}),
           "S5": (S.figS5, dict(zip("abc", "def")), 62.0, {})}
# composed figures: name -> list of (source pdf, clip in mm from the top-left of the source or None).
# Parts are stacked top to bottom; srep_compose.py trims blank rows above and below each clip and leaves GAP_MM between parts.
LAYOUT = {
    "Figure1": [("srep/parts/E1/Figure1.pdf", None), ("srep/parts/D1/Figure1_d.pdf", None),
                ("srep/parts/S1/Figure1.pdf", [0, 69.5, 183, 143.5])],
    "Figure2": [("srep/parts/E2/Figure2.pdf", None), ("srep/parts/F2/Figure2_pairs.pdf", None)],
    "Figure3": [("srep/parts/F3/Figure3.pdf", None)],
    "Figure4": [("srep/parts/E3/Figure3.pdf", None), ("srep/parts/S2/Figure2.pdf", [0, 0, 183, 52]),
                ("srep/parts/S2/Figure2.pdf", [0, 52, 183, 122])],
    "Figure5": [("srep/parts/E4/Figure4.pdf", None), ("srep/parts/S5/FigureS5.pdf", None)],
    "Figure6": [("er_supp/FigureS1.pdf", None), ("srep/parts/S3/Figure3.pdf", [0, 55.5, 183, 118])],
    "Figure7": [("srep/parts/E5/Figure5.pdf", None), ("srep/parts/S4/Figure4.pdf", None)],
    "supp/SupplementaryFigS1": [("srep/parts/S3/Figure3.pdf", [0, 0, 183, 55.5])],
    "supp/SupplementaryFigS2": [("er_supp/FigureS2.pdf", None)],
    "supp/SupplementaryFigS3": [("er_supp/FigureS3.pdf", None)],
    "supp/SupplementaryFigS4": [("er_supp/FigureS6.pdf", None)],
}
GAP_MM = 4.0
# source data: name -> list of (folder, stem, letter map or None, letters kept or None, letter for unlettered files)
SOURCE = {
    "Figure1": [("srep/parts/E1", "Figure1", None, None, None), ("srep/parts/S1", "Figure1", dict(zip("bcde", "defg")), "bcde", None)],
    "Figure2": [("srep/parts/E2", "Figure2", None, None, None), ("srep/parts/F2", "Figure2_pairs", None, None, None)],
    "Figure3": [("srep/parts/F3", "Figure3", None, None, None)],
    "Figure4": [("srep/parts/E3", "Figure3", None, None, "a_b_c"), ("srep/parts/S2", "Figure2", dict(zip("abcdef", "defghi")), None, None)],
    "Figure5": [("srep/parts/E4", "Figure4", None, None, None), ("srep/parts/S5", "FigureS5", dict(zip("abc", "def")), None, None)],
    "Figure6": [("er_supp", "FigureS1", None, None, None), ("srep/parts/S3", "Figure3", dict(zip("cde", "fgh")), "cde", None)],
    "Figure7": [("srep/parts/E5", "Figure5", None, None, "a"), ("srep/parts/S4", "Figure4", dict(zip("abc", "bcd")), None, None)],
    "supp/SupplementaryFigS1": [("srep/parts/S3", "Figure3", None, "ab", None)],
    "supp/SupplementaryFigS2": [("er_supp", "FigureS2", None, None, None)],
    "supp/SupplementaryFigS3": [("er_supp", "FigureS3", None, None, None)],
    "supp/SupplementaryFigS4": [("er_supp", "FigureS6", None, None, None)],
}


def violin(ax, v, pos, col, width=0.55, alpha=0.3):
    parts = ax.violinplot(v, positions=[pos], widths=width, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(col); b.set_alpha(alpha); b.set_edgecolor("none")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([pos, pos], [q1, q3], color=col, lw=1.0, zorder=3)
    ax.scatter([pos], [med], s=11, color=col, edgecolor="white", lw=0.4, zorder=4)
    return med, q1, q3


def fig3():
    from scipy.stats import spearmanr
    style.setup()
    H = 66.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    P = pd.read_csv(ER("pairs_perpair.csv"))
    S = pd.read_csv(ER("pairs_summary.csv"))
    nl = P[P.dataset == "NLST"].copy()
    nl["added_noise"] = np.sqrt(nl.noise_sharp ** 2 - nl.noise_soft ** 2)
    nl["imat_diff_native"] = nl.imat_pct_sharp_native - nl.imat_pct_soft_native

    # a: native kernel effect against the added noise, by vendor
    a = ax_mm(fig, H, 12, 12, 38, 42)
    for v, lab in VENDORS:
        d = nl[nl.vendor == v]
        a.scatter(d.added_noise, d.imat_diff_native, s=5, marker=VMK[v], color=VCOL[v], alpha=0.6, lw=0, label=f"{lab} ({len(d)})")
    rho = spearmanr(nl.added_noise, nl.imat_diff_native)[0]
    a.text(0.97, 0.04, f"Spearman ρ = {rho:.2f}", transform=a.transAxes, fontsize=5.2, ha="right", va="bottom")
    a.set_xlim(0, 450); a.set_ylim(0, 40)
    plain(a, "x", (0, 100, 200, 300, 400)); plain(a, "y", (0, 10, 20, 30, 40))
    a.set_xlabel("Noise added by the sharp kernel (HU)", fontsize=5.8)  # (σ²sharp − σ²soft)^1/2, defined in the legend
    a.set_ylabel("Sharp − soft intramuscular fat\n(percentage points)", fontsize=5.8)
    a.set_title("NLST pairs, native", fontsize=6, pad=3)
    a.legend(loc="upper left", fontsize=4.8, handletextpad=0.1, markerscale=1.6, labelspacing=0.3)

    # b: remaining difference by vendor under each strategy (native estimate printed above)
    b = ax_mm(fig, H, 62, 12, 42, 42)
    ests = [("fixed matching", "Both filtered (fixed)", E.C_FIX, "o", E.C_FIX),
            ("sharp-only, pairwise target", "Sharp only, to paired soft noise", E.C_ADA, "o", E.C_ADA),
            ("sharp-only, fixed target", "Sharp only, one target per dataset", E.C_ADA, "o", "white")]
    rows_b = []
    for i, (v, lab) in enumerate(VENDORS):
        g = lambda est: S[(S.dataset == "NLST") & (S.stratum == f"vendor={v}") & (S.measure == "imat_pct") & (S.estimand == est)].iloc[0]
        for j, (est, _, col, mk, fc) in enumerate(ests):
            r = g(est)
            x = i + (j - 1) * 0.22
            b.errorbar(x, r.hl, yerr=[[r.hl - r.lo], [r.hi - r.hl]], fmt=mk, ms=3, color=col, mfc=fc, capsize=0, lw=0.9, zorder=3)
            rows_b.append(dict(vendor=v, estimand=est, n=int(r.n), hl=r.hl, lo=r.lo, hi=r.hi))
        rn = g("native")
        rows_b.append(dict(vendor=v, estimand="native", n=int(rn.n), hl=rn.hl, lo=rn.lo, hi=rn.hi))
        b.text(i, 2.55, f"native\n{style.fnum(rn.hl, '{:.1f}')}", fontsize=4.6, ha="center", va="top", color=E.C_NAT)
    b.axhline(0, color=style.INK2, lw=0.5, zorder=0)
    b.set_xticks(range(4))
    b.set_xticklabels([f"{lab}\n({int(S[(S.dataset == 'NLST') & (S.stratum == f'vendor={v}') & (S.measure == 'imat_pct') & (S.estimand == 'native')].n.iloc[0])})"
                       for v, lab in VENDORS], fontsize=5.2)
    b.set_xlim(-0.5, 3.5); b.set_ylim(-7.8, 2.6)
    plain(b, "y", (-6, -4, -2, 0, 2))
    b.set_ylabel("Remaining sharp − soft intramuscular fat\n(percentage points)", fontsize=5.8)
    b.set_title("NLST pairs by vendor (HL, 95% CI)", fontsize=6, pad=3)
    b.legend(handles=[Line2D([0], [0], marker=mk, ls="", ms=3, color=col, mfc=fc, label=lab) for _, lab, col, mk, fc in ests],
             loc="lower left", fontsize=4.6, handletextpad=0.2, labelspacing=0.3)

    # c, d: unpaired clinical CTs
    C = pd.read_csv(ER("cohort_kernel_percase.csv"))
    sh, so = C[C.kernel_type == "sharp"], C[C.kernel_type == "soft"]
    c = ax_mm(fig, H, 117, 12, 26, 42)
    rows_c = []
    for pos, (lab, v, col) in enumerate([("Sharp,\nnative", sh.noise_native, KC["sharp"]),
                                         ("Sharp,\nsharp only", sh.noise_after, KC["sharp"]),
                                         ("Soft,\nnative", so.noise_native, KC["soft"])]):
        v = np.log10(v.dropna())
        med, q1, q3 = violin(c, v, pos, col, alpha=0.3 if pos != 1 else 0.15)
        rows_c.append(dict(group=lab.replace("\n", " "), n=len(v), median=10 ** med, q1=10 ** q1, q3=10 ** q3))
    c.set_xticks(range(3)); c.set_xticklabels([r["group"].replace(", ", ",\n") for r in rows_c], fontsize=5.0)
    yt = [10, 30, 100, 300]
    c.set_yticks(np.log10(yt)); c.set_yticklabels([str(t) for t in yt]); c.set_ylim(np.log10(8), np.log10(400))
    c.set_xlim(-0.6, 2.6)
    c.set_ylabel("Muscle noise (HU)", fontsize=5.8)
    c.set_title(f"Clinical CTs ({len(sh)} sharp, {len(so)} soft)", fontsize=6, pad=3)

    d = ax_mm(fig, H, 154, 12, 27, 42)
    rows_d = []
    for i, (col_, lab) in enumerate([("imat_native", "Native"), ("imat_fixed", "Both\nfiltered"), ("imat_adapt", "Sharp\nonly")]):
        for k, (kt, df) in enumerate([("soft", so), ("sharp", sh)]):
            v = df[col_].dropna()
            med, q1, q3 = violin(d, v, i + (k - 0.5) * 0.36, KC[kt], width=0.34)
            rows_d.append(dict(processing=col_, kernel=kt, n=len(v), median=med, q1=q1, q3=q3))
    d.set_xticks(range(3)); d.set_xticklabels(["Native", "Both\nfiltered", "Sharp\nonly"], fontsize=5.0)
    d.set_xlim(-0.6, 2.6); d.set_ylim(0, 90)
    plain(d, "y", (0, 20, 40, 60, 80))
    d.set_ylabel("T12 intramuscular fat (%)", fontsize=5.8)
    d.legend(handles=[Patch(color=KC["sharp"], alpha=0.6, label="Sharp kernel"), Patch(color=KC["soft"], alpha=0.6, label="Soft kernel")],
             loc="upper right", fontsize=4.8, handlelength=0.9, handletextpad=0.3)
    d.set_title("Clinical CTs", fontsize=6, pad=3)

    for t, x, y in [("a", 1, 63), ("b", 51, 63), ("c", 106, 63), ("d", 144, 63)]:
        label(fig, H, x, y, t)
    E.src(nl[["pid", "vendor", "noise_soft", "noise_sharp", "added_noise", "imat_diff_native"]], "Figure3", "a_pairs")
    E.src(pd.DataFrame(rows_b), "Figure3", "b_vendor")
    E.src(C, "Figure3", "c_d_clinical_cts")
    E.src(pd.DataFrame(rows_c), "Figure3", "c_noise_summary")
    E.src(pd.DataFrame(rows_d), "Figure3", "d_imat_summary")
    return style.save(fig, "Figure3")


FS, FS_X = 5.5, 5.0  # flow diagram text (pt): main boxes, exclusion boxes
LH = lambda fs: fs * 1.25 / 72 * 25.4  # line height (mm)
PAD = 1.2  # inner box margin (mm)


def flow_counts():
    """Participant counts for Figure 1a-b, from the result files (fixed numbers: see er_figures.fig1)."""
    pw = pd.read_csv(ER("pairs_perpair.csv"))
    nl = pw[pw.dataset == "NLST"]
    meta = pd.read_csv(ER("pairs_meta.csv"))
    err = pd.read_csv(ER("pairs_errors.csv"))
    rec = pd.read_csv(S.RR("recon_pairs_dicom.csv"))
    rg = pd.read_csv(os.path.join(style.ROOT, "results", "public_rg_bodycomp.csv"))
    v = nl.vendor.value_counts()
    return dict(nlst=616, cand=len(pd.read_csv(ER("nlst_pairs_dicom.csv"))), meas=int((meta.dataset == "NLST").sum()),
                outside=int(((err.dataset == "NLST") & (err.error == "T12 outside partner")).sum()), pairs=len(nl),
                vend=v, rg=rg.case_id.nunique(), rec=len(rec), rec_pt=rec.case_id.nunique(),
                rg_meas=int((meta.dataset == "RG").sum()), rg_pairs=int((pw.dataset == "RG").sum()),
                rg_vend=pw[pw.dataset == "RG"].vendor.value_counts(), nl=nl)


def fig1_flow():
    """Figure 1a-c. a-b: participant flow as a plain CONSORT-style diagram (outlined boxes, exclusions to the right of
    each arrow, rows aligned across the three flows); c: the processing strategies as a ruled table."""
    style.setup()
    k = flow_counts()
    vd, rv = k["vend"], k["rg_vend"]
    nl_c = [dict(m=["NLST participants with staged", "lung cancer: last screening CT", f"n = {k['nlst']}"],
                 x=["No other-kernel series in", f"the same study, n = {k['nlst'] - k['cand']}"]),
            dict(m=["Other-kernel series of the", f"same acquisition, n = {k['cand']}"],
                 x=[f"Not measurable at T12, n = {k['cand'] - k['meas']}",
                    f"No complete T12 measurement, {k['cand'] - k['meas'] - k['outside']}",
                    f"T12 outside the partner series, {k['outside']}"]),
            dict(m=["Measurable at T12", f"n = {k['meas']}"],
                 x=["<95% of the mask inside the", f"partner's field of view, n = {k['meas'] - k['pairs']}"]),
            dict(m=[f"NLST pairs analysed, n = {k['pairs']}", f"GE {vd['GE']}, Siemens {vd['SIEMENS']},",
                    f"Toshiba {vd['TOSHIBA']}, Philips {vd['PHILIPS']}"])]
    rg_c = [dict(m=["NSCLC-Radiogenomics (RG)", f"staging CTs, n = {k['rg']}"],
                 x=["No alternative reconstruction", f"of the same acquisition, n = {k['rg'] - k['rec_pt']}"]),
            dict(m=["Alternative reconstructions,", f"n = {k['rec']} ({k['rec_pt']} patients)"],
                 x=[f"Not measurable at T12, n = {k['rec'] - k['rg_meas']}"]),
            dict(m=["Measurable at T12", f"n = {k['rg_meas']}"], x=[f"Same kernel class, n = {k['rg_meas'] - k['rg_pairs']}"]),
            dict(m=[f"RG pairs analysed, n = {k['rg_pairs']}", f"Siemens {rv['SIEMENS']}, GE {rv['GE']}"])]
    sv_c = [dict(m=["Resected patients, four public", "collections, measured, n = 312"],
                 x=["Missing age, sex or follow-up, n = 1", "Incomplete T5–T12 coverage, n = 37"]),
            dict(m=["Clinical cohort, n = 274", "RG 176; TCGA and CPTAC 98"],
                 x=["Not in the primary population:", "TCGA and CPTAC, 98", "CT not 0–180 d before surgery, 3",
                    "CT date >3 d from the recorded date, 13"]),
            dict(m=["RG, date-concordant", "n = 160 (54 deaths)"], x=["Aged <65 years, n = 39"]),
            dict(m=["Primary population", "n = 121 (44 deaths)"])]
    cols = [nl_c, rg_c, sv_c]
    # widths from the text, then the spare width spread evenly over the gaps
    sc = plt.figure(); rend = sc.canvas.get_renderer()
    tw = lambda s, fs: sc.text(0, 0, s, fontsize=fs).get_window_extent(rend).width / sc.dpi * 25.4
    wm = [max(tw(s, FS) for r in c for s in r["m"]) + 2 * PAD for c in cols]
    wx = [max(tw(s, FS_X) for r in c for s in r.get("x", [])) + 2 * PAD for c in cols]
    plt.close(sc)
    hm = [max(len(c[i]["m"]) for c in cols) * LH(FS) + 2 * PAD for i in range(4)]
    hx = [max(len(c[i].get("x", [])) for c in cols) * LH(FS_X) + 2 * PAD for i in range(3)]
    gx, x0 = 2.5, 3.0  # main box to exclusion box; left margin
    spare = W - 1.0 - x0 - sum(wm) - sum(wx) - 3 * gx
    gc = spare / 2  # between the three flows
    top = [5.0]
    for i in range(3):
        top.append(top[-1] + hm[i] + hx[i] + 3.0)
    note_t = top[3] + hm[3] + 1.2
    c_top = note_t + 2 * LH(5.0) + 2.0  # top of the panel letter c
    rows = [("Native", "Thresholds applied to the images as reconstructed", E.C_NAT),
            ("Both filtered (fixed)", "5-mm slab average and in-plane Gaussian filter (σ = 0.75 mm) applied to every CT", E.C_FIX),
            ("Sharp only (adaptive)", "Sharp-kernel CT smoothed until its muscle noise equals the soft-kernel level; "
                                      "soft-kernel CT unchanged", E.C_ADA)]
    rh = 3.6
    H = c_top + 4.5 + rh * (len(rows) + 1) + 1.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")  # y in mm from the top
    arr = dict(arrowstyle="-|>", lw=0.5, color=style.INK, mutation_scale=4.5, shrinkA=0, shrinkB=0)

    def rect(x, t, w, h, lines, fs, centre):
        ax.add_patch(Rectangle((x, t), w, h, fc="white", ec=style.INK, lw=0.5))
        ax.text(x + w / 2 if centre else x + PAD, t + h / 2, "\n".join(lines), fontsize=fs, ha="center" if centre else "left",
                va="center", ma="center" if centre else "left", linespacing=1.25, color=style.INK)

    xs = []
    x = x0
    for c, a, b in zip(cols, wm, wx):
        xs.append(x); x += a + gx + b + gc
    for c, xm, a, b in zip(cols, xs, wm, wx):
        xc = xm + a / 2
        for i, r in enumerate(c):
            rect(xm, top[i], a, hm[i], r["m"], FS, True)
            if i < 3:
                ax.annotate("", (xc, top[i + 1]), (xc, top[i] + hm[i]), arrowprops=arr)
            if r.get("x"):
                h = len(r["x"]) * LH(FS_X) + 2 * PAD
                ym = top[i] + hm[i] + hx[i] / 2 + 1.5
                rect(xm + a + gx, ym - h / 2, b, h, r["x"], FS_X, False)
                ax.annotate("", (xm + a + gx, ym), (xc, ym), arrowprops=arr)
    ax.text(xs[2], note_t, "Sensitivity population, interval criterion only: n = 130 (46 deaths)\n"
            "Pooled population aged ≥65 years: n = 195 (66 deaths)", fontsize=5.0, va="top", color=style.INK2,
            linespacing=1.25)
    # c: processing strategies, ruled table
    xn, xd, yt = x0 + 6.0, x0 + 36.0, c_top + 4.5
    for y, lw in ((yt, 0.6), (yt + rh, 0.4), (yt + rh * (len(rows) + 1), 0.6)):
        ax.plot([x0, W - 1.0], [y, y], color=style.INK, lw=lw, solid_capstyle="butt")
    ax.text(x0, yt + rh / 2, "Processing strategy", fontsize=FS, va="center", color=style.INK)
    ax.text(xd, yt + rh / 2, "Image processing before thresholding (segmentation masks identical under all three)",
            fontsize=FS, va="center", color=style.INK)
    for j, (t, d, col) in enumerate(rows):
        y = yt + rh * (j + 1.5)
        ax.plot([x0, x0 + 4.0], [y, y], color=col, lw=1.6, solid_capstyle="butt")
        ax.text(xn, y, t, fontsize=FS, va="center", color=style.INK)
        ax.text(xd, y, d, fontsize=FS, va="center", color=style.INK)
    for t, xl, yl in (("a", xs[0], top[0]), ("b", xs[2], top[0]), ("c", x0, c_top + 4.5)):
        style.panel_label(None, t, (xl - 2.5) / W, 1 - (yl - 4.5) / H, fig=fig)
    E.src(k["nl"][["pid", "vendor", "equal_thickness"]], "Figure1", "a_nlst_pairs")
    return style.save(fig, "Figure1")


def fig1_pipeline(case_id="R01-049"):
    """Figure 1d: measurement steps on one RG staging CT, shown as four images with plain captions."""
    import fig1_design_pipeline as F
    from bodycomp_metrics import Case, vertebra_info
    style.setup()
    case = Case(os.path.join(style.ROOT, "data", "nifti", "public_rg", f"{case_id}.nii.gz"),
                os.path.join(style.ROOT, "data", "seg", "public_rg", case_id))
    info = {lv: vertebra_info(case.segs["total"], lv)[0] for lv in ["T4", "T8", "T12"]}
    tot = case.segs["total"]
    z8, z12 = int(round(info["T8"])), int(round(info["T12"]))
    cr = F.centred_crop(F.disp(case.body[:, :, z8]), 150, 190)
    w, gap, x0, t0 = 41.0, 5.0, 3.0, 4.5
    h = w * 300 / 380
    H = t0 + h + 5.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    axs = [ax_mm(fig, H, x0 + i * (w + gap), H - t0 - h, w, h) for i in range(4)]
    axs[0].imshow(F.window(F.disp(case.raw[:, :, z8]))[cr], cmap="gray", interpolation="lanczos")
    axs[1].imshow(F.seg_thumbnail(case, z8, cr), interpolation="lanczos")
    xmid = int(np.median(np.nonzero(case.cord.any(axis=(1, 2)))[0]))
    nz = tot.shape[2]
    zlo, zhi = max(int(info["T12"] - 40 / case.zooms[2]), 0), min(int(info["T4"] + 40 / case.zooms[2]), nz - 1)
    sag = np.flipud(F.window(case.raw[xmid, :, zlo:zhi + 1], 300, 1500).T)[:, ::-1]
    axs[2].imshow(sag, cmap="gray", aspect="auto", interpolation="lanczos")
    for lv in ["T4", "T8", "T12"]:
        yy = zhi - info[lv]
        axs[2].axhline(yy, color=style.TISSUE["muscle"], lw=0.6)
        axs[2].text(tot.shape[1] * 0.035, yy - 3, lv, color="white", fontsize=5, va="bottom", fontweight="bold")
    m12 = {kk: F.disp(v) for kk, v in case.slice_masks(z12).items()}
    cr12 = F.centred_crop(m12["body"], 150, 190)
    fat = {style.TISSUE["sat"], style.TISSUE["ift"], style.TISSUE["imat"]}
    rgb = np.repeat(F.window(F.disp(case.ct[:, :, z12]))[..., None], 3, axis=2)
    for kk in ("sat", "ift", "muscle", "imat"):  # fat drawn translucent, as in Figure 1f
        a = 0.42 if style.TISSUE[kk] in fat else 0.65
        rgb[m12[kk]] = (1 - a) * rgb[m12[kk]] + a * F.hex_rgb(style.TISSUE[kk])
    axs[3].imshow(rgb[cr12], interpolation="lanczos")
    caps = ["Staging CT at T8, native", "Deep-learning segmentation", "Vertebral-body levels",
            "T12 tissue compartments, fixed filter"]
    for a, c in zip(axs, caps):
        F.image_ax(a)
        a.text(0.5, -1.2 / h, c, transform=a.transAxes, fontsize=5.5, ha="center", va="top", color=style.INK)
    style.panel_label(None, "d", 0.5 / W, 1 - 0.3 / H, fig=fig)
    return style.save(fig, "Figure1_d")


def fig2_pairs():
    """Figure 2c-f: soft- against sharp-kernel values of each pair (same mask)."""
    style.setup()
    H = 58.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    P = pd.read_csv(ER("pairs_perpair.csv"))
    nl = P[P.dataset == "NLST"]
    panels = [("imat_pct", "native", "Intramuscular fat, native", "%", (0, 50)),
              ("imat_pct", "adapt_pair", "Intramuscular fat, sharp only", "%", (0, 50)),
              ("smd_window_hu", "native", "Muscle-window density, native", "HU", (10, 80)),
              ("smd_all_hu", "native", "Whole-mask density, native", "HU", (-20, 75))]
    rows = []
    for i, (m, sharp, ttl, unit, lim) in enumerate(panels):
        ax = ax_mm(fig, H, 14 + i * 45, 11, 32, 36)
        if sharp == "adapt_pair":
            ax.fill_between(lim, [lim[0] - 2, lim[1] - 2], [lim[0] + 2, lim[1] + 2], color=style.WASH, lw=0, zorder=0)
        ax.plot(lim, lim, color=style.HAIR, lw=0.6, zorder=1)
        for ds, mk in [("NLST", "o"), ("RG", "D")]:
            d = P[P.dataset == ds]
            x, y = d[f"{m}_soft_native"], d[f"{m}_sharp_{sharp}"]
            ax.scatter(x, y, s=4 if ds == "NLST" else 9, marker=mk, facecolor="none" if ds == "RG" else style.INK,
                       edgecolor=style.INK, lw=0.5, alpha=0.45 if ds == "NLST" else 1, zorder=2 if ds == "NLST" else 3)
            rows.append(pd.DataFrame(dict(panel="cdef"[i], measure=m, sharp_processing=sharp, dataset=ds, pid=d.pid,
                                          soft_native=x, sharp=y)))
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(f"Soft reconstruction ({unit})", fontsize=5.8)
        ax.set_ylabel(f"Sharp reconstruction ({unit})", fontsize=5.8)
        ax.set_title(ttl, fontsize=6, pad=3)
        dn = nl[f"{m}_sharp_{sharp}"] - nl[f"{m}_soft_native"]
        if i == 0:
            n_all = int((P[f"{m}_sharp_native"] > P[f"{m}_soft_native"]).sum())
            txt = f"sharp > soft in\n{n_all} of {len(P)} pairs"
        elif i == 1:
            txt = f"NLST within ±2 points:\n{100 * (dn.abs() <= 2).mean():.0f}% of pairs"
        else:
            txt = f"NLST median difference\n{style.fnum(dn.median(), '{:.1f}')} HU"
        ax.text(0.97, 0.04, txt, transform=ax.transAxes, fontsize=4.8, ha="right", va="bottom")
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=style.INK, label=f"NLST pairs ({len(nl)})"),
                        Line2D([0], [0], marker="D", ls="", ms=3, mfc="white", mec=style.INK,
                               label=f"NSCLC-Radiogenomics pairs ({int((P.dataset == 'RG').sum())})")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2, fontsize=5, frameon=False)
    for t, x in zip("cdef", (1, 46, 91, 136)):
        label(fig, H, x, 54, t)
    E.src(pd.concat(rows), "Figure2_pairs", "c_d_e_f")
    return style.save(fig, "Figure2_pairs")


def palette():
    """Apply the Sci Rep colour code to the ER figure modules; returns the original values for restoring."""
    saved = dict(E_KCOL=dict(E.KCOL), S_KCOL=dict(S.KCOL), C_SOFTSHIFT=E.C_SOFTSHIFT, PHANTOM_COL=dict(E.PHANTOM_COL),
                 C_ERO=E.C_ERO, C_SAFIX=S.C_SAFIX, C_KM=S.C_KM, F2F_XMAX=S.F2F_XMAX)
    E.KCOL.update(KC); S.KCOL.update(KC)
    E.C_SOFTSHIFT = KC["soft"]  # Figure 5a: the shift of the soft-kernel value
    E.PHANTOM_COL = {2: "#b3b1a8", 3: style.INK2, 5: style.INK}  # Figure 5b: inclusion size as a grey ramp, not orange
    E.C_ERO = C_ERO  # Figure 7a
    S.C_SAFIX = E.C_FIX  # Figure 6h: the fixed filter is black everywhere
    S.C_KM = style.INK  # Figure 7b
    S.F2F_XMAX = 2.4  # Figure 4i: no points on the axis edge
    return saved


def unpalette(saved):
    E.KCOL.clear(); E.KCOL.update(saved["E_KCOL"]); S.KCOL.clear(); S.KCOL.update(saved["S_KCOL"])
    E.C_SOFTSHIFT, E.PHANTOM_COL, E.C_ERO = saved["C_SOFTSHIFT"], saved["PHANTOM_COL"], saved["C_ERO"]
    S.C_SAFIX, S.C_KM, S.F2F_XMAX = saved["C_SAFIX"], saved["C_KM"], saved["F2F_XMAX"]


MIN_PT = 5.0  # Nature portfolio artwork: text 5-7 pt at final size (subscripts excepted)


def text_floor():
    """Raise every font size below MIN_PT to MIN_PT (all text sizes pass through FontProperties.set_size)."""
    from matplotlib.font_manager import FontProperties
    orig = FontProperties.set_size

    def set_size(self, size):
        orig(self, size)
        if self.get_size_in_points() < MIN_PT:
            orig(self, MIN_PT)
    FontProperties.set_size = set_size
    return lambda: setattr(FontProperties, "set_size", orig)


def render_parts():
    """Render every part of the composed figures into figures/srep/parts/ with the Sci Rep colour code: the ER main
    figures, the former supplementary figures with new panel letters, Figure 7a with its letter, and the new panels."""
    orig_label, orig_save = style.panel_label, style.save
    saved = palette()
    restore_text = text_floor()
    try:
        for key, (fn, lmap, H, down) in RELABEL.items():
            style.OUT = os.path.join(PARTS, key)

            def relabel(ax, t, x=0.0, y=1.0, *a, _m=lmap, _d=down, _H=H, **k):
                return orig_label(ax, _m.get(t, t), x, y - _d.get(t, 0) / _H, *a, **k)
            style.panel_label = relabel
            print(key, fn())
        style.panel_label = orig_label

        def save_with_a(fig, name, **k):
            style.panel_label(None, "a", 1 / W, 69.6 / 70.0, fig=fig)  # on the legend line, not below it
            return orig_save(fig, name, **k)
        for key, fn in (("E2", E.fig2), ("E3", E.fig3), ("E4", E.fig4)):
            style.OUT = os.path.join(PARTS, key)
            print(key, fn())
        style.OUT = os.path.join(PARTS, "E5")
        style.save = save_with_a
        print("E5", E.fig5())
        style.save = orig_save
        for key, fn in (("E1", fig1_flow), ("D1", fig1_pipeline), ("F2", fig2_pairs), ("F3", fig3)):
            style.OUT = os.path.join(PARTS, key)
            print(key, fn())
    finally:
        style.panel_label, style.save, style.OUT = orig_label, orig_save, SREP
        unpalette(saved)
        restore_text()


def compose():
    import json
    import subprocess
    spec = {name: [(os.path.join(FIGS, p), clip) for p, clip in parts] for name, parts in LAYOUT.items()}
    js = os.path.join(PARTS, "layout.json")
    json.dump(dict(out=SREP, width_mm=W, gap_mm=GAP_MM, figures=spec), open(js, "w"), indent=1)
    subprocess.run([COMPOSE_PY, "-I", os.path.join(HERE, "srep_compose.py"), js], check=True)


def source_data():
    """Source data per composed figure, renamed to the new figure and panel letters."""
    import re
    for name, items in SOURCE.items():
        out = os.path.join(SREP, os.path.dirname(name), "source_data")
        os.makedirs(out, exist_ok=True)
        base = os.path.basename(name)
        for folder, stem, lmap, keep, default in items:
            for f in sorted(glob.glob(os.path.join(FIGS, folder, "source_data", f"{stem}_*.csv"))):
                suf = os.path.basename(f)[len(stem) + 1:-4]
                m = re.match(r"^((?:[a-i]_)*[a-i])(?:_|$)", suf)
                letters = m.group(1).split("_") if m else []
                if keep and not set(letters) & set(keep):
                    continue
                if lmap and letters:
                    suf = "_".join(lmap.get(x, x) for x in letters) + suf[len(m.group(1)):]
                elif not letters and default:
                    suf = f"{default}_{suf}"
                shutil.copy2(f, os.path.join(out, f"{base}_{suf}.csv"))
    print("source data written")


if __name__ == "__main__":
    for d in (os.path.join(SREP, "source_data"), os.path.join(SREP, "supp", "source_data")):
        for f in glob.glob(os.path.join(d, "*.csv")):
            os.remove(f)
    for f in glob.glob(os.path.join(SREP, "Figure*.*")):
        os.remove(f)
    render_parts()
    compose()
    source_data()
