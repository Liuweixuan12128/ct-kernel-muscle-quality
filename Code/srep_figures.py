"""Main figures for the Scientific Reports version: seven multi-panel figures. Output: figures/srep/.

Each figure is composed (vector PDF, figures/srep_compose.py, PyMuPDF in the r45 environment) from parts rendered here
into figures/srep/parts/:
Figure1  a-c participant flow and processing strategies (fig1_flow, plain CONSORT-style boxes and a ruled table; same
         counts as er_figures.fig1) + d measurement steps (fig1_pipeline) + e-g former Supp. Fig. S1 c-e
Figure2  a-b kernel effect (er_figures.fig2) + c-f per-pair soft vs sharp values (fig2_pairs)
Figure3  a-d vendor residuals and unpaired clinical CTs, with noise-referenced filtering (fig3)
Figure4  a-c paired noise decomposition (er_figures.fig3) + d-i cohort-level noise analysis (former Supp. Fig. S2)
Figure5  a-f soft-kernel noise bias and noise-referenced filtering (fig5_noise_ref; revision 1)
Figure6  a-c filter strength and mask erosion (er_figures.fig4) + d-f separate segmentation and slice position (fig6_seg)
Figure7  a-b survival per fixed unit under every strategy and Kaplan-Meier curve (fig7_survival; revision 1)
Supplementary Figs: S1 = former S3 a-b; S2, S3, S4 = figures/er_supp/FigureS2, FigureS3, FigureS6; S5 = Comp2Comp and
separate examinations (former Figure 6); S6 = survival per SD, prespecified exposures and processing steps (former Figure
7a, c, d); S7 = phantom by processing step (former Figure 5d, e).
Former supplementary figures are re-rendered with new panel letters (style.panel_label remapped); content unchanged.

One colour code across all seven figures (the ER figures keep their own; patched in render_parts):
processing strategy  native grey, both filtered (fixed) black, sharp only blue, noise-referenced magenta, eroded mask brown;
kernel class         sharp orange, soft petrol (not blue, which means sharp-only filtering);
processing steps     native grey, 5-mm slab light blue, Gaussian blue, slab + Gaussian black (= the fixed filter).
Parts are stacked with their blank margins trimmed (srep_compose.py), so the figures carry no empty bands.
No text below 5 pt (text_floor), as the Nature portfolio artwork guide asks.

Data: results/er/*, results/r1005/*, results/c2c/*, results/srep/* (revision 1: stats/srep_noise_reference.py,
srep_phantom_noise_ref.py, srep_survival_units.R, srep_sharp_own_seg_summary.py, pipeline/srep_sharp_own_seg.py);
cohort_kernel_percase.csv from stats/srep_cohort_kernel_percase.R.
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
SR = lambda f: os.path.join(style.ROOT, "results", "srep", f)  # revision 1 results
VENDORS = [("GE", "GE"), ("SIEMENS", "Siemens"), ("TOSHIBA", "Toshiba"), ("PHILIPS", "Philips")]
# vendor colours avoid the kernel and strategy colours; markers differ as well (GE vs Siemens is weak under protanopia)
VCOL = {"GE": "#4a3aa7", "SIEMENS": "#3a8a3f", "TOSHIBA": "#d9a400", "PHILIPS": "#7a5230"}
VMK = {"GE": "o", "SIEMENS": "s", "TOSHIBA": "^", "PHILIPS": "v"}
KC = {"sharp": style.SERIES[1], "soft": "#006d7c"}  # kernel class; CVD-checked against the strategy colours
C_ERO = "#9a6a2f"
C_NREF = "#a8327f"  # noise-referenced filtering (revision 1); CVD-checked against grey, black and blue
NREF_R = 25  # reference muscle noise (HU)
SREP = os.path.join(style.ROOT, "figures", "srep")
PARTS = os.path.join(SREP, "parts")
COMPOSE_PY = "/users/yc4229/.conda/envs/r45/bin/python"
FIGS = os.path.join(style.ROOT, "figures")
# former supplementary figures re-rendered with new panel letters:
# key -> (function, letter map, figure height in mm, letters moved down by mm to sit just above their panel)
RELABEL = {"S1": (S.fig1, dict(zip("bcde", "defg")), 143.5, {}),
           "S2": (S.fig2, dict(zip("abcdef", "defghi")), 122.0, dict(a=5, b=5, c=5)),
           "S3": (S.fig3, dict(zip("cde", "fgh")), 118.0, {}),
           "S4": (S.fig4, {}, 112.0, dict(c=2.5)),
           "S5": (S.figS5, {}, 62.0, {})}
# composed figures: name -> list of (source pdf, clip in mm from the top-left of the source or None[, dx in mm]).
# Parts are stacked top to bottom; srep_compose.py trims blank rows above and below each clip and leaves GAP_MM between
# parts; dx moves a clip to the left (Supplementary Fig. S6 b, whose Kaplan-Meier neighbour is in Figure 7).
# Revision 1: Figure 5 (noise reference), Figure 6 d-f (separate segmentation, slice position) and Figure 7 (per-unit HRs)
# are new; the former Figure 6 (Comp2Comp, separate examinations), Figure 7a, c, d and Figure 5d, e moved to the supplement.
LAYOUT = {
    "Figure1": [("srep/parts/E1/Figure1.pdf", None), ("srep/parts/D1/Figure1_d.pdf", None),
                ("srep/parts/S1/Figure1.pdf", [0, 69.5, 183, 143.5])],
    "Figure2": [("srep/parts/E2/Figure2.pdf", None), ("srep/parts/F2/Figure2_pairs.pdf", None)],
    "Figure3": [("srep/parts/F3/Figure3.pdf", None)],
    "Figure4": [("srep/parts/E3/Figure3.pdf", None), ("srep/parts/S2/Figure2.pdf", [0, 0, 183, 52]),
                ("srep/parts/S2/Figure2.pdf", [0, 52, 183, 122])],
    "Figure5": [("srep/parts/F5/Figure5.pdf", None)],
    "Figure6": [("srep/parts/E4/Figure4.pdf", None), ("srep/parts/F6/Figure6.pdf", None)],
    "Figure7": [("srep/parts/F7/Figure7.pdf", None)],
    "supp/SupplementaryFigS1": [("srep/parts/S3/Figure3.pdf", [0, 0, 183, 55.5])],
    "supp/SupplementaryFigS2": [("er_supp/FigureS2.pdf", None)],
    "supp/SupplementaryFigS3": [("er_supp/FigureS3.pdf", None)],
    "supp/SupplementaryFigS4": [("er_supp/FigureS6.pdf", None)],
    "supp/SupplementaryFigS5": [("er_supp/FigureS1.pdf", None), ("srep/parts/S3/Figure3.pdf", [0, 55.5, 183, 118])],
    "supp/SupplementaryFigS6": [("srep/parts/E5/Figure5.pdf", None), ("srep/parts/S4/Figure4.pdf", [64, 0, 183, 62.3], 63),
                                ("srep/parts/S4/Figure4.pdf", [0, 62.4, 183, 112])],
    "supp/SupplementaryFigS7": [("srep/parts/S5/FigureS5.pdf", [0, 0, 124, 62])],
    "supp/SupplementaryFigS8": [("srep/parts/G8/FigureS8.pdf", None)],
}
GAP_MM = 4.0
# source data: name -> list of (folder, stem, letter map or None, letters kept or None, letter for unlettered files)
SOURCE = {
    "Figure1": [("srep/parts/E1", "Figure1", None, None, None), ("srep/parts/S1", "Figure1", dict(zip("bcde", "defg")), "bcde", None)],
    "Figure2": [("srep/parts/E2", "Figure2", None, None, None), ("srep/parts/F2", "Figure2_pairs", None, None, None)],
    "Figure3": [("srep/parts/F3", "Figure3", None, None, None)],
    "Figure4": [("srep/parts/E3", "Figure3", None, None, "a_b_c"), ("srep/parts/S2", "Figure2", dict(zip("abcdef", "defghi")), None, None)],
    "Figure5": [("srep/parts/F5", "Figure5", None, None, None)],
    "Figure6": [("srep/parts/E4", "Figure4", None, None, None), ("srep/parts/F6", "Figure6", None, None, None)],
    "Figure7": [("srep/parts/F7", "Figure7", None, None, None)],
    "supp/SupplementaryFigS1": [("srep/parts/S3", "Figure3", None, "ab", None)],
    "supp/SupplementaryFigS2": [("er_supp", "FigureS2", None, None, None)],
    "supp/SupplementaryFigS3": [("er_supp", "FigureS3", None, None, None)],
    "supp/SupplementaryFigS4": [("er_supp", "FigureS6", None, None, None)],
    "supp/SupplementaryFigS5": [("er_supp", "FigureS1", None, None, None), ("srep/parts/S3", "Figure3", dict(zip("cde", "fgh")), "cde", None)],
    "supp/SupplementaryFigS6": [("srep/parts/E5", "Figure5", None, None, "a"), ("srep/parts/S4", "Figure4", dict(zip("bc", "bc")), "bc", None)],
    "supp/SupplementaryFigS7": [("srep/parts/S5", "FigureS5", None, "ab", None)],
    "supp/SupplementaryFigS8": [("srep/parts/G8", "FigureS8", None, None, None)],
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
    """Figure 3: kernel effect and its correction by vendor (a, b) and in the unpaired clinical CTs (c, d), with the
    noise-referenced strategy of revision 1 (results/srep/noise_ref_*.csv)."""
    from scipy.stats import spearmanr
    style.setup()
    H = 112.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    P = pd.read_csv(ER("pairs_perpair.csv"))
    S = pd.read_csv(ER("pairs_summary.csv"))
    NR = pd.read_csv(SR("noise_ref_summary.csv"))
    nl = P[P.dataset == "NLST"].copy()
    nl["added_noise"] = np.sqrt(nl.noise_sharp ** 2 - nl.noise_soft ** 2)
    nl["imat_diff_native"] = nl.imat_pct_sharp_native - nl.imat_pct_soft_native

    # a: native kernel effect against the added noise, by vendor
    a = ax_mm(fig, H, 13, 66, 44, 38)
    for v, lab in VENDORS:
        d = nl[nl.vendor == v]
        a.scatter(d.added_noise, d.imat_diff_native, s=5, marker=VMK[v], color=VCOL[v], alpha=0.6, lw=0, label=f"{lab} ({len(d)})")
    a.set_xlim(0, 450); a.set_ylim(0, 40)
    plain(a, "x", (0, 100, 200, 300, 400)); plain(a, "y", (0, 10, 20, 30, 40))
    a.set_xlabel("Noise added by the sharp kernel (HU)", fontsize=5.8)  # (σ²sharp − σ²soft)^1/2, defined in the legend
    a.set_ylabel("Sharp − soft intramuscular fat\n(percentage points)", fontsize=5.8)
    a.set_title("NLST pairs, native", fontsize=6, pad=3)
    a.legend(loc="upper left", fontsize=4.8, handletextpad=0.1, markerscale=1.6, labelspacing=0.3)

    # b: remaining difference by vendor under each strategy (native estimate printed above)
    b = ax_mm(fig, H, 76, 66, 100, 38)
    ests = [(S, "fixed matching", "Both filtered (fixed)", E.C_FIX, None),
            (S, "sharp-only, pairwise target", "Sharp only, to paired soft noise", E.C_ADA, None),
            (S, "sharp-only, fixed target", "Sharp only, one target per dataset", E.C_ADA, "white"),
            (NR, f"noise-referenced {NREF_R} HU", f"Noise-referenced, {NREF_R} HU", C_NREF, None)]
    rows_b = []
    for i, (v, lab) in enumerate(VENDORS):
        for j, (tab, est, _, col, fc) in enumerate(ests):
            r = tab[(tab.dataset == "NLST") & (tab.stratum == f"vendor={v}") & (tab.measure == "imat_pct") & (tab.estimand == est)].iloc[0]
            x = i + (j - 1.5) * 0.17
            b.errorbar(x, r.hl, yerr=[[r.hl - r.lo], [r.hi - r.hl]], fmt="o", ms=3, color=col, mfc=fc or col, capsize=0, lw=0.9, zorder=3)
            rows_b.append(dict(vendor=v, estimand=est, n=int(r.n), hl=r.hl, lo=r.lo, hi=r.hi))
        rn = S[(S.dataset == "NLST") & (S.stratum == f"vendor={v}") & (S.measure == "imat_pct") & (S.estimand == "native")].iloc[0]
        rows_b.append(dict(vendor=v, estimand="native", n=int(rn.n), hl=rn.hl, lo=rn.lo, hi=rn.hi))
        b.text(i, 2.55, f"native\n{style.fnum(rn.hl, '{:.1f}')}", fontsize=5, ha="center", va="top", color=E.C_NAT)
    b.axhline(0, color=style.INK2, lw=0.5, zorder=0)
    b.set_xticks(range(4))
    b.set_xticklabels([f"{lab} ({int(nl.vendor.eq(v).sum())})" for v, lab in VENDORS], fontsize=5.4)
    b.set_xlim(-0.5, 3.5); b.set_ylim(-7.8, 2.6)
    plain(b, "y", (-6, -4, -2, 0, 2))
    b.set_ylabel("Remaining sharp − soft\nintramuscular fat (points)", fontsize=5.8)
    b.set_title("NLST pairs by vendor (Hodges–Lehmann estimate, 95% CI)", fontsize=6, pad=3)
    b.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=col, mfc=fc or col, label=lab) for _, _, lab, col, fc in ests],
             loc="lower left", fontsize=5, handletextpad=0.2, labelspacing=0.3, ncol=2, columnspacing=1.2)

    # c, d: unpaired clinical CTs
    NE = pd.read_csv(SR("noise_est_cohort.csv"))
    NE["noise"] = NE.liv_mad.fillna(NE.es_mad)
    C = (pd.read_csv(ER("cohort_kernel_percase.csv"))  # sigma_used: sharp-only filter of each CT
         .merge(pd.read_csv(SR("noise_ref_exposures.csv"))[["case_id", f"noise_r{NREF_R}", f"imat_r{NREF_R}"]], on="case_id", how="left")
         .merge(NE[NE.sigma == 0][["case_id", "noise"]].rename(columns={"noise": "noise_native_liver"}), on="case_id", how="left"))
    C = C.merge(NE[["case_id", "sigma", "noise"]].rename(columns={"sigma": "sigma_used", "noise": "noise_after_liver"}),
                on=["case_id", "sigma_used"], how="left")
    sh, so = C[C.kernel_type == "sharp"], C[C.kernel_type == "soft"]
    c = ax_mm(fig, H, 13, 11, 70, 38)
    rows_c = []
    groups = [("Sharp, native", sh.noise_native_liver, KC["sharp"], 0.3), ("Sharp, sharp only", sh.noise_after_liver, KC["sharp"], 0.15),
              (f"Sharp, {NREF_R} HU", sh[f"noise_r{NREF_R}"], KC["sharp"], 0.15),
              ("Soft, native", so.noise_native_liver, KC["soft"], 0.3), (f"Soft, {NREF_R} HU", so[f"noise_r{NREF_R}"], KC["soft"], 0.15)]
    for pos, (lab, v, col, al) in enumerate(groups):
        v = np.log10(v.dropna())
        med, q1, q3 = violin(c, v, pos, col, alpha=al)
        rows_c.append(dict(group=lab, n=len(v), median=10 ** med, q1=10 ** q1, q3=10 ** q3))
    c.axhline(np.log10(NREF_R), color=C_NREF, lw=0.6, ls=(0, (3, 2)), zorder=0)
    c.set_xticks(range(len(groups))); c.set_xticklabels([g[0].replace(", ", ",\n") for g in groups], fontsize=5.2)
    yt = [10, 25, 50, 100, 300]
    c.set_yticks(np.log10(yt)); c.set_yticklabels([str(t) for t in yt]); c.set_ylim(np.log10(8), np.log10(400))
    c.set_xlim(-0.6, len(groups) - 0.4)
    c.set_ylabel("Image noise, liver (HU)", fontsize=5.8)
    c.set_title(f"Clinical CTs ({len(sh)} sharp, {len(so)} soft)", fontsize=6, pad=3)

    d = ax_mm(fig, H, 104, 11, 72, 38)
    rows_d = []
    strat = [("imat_native", "Native"), ("imat_fixed", "Both filtered"), ("imat_adapt", "Sharp only"),
             (f"imat_r{NREF_R}", "Noise-\nreferenced")]
    for i, (col_, lab) in enumerate(strat):
        for k, (kt, df) in enumerate([("soft", so), ("sharp", sh)]):
            v = df[col_].dropna()
            med, q1, q3 = violin(d, v, i + (k - 0.5) * 0.36, KC[kt], width=0.34)
            rows_d.append(dict(processing=col_, kernel=kt, n=len(v), median=med, q1=q1, q3=q3))
    d.set_xticks(range(len(strat))); d.set_xticklabels([s_[1].replace("Both filtered", "Both\nfiltered").replace("Sharp only", "Sharp\nonly") for s_ in strat], fontsize=5.2)
    d.set_xlim(-0.6, len(strat) - 0.4); d.set_ylim(0, 90)
    plain(d, "y", (0, 20, 40, 60, 80))
    d.set_ylabel("T12 intramuscular fat (%)", fontsize=5.8)
    d.legend(handles=[Patch(color=KC["sharp"], alpha=0.6, label="Sharp kernel"), Patch(color=KC["soft"], alpha=0.6, label="Soft kernel")],
             loc="upper right", fontsize=5, handlelength=0.9, handletextpad=0.3)
    d.set_title("Clinical CTs", fontsize=6, pad=3)

    for t, x, y in [("a", 1, 110), ("b", 63, 110), ("c", 1, 55), ("d", 92, 55)]:
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
    sv_c = [dict(m=["Resected patients, five public", "collections, measured, n = 312"],
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
                                      "soft-kernel CT unchanged", E.C_ADA),
            ("Noise-referenced", f"Every CT whose image noise (measured in the liver) exceeds {NREF_R} HU smoothed (in-plane Gaussian) "
                                 f"to {NREF_R} HU, whatever its kernel; CTs at or below the reference unchanged", C_NREF)]
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
    ax.text(xd, yt + rh / 2, "Image processing before thresholding (segmentation masks identical under all strategies)",
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


def noise_only(sd, mu=40.0):
    """Intramuscular fat (%) produced by noise of SD sd alone in uniform tissue of mean mu (normal model)."""
    from scipy.stats import norm
    sd = np.maximum(np.asarray(sd, float), 1e-6)
    f = norm.cdf((-30 - mu) / sd) - norm.cdf((-190 - mu) / sd)
    m = norm.cdf((150 - mu) / sd) - norm.cdf((-29 - mu) / sd)
    return 100 * f / (f + m)


def fig5_noise_ref():
    """Figure 5 (revision 1): soft reconstructions also carry noise bias; filtering to an absolute noise reference.
    a noise-only intramuscular fat; b change of the soft image under fixed and noise-referenced filtering; c pairs after
    noise-referenced filtering; d reference level; e noise texture; f phantom detail."""
    style.setup()
    H = 116.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    NP = pd.read_csv(SR("noise_ref_perpair.csv"))
    NS = pd.read_csv(SR("noise_ref_summary.csv"))
    C = pd.read_csv(ER("cohort_kernel_percase.csv"))
    CL = C[["case_id", "kernel_type"]].merge(pd.read_csv(SR("noise_ref_exposures.csv"))[["case_id", "noise_native"]], on="case_id")
    MK = {"NLST": "o", "RG": "D"}
    t = f"r{NREF_R}"

    # a: noise-only intramuscular fat
    a = ax_mm(fig, H, 13, 68, 42, 38)
    sd = np.geomspace(5, 200, 200)
    rows_a = []
    for mu, ls in [(30, (0, (1, 1.2))), (40, "-"), (50, (0, (3, 1.5)))]:
        y = noise_only(sd, mu)
        a.plot(sd, y, color=style.INK, lw=0.9 if mu == 40 else 0.7, ls=ls, label=f"{mu} HU tissue")
        rows_a += [dict(tissue_hu=mu, noise_sd=s_, noise_only_imat=v) for s_, v in zip(sd, y)]
    meds = [("NLST", "soft", NP[NP.dataset == "NLST"].noise_soft.median()), ("NLST", "sharp", NP[NP.dataset == "NLST"].noise_sharp.median()),
            ("RG", "soft", NP[NP.dataset == "RG"].noise_soft.median()), ("RG", "sharp", NP[NP.dataset == "RG"].noise_sharp.median()),
            ("Clinical", "soft", CL[CL.kernel_type == "soft"].noise_native.median()), ("Clinical", "sharp", CL[CL.kernel_type == "sharp"].noise_native.median())]
    for ds, k, v in meds:
        a.scatter(v, noise_only(v), s=14, marker={"NLST": "o", "RG": "D", "Clinical": "s"}[ds], color=KC[k], edgecolor="white", lw=0.4, zorder=4)
        rows_a.append(dict(group=f"{ds} {k}, median native noise", noise_sd=v, noise_only_imat=float(noise_only(v))))
    a.axvline(NREF_R, color=C_NREF, lw=0.7, ls=(0, (3, 2)), zorder=0)
    a.set_xscale("log"); plain(a, "x", (5, 10, 25, 50, 100, 200)); a.set_xlim(5, 200); a.set_ylim(-1, 45)
    plain(a, "y", (0, 10, 20, 30, 40))
    a.set_xlabel("Image noise SD (HU)", fontsize=6); a.set_ylabel("Intramuscular fat from noise alone (%)", fontsize=6)
    h1 = a.legend(loc="upper left", fontsize=5, handlelength=1.8, labelspacing=0.3)
    a.add_artist(h1)
    a.legend(handles=[Line2D([0], [0], marker=m_, ls="", ms=3.2, color=style.INK2, label=l_)
                      for l_, m_ in [("NLST pairs", "o"), ("RG pairs", "D"), ("Clinical CTs", "s")]],
             loc="center left", bbox_to_anchor=(0.0, 0.52), fontsize=5, handletextpad=0.2, labelspacing=0.3)

    # b: change of the soft image itself, observed against the change predicted by adding the removed noise back
    b = ax_mm(fig, H, 73, 68, 38, 38)
    AP = pd.read_csv(SR("noise_ref_alt_perpair.csv"))
    rows_b = []
    for ds in ["NLST", "RG"]:
        d = AP[AP.dataset == ds]
        for lab, col, filt, pred in [("fixed", E.C_FIX, d.imat_soft_fixed_grid, d.pred_native_fixed),
                                     ("noise-referenced", C_NREF, d.imat_pct_soft_liv_mad_inplane, d.pred_native_liv_mad_inplane)]:
            ok = pred.notna()
            x_, y_ = (filt - pred)[ok], (filt - d.imat_soft_native)[ok]
            b.scatter(x_, y_, s=4 if ds == "NLST" else 10, marker=MK[ds], facecolor=col if ds == "NLST" else "white",
                      edgecolor=col, lw=0.5, alpha=0.45 if ds == "NLST" else 0.95, zorder=3 if ds == "RG" else 2)
            rows_b += [dict(dataset=ds, pid=p_, processing=lab, predicted_change=u, observed_change=v)
                       for p_, u, v in zip(d.pid[ok], x_, y_)]
    b.plot([-20, 2], [-20, 2], color=style.HAIR, lw=0.6, zorder=1)
    b.axhline(0, color=style.HAIR, lw=0.4, zorder=0); b.axvline(0, color=style.HAIR, lw=0.4, zorder=0)
    b.set_xlim(-16, 2); b.set_ylim(-16, 2); plain(b, "x", (-15, -10, -5, 0)); plain(b, "y", (-15, -10, -5, 0))
    b.set_xlabel("Change predicted from the noise removed\n(percentage points)", fontsize=6)
    b.set_ylabel("Observed change of soft intramuscular fat\n(percentage points)", fontsize=6)
    b.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=E.C_FIX, label="Both filtered (fixed)"),
                      Line2D([0], [0], marker="o", ls="", ms=3, color=C_NREF, label=f"Noise-referenced ({NREF_R} HU)")],
             loc="upper left", fontsize=5, handletextpad=0.3, labelspacing=0.3)

    # c: pairs after noise-referenced filtering
    c = ax_mm(fig, H, 133, 68, 42, 38)
    c.fill_between([0, 50], [-2, 48], [2, 52], color=style.WASH, lw=0, zorder=0)
    c.plot([0, 50], [0, 50], color=style.HAIR, lw=0.6, zorder=1)
    for ds in ["NLST", "RG"]:
        d = NP[NP.dataset == ds]
        c.scatter(d[f"imat_pct_soft_{t}"], d[f"imat_pct_sharp_{t}"], s=4 if ds == "NLST" else 10, marker=MK[ds],
                  facecolor=C_NREF if ds == "NLST" else "white", edgecolor=C_NREF, lw=0.5, alpha=0.5 if ds == "NLST" else 0.95, zorder=3)
    c.set_xlim(0, 40); c.set_ylim(0, 40); plain(c, "x", (0, 10, 20, 30, 40)); plain(c, "y", (0, 10, 20, 30, 40))
    c.set_xlabel(f"Soft reconstruction, {NREF_R} HU (%)", fontsize=6); c.set_ylabel(f"Sharp reconstruction, {NREF_R} HU (%)", fontsize=6)
    c.set_title("Intramuscular fat, noise-referenced", fontsize=6, pad=3)
    c.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=C_NREF, label="NLST pairs"),
                      Line2D([0], [0], marker="D", ls="", ms=3, mfc="white", color=C_NREF, label="RG pairs")],
             loc="upper left", fontsize=5, handletextpad=0.2)

    # d: reference level
    d_ = ax_mm(fig, H, 13, 11, 42, 38)
    refs = [15, 20, 25, 30, 35, 40, 45]
    rows_d = []
    for ds, ls in [("NLST", "-"), ("RG", (0, (3, 1.5)))]:
        for est, col in [("noise-referenced {} HU", C_NREF), ("soft shift, noise-referenced {} HU", KC["soft"])]:
            s_ = pd.concat([NS[(NS.dataset == ds) & (NS.stratum == "all") & (NS.measure == "imat_pct") & (NS.estimand == est.format(r))].assign(ref=r) for r in refs])
            d_.fill_between(s_.ref, s_.lo, s_.hi, color=col, alpha=0.15, lw=0)
            d_.plot(s_.ref, s_.hl, color=col, lw=0.9, ls=ls, marker=MK[ds], ms=2.4, mfc=col if ds == "NLST" else "white")
            rows_d += s_[["dataset", "ref", "estimand", "n", "hl", "lo", "hi"]].to_dict("records")
    d_.axhline(0, color=style.HAIR, lw=0.5, zorder=0); d_.axvline(NREF_R, color=C_NREF, lw=0.6, ls=(0, (3, 2)), zorder=0)
    d_.set_xlim(13, 47); plain(d_, "x", refs); d_.set_ylim(-7, 3.5); plain(d_, "y", (-6, -4, -2, 0, 2))
    d_.set_xlabel("Reference image noise (HU)", fontsize=6); d_.set_ylabel("Intramuscular fat (percentage points)", fontsize=6)
    d_.legend(handles=[Line2D([0], [0], color=C_NREF, lw=0.9, label="Sharp − soft"),
                       Line2D([0], [0], color=KC["soft"], lw=0.9, label="Soft, change from native"),
                       Line2D([0], [0], color=style.INK2, lw=0.9, marker="o", ms=2.4, label="NLST"),
                       Line2D([0], [0], color=style.INK2, lw=0.9, ls=(0, (3, 1.5)), marker="D", mfc="white", ms=2.4, label="RG")],
              loc="lower right", fontsize=5, handlelength=2.0, labelspacing=0.3, ncol=2, columnspacing=0.8)

    # e: noise texture (NLST pairs; eroded erector spinae)
    e = ax_mm(fig, H, 73, 11, 42, 38)
    T = pd.read_csv(SR("sharp_own_seg.csv"))
    T = T[(T.dataset == "NLST") & T.error.isna()] if "error" in T else T[T.dataset == "NLST"]
    grp = [("corr_soft_native", "Soft,\nnative", KC["soft"], 0.3), ("corr_sharp_native", "Sharp,\nnative", KC["sharp"], 0.3),
           ("corr_sharp_pair", "Sharp, to\nsoft noise", KC["sharp"], 0.15), ("corr_soft_r25", f"Soft,\n{NREF_R} HU", KC["soft"], 0.15),
           ("corr_sharp_r25", f"Sharp,\n{NREF_R} HU", KC["sharp"], 0.15)]
    rows_e = []
    for i, (col_, lab, col, al) in enumerate(grp):
        v = T[col_].dropna()
        med, q1, q3 = violin(e, v, i, col, alpha=al)
        rows_e.append(dict(image=lab.replace("\n", " "), n=len(v), median=med, q1=q1, q3=q3))
    e.set_xticks(range(len(grp))); e.set_xticklabels([g_[1] for g_ in grp], fontsize=5)
    e.set_xlim(-0.6, len(grp) - 0.4); e.set_ylim(0, 2.0); plain(e, "y", (0, 0.5, 1.0, 1.5, 2.0))
    e.set_ylabel("Noise correlation length (mm)", fontsize=6)
    e.set_title(f"NLST pairs ({len(T)}), erector spinae", fontsize=6, pad=3)

    # f: digital phantom, recovery of fat inclusions by size
    PH = pd.read_csv(SR("phantom_noise_ref_summary.csv"))
    rows_f = []
    for k, (sdv, corr, ttl) in enumerate([(45, 0.0, "White noise, 45 HU"), (120, 0.0, "White noise, 120 HU")]):
        f = ax_mm(fig, H, 133 + k * 24, 11, 19, 38)
        for meth, col, ls in [("native", E.C_NAT, "-"), ("fixed", E.C_FIX, "-"), ("noise-referenced", C_NREF, "-"),
                              ("slab + noise-referenced", C_NREF, (0, (2, 1.2)))]:
            r = PH[(PH.noise_sd == sdv) & (PH.corr_mm == corr) & (PH.method == meth)].iloc[0]
            dia = [1, 2, 3, 5, 10]
            f.plot(dia, [r[f"recovery_{x}mm_mean"] for x in dia], color=col, lw=0.9, ls=ls, marker="o", ms=2.2,
                   mfc=col if ls == "-" else "white")
            rows_f.append(dict(noise_sd=sdv, corr_mm=corr, method=meth, sigma_mm=r.sigma_mean, false_fat_pct=r.false_fat_pct_mean,
                               **{f"recovery_{x}mm": r[f"recovery_{x}mm_mean"] for x in dia}))
        f.set_xscale("log"); plain(f, "x", (1, 2, 5, 10)); f.set_xlim(0.8, 12); f.set_ylim(0, 1.05)
        plain(f, "y", (0, 0.5, 1.0) if k == 0 else ())
        if k:
            f.tick_params(axis="y", labelleft=False)
        f.set_title(ttl, fontsize=5.5, pad=3)
        f.set_xlabel("Inclusion (mm)", fontsize=6)
        if k == 0:
            f.set_ylabel("Recovered fraction (phantom)", fontsize=6)
            f.legend(handles=[Line2D([0], [0], color=c_, lw=0.9, ls=l2, label=l_) for l_, c_, l2 in
                              [("Native", E.C_NAT, "-"), ("Both filtered", E.C_FIX, "-"), ("Noise-ref.", C_NREF, "-"),
                               ("Slab + noise-ref.", C_NREF, (0, (2, 1.2)))]],
                     loc="lower right", fontsize=5, handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)
    for t_, x, y in [("a", 1, 113), ("b", 61, 113), ("c", 121, 113), ("d", 1, 56), ("e", 61, 56), ("f", 121, 56)]:
        label(fig, H, x, y, t_)
    E.src(pd.DataFrame(rows_a), "Figure5", "a_noise_only"); E.src(pd.DataFrame(rows_b), "Figure5", "b_soft_change")
    E.src(NP[["dataset", "pid", "vendor", f"imat_pct_soft_{t}", f"imat_pct_sharp_{t}", f"sigma_soft_{t}", f"sigma_sharp_{t}"]], "Figure5", "c_pairs")
    E.src(pd.DataFrame(rows_d), "Figure5", "d_reference"); E.src(pd.DataFrame(rows_e), "Figure5", "e_texture")
    E.src(pd.DataFrame(rows_f), "Figure5", "f_phantom")
    return style.save(fig, "Figure5")


def fig6_seg():
    """Figure 6 d-f (revision 1): sharp reconstructions segmented on their own (d, e) and measurement-slice position (f)."""
    from fig_revision import panel_f
    style.setup()
    H = 56.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    O = pd.read_csv(SR("sharp_own_seg.csv"))
    O = O[O["error"].isna()] if "error" in O else O
    SS = pd.read_csv(SR("sharp_own_seg_summary.csv"))
    # d: agreement of the separately drawn sharp-image muscle mask with the soft mask on the same slice
    d = ax_mm(fig, H, 13, 11, 36, 36)
    rows_d = []
    for ds, mk in [("NLST", "o"), ("RG", "D")]:
        v = O[O.dataset == ds].dice_muscle.dropna()
        if ds == "NLST":
            d.hist(v, bins=np.arange(0.60, 1.0001, 0.01), color=style.MUTED, edgecolor="white", lw=0.3)
        else:
            d.scatter(v, np.full(len(v), -4), marker=mk, s=10, facecolor="white", edgecolor=style.INK, lw=0.5, clip_on=False, zorder=3)
        rows_d += [dict(dataset=ds, dice_muscle=x) for x in v]
    d.set_xlim(0.60, 1.0); plain(d, "x", (0.6, 0.7, 0.8, 0.9, 1.0))
    d.set_ylim(-8, None)
    d.spines["left"].set_bounds(0, d.get_ylim()[1])
    d.set_xlabel("Dice, sharp-image mask vs soft-image mask", fontsize=6); d.set_ylabel("NLST pairs", fontsize=6)
    d.set_title("T12 muscle masks", fontsize=6, pad=3)
    # e: sharp - soft difference with the shared mask and with each image's own masks, NLST, per strategy
    strat = [("native", "Native", E.C_NAT), ("fixed", "Fixed", E.C_FIX), ("sharp-only", "Sharp\nonly", E.C_ADA),
             ("noise-referenced 25 HU", f"{NREF_R} HU", C_NREF)]
    rows_e = []
    for k, (m, ttl, lim, ticks) in enumerate([("imat_pct", "Intramuscular fat (points)", (-3, 20), (0, 5, 10, 15, 20)),
                                               ("sma_cm2", "Muscle area (cm²)", (-24, 3), (-20, -15, -10, -5, 0))]):
        e = ax_mm(fig, H, 64 + k * 34, 11, 26, 36)
        for i, (st, lab, col) in enumerate(strat):
            for j, (mask, fc) in enumerate([("shared mask", col), ("own masks", "white")]):
                r = SS[(SS.dataset == "NLST") & (SS["item"] == f"{st} | {mask} | {m}")].iloc[0]
                x = i + (j - 0.5) * 0.3
                e.errorbar(x, r.value, yerr=[[r.value - r.lo], [r.hi - r.value]], fmt="o", ms=2.8, color=col, mfc=fc,
                           capsize=0, lw=0.8, zorder=3)
                rows_e.append(dict(dataset="NLST", measure=m, strategy=st, masks=mask, n=r.n, hl=r.value, lo=r.lo, hi=r.hi))
        e.axhline(0, color=style.INK2, lw=0.5, zorder=0)
        e.set_xticks(range(len(strat))); e.set_xticklabels([s_[1] for s_ in strat], fontsize=5)
        e.set_xlim(-0.6, len(strat) - 0.4); e.set_ylim(*lim); plain(e, "y", ticks)
        e.set_title(ttl, fontsize=6, pad=3)
        if k == 0:
            e.set_ylabel("Sharp − soft (HL, 95% CI)", fontsize=6)
            e.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=style.INK2, label="Shared mask"),
                              Line2D([0], [0], marker="o", ls="", ms=3, color=style.INK2, mfc="white", label="Own masks")],
                     loc="upper right", fontsize=5, handletextpad=0.2, labelspacing=0.3)
    f = ax_mm(fig, H, 140, 11, 40, 36)
    sf = panel_f(f)
    f.set_ylabel("Median absolute change", fontsize=6)
    for t_, x, y in [("d", 1, 54), ("e", 52, 54), ("f", 128, 54)]:
        label(fig, H, x, y, t_)
    E.src(pd.DataFrame(rows_d), "Figure6", "d_dice"); E.src(pd.DataFrame(rows_e), "Figure6", "e_own_masks")
    E.src(sf, "Figure6", "f_offset")
    return style.save(fig, "Figure6")


def figS8_noise():
    """Supplementary Fig. S8 (revision 2): noise estimators and routine-dose pairs.
    a muscle/liver noise ratio against intramuscular fat; b muscle against liver noise; c, d noise-referenced filtering with
    each estimator, in-plane and with a 5-mm slab; e routine-dose TCGA/CPTAC same-acquisition pairs."""
    style.setup()
    H = 112.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    T = pd.read_csv(SR("noise_est_test_perimage.csv"))
    rows_a = []
    a = ax_mm(fig, H, 13, 66, 44, 38)
    for (ds, k), mk in [(("NLST", "soft"), "o"), (("Clinical", "soft"), "s"), (("Clinical", "sharp"), "s")]:
        d = T[(T.dataset == ds) & (T.kernel == k)]
        a.scatter(d.fat_ref, d.ratio, s=4, marker=mk, color=KC[k], alpha=0.45 if ds == "NLST" else 0.7, lw=0,
                  label=f"{ds}, {k} ({len(d)})")
        q = pd.qcut(d.fat_ref, 3)
        m = d.groupby(q, observed=True)[["fat_ref", "ratio"]].median()
        a.plot(m.fat_ref, m.ratio, color=KC[k], lw=1.0, ls="-" if ds == "NLST" else (0, (3, 1.5)), marker=mk, ms=3, mfc="white")
        rows_a += d.assign(group=f"{ds} {k}")[["group", "id", "fat_ref", "ratio", "es_sd", "liv_sd"]].to_dict("records")
    a.axhline(1, color=style.HAIR, lw=0.6, zorder=0)
    a.set_xlim(0, 35); a.set_ylim(0.5, 2.0); plain(a, "x", (0, 10, 20, 30)); plain(a, "y", (0.5, 1, 1.5, 2))
    a.set_xlabel("Intramuscular fat, fixed filter (%)", fontsize=6); a.set_ylabel("Muscle noise / liver noise", fontsize=6)
    a.legend(loc="upper right", fontsize=5, markerscale=1.6, handletextpad=0.2, labelspacing=0.3)
    b = ax_mm(fig, H, 73, 66, 38, 38)
    for (ds, k), mk in [(("NLST", "soft"), "o"), (("NLST", "sharp"), "o"), (("Clinical", "soft"), "s"), (("Clinical", "sharp"), "s")]:
        d = T[(T.dataset == ds) & (T.kernel == k)]
        b.scatter(d.liv_sd, d.es_sd, s=4, marker=mk, color=KC[k], alpha=0.4, lw=0)
    b.plot([8, 500], [8, 500], color=style.HAIR, lw=0.6)
    b.set_xscale("log"); b.set_yscale("log"); plain(b, "x", (10, 30, 100, 300)); plain(b, "y", (10, 30, 100, 300))
    b.set_xlim(8, 500); b.set_ylim(8, 500)
    b.set_xlabel("Liver noise, native (HU)", fontsize=6); b.set_ylabel("Muscle noise, native (HU)", fontsize=6)
    S = pd.read_csv(SR("noise_ref_alt_summary.csv"))
    ests = [("es_sd", "Muscle SD"), ("es_mad", "Muscle MAD"), ("liv_sd", "Liver SD"), ("liv_mad", "Liver MAD")]
    rows_c = []
    for k, (estd, ttl, lim) in enumerate([("sharp - soft", "Sharp − soft (points)", (-1.0, 1.0)), ("soft change", "Soft value change (points)", (-6, 0))]):
        c = ax_mm(fig, H, 128 + k * 28, 66, 22, 38)
        for i, (e, lab) in enumerate(ests):
            for j, (v, fc) in enumerate([("inplane", C_NREF), ("slab", "white")]):
                r = S[(S.dataset == "NLST") & (S.estimator == e) & (S.variant == v) & (S.measure == "imat_pct") & (S.estimand == estd)].iloc[0]
                x = i + (j - 0.5) * 0.3
                c.errorbar(x, r.hl, yerr=[[r.hl - r.lo], [r.hi - r.hl]], fmt="o", ms=2.8, color=C_NREF, mfc=fc, capsize=0, lw=0.8)
                rows_c.append(dict(estimand=estd, estimator=e, variant=v, n=r.n, hl=r.hl, lo=r.lo, hi=r.hi))
        c.axhline(0, color=style.INK2, lw=0.5, zorder=0)
        c.set_xticks(range(4)); c.set_xticklabels([l_ for _, l_ in ests], fontsize=5, rotation=90)
        c.set_xlim(-0.6, 3.6); c.set_ylim(*lim)
        c.set_title(ttl, fontsize=5.6, pad=3)
        if k == 0:
            c.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=C_NREF, label="In-plane"),
                              Line2D([0], [0], marker="o", ls="", ms=3, color=C_NREF, mfc="white", label="5-mm slab first")],
                     loc="upper left", fontsize=5, handletextpad=0.2, labelspacing=0.3)
    # e: routine-dose TCGA/CPTAC pairs
    rows_e = []
    if os.path.exists(SR("tcga_recon_summary_perpair.csv")):
        TP = pd.read_csv(SR("tcga_recon_summary_perpair.csv"))
        for k, (cs, cl, ttl) in enumerate([("imat_soft_native", "imat_sharp_native", "Native"),
                                            (f"imat_soft_r{NREF_R}", f"imat_sharp_r{NREF_R}", f"Noise-referenced, {NREF_R} HU")]):
            e = ax_mm(fig, H, 13 + k * 46, 11, 34, 34)
            e.plot([0, 60], [0, 60], color=style.HAIR, lw=0.6)
            e.fill_between([0, 60], [-2, 58], [2, 62], color=style.WASH, lw=0, zorder=0)
            e.scatter(TP[cs], TP[cl], s=9, marker="s", facecolor=C_NREF if k else E.C_NAT, edgecolor="white", lw=0.3, zorder=3)
            lim_ = 50 if k == 0 else 30
            e.set_xlim(0, lim_); e.set_ylim(0, lim_)
            e.set_xlabel("Soft reconstruction (%)", fontsize=6); e.set_ylabel("Sharp reconstruction (%)", fontsize=6)
            e.set_title(ttl, fontsize=6, pad=3)
        rows_e = TP.to_dict("records")
    for t_, x, y in [("a", 1, 110), ("b", 62, 110), ("c", 117, 110), ("d", 1, 54)]:
        label(fig, H, x, y, t_)
    E.src(pd.DataFrame(rows_a), "FigureS8", "a_b_noise_ratio"); E.src(pd.DataFrame(rows_c), "FigureS8", "c_estimators")
    if rows_e:
        E.src(pd.DataFrame(rows_e), "FigureS8", "d_routine_dose_pairs")
    return style.save(fig, "FigureS8")


def fig7_survival():
    """Figure 7 (revision 1): survival associations per fixed unit under every processing strategy, and overall survival."""
    style.setup()
    H = 62.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    U = pd.read_csv(SR("survival_units.csv"))
    U = U[U.scale == "unit"]
    meas = [("imat", "Intramuscular fat,\nper 5 points higher"), ("window", "Muscle-window density,\nper 10 HU lower"),
            ("whole", "Whole-mask density,\nper 10 HU lower"), ("area", "Muscle area,\nper 10 cm² lower")]
    strat = [("native", "Native", E.C_NAT), ("fixed", "Both filtered (fixed)", E.C_FIX), ("sharp_only", "Sharp only", E.C_ADA),
             ("noise_ref_25", f"Noise-referenced ({NREF_R} HU)", C_NREF)]
    pops = [("primary_121", "Primary population (121 patients, 44 deaths)", True), ("pooled_195", "Pooled, aged ≥65 years (195, 66)", False)]
    a = ax_mm(fig, H, 13, 11, 104, 40)
    rows = []
    for i, (m, _) in enumerate(meas):
        for j, (p, _, lab, col) in enumerate([(s_[0], None, s_[1], s_[2]) for s_ in strat]):
            for k, (pop, _, filled) in enumerate(pops):
                r = U[(U.population == pop) & (U.exposure == m) & (U.processing == p)].iloc[0]
                x = i + (j - 1.5) * 0.19 + (k - 0.5) * 0.08
                a.plot([x, x], [r.lo, r.hi], color=col, lw=0.8)
                a.scatter(x, r.HR, s=10, color=col if filled else "white", edgecolor=col, lw=0.7, zorder=3)
                rows.append(dict(exposure=m, processing=p, population=pop, n=r.n, events=r.events, HR=r.HR, lo=r.lo, hi=r.hi))
    a.axhline(1, color=style.HAIR, lw=0.6, zorder=0)
    a.set_yscale("log"); plain(a, "y", (0.75, 1, 1.5, 2, 3)); a.set_ylim(0.7, 2.6)
    a.set_xticks(range(len(meas))); a.set_xticklabels([m_[1] for m_ in meas], fontsize=5.4)
    a.set_xlim(-0.5, len(meas) - 0.5)
    a.set_ylabel("Adjusted HR (95% CI)", fontsize=6)
    a.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=c_, label=l_) for _, l_, c_ in strat]
             + [Line2D([0], [0], marker="o", ls="", ms=3, color=style.INK2, label="Primary"),
                Line2D([0], [0], marker="o", ls="", ms=3, color=style.INK2, mfc="white", label="Pooled, ≥65 years")],
             loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=5, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)
    # b: Kaplan-Meier, primary population
    b = ax_mm(fig, H, 133, 19, 46, 32)
    rt_ax = ax_mm(fig, H, 133, 6, 46, 5)
    km = pd.read_csv(S.RR("km_primary121.csv"))
    rt = pd.read_csv(S.RR("km_primary121_risktable.csv"))
    tt = np.r_[0, km.time_years]
    b.fill_between(tt, np.r_[1, km.lower], np.r_[1, km.upper], step="post", color=style.INK, alpha=0.12, lw=0)
    b.step(tt, np.r_[1, km.surv], where="post", color=style.INK, lw=1.0)
    cen = km[km.n_censor > 0]
    b.scatter(cen.time_years, cen.surv, marker="|", s=8, color=style.INK, lw=0.5)
    b.set_xlim(0, 6); b.set_ylim(0, 1.02); b.set_xticks(range(7)); plain(b, "y", (0, 0.25, 0.5, 0.75, 1))
    b.set_xlabel("Years after resection", fontsize=6); b.set_ylabel("Overall survival", fontsize=6)
    rt_ax.set_xlim(0, 6); rt_ax.set_ylim(0, 1); rt_ax.set_axis_off()
    for _, r in rt.iterrows():
        rt_ax.text(r.time_years, 0.3, f"{int(r.n_risk)}", fontsize=5, ha="center", va="center")
    rt_ax.text(-0.05, 1.05, "Number at risk", fontsize=5, ha="left", va="bottom", color=style.INK2)
    for t_, x, y in [("a", 1, 60), ("b", 122, 60)]:
        label(fig, H, x, y, t_)
    E.src(pd.DataFrame(rows), "Figure7", "a_hr_per_unit"); E.src(km, "Figure7", "b_km"); E.src(rt, "Figure7", "b_risktable")
    return style.save(fig, "Figure7")


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
        for key, fn in (("E1", fig1_flow), ("D1", fig1_pipeline), ("F2", fig2_pairs), ("F3", fig3), ("F5", fig5_noise_ref),
                        ("F6", fig6_seg), ("F7", fig7_survival), ("G8", figS8_noise)):
            style.OUT = os.path.join(PARTS, key)
            print(key, fn())
    finally:
        style.panel_label, style.save, style.OUT = orig_label, orig_save, SREP
        unpalette(saved)
        restore_text()


def compose():
    import json
    import subprocess
    spec = {name: [(os.path.join(FIGS, p), *rest) for p, *rest in parts] for name, parts in LAYOUT.items()}
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
