# ER revision: copy of figures/r1005_figures.py with labels aligned to the ER manuscript terminology
# ("fixed filter" for the published noise-matching filter; "Slab + Gauss." for the combined step). Output: figures/er_supp/
"""Figures for the 1005 revision. All data from results/r1005/* (and the measurement tables they derive from).
Writes to figures/r1005/ only (the earlier figure set in figures/out/ is left untouched).

Figure1   cohorts, analysis roles and the measurement pipeline
Figure2   why native threshold measures depend on the reconstruction kernel: the noise mechanism
Figure3   same-acquisition reconstructions and separate examinations
Figure4   survival associations and their robustness to processing
FigureS5  digital reference (50 realisations) and measurement-slice offset
FigureS6  the extreme repeat-examination case R01-036

Usage: env/bin/python figures/r1005_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
import style  # noqa: E402

style.OUT = os.path.join(style.ROOT, "figures", "er_supp")
R = lambda *p: os.path.join(style.ROOT, *p)
RR = lambda f: R("results", "r1005", f)
W = 183.0
COND = ["native", "slab", "gauss", "both"]
COND_LAB = {"native": "Native", "slab": "5-mm slab", "gauss": "Gaussian", "both": "Slab + Gaussian"}
COND_TICK = {"native": "Native", "slab": "Slab", "gauss": "Gauss.", "both": "Slab +\nGauss."}
COND_COL = {"native": style.MUTED, "slab": "#9fb7d4", "gauss": style.SERIES[0], "both": style.INK}
KCOL = {"sharp": style.SERIES[1], "soft": style.SERIES[0]}
KLAB = {"sharp": "Sharp kernel", "soft": "Soft kernel"}
C_SAFIX = style.SERIES[0]  # Figure 3e same-acquisition pairs after the fixed filter
C_KM = style.SERIES[0]  # Figure 4a Kaplan-Meier curve
F2F_XMAX = 2.2  # Figure 2f x-axis limit (noise correlation length, mm)


def ax_mm(fig, H, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def label(fig, H, x, y, t):
    style.panel_label(None, t, x / W, y / H, fig=fig)


def plain_log(ax, axis="x", ticks=(0.5, 1, 2, 4)):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(NullLocator())


def src(df, fig, panel):
    style.source_data(df, fig, panel)


# =================================================================== Figure 1
def fig1(case_id="R01-049"):
    import fig1_design_pipeline as F
    from bodycomp_metrics import Case, vertebra_info

    fat_cols = {style.TISSUE["sat"], style.TISSUE["ift"], style.TISSUE["imat"]}
    paint0 = F.paint

    def paint(gray, layers, alpha=0.65):
        # fat compartments drawn translucent so that the underlying fat-window voxels stay visible
        rgb = np.repeat(gray[..., None], 3, axis=2)
        for m, col in layers:
            a = min(alpha, 0.42) if col in fat_cols else alpha
            rgb[m] = (1 - a) * rgb[m] + a * F.hex_rgb(col)
        return rgb

    F.paint = paint
    F.T = lambda s: {"Harmonised": "Fixed filter", "harmonised": "fixed filter"}.get(s, s)

    def panel_overview(fig, A):
        pc = pd.read_csv(R("results", "public_pooled", "cohort_description.csv")).set_index("cohort")
        tc_deaths = int(pc.loc[["CPTAC", "TCGA-LUSC", "TCGA-LUAD"], "deaths"].sum())
        n_tc = int(pc.loc[["CPTAC", "TCGA-LUSC", "TCGA-LUAD"], "n"].sum())
        ver = pd.read_csv(RR("recon_pairs_dicom.csv"))
        cards = [
            ("NSCLC-Radiogenomics · measurement", "Staging CT, soft and sharp reconstructions",
             "202", "staging CTs measured",
             ["49 separate repeat CTs", f"{int(ver.same_acquisition.sum())} same-acquisition pairs"],
             [("Noise model", True), ("Kernel experiment", True), ("Repeat agreement", True)]),
            ("NSCLC-Radiogenomics · survival", "CT 0–180 d before resection, date-concordant",
             "121", "aged ≥65 y · 44 deaths",
             ["160 all ages", "130 interval-only (sensitivity)"],
             [("Primary analysis", True), ("Processing robustness", True)]),
            ("TCGA-LUAD/LUSC · CPTAC", "Diagnostic CT; CT-to-surgery interval unknown",
             f"{n_tc}", f"all ages · {tc_deaths} deaths",
             ["pooled 274, 195 aged ≥65 y", "3 collections"],
             [("Pooled secondary", True), ("Leave-one-cohort-out", True)]),
        ]
        rows = []
        rend = fig.canvas.get_renderer()
        CH = 25.0
        for i, (name, ct, big, what, sub, tags) in enumerate(cards):
            ax = A(3 + i * 60.3, 115.5, 57.5, CH)
            F.image_ax(ax); ax.set_xlim(0, 57.5); ax.set_ylim(0, CH)
            ax.add_patch(FancyBboxPatch((0, 0), 57.5, CH, boxstyle="round,pad=0,rounding_size=1.6", fc="#f4f3ee",
                                        ec="none", zorder=0))
            ax.text(2.4, CH - 2.2, name, fontsize=6.3, fontweight="bold", va="top", color=style.INK)
            ax.text(2.4, CH - 5.6, ct, fontsize=5.2, va="top", color=style.INK2)
            tb = ax.text(2.4, 9.6, big, fontsize=15, fontweight="bold", va="baseline", color=style.INK)
            wbig = F.w_data(ax, tb, rend)
            ax.text(2.4 + wbig + 1.6, 13.2, what, fontsize=5.2, va="baseline", color=style.INK)
            ax.text(2.4 + wbig + 1.6, 9.8, "  ·  ".join(sub), fontsize=5.0, va="baseline", color=style.INK2)
            x = 2.4
            for t, done in tags:
                tt = ax.text(x + 1.3, 3.9, t, fontsize=4.8, ha="left", va="center", color=style.INK, zorder=3)
                w = F.w_data(ax, tt, rend) + 2.6
                ax.add_patch(FancyBboxPatch((x, 1.6), w, 4.6, boxstyle="round,pad=0,rounding_size=2.2",
                                            fc="#e6e5dd", ec="none", zorder=2))
                x += w + 1.0
            rows.append(dict(cohort=name, ct=ct, n=big, details="; ".join([what] + sub),
                             uses="; ".join(t for t, _ in tags)))
        return ax, pd.DataFrame(rows)

    style.setup()
    case = Case(R("data", "nifti", "public_rg", f"{case_id}.nii.gz"), R("data", "seg", "public_rg", case_id))
    info = {lv: vertebra_info(case.segs["total"], lv)[0] for lv in ["T4", "T5", "T8", "T12"]}
    Wf, H = 183.0, 143.5
    fig = plt.figure(figsize=(Wf * style.MM, H * style.MM))
    A = lambda *a: F.mm_ax(fig, *a, W=Wf, H=H)
    ax_a, sa = panel_overview(fig, A)
    F.panel_pipeline(fig, A, case, info)
    ax_c, sc = F.panel_3d(fig, A, case_id)
    dax = [A(101 + i * 27.3, 42, 26, 21) for i in range(3)]
    sd = F.panel_d_axes(fig, dax, case, ["T4", "T8", "T12"], legend=False)
    ax_e = A(111, 8.5, 69, 22)
    prof = F.muscle_profile(case, info)
    F.panel_e(ax_e, case, info, prof)
    for t, x, y in [("a", 1, 143), ("b", 1, 112), ("c", 1, 70.5), ("d", 99, 70.5), ("e", 99, 35)]:
        style.panel_label(None, t, x / Wf, y / H, fig=fig)
    src(sa, "Figure1", "a_cohorts"); src(sc, "Figure1", "c_levels"); src(sd, "Figure1", "d_compartments")
    src(prof, "Figure1", "e_muscle_profile")
    F.paint = paint0
    return style.save(fig, "Figure1")


# =================================================================== Figure 2
def fig2(noise_case="R01-064"):
    import fig1_design_pipeline as F
    from bodycomp_metrics import Case, vertebra_info
    F.T = lambda s: {"Harmonised": "Fixed filter", "harmonised": "fixed filter"}.get(s, s)
    style.setup()
    H = 122.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    nm = pd.read_csv(RR("noise_model_per_scan.csv"))
    nm = nm[nm.sigma_source == "residual"].dropna(subset=["sigma_hu", "imat_observed"])
    summ = pd.read_csv(RR("noise_model_summary.csv"))
    summ = summ[summ.sigma_source == "residual"].set_index(["condition", "stratum"])
    th = pd.read_csv(RR("noise_model_theory.csv"))

    # a, b: native vs noise-matched image and liver histogram (example sharp-kernel scan)
    a1, a2 = ax_mm(fig, H, 3, 79, 30, 30), ax_mm(fig, H, 34, 79, 30, 30)
    bh = ax_mm(fig, H, 75, 82, 36, 26)
    case = Case(R("data", "nifti", "public_rg", f"{noise_case}.nii.gz"), R("data", "seg", "public_rg", noise_case))
    zn = int(round(vertebra_info(case.segs["total"], "T12")[0]))
    sa, sb = F.panel_b_axes(a1, a2, bh, case, zn)
    a1.text(0.0, -0.06, "orange: voxels in the fat window (−190 to −30 HU)", transform=a1.transAxes, fontsize=5,
            va="top", color=style.INK2)

    # c: native fat fraction tracks image noise as the noise-only model predicts
    c = ax_mm(fig, H, 128, 82, 50, 28)
    t25 = th[th.mu == 25].sort_values("sigma")
    t45 = th[th.mu == 45].sort_values("sigma")
    c.fill_between(t25.sigma, t45.imat_noise_only, t25.imat_noise_only, color=style.WASH, lw=0, zorder=0)
    c.plot(t25.sigma, t25.imat_noise_only, color=style.INK, lw=0.9, zorder=1)
    for k in ["sharp", "soft"]:
        s = nm[(nm.condition == "native") & (nm.kernel_type == k)]
        c.scatter(s.sigma_hu, s.imat_observed, s=5, facecolor="none", edgecolor=KCOL[k], lw=0.5, zorder=2)
    s = nm[nm.condition == "both"]
    c.scatter(s.sigma_hu, s.imat_observed, s=3, color=style.INK, lw=0, zorder=3)
    c.set_xscale("log"); plain_log(c, "x", (5, 10, 20, 50, 100))
    c.set_xlim(4, 160); c.set_ylim(0, 45)
    c.set_xlabel("Image noise SD (HU)", fontsize=6)
    c.set_ylabel("Intramuscular fat (%)", fontsize=6)
    c.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=2.8, mfc="none", mec=KCOL["sharp"], label="Native, sharp"),
                      Line2D([0], [0], marker="o", ls="", ms=2.8, mfc="none", mec=KCOL["soft"], label="Native, soft"),
                      Line2D([0], [0], marker="o", ls="", ms=2.4, color=style.INK, label="Fixed filter"),
                      Line2D([0], [0], color=style.INK, lw=0.9, label="Noise only, μ = 25–45 HU")],
             loc="upper left", fontsize=4.8, handletextpad=0.3, labelspacing=0.25, handlelength=1.2)

    # d: the muscle-window mean is biased upwards by truncation, as predicted
    d = ax_mm(fig, H, 14, 13, 44, 44)
    nat = nm[nm.condition == "native"]
    for k in ["sharp", "soft"]:
        s = nat[nat.kernel_type == k]
        d.scatter(s.window_bias_noise_only, s.window_bias_observed, s=5, facecolor="none", edgecolor=KCOL[k], lw=0.5)
    d.plot([0, 35], [0, 35], color=style.HAIR, lw=0.6, zorder=0)
    d.set_xlim(0, 35); d.set_ylim(0, 35)
    d.set_xlabel("Predicted by the noise model (HU)", fontsize=6)
    d.set_ylabel("Observed window − whole-mask mean (HU)", fontsize=6)
    r_all = summ.loc[("native", "all"), "window_bias_pearson_r"]
    r_fix = summ.loc[("native", "all"), "window_bias_pearson_r_mu_fixed"]
    d.text(0.03, 0.97, f"Native images, n = {len(nat)}\n$\\mathit{{r}}$ = {r_all:.2f} "
                       f"(noise term alone {r_fix:.2f})", transform=d.transAxes, fontsize=5.2, va="top")
    d.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=2.8, mfc="none", mec=KCOL[k], label=KLAB[k])
                      for k in ["sharp", "soft"]], loc="lower right", fontsize=4.8, handletextpad=0.1)
    d.set_title("Truncation bias of the window mean", fontsize=6, pad=3)

    # e: processing effect by kernel (paired, development sample)
    e = ax_mm(fig, H, 75, 13, 44, 44)
    e2 = pd.read_csv(R("results", "audit", "e2_ablation_public_rg.csv"))
    sp = pd.read_csv(R("results", "audit", "splits.csv"))
    e2 = e2[e2.case_id.isin(sp[sp.split == "development"].case_id)]
    ser = pd.read_csv(R("data", "public_rg", "series_candidates.csv"))
    ser = ser[ser.chosen == True]
    kt = pd.DataFrame({"case_id": ser.PatientID, "kernel_type": np.where(
        ser.kernel.fillna("").str.contains("LUNG|BONE|B4|B5|B6|B7|I70|SHARP|YB|YA", case=False), "sharp", "soft")})
    e2 = e2.merge(kt, on="case_id")
    rows = []
    for j, k in enumerate(["sharp", "soft"]):
        s = e2[e2.kernel_type == k]
        x0 = j * 2.2
        for _, r in s.iterrows():
            e.plot([x0, x0 + 1], [r.native_imat_pct, r.both_imat_pct], color=KCOL[k], alpha=0.18, lw=0.5)
        mn, mb = s.native_imat_pct.median(), s.both_imat_pct.median()
        e.plot([x0, x0 + 1], [mn, mb], color=KCOL[k], lw=1.6, marker="o", ms=3, zorder=3)
        e.text(x0 - 0.12, mn, f"{mn:.1f}", fontsize=5.2, ha="right", va="center", color=KCOL[k])
        e.text(x0 + 1.12, mb, f"{mb:.1f}", fontsize=5.2, ha="left", va="center", color=KCOL[k])
        e.text(x0 + 0.5, 47, f"{KLAB[k]}\nn = {len(s)}", fontsize=5.2, ha="center", va="top", color=KCOL[k])
        rows.append(dict(kernel=k, n=len(s), native_median=mn, noise_matched_median=mb,
                         median_paired_change=(s.both_imat_pct - s.native_imat_pct).median()))
    e.set_xticks([0, 1, 2.2, 3.2]); e.set_xticklabels(["Native", "Fixed\nfilter"] * 2, fontsize=5.2)
    e.set_xlim(-0.7, 3.9); e.set_ylim(0, 48)
    e.set_ylabel("T12 intramuscular fat (%)", fontsize=6)
    e.set_title("Processing changes sharp-kernel scans", fontsize=6, pad=3)

    # f: amplitude vs correlation length
    f = ax_mm(fig, H, 135, 13, 44, 44)
    n = pd.read_csv(R("results", "audit", "e4_noise_fov.csv"))
    n = n[n.region.isin(["liver", "spleen", "subcutaneous_fat"])].merge(kt, on="case_id", how="left")
    for img, filled in [("native", False), ("harmonised", True)]:
        for k in ["sharp", "soft"]:
            s = n[(n.image == img) & (n.kernel_type == k)]
            f.scatter(s.corr_length_mm, s.residual_sd, s=3, facecolor=KCOL[k] if filled else "none",
                      edgecolor=KCOL[k], lw=0.4, alpha=0.6, rasterized=True)
    f.set_yscale("log"); plain_log(f, "y", (5, 10, 20, 50, 100))
    f.set_xlim(0, F2F_XMAX)
    f.set_xlabel("Noise correlation length (mm)", fontsize=6)
    f.set_ylabel("Residual noise SD (HU)", fontsize=6)
    f.set_title("Smoothing lowers amplitude, lengthens correlation", fontsize=6, pad=3)
    f.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=2.8, mfc="none", mec=style.INK, label="Native"),
                      Line2D([0], [0], marker="o", ls="", ms=2.8, color=style.INK, label="Fixed filter")],
             loc="upper right", fontsize=4.8, handletextpad=0.1)
    for t, x, y in [("a", 1, 120), ("b", 66, 120), ("c", 117, 120), ("d", 1, 64), ("e", 63, 64), ("f", 123, 64)]:
        label(fig, H, x, y, t)
    src(sa, "Figure2", "a_noise"); src(sb, "Figure2", "b_histogram")
    src(nm, "Figure2", "c_d_noise_model_per_scan"); src(th, "Figure2", "c_theory")
    src(pd.DataFrame(rows), "Figure2", "e_by_kernel"); src(n, "Figure2", "f_noise")
    return style.save(fig, "Figure2")


# =================================================================== Figure 3
SA_MEAS = [("T12 intramuscular fat (%)", "Intramuscular fat", "percentage points"),
           ("T12 muscle-window density (HU)", "Window density", "HU"),
           ("T12 whole-mask density (HU)", "Whole-mask density", "HU"),
           ("T12 muscle area (cm²)", "Muscle area", "cm²")]


def fig3():
    style.setup()
    H = 118.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))

    # a: one helix, two reconstructions
    a = ax_mm(fig, H, 13, 70, 42, 38)
    ex = pd.read_csv(RR("recon_example_slice_times.csv"))
    t0 = ex.acq_time_s.min()
    for i, (s, g) in enumerate(ex.groupby("series", sort=False)):
        big = "primary" in s
        a.scatter(g.z_mm, g.acq_time_s - t0, s=9 if big else 1.5, facecolor="none" if big else KCOL["sharp"],
                  edgecolor=KCOL["soft"] if big else KCOL["sharp"], lw=0.6 if big else 0, zorder=3 - i,
                  label=s.replace("primary", "Soft").replace("alternative", "Sharp")
                  if "B31" in s or "B45" in s else s)
    a.set_xlabel("Table position (mm)", fontsize=6)
    a.set_ylabel("Slice acquisition time (s)", fontsize=6)
    a.legend(loc="upper right", fontsize=4.8, handletextpad=0.1, markerscale=1.2)
    ver = pd.read_csv(RR("recon_pairs_dicom.csv"))
    a.text(0.03, 0.03, f"{int(ver.same_acquisition.sum())}/{len(ver)} pairs: same frame of reference,\n"
                       f"acquisition number and slice timing\n(median Δt ≤ {ver.median_time_diff_s.max():.2f} s)",
           transform=a.transAxes, fontsize=4.8, ha="left", va="bottom", color=style.INK2)
    a.set_title("One acquisition, two reconstructions", fontsize=6, pad=3)

    # b: sharp minus soft within the same acquisition
    kc = pd.read_csv(RR("recon_kernel_contrasts.csv"))
    ks = pd.read_csv(RR("recon_kernel_summary.csv"))
    kc = kc[kc.sharp_minus_soft.notna()]
    rows = []
    for i, (meas, ttl, unit) in enumerate(SA_MEAS):
        b = ax_mm(fig, H, 75 + i * 27, 70, 19, 38)
        w = kc[kc.measure == meas].pivot_table(index="alt_id", columns="condition", values="sharp_minus_soft")
        for _, r in w.iterrows():
            b.plot([0, 1], [r.native, r.both], color=style.HAIR, lw=0.5, zorder=1)
        b.scatter(np.zeros(len(w)), w.native, s=5, facecolor="none", edgecolor=style.MUTED, lw=0.5, zorder=2)
        b.scatter(np.ones(len(w)), w.both, s=5, color=style.INK, lw=0, zorder=2)
        for j, cnd in enumerate(["native", "both"]):
            s = ks[(ks.measure == meas) & (ks.condition == cnd)].iloc[0]
            b.errorbar(j + 0.28, s.median_sharp_minus_soft, yerr=[[s.median_sharp_minus_soft - s.ci_lo],
                                                                  [s.ci_hi - s.median_sharp_minus_soft]],
                       fmt="o", ms=3, color=style.TISSUE["muscle"], lw=0.9, capsize=0, zorder=4)
            rows.append(dict(measure=meas, condition=cnd, pairs=int(s.pairs), median=s.median_sharp_minus_soft,
                             ci_lo=s.ci_lo, ci_hi=s.ci_hi))
        b.axhline(0, color=style.INK2, lw=0.5, zorder=0)
        b.set_xticks([0, 1]); b.set_xticklabels(["Native", "Fixed\nfilter"], fontsize=5)
        b.set_xlim(-0.35, 1.55)
        b.set_title(ttl, fontsize=5.8, pad=3)
        b.set_ylabel(f"Sharp − soft ({unit})", fontsize=5.5, labelpad=1)
    fig.text(75 / W, 113 / H, f"Same acquisition, sharp minus soft reconstruction ({int(w.shape[0])} pairs; "
             "red: median, 95% CI)", fontsize=6, color=style.INK)

    # c: repeat-examination ICC by processing condition
    t2 = pd.read_csv(RR("table2_by_kernel.csv"))
    t2 = t2[t2.stratum == "all"].set_index("measure")
    cm = [("T12 muscle area (cm²)", "Muscle\narea"), ("T12 muscle-window density (HU)", "Window\ndensity"),
          ("T12 whole-mask density (HU)", "Whole-mask\ndensity"), ("T12 intramuscular fat (%)", "Intramusc.\nfat %"),
          ("T5–T12 muscle volume (cm³)", "Muscle\nvolume")]
    c = ax_mm(fig, H, 13, 12, 52, 38)
    for i, (m, lab) in enumerate(cm):
        for j, cnd in enumerate(COND):
            c.scatter(i + (j - 1.5) * 0.17, t2.loc[m, f"icc_{cnd}"], s=12, color=COND_COL[cnd], zorder=3,
                      edgecolor="white", lw=0.4)
    c.axhspan(0.9, 1.0, color=style.WASH, lw=0, zorder=0)
    c.set_xticks(range(len(cm))); c.set_xticklabels([l for _, l in cm], fontsize=5)
    c.set_xlim(-0.6, len(cm) - 0.4); c.set_ylim(0, 1.02)
    c.set_ylabel("ICC, 49 separate repeat CTs", fontsize=6)
    c.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.2, color=COND_COL[k], label=COND_LAB[k]) for k in COND],
             loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=4.8, handletextpad=0.1, columnspacing=0.6)

    # d: Bland-Altman, noise-matched T12 muscle area, extreme case marked
    d = ax_mm(fig, H, 80, 12, 40, 38)
    pr = pd.read_csv(RR("repeat_pairs_area_volume.csv"))
    ag = pd.read_csv(RR("repeat_agreement_outlier.csv"))
    ag = ag[(ag.measure == "T12 muscle area (cm²)") & (ag.condition == "both")]
    mean_ = (pr.both_sma_cm2_1 + pr.both_sma_cm2_2) / 2
    diff = pr.both_sma_cm2_2 - pr.both_sma_cm2_1
    out = pr.case_id == "R01-036"
    d.scatter(mean_[~out], diff[~out], s=5, color=style.INK, lw=0)
    d.scatter(mean_[out], diff[out], s=14, facecolor="none", edgecolor=style.TISSUE["muscle"], lw=0.8)
    d.text(mean_[out].iloc[0], diff[out].iloc[0] + 3, "R01-036", fontsize=5, ha="center", va="bottom",
           color=style.TISSUE["muscle"])
    for _, r in ag.iterrows():
        ls = (0, (3, 2)) if pd.isna(r.excluding) or r.excluding == "" else (0, (1, 1.5))
        col = style.MUTED if ls == (0, (3, 2)) else style.INK
        for v in (r.loa_lo, r.loa_hi):
            d.axhline(v, color=col, lw=0.7, ls=ls)
    d.axhline(0, color=style.HAIR, lw=0.5)
    all_ = ag[ag.excluding.isna() | (ag.excluding == "")].iloc[0]
    ex_ = ag[ag.excluding == "R01-036"].iloc[0]
    d.text(0.98, 0.98, f"all 49: LoA {style.fnum(all_.loa_lo, '{:.1f}')} to {all_.loa_hi:.1f}, ICC {all_.icc:.3f}\n"
                       f"without R01-036: {style.fnum(ex_.loa_lo, '{:.1f}')} to {ex_.loa_hi:.1f}, ICC {ex_.icc:.3f}",
           transform=d.transAxes, fontsize=4.8, ha="right", va="top")
    d.set_xlabel("Mean of two CTs (cm²)", fontsize=6)
    d.set_ylabel("CT 2 − CT 1, T12 muscle area (cm²)", fontsize=6)
    d.set_ylim(-40, 25)

    # e: magnitude of differences: same acquisition vs separate examinations
    e = ax_mm(fig, H, 143, 12, 37, 38)
    e2 = pd.read_csv(R("results", "audit", "e2_ablation_public_rg.csv"))
    s2 = pd.read_csv(R("results", "audit", "e2_ablation_public_rg_scan2.csv"))
    pp = e2.merge(s2, on="case_id", suffixes=("_1", "_2"))
    pp = pp[(pp.vol_available_1 == 1) & (pp.vol_available_2 == 1)]
    kcall = pd.read_csv(RR("recon_kernel_contrasts.csv"))
    keymap = {"T12 intramuscular fat (%)": "imat_pct", "T12 muscle-window density (HU)": "smd_window_hu",
              "T12 whole-mask density (HU)": "smd_all_hu", "T12 muscle area (cm²)": "sma_cm2"}
    rows_e = []
    for i, (meas, ttl, unit) in enumerate(SA_MEAS):
        y = len(SA_MEAS) - 1 - i
        k = keymap[meas]
        sa_nat = kcall[(kcall.measure == meas) & (kcall.condition == "native")].abs_diff.median()
        sa_nm = kcall[(kcall.measure == meas) & (kcall.condition == "both")].abs_diff.median()
        se_nm = (pp[f"both_{k}_2"] - pp[f"both_{k}_1"]).abs().median()
        e.plot([sa_nm, se_nm], [y, y], color=style.HAIR, lw=0.8, zorder=1)
        e.scatter(sa_nat, y, s=11, facecolor="none", edgecolor=style.MUTED, lw=0.7, zorder=2)
        e.scatter(sa_nm, y, s=11, color=C_SAFIX, zorder=3)
        e.scatter(se_nm, y, s=11, facecolor="white", edgecolor=style.INK, lw=0.8, zorder=3)
        rows_e.append(dict(measure=meas, same_acq_native=sa_nat, same_acq_noise_matched=sa_nm,
                           separate_ct_noise_matched=se_nm))
    e.set_xscale("log"); plain_log(e, "x", (0.3, 1, 3, 10))
    e.set_xlim(0.2, 15)
    e.set_yticks(range(len(SA_MEAS))); e.set_yticklabels([t for _, t, _ in SA_MEAS][::-1], fontsize=5.2)
    e.set_xlabel("Median absolute difference", fontsize=6)
    e.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, mfc="none", mec=style.MUTED, label="Same acq., native"),
                      Line2D([0], [0], marker="o", ls="", ms=3, color=C_SAFIX, label="Same acq., fixed filter"),
                      Line2D([0], [0], marker="o", ls="", ms=3, mfc="white", mec=style.INK, label="Separate CTs, fixed filter")],
             loc="lower left", bbox_to_anchor=(-0.9, 1.0), fontsize=4.8, handletextpad=0.1, labelspacing=0.2)
    for t, x, y in [("a", 1, 116), ("b", 66, 116), ("c", 1, 61), ("d", 68, 61), ("e", 122, 61)]:
        label(fig, H, x, y, t)
    src(ex, "Figure3", "a_slice_times"); src(ver, "Figure3", "a_pairs_dicom"); src(pd.DataFrame(rows), "Figure3", "b_summary")
    src(kc, "Figure3", "b_pairs"); src(t2.reset_index(), "Figure3", "c_icc"); src(pr, "Figure3", "d_pairs")
    src(pd.DataFrame(rows_e), "Figure3", "e")
    return style.save(fig, "Figure3")


# =================================================================== Figure 4
FOREST = [("esm_area", "Paraspinal area", "lower"), ("esm_hu", "Paraspinal density", "lower"),
          ("smd3d", "T5–T12 window density", "lower"), ("imat_pct", "Intramuscular fat %", "higher"),
          ("smd_all", "Whole-mask density*", "lower"), ("bmd", "Trabecular attenuation", "lower"),
          ("vat", "Internal fat area", "higher"), ("sat", "Subcutaneous fat area", "higher")]
POPS = [("primary_121", "Primary, date-concordant (121/44)", style.INK, "o", True),
        ("interval_only_130", "Interval-only sensitivity (130/46)", style.INK, "o", False),
        ("pooled_195", "Pooled four cohorts (195/66)", style.MUTED, "s", True)]


def fig4():
    style.setup()
    H = 112.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    # a: KM
    a = ax_mm(fig, H, 13, 72, 44, 30)
    rta = ax_mm(fig, H, 13, 57, 44, 5)
    km = pd.read_csv(RR("km_primary121.csv"))
    rt = pd.read_csv(RR("km_primary121_risktable.csv"))
    t = np.r_[0, km.time_years]
    a.fill_between(t, np.r_[1, km.lower], np.r_[1, km.upper], step="post", color=C_KM, alpha=0.15, lw=0)
    a.step(t, np.r_[1, km.surv], where="post", color=C_KM, lw=1.0)
    cen = km[km.n_censor > 0]
    a.scatter(cen.time_years, cen.surv, marker="|", s=8, color=C_KM, lw=0.5)
    a.set_xlim(0, 6); a.set_ylim(0, 1.02); a.set_xticks(range(7)); a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a.set_xlabel("Years after resection", fontsize=6); a.set_ylabel("Overall survival", fontsize=6)
    a.set_title("Primary population, 121 patients, 44 deaths", fontsize=6, pad=3)
    rta.set_xlim(0, 6); rta.set_ylim(0, 1); rta.set_axis_off()
    for _, r in rt.iterrows():
        rta.text(r.time_years, 0.3, f"{int(r.n_risk)}", fontsize=5, ha="center", va="center")
    rta.text(-0.05, 1.05, "Number at risk", fontsize=5, ha="left", va="bottom", color=style.INK2)

    # b: forest
    t4 = pd.read_csv(RR("table4_reproduced.csv"))
    b = ax_mm(fig, H, 104, 56, 52, 41)
    rows = []
    for i, (v, lab, arr) in enumerate(FOREST):
        y = len(FOREST) - 1 - i
        for j, (p, plab, col, mk, filled) in enumerate(POPS):
            r = t4[(t4.population == p) & (t4.exposure == v)].iloc[0]
            yy = y + 0.22 - j * 0.22
            b.plot([r.lo, r.hi], [yy, yy], color=col, lw=0.8)
            b.scatter(r.HR, yy, s=11, marker=mk, facecolor=col if filled else "white", edgecolor=col, lw=0.7, zorder=3)
            rows.append(dict(exposure=lab, population=p, n=r.n, events=r.events, HR=r.HR, lo=r.lo, hi=r.hi, P=r.P,
                             q=r.q))
    b.axvline(1, color=style.HAIR, lw=0.6, zorder=0)
    b.set_xscale("log"); plain_log(b, "x", (0.5, 1, 2, 4)); b.set_xlim(0.45, 4.5)
    b.set_yticks(range(len(FOREST))); b.set_yticklabels([f"{l}, {a_}" for _, l, a_ in FOREST][::-1], fontsize=5.4)
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.set_ylim(-0.6, len(FOREST) - 0.4)
    b.set_xlabel("Adjusted HR per SD in the stated direction (95% CI)", fontsize=6)
    b.legend(handles=[Line2D([0], [0], marker=mk, ls="-", lw=0.8, ms=3.2, color=col,
                             mfc=col if filled else "white", label=pl) for _, pl, col, mk, filled in POPS],
             loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=4.8, handletextpad=0.3, labelspacing=0.2,
             borderaxespad=0.1)

    # c: robustness to processing, primary population, by kernel
    hs = pd.read_csv(RR("hr_by_processing_condition.csv"))
    hs = hs[hs.population == "primary_121"]
    cexp = [("imat_pct", "Intramuscular fat %, higher"), ("smd_window_hu", "Muscle-window density, lower"),
            ("smd_all_hu", "Whole-mask density, lower"), ("sma_cm2", "T12 muscle area, lower")]
    strata = [("all", "All (121/44)", style.INK), ("sharp", "Sharp kernel (74/25)", KCOL["sharp"]),
              ("soft", "Soft kernel (47/19)", KCOL["soft"])]
    for i, (v, ttl) in enumerate(cexp):
        c = ax_mm(fig, H, 13 + i * 43.5, 9, 35, 32)
        for j, (k, kl, col) in enumerate(strata):
            s = hs[(hs.exposure == v) & (hs.kernel == k)].set_index("condition").loc[COND]
            x = np.arange(4) + (j - 1) * 0.17
            c.errorbar(x, s.HR, yerr=[s.HR - s.lo, s.hi - s.HR], fmt="o-", ms=2.6, lw=0.8, color=col, capsize=0,
                       elinewidth=0.6, mfc=col if k != "soft" else "white")
        c.axhline(1, color=style.HAIR, lw=0.6, zorder=0)
        c.set_yscale("log"); plain_log(c, "y", (0.5, 1, 2, 4)); c.set_ylim(0.4, 5.5)
        c.set_xticks(range(4)); c.set_xticklabels([COND_TICK[k] for k in COND], fontsize=5)
        c.set_xlim(-0.5, 3.5)
        c.set_title(ttl, fontsize=6, pad=3)
        if i == 0:
            c.set_ylabel("HR per SD (95% CI)", fontsize=6)
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="-", lw=0.8, ms=3, color=col,
                               mfc=col if k != "soft" else "white", label=kl) for k, kl, col in strata],
               loc="upper left", bbox_to_anchor=(13 / W, 50 / H), ncol=3, fontsize=5, handletextpad=0.3,
               columnspacing=1.0, frameon=False)
    for t_, x, y in [("a", 1, 109), ("b", 66, 109), ("c", 1, 51)]:
        label(fig, H, x, y, t_)
    src(km, "Figure4", "a_km"); src(rt, "Figure4", "a_risktable"); src(pd.DataFrame(rows), "Figure4", "b_forest")
    src(hs, "Figure4", "c_processing")
    return style.save(fig, "Figure4")


# =================================================================== Supplementary
def figS5():
    from fig_revision import panel_f
    style.setup()
    H = 62.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ps = pd.read_csv(RR("phantom_summary.csv"))
    a = ax_mm(fig, H, 13, 11, 45, 42)
    b = ax_mm(fig, H, 75, 11, 45, 42)
    for cnd in COND:
        for corr, ls in [(0.0, "-"), (1.5, (0, (2, 1.5)))]:
            s = ps[(ps.condition == cnd) & (ps.corr_mm == corr)].sort_values("sd_hu")
            a.fill_between(s.sd_hu, s.fp_fat_pct_mean - s.fp_fat_pct_ci, s.fp_fat_pct_mean + s.fp_fat_pct_ci,
                           color=COND_COL[cnd], alpha=0.2, lw=0)
            a.plot(s.sd_hu, s.fp_fat_pct_mean, color=COND_COL[cnd], ls=ls, lw=0.9)
            s75 = ps[(ps.condition == cnd) & (ps.corr_mm == corr) & (ps.sd_hu == 75)].iloc[0]
            dia = [1, 2, 3, 5, 10]
            m = [s75[f"recovery_{x}mm_mean"] for x in dia]
            ci = [s75[f"recovery_{x}mm_ci"] for x in dia]
            b.errorbar(dia, m, yerr=ci, color=COND_COL[cnd], ls=ls, lw=0.9, marker="o", ms=2.2, capsize=0, elinewidth=0.6)
    a.set_xlabel("Added noise SD (HU)", fontsize=6); a.set_ylabel("False fat in fat-free block (%)", fontsize=6)
    a.set_xticks([10, 25, 50, 75, 120])
    b.set_xscale("log"); plain_log(b, "x", (1, 2, 3, 5, 10))
    b.set_xlabel("Fat inclusion diameter (mm)", fontsize=6); b.set_ylabel("Recovered fraction, noise SD 75 HU", fontsize=6)
    b.set_ylim(0, 1.05)
    a.legend(handles=[Line2D([0], [0], color=COND_COL[k], lw=1, label=COND_LAB[k]) for k in COND],
             loc="upper left", fontsize=4.8, handlelength=1.6)
    b.legend(handles=[Line2D([0], [0], color=style.INK2, lw=1, ls="-", label="white noise"),
                      Line2D([0], [0], color=style.INK2, lw=1, ls=(0, (2, 1.5)), label="correlated, 1.5 mm")],
             loc="lower right", fontsize=4.8, handlelength=1.8)
    c = ax_mm(fig, H, 137, 11, 42, 42)
    sf = panel_f(c)
    c.set_ylabel("Median absolute change", fontsize=6)
    for t, x, y in [("a", 1, 60), ("b", 63, 60), ("c", 125, 60)]:
        label(fig, H, x, y, t)
    src(ps, "FigureS5", "a_b_phantom"); src(sf, "FigureS5", "c_offset")
    return style.save(fig, "FigureS5")


def figS6(cid="R01-036"):
    from bodycomp_metrics import Case, vertebra_info
    import fig1_design_pipeline as F
    style.setup()
    H = 62.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    rows = []
    for i, (cohort, ttl) in enumerate([("public_rg", "CT 1: contrast chest CT, supine"),
                                       ("public_rg_scan2", "CT 2: PET/CT, prone, 7 days earlier")]):
        case = Case(R("data", "nifti", cohort, f"{cid}.nii.gz"), R("data", "seg", cohort, cid))
        z = int(round(vertebra_info(case.segs["total"], "T12")[0]))
        m = {k: F.disp(v) for k, v in case.slice_masks(z).items()}
        cr = F.centred_crop(m["body"], 230, 250)
        img = F.paint(F.window(F.disp(case.ct[:, :, z])), [(m["muscle"], style.TISSUE["muscle"])], alpha=0.5)
        ax = ax_mm(fig, H, 3 + i * 50, 8, 46, 46)
        ax.imshow(img[cr], interpolation="lanczos"); F.image_ax(ax)
        area = float(case.slice_masks(z)["muscle"].sum() * case.px_cm2)
        ax.set_title(f"{ttl}\nT12 muscle {area:.0f} cm², slice spacing {case.zooms[2]:.2f} mm", fontsize=5.6, pad=2)
        F.scalebar(ax, case.zooms[0] * 1.0)
        rows.append(dict(ct=ttl, t12_muscle_cm2=area, slice_mm=float(case.zooms[2])))
    h = ax_mm(fig, H, 118, 11, 60, 40)
    bc = pd.read_csv(R("results", "public_rg_bodycomp.csv"))
    h.hist(bc.T12_sma.dropna(), bins=40, color=style.WASH, edgecolor=style.MUTED, lw=0.3)
    v = float(bc.loc[bc.case_id == cid, "T12_sma"].iloc[0])
    h.axvline(v, color=style.TISSUE["muscle"], lw=0.9)
    h.text(v, h.get_ylim()[1] * 0.95, f"{cid}  ", fontsize=5, color=style.TISSUE["muscle"], va="top", ha="right")
    h.set_xlabel(f"T12 muscle area, fixed filter, {int(bc.T12_sma.notna().sum())} staging CTs (cm²)", fontsize=6); h.set_ylabel("CTs", fontsize=6)
    from matplotlib.ticker import MaxNLocator
    h.yaxis.set_major_locator(MaxNLocator(integer=True))
    for t, x, y in [("a", 1, 60), ("b", 108, 60)]:
        label(fig, H, x, y, t)
    src(pd.DataFrame(rows), "FigureS6", "a")
    return style.save(fig, "FigureS6")



# =================================================================== supplementary S2, S3 (house style)
def figS2():
    """Leave-one-cohort-out change in C (values of Supplementary Table S4)."""
    style.setup()
    H = 52.0
    fig = plt.figure(figsize=(120 * style.MM, H * style.MM))
    d = pd.DataFrame([("CPTAC", 43, 9, 0.512, 0.656, 0.143, 0.032, 0.305),
                      ("TCGA-LUAD", 22, 11, 0.690, 0.669, -0.021, -0.130, 0.064),
                      ("TCGA-LUSC", 33, 12, 0.565, 0.684, 0.119, -0.081, 0.330),
                      ("NSCLC-Radiogenomics", 137, 52, 0.662, 0.640, -0.022, -0.071, 0.023)],
                     columns=["cohort", "n", "events", "c_clin", "c_bc", "dc", "lo", "hi"])
    ax = fig.add_axes([46 / 120, 10 / H, 52 / 120, 38 / H])
    for i, r in d.iloc[::-1].reset_index(drop=True).iterrows():
        ax.plot([r.lo, r.hi], [i, i], color=style.INK, lw=0.8)
        ax.scatter(r.dc, i, s=14, color=style.INK, zorder=3)
        ax.text(1.03, i, f"{r.c_clin:.3f} / {r.c_bc:.3f}", fontsize=5, va="center", ha="left", color=style.INK2,
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color=style.HAIR, lw=0.6, zorder=0)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{c} ({n}/{e})" for c, n, e in d[["cohort", "n", "events"]].values[::-1]],
                                                fontsize=5.4)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.2, 0.35); ax.set_ylim(-0.6, 3.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fnum(v, "{:.1f}")))
    ax.set_xlabel("Change in Harrell C, clinical + body composition − clinical", fontsize=6)
    ax.text(1.03, 3.55, "C: clinical / + BC", fontsize=5, ha="left", va="bottom", color=style.INK2,
            transform=ax.get_yaxis_transform())
    src(d, "FigureS2", "loco")
    return style.save(fig, "FigureS2")


def figS3():
    """Intramuscular-fat HR across eligibility and adjustment sensitivities (values of Supplementary Table S3)."""
    style.setup()
    H = 62.0
    fig = plt.figure(figsize=(120 * style.MM, H * style.MM))
    d = [("Primary, date-concordant (121/44)", 1.42, 1.12, 1.81),
         ("Interval-only population (130/46)", 1.35, 1.06, 1.70),
         ("CT within 90 days of surgery (110/40)", 1.47, 1.16, 1.86),
         ("No T12 edge truncation (114/44)", 1.44, 1.14, 1.83),
         ("Soft kernel only (47/19)", np.nan, np.nan, np.nan),
         ("+ contrast-agent field, three levels (121/44)", 1.42, 1.10, 1.83),
         ("+ binary contrast field (121/44)", 1.43, 1.11, 1.84),
         ("+ smoking status (121/44)", 1.46, 1.14, 1.89),
         ("Recorded M0 only (95/41)", 1.56, 1.23, 2.00),
         ("Recorded M1 excluded (120/43)", 1.46, 1.15, 1.86),
         ("M0 with separate T and N terms (95/41)", np.nan, np.nan, np.nan)]
    d = pd.DataFrame(d, columns=["model", "HR", "lo", "hi"])
    ax = fig.add_axes([58 / 120, 10 / H, 58 / 120, 48 / H])
    for i, r in d.iloc[::-1].reset_index(drop=True).iterrows():
        if np.isnan(r.HR):
            ax.text(1.02, i, "not estimable (separation)", fontsize=5, va="center", color=style.INK2)
            continue
        first = r.model.startswith("Primary")
        ax.plot([r.lo, r.hi], [i, i], color=style.INK, lw=0.8)
        ax.scatter(r.HR, i, s=14, color=style.INK if first else "white", edgecolor=style.INK, lw=0.7, zorder=3)
    ax.axvline(1, color=style.HAIR, lw=0.6, zorder=0)
    ax.set_xscale("log"); plain_log(ax, "x", (0.8, 1, 1.5, 2)); ax.set_xlim(0.8, 2.2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.model[::-1], fontsize=5.2)
    ax.set_ylim(-0.7, len(d) - 0.4)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Intramuscular fat %, HR per SD (95% CI)", fontsize=6)
    src(d, "FigureS3", "sensitivity")
    return style.save(fig, "FigureS3")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "S2", "S3", "S5", "S6"]
    for w in which:
        print(globals()[f"fig{w}"]())
