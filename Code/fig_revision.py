"""Revision figures (technical error, protocol robustness, the rejected method iteration, and the recomputed
clinical analysis). Everything is drawn from results/audit/* and results/e5/*.

RevisionFig1  measurement error and protocol robustness (E2, E3, E4)
RevisionFig2  the rejected denoiser (A1) and the recomputed clinical analysis (E5)

Usage: env/bin/python figures/fig_revision.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style  # noqa: E402

R = lambda *p: os.path.join(style.ROOT, *p)
W = 183.0
COND = ["native", "slab", "gauss", "both"]
COND_LAB = {"native": "Native", "slab": "Slab only", "gauss": "Gaussian only", "both": "Slab + Gaussian"}
COND_COL = {"native": style.MUTED, "slab": "#9fb7d4", "gauss": style.SERIES[0], "both": style.INK}


def ax_mm(fig, H, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def label(fig, H, x, y, t):
    style.panel_label(None, t, x / W, y / H, fig=fig)


def icc21(x, y):
    M = np.c_[x, y]; n, k = M.shape; g = M.mean()
    msr = k * ((M.mean(1) - g) ** 2).sum() / (n - 1); msc = n * ((M.mean(0) - g) ** 2).sum() / (k - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + g) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


# ------------------------------------------------------------------ Figure R1
E2_MEAS = [("sma_cm2", "Muscle\narea"), ("smd_window_hu", "Density,\nwindow"),
           ("smd_all_hu", "Density,\nwhole mask"), ("imat_pct", "Intramuscular\nfat %"),
           ("vol_muscle_cm3", "Muscle\nvolume")]


def panel_a(ax):
    d = pd.read_csv(R("results", "audit", "e2_ablation_public_rg.csv"))
    sp = pd.read_csv(R("results", "audit", "splits.csv"))
    d = d[d.case_id.isin(set(sp[sp.split == "development"].case_id))]
    rng = np.random.default_rng(3)
    rows = []
    for i, (m, lab) in enumerate(E2_MEAS):
        base = d[f"native_{m}"].astype(float)
        diff = (d[f"both_{m}"].astype(float) - base) / base.abs() * 100
        diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(diff)), diff, s=2.2, color=style.INK2, alpha=0.3, lw=0,
                   rasterized=True)
        ax.plot([i - 0.3, i + 0.3], [diff.median()] * 2, color=style.TISSUE["muscle"], lw=1.4, zorder=3)
        rows.append(dict(measure=lab.replace("\n", " "), n=len(diff), median_pct=diff.median(),
                         q1=diff.quantile(.25), q3=diff.quantile(.75)))
    ax.axhline(0, color=style.HAIR, lw=0.6, zorder=0)
    ax.set_xticks(range(len(E2_MEAS)))
    ax.set_xticklabels([l for _, l in E2_MEAS], fontsize=5.2)
    ax.set_xlim(-0.6, len(E2_MEAS) - 0.4)
    ax.set_ylim(-80, 80)
    ax.set_ylabel("Harmonised vs native (%)", fontsize=6)
    ax.set_title("Processing changes the measurement", fontsize=6, pad=3)
    return pd.DataFrame(rows)


def panel_b(ax):
    a = pd.read_csv(R("results", "audit", "e2_ablation_public_rg.csv")).set_index("case_id")
    b = pd.read_csv(R("results", "audit", "e2_ablation_public_rg_scan2.csv")).set_index("case_id")
    ids = [i for i in a.index if i in b.index]
    rows = []
    for i, (m, lab) in enumerate(E2_MEAS):
        for c in COND:
            x = a.loc[ids, f"{c}_{m}"].astype(float).values
            y = b.loc[ids, f"{c}_{m}"].astype(float).values
            ok = np.isfinite(x) & np.isfinite(y)
            rows.append(dict(measure=lab.replace("\n", " "), condition=c, n=int(ok.sum()), icc=icc21(x[ok], y[ok])))
    r = pd.DataFrame(rows)
    for i, (m, lab) in enumerate(E2_MEAS):
        s = r[r.measure == lab.replace("\n", " ")].set_index("condition")
        for j, c in enumerate(COND):
            ax.scatter(i + (j - 1.5) * 0.17, s.loc[c, "icc"], s=13, color=COND_COL[c], zorder=3,
                       edgecolor="white", lw=0.4)
    ax.axhspan(0.9, 1.0, color=style.WASH, lw=0, zorder=0)
    ax.set_xticks(range(len(E2_MEAS)))
    ax.set_xticklabels([l for _, l in E2_MEAS], fontsize=5.2)
    ax.set_xlim(-0.6, len(E2_MEAS) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Scan–rescan ICC (49 pairs)", fontsize=6)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.4, color=COND_COL[c], label=COND_LAB[c])
                       for c in COND], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=5,
              handletextpad=0.2, columnspacing=0.9)
    return r


E3_MEAS = ["T12 muscle area (cm²)", "T12 density, window mean (HU)", "T12 intramuscular fat (%)",
           "T12 subcutaneous fat (cm²)", "T12 visceral fat (cm²)", "T5–T12 muscle volume (cm³)"]


def panel_c(ax):
    pr = pd.read_csv(R("results", "audit", "e3_pairs_strata.csv"))
    pr = pr[pr.stratum == "all pairs"].set_index("measure")
    rc = pd.read_csv(R("results", "audit", "e3_recon_effects.csv"))
    rows = []
    y = np.arange(len(E3_MEAS))[::-1]
    for yi, m in zip(y, E3_MEAS):
        p_val = pr.loc[m, "median_abs_diff"]
        r_val = rc[rc.measure == m]["diff"].abs().median()
        ax.annotate("", xy=(p_val, yi), xytext=(r_val, yi),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=style.HAIR))
        ax.scatter(r_val, yi, s=16, color=style.SERIES[0], zorder=3, label="_")
        ax.scatter(p_val, yi, s=16, facecolor="white", edgecolor=style.INK, lw=0.8, zorder=3)
        rows.append(dict(measure=m, same_acquisition=r_val, separate_scans=p_val))
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace(" (", "\n(") for m in E3_MEAS], fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 60)
    ax.set_xticks([0.5, 1, 5, 10, 50])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Median |difference| (measurement units)", fontsize=6)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.6, color=style.SERIES[0],
                              label="Same acquisition,\ndifferent reconstruction (21)"),
                       Line2D([0], [0], marker="o", ls="", ms=3.6, mfc="white", mec=style.INK,
                              label="Two separate CTs (49)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=5, handletextpad=0.2, labelspacing=0.4)
    return pd.DataFrame(rows)


def panel_d(ax):
    n = pd.read_csv(R("results", "audit", "e4_noise_fov.csv"))
    n = n[n.region.isin(["liver", "spleen", "subcutaneous_fat"])]
    t = pd.read_csv(R("results", "audit", "timing.csv"))[["case_id", "kernel_type"]]
    n = n.merge(t, on="case_id", how="left")
    for img, mk, fc in [("native", "o", "white"), ("harmonised", "o", style.INK)]:
        for kt, col in [("sharp", style.TISSUE["muscle"]), ("soft", style.SERIES[0])]:
            s = n[(n.image == img) & (n.kernel_type == kt)]
            ax.scatter(s.corr_length_mm, s.residual_sd, s=3, marker=mk,
                       facecolor=col if fc != "white" else "none", edgecolor=col, lw=0.4, alpha=0.6,
                       rasterized=True)
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, 2.2)
    ax.set_xlabel("Noise correlation length (mm)", fontsize=6)
    ax.set_ylabel("Residual SD (HU)", fontsize=6)
    ax.set_title("Harmonisation trades amplitude for correlation", fontsize=6, pad=3)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.2, mfc="none", mec=style.INK, label="Native"),
                       Line2D([0], [0], marker="o", ls="", ms=3.2, color=style.INK, label="Harmonised"),
                       Line2D([0], [0], marker="s", ls="", ms=3.2, color=style.TISSUE["muscle"], label="Sharp kernel"),
                       Line2D([0], [0], marker="s", ls="", ms=3.2, color=style.SERIES[0], label="Soft kernel")],
              loc="upper right", fontsize=4.8, handletextpad=0.2, labelspacing=0.3)
    return n


def panel_e(ax1, ax2):
    p = pd.read_csv(R("results", "audit", "e4_phantom.csv"))
    g = p.groupby(["condition", "corr_mm", "sd_hu"]).agg(fp=("fp_fat_pct", "mean"),
                                                         rec=("recovery_2mm", "mean")).reset_index()
    for c in COND:
        for corr, ls in [(0.0, "-"), (1.5, (0, (2, 1.5)))]:
            s = g[(g.condition == c) & (g.corr_mm == corr)].sort_values("sd_hu")
            ax1.plot(s.sd_hu, s.fp, linestyle=ls, color=COND_COL[c], lw=1.0)
            ax2.plot(s.sd_hu, s.rec, linestyle=ls, color=COND_COL[c], lw=1.0)
    ax1.set_ylabel("False fat in fat-free block (%)", fontsize=6)
    ax1.set_title("Digital reference (colours as in b)", fontsize=6, pad=3, loc="left")
    ax2.set_ylabel("Recovery of a 2 mm\nfat inclusion", fontsize=6)
    for a in (ax1, ax2):
        a.set_xlabel("Added noise SD (HU)", fontsize=6)
        a.set_xticks([10, 50, 120])
    ax1.legend(handles=[Line2D([0], [0], color=style.INK, ls="-", lw=1, label="white noise"),
                        Line2D([0], [0], color=style.INK, ls=(0, (2, 1.5)), lw=1, label="correlated, 1.5 mm")],
               loc="upper left", fontsize=4.8, handlelength=1.4, handletextpad=0.3, borderaxespad=0.2)
    return g


def panel_f(ax):
    l = pd.read_csv(R("results", "audit", "e4_localisation.csv"))
    b = l[l["shift"] == 0].set_index("case_id")
    rows = []
    for m, lab, col in [("sma_cm2", "Muscle area (cm²)", style.TISSUE["muscle"]),
                        ("smd_hu", "Density (HU)", style.INK),
                        ("imat_pct", "Intramuscular fat (points)", style.TISSUE["imat"])]:
        xs, ys, lo, hi = [], [], [], []
        for s in sorted(l["shift"].unique()):
            x = l[l["shift"] == s].set_index("case_id")
            ids = x.index.intersection(b.index)
            d = (x.loc[ids, m] - b.loc[ids, m]).abs().astype(float)
            xs.append(x.loc[ids, "shift_mm"].median())
            ys.append(d.median()); lo.append(d.quantile(.25)); hi.append(d.quantile(.75))
            rows.append(dict(measure=lab, shift=s, median_abs=d.median()))
        o = np.argsort(xs)
        xs, ys, lo, hi = np.array(xs)[o], np.array(ys)[o], np.array(lo)[o], np.array(hi)[o]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=col, lw=1.1, label=lab)
    ax.set_xlabel("Measurement slice shifted (mm)", fontsize=6)
    ax.set_ylabel("Median |change|", fontsize=6)
    ax.legend(fontsize=5, loc="upper center", handlelength=1.2)
    return pd.DataFrame(rows)


def figure1():
    H = 150.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    a = ax_mm(fig, H, 16, 104, 62, 34); sa = panel_a(a)
    b = ax_mm(fig, H, 104, 104, 62, 34); sb = panel_b(b)
    c = ax_mm(fig, H, 34, 58, 44, 30); sc = panel_c(c)
    d = ax_mm(fig, H, 104, 58, 62, 34); sd = panel_d(d)
    e1 = ax_mm(fig, H, 16, 12, 34, 30)
    e2 = ax_mm(fig, H, 64, 12, 34, 30); se = panel_e(e1, e2)
    f = ax_mm(fig, H, 117, 12, 49, 30); sf = panel_f(f)
    for x, y, t in [(1, 149, "a"), (89, 149, "b"), (1, 100, "c"), (89, 100, "d"), (1, 50, "e"), (102, 50, "f")]:
        label(fig, H, x, y, t)
    for df, p in [(sa, "a_ablation"), (sb, "b_icc"), (sc, "c_protocol"), (sd, "d_noise"), (se, "e_phantom"),
                  (sf, "f_localisation")]:
        style.source_data(df, "RevisionFig1", p)
    print(style.save(fig, "RevisionFig1"))


# ------------------------------------------------------------------ Figure R2
A1_COND = ["native", "gauss", "both", "bsn_r0", "bsn_r1", "bsn_r1_slab"]
A1_LAB = {"native": "Native", "gauss": "Gaussian", "both": "Published pipeline", "bsn_r0": "Blind spot, 1 px",
          "bsn_r1": "Blind spot, matched", "bsn_r1_slab": "Blind spot + slab"}
A1_COL = {"native": style.MUTED, "gauss": "#9fb7d4", "both": style.INK, "bsn_r0": "#f0b27a",
          "bsn_r1": style.SERIES[1], "bsn_r1_slab": "#c46a2a"}


def panel_a1_phantom(ax1, ax2):
    d = pd.read_csv(R("results", "audit", "a1_phantom_nativecorr.csv"))
    g = d.groupby("condition").agg(fp=("fp_fat_pct", "mean"), rec=("recovery_2mm", "mean")).reindex(A1_COND)
    y = np.arange(len(A1_COND))[::-1]
    for ax, col, lab in [(ax1, "fp", "False fat (%)"), (ax2, "rec", "2 mm recovery")]:
        ax.barh(y, g[col].values, height=0.62, color=[A1_COL[c] for c in A1_COND], lw=0)
        ax.set_yticks(y)
        ax.set_yticklabels([A1_LAB[c] for c in A1_COND] if ax is ax1 else [], fontsize=5.2)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(lab, fontsize=6)
    ax1.set_title("Phantom at the noise correlation measured on native CT", fontsize=6, pad=3, loc="left")
    return g.reset_index()


def panel_a1_patient(ax):
    r = pd.read_csv(R("results", "audit", "a1_reproducibility.csv"))
    meas = [("sma_cm2", "Muscle area"), ("smd_window_hu", "Density, window"), ("smd_all_hu", "Density, whole mask"),
            ("imat_pct", "Intramuscular fat %"), ("vat_cm2", "Visceral fat")]
    for i, (m, lab) in enumerate(meas):
        s = r[r.measure == m].set_index("condition")
        for j, c in enumerate(A1_COND):
            if c in s.index:
                ax.scatter(i + (j - 2.5) * 0.13, s.loc[c, "icc"], s=11, color=A1_COL[c], zorder=3,
                           edgecolor="white", lw=0.3)
    ax.axhspan(0.9, 1.0, color=style.WASH, lw=0, zorder=0)
    ax.set_xticks(range(len(meas)))
    ax.set_xticklabels([l for _, l in meas], fontsize=5, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Scan–rescan ICC", fontsize=6)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.2, color=A1_COL[c], label=A1_LAB[c])
                       for c in A1_COND], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=4.8,
              handletextpad=0.2, columnspacing=0.8)
    return r


E5_EXPO = [("imat_pct", "Intramuscular fat %, per SD higher"), ("smd_all", "Density, whole mask, per SD lower"),
           ("smd3d", "T5–T12 density, window, per SD lower"), ("esm_hu", "Erector spinae density, per SD lower"),
           ("esm_area", "Erector spinae area, per SD lower"), ("mvr", "Muscle-to-vertebra ratio, per SD lower"),
           ("bmd", "Trabecular attenuation, per SD lower"), ("vat", "Visceral fat, per SD higher")]


def panel_e5(ax):
    r = pd.read_csv(R("results", "e5", "cox_primary_secondary.csv"))
    prim = r[r.population == "primary: timing verified, age >= 65"].set_index("exposure")
    prim_all = r[r.population == "primary: timing verified, all ages"].set_index("exposure")
    sec = r[r.population == "secondary: pooled cohorts, age >= 65"].set_index("exposure")
    y = np.arange(len(E5_EXPO))[::-1]
    ax.axvline(1, color=style.HAIR, lw=0.6, zorder=0)
    rows = []
    for yi, (e, lab) in zip(y, E5_EXPO):
        for src, off, fc, ec in [(sec, 0.26, "white", style.INK), (prim_all, 0.0, style.MUTED, style.MUTED),
                                 (prim, -0.26, style.INK, style.INK)]:
            if e in src.index:
                s = src.loc[e]
                ax.plot([s.lo, s.hi], [yi + off] * 2, color=style.INK2, lw=0.7)
                ax.scatter(s.HR, yi + off, s=15, marker="s", facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
                rows.append(dict(population=s.population, exposure=e, HR=s.HR, lo=s.lo, hi=s.hi, p=s.p,
                                 q=s.get("q_fdr", np.nan)))
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in E5_EXPO], fontsize=5.2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 3.2)
    ax.set_xticks([0.5, 1, 2, 3])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Adjusted HR per SD, aged ≥65 years", fontsize=6)
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=3.6, color=style.INK,
                              label="Primary, aged ≥65 y (130; 46 deaths)"),
                       Line2D([0], [0], marker="s", ls="", ms=3.6, color=style.MUTED,
                              label="Primary, all ages (173; 56 deaths)"),
                       Line2D([0], [0], marker="s", ls="", ms=3.6, mfc="white", mec=style.INK,
                              label="Pre-specified pooled, aged ≥65 y (195; 66 deaths)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=4.8, handletextpad=0.2, labelspacing=0.3)
    return pd.DataFrame(rows)


def panel_e5_sens(ax):
    s = pd.read_csv(R("results", "e5", "cox_sensitivity.csv"))
    p = pd.read_csv(R("results", "e5", "cox_primary_secondary.csv"))
    p = p[p.population == "primary: timing verified, age >= 65"]
    keep = ["imat_pct", "smd_all"]
    labs = {"imat_pct": "Intramuscular fat %", "smd_all": "Density, whole mask"}
    groups = [("primary: timing verified, age >= 65", "Primary"),
              ("sensitivity: CT within 90 days", "CT ≤90 days"),
              ("sensitivity: T12 not truncated", "T12 not truncated"),
              ("sensitivity: unenhanced only", "Unenhanced only")]
    allr = pd.concat([p, s])
    y = np.arange(len(groups))[::-1]
    ax.axvline(1, color=style.HAIR, lw=0.6, zorder=0)
    rows = []
    for yi, (pop, lab) in zip(y, groups):
        for e, off, col in [("imat_pct", 0.17, style.TISSUE["imat"]), ("smd_all", -0.17, style.TISSUE["muscle"])]:
            g = allr[(allr.population == pop) & (allr.exposure == e)]
            if g.empty:
                continue
            g = g.iloc[0]
            ax.plot([g.lo, g.hi], [yi + off] * 2, color=style.INK2, lw=0.7)
            ax.scatter(g.HR, yi + off, s=14, marker="o", color=col, zorder=3)
            rows.append(dict(population=pop, exposure=e, n=g.n, events=g.events, HR=g.HR, lo=g.lo, hi=g.hi, p=g.p))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n(n={int(allr[(allr.population == pop) & (allr.exposure == 'imat_pct')].n.iloc[0])})"
                        if not allr[(allr.population == pop) & (allr.exposure == 'imat_pct')].empty else l
                        for pop, l in groups], fontsize=5.2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xscale("log"); ax.set_xlim(0.4, 3.2); ax.set_xticks([0.5, 1, 2, 3])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("HR per SD, primary population", fontsize=6)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.4, color=style.TISSUE["imat"],
                              label="Intramuscular fat %"),
                       Line2D([0], [0], marker="o", ls="", ms=3.4, color=style.TISSUE["muscle"],
                              label="Whole-mask density")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=4.8, handletextpad=0.2,
              labelspacing=0.3)
    return pd.DataFrame(rows)


def figure2():
    H = 118.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    a1 = ax_mm(fig, H, 36, 74, 26, 30)
    a2 = ax_mm(fig, H, 68, 74, 26, 30)
    sa = panel_a1_phantom(a1, a2)
    b = ax_mm(fig, H, 116, 74, 58, 30); sb = panel_a1_patient(b)
    c = ax_mm(fig, H, 62, 14, 40, 44); sc = panel_e5(c)
    d = ax_mm(fig, H, 140, 14, 34, 44); sd = panel_e5_sens(d)
    for x, y, t in [(1, 117, "a"), (100, 117, "b"), (1, 64, "c"), (110, 64, "d")]:
        label(fig, H, x, y, t)
    for df, p in [(sa, "a_phantom"), (sb, "b_patient_icc"), (sc, "c_primary_secondary"), (sd, "d_sensitivity")]:
        style.source_data(df, "RevisionFig2", p)
    print(style.save(fig, "RevisionFig2"))




# ------------------------------------------------------------------ deliverable figures
PHENO_COL = {"normal": style.MUTED, "low muscle": style.SERIES[0], "myosteatosis": style.SERIES[1],
             "both": style.INK}
PHENO_LAB = {"normal": "Neither", "low muscle": "Low area only", "myosteatosis": "Low density only",
             "both": "Both low"}


def panel_km_primary(ax, rta):
    km = pd.read_csv(R("results", "e5", "km_primary.csv"))
    ph = pd.read_csv(R("results", "e5", "cox_phenotype_primary.csv")).set_index("term")
    rt0 = pd.read_csv(R("results", "e5", "km_primary_risktable.csv"))
    n0 = rt0[rt0.time_years == 0].set_index("phenotype").n_risk
    for p_, col in PHENO_COL.items():
        s = km[km.phenotype == p_]
        if s.empty:
            continue
        ax.step(np.r_[0, s.time_years], np.r_[1, s.surv], where="post", color=col,
                lw=1.1 if p_ == "both" else 0.9, label=f"{PHENO_LAB[p_]} ({n0.get(p_, 0)})",
                zorder=3 if p_ == "both" else 2)
    ax.set_xlim(0, 5); ax.set_ylim(0, 1.18); ax.set_xticks(range(6)); ax.set_yticks([0, 0.5, 1.0])
    ax.spines["left"].set_bounds(0, 1)
    ax.set_xlabel("Years after surgery", fontsize=6)
    ax.set_ylabel("Overall survival", fontsize=6)
    ax.legend(loc="lower left", fontsize=5, handlelength=1.2, labelspacing=0.25, borderaxespad=0.15)
    hb = ph.loc["phenotypeboth"]
    ax.text(0.99, 1.0, f"Both vs neither: HR {hb['exp(coef)']:.2f} ({hb['lower .95']:.2f}–{hb['upper .95']:.2f}), "
            f"$\\mathit{{P}}$ = {hb.p:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5,
            color=style.INK)
    # numbers at risk
    rta.set_xlim(0, 5); rta.set_ylim(-0.5, 3.5); rta.set_axis_off()
    rt = pd.read_csv(R("results", "e5", "km_primary_risktable.csv"))
    for i, p_ in enumerate(PHENO_COL):
        yy = 3 - i
        rta.plot([-0.52, -0.3], [yy, yy], color=PHENO_COL[p_], lw=1.4, clip_on=False, solid_capstyle="butt")
        for _, r in rt[rt.phenotype == p_].iterrows():
            rta.text(r.time_years, yy, f"{int(r.n_risk)}", fontsize=5, ha="center", va="center", color=style.INK)
    rta.text(-0.52, 3.9, "Number at risk", fontsize=5.2, ha="left", va="bottom", color=style.INK)
    return km


def figure_clinical():
    H = 78.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    a = ax_mm(fig, H, 16, 28, 46, 38)
    rta = ax_mm(fig, H, 16, 9, 46, 10)
    sa = panel_km_primary(a, rta)
    b = ax_mm(fig, H, 104, 12, 38, 54); sb = panel_e5(b)
    c = ax_mm(fig, H, 154, 12, 24, 54); sc = panel_e5_sens(c)
    for x, yy, t in [(1, 77, "a"), (88, 77, "b"), (140, 77, "c")]:
        label(fig, H, x, yy, t)
    for df, p_ in [(sa, "a_km"), (sb, "b_primary_secondary"), (sc, "c_sensitivity")]:
        style.source_data(df, "RevFig5", p_)
    print(style.save(fig, "RevFig5"))


def figure_a1_ed():
    H = 92.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    a1 = ax_mm(fig, H, 36, 48, 26, 30)
    a2 = ax_mm(fig, H, 68, 48, 26, 30)
    sa = panel_a1_phantom(a1, a2)
    b = ax_mm(fig, H, 118, 48, 56, 30); sb = panel_a1_patient(b)
    c = ax_mm(fig, H, 36, 10, 58, 26)
    dr = pd.read_csv(R("results", "audit", "a1_drift.csv"))
    base = dr[dr.condition == "native"].set_index(["case_id", "scan"]).mean_hu
    dr["drift"] = dr.apply(lambda r: r.mean_hu - base.get((r.case_id, r.scan), np.nan), axis=1)
    rng = np.random.default_rng(5)
    conds = [c_ for c_ in A1_COND if c_ != "native"]
    for i, c_ in enumerate(conds):
        v = dr[dr.condition == c_]["drift"].dropna()
        c.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=2.5, color=A1_COL[c_], alpha=0.5, lw=0, rasterized=True)
        c.plot([i - 0.28, i + 0.28], [v.median()] * 2, color=style.INK, lw=1.2, zorder=3)
    c.axhline(0, color=style.HAIR, lw=0.6, zorder=0)
    for s_ in (-2, 2):
        c.axhline(s_, color=style.TISSUE["muscle"], lw=0.6, ls=(0, (2, 2)), zorder=1)
    c.set_xticks(range(len(conds))); c.set_xticklabels([A1_LAB[c_] for c_ in conds], fontsize=5, rotation=15,
                                                       ha="right")
    c.set_ylim(-25, 25)
    c.set_ylabel("Liver HU drift vs native", fontsize=6)
    c.text(0.99, 0.97, "dashed: ±2 HU stop rule", transform=c.transAxes, ha="right", va="top", fontsize=5,
           color=style.TISSUE["muscle"])
    d = ax_mm(fig, H, 124, 10, 50, 26)
    sd = pd.read_csv(R("results", "audit", "a1_drift.csv")).groupby("condition").sd_hu.median().reindex(A1_COND)
    d.bar(range(len(A1_COND)), sd.values, color=[A1_COL[c_] for c_ in A1_COND], width=0.66, lw=0)
    d.set_xticks(range(len(A1_COND))); d.set_xticklabels([A1_LAB[c_] for c_ in A1_COND], fontsize=5, rotation=15,
                                                         ha="right")
    d.set_ylabel("Residual SD in liver (HU)", fontsize=6)
    for x, yy, t in [(1, 91, "a"), (102, 91, "b"), (1, 40, "c"), (106, 40, "d")]:
        label(fig, H, x, yy, t)
    for df, p_ in [(sa, "a_phantom"), (sb, "b_icc"), (dr, "c_drift"), (sd.reset_index(), "d_sd")]:
        style.source_data(df, "RevED3", p_)
    print(style.save(fig, "RevED3"))


def figure_a3():
    """Extended Data: measurement-level intervals from paired images (split conformal)."""
    H = 78.0
    iv = pd.read_csv(R("results", "audit", "a3_intervals.csv"))
    cov = pd.read_csv(R("results", "audit", "a3_coverage.csv"))
    meas = ["T12 muscle area (cm²)", "T12 radiodensity, window mean (HU)",
            "T12 radiodensity, whole-mask mean (HU)", "T12 intramuscular fat (%)",
            "T12 subcutaneous fat (cm²)", "T12 visceral fat (cm²)", "T5–T12 muscle volume (cm³)"]
    short = {"T12 muscle area (cm²)": "muscle area", "T12 radiodensity, window mean (HU)": "radiodensity, window mean",
             "T12 radiodensity, whole-mask mean (HU)": "radiodensity, whole-mask mean",
             "T12 intramuscular fat (%)": "intramuscular fat", "T12 subcutaneous fat (cm²)": "subcutaneous fat",
             "T12 visceral fat (cm²)": "visceral fat", "T5–T12 muscle volume (cm³)": "T5–T12 muscle volume"}
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    COLK = {"protocol": style.SERIES[0], "repeat": style.INK}
    LABK = {"protocol": "Different reconstruction of the same acquisition (26 pairs)",
            "repeat": "Repeat examination (49 pairs)"}
    y = np.arange(len(meas))[::-1]
    a = ax_mm(fig, H, 52, 12, 44, 52)
    for kind in ("protocol", "repeat"):
        s = iv[(iv.kind == kind) & (iv.level == "90%")].set_index("measure")
        for yi, m in zip(y, meas):
            if m in s.index:
                a.scatter(s.loc[m, "half_width_over_sd"], yi + (0.18 if kind == "protocol" else -0.18), s=16,
                          color=COLK[kind], zorder=3)
    a.axvline(1, color=style.TISSUE["muscle"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    a.text(1.02, len(meas) - 0.4, "one between-patient SD", fontsize=5, color=style.TISSUE["muscle"], va="top")
    a.set_yticks(y); a.set_yticklabels([short[m] for m in meas], fontsize=5.4)
    a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
    a.set_xlim(0, 1.8)
    a.set_xlabel("90% interval half-width (between-patient SD)", fontsize=6)
    a.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.6, color=COLK[k], label=LABK[k])
                      for k in ("protocol", "repeat")], loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=5,
             handletextpad=0.2, labelspacing=0.3)
    b = ax_mm(fig, H, 118, 12, 30, 52)
    for kind in ("protocol", "repeat"):
        s = iv[(iv.kind == kind) & (iv.level == "90%")].set_index("measure")
        for yi, m in zip(y, meas):
            if m in s.index:
                b.scatter(s.loc[m, "half_width"], yi + (0.18 if kind == "protocol" else -0.18), s=16,
                          color=COLK[kind], zorder=3)
    b.set_yticks(y); b.set_yticklabels([])
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.set_xscale("log"); b.set_xlim(0.8, 300)
    b.set_xticks([1, 10, 100]); b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    b.xaxis.set_minor_formatter(plt.NullFormatter())
    b.set_xlabel("90% interval half-width\n(measurement units)", fontsize=6)
    c = ax_mm(fig, H, 162, 12, 16, 52)
    for kind in ("protocol", "repeat"):
        s = cov[cov.kind == kind]
        for lev, mk in [("80%", "o"), ("90%", "s")]:
            v = s[s.level == lev].loo_coverage.mean()
            c.scatter(float(lev[:-1]) / 100, v, s=18, marker=mk, color=COLK[kind], zorder=3)
    c.plot([0.75, 0.95], [0.75, 0.95], color=style.HAIR, lw=0.7, zorder=0)
    c.set_xlim(0.75, 0.95); c.set_ylim(0.75, 0.95)
    c.set_xticks([0.8, 0.9]); c.set_yticks([0.8, 0.9])
    c.set_xlabel("Target coverage", fontsize=6)
    c.set_ylabel("Leave-one-out coverage", fontsize=6)
    c.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.4, color=style.INK2, label="80%"),
                      Line2D([0], [0], marker="s", ls="", ms=3.4, color=style.INK2, label="90%")],
             loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=5, handletextpad=0.2)
    for x, yy, t in [(1, 77, "a"), (100, 77, "b"), (150, 77, "c")]:
        label(fig, H, x, yy, t)
    style.source_data(iv, "RevED5", "ab_intervals")
    style.source_data(cov, "RevED5", "c_coverage")
    print(style.save(fig, "RevED5"))


if __name__ == "__main__":
    style.setup()
    import sys as _s
    which = _s.argv[1:] or ["1", "2", "clin", "a1", "a3"]
    if "1" in which:
        figure1()
    if "2" in which:
        figure2()
    if "clin" in which:
        figure_clinical()
    if "a1" in which:
        figure_a1_ed()
    if "a3" in which:
        figure_a3()
