"""ER revision: analysis of same-acquisition sharp/soft reconstruction pairs measured with one shared mask.

Eligibility: kernel-discordant pairs, DICOM-verified single acquisition, mapped mask retaining >=95% of the
segmented-series mask (excludes partner series with a smaller reconstruction field).
Estimands (sharp minus soft, per pair; Hodges-Lehmann median with Wilcoxon-based 95% CI):
  native            both reconstructions as reconstructed
  fixed matching    published filter (5-mm slab + Gaussian 0.75 mm) applied to both
  sharp-only        sharp image filtered (in-plane Gaussian) to the soft partner's muscle noise (pairwise target);
                    soft image unchanged -> tests whether processed sharp equals native soft
  sharp-only, fixed target   same, but to one fixed muscle-noise target per dataset (median native soft noise),
                    i.e. what can be applied without a partner
  soft shift        soft(fixed matching) - soft(native): how far fixed matching moves the soft value itself
Noise-convolution decomposition: soft voxel histogram + Gaussian noise of variance (sigma_sharp^2 - sigma_soft^2)
predicts sharp intramuscular fat and muscle-window mean; fraction of the native sharp-soft difference explained.
Outputs: results/er/pairs_summary.csv, results/er/pairs_perpair.csv, results/er/pairs_sigma_sweep.csv,
         results/er/pairs_decomposition.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)
OUT = R("results", "er")
MEAS = ["imat_pct", "smd_window_hu", "smd_all_hu", "sma_cm2", "esm_hu"]


def hodges_lehmann(d, alpha=0.05):
    """HL estimator of the median paired difference with the Wilcoxon signed-rank confidence interval."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 6:
        return np.nan, np.nan, np.nan, n
    w = np.sort((d[:, None] + d[None, :])[np.triu_indices(n)] / 2)
    m = len(w)
    z = norm.ppf(1 - alpha / 2)
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(k, 0)
    return float(np.median(w)), float(w[k]), float(w[m - 1 - k]), n


def pick(df, cond, sigma=None):
    s = df[df.condition == cond]
    if sigma is not None:
        s = s[np.isclose(s.sigma, sigma)]
    return s.set_index(["dataset", "pid", "role"])


def main():
    L = pd.read_csv(os.path.join(OUT, "pairs_long.csv"))
    M = pd.read_csv(os.path.join(OUT, "pairs_meta.csv"))
    M["pid"] = M.pid.astype(str)
    L["pid"] = L.pid.astype(str)
    # eligibility
    roles = L[L.condition == "native"].pivot_table(index=["dataset", "pid"], columns="role", values="sharp")
    disc = roles[(roles.A + roles.B) == 1].reset_index()[["dataset", "pid"]]
    M = M.merge(disc, on=["dataset", "pid"])
    # retained mask area inside the partner's reconstructed field of view (area-based; pixel sizes may differ)
    M["mask_ratio"] = M.mask_valid_area_ratio
    nv = pd.read_csv(os.path.join(OUT, "nlst_pairs_dicom.csv"))
    nv["pid"] = nv.pid.astype(str)
    rv = pd.read_csv(R("results", "r1005", "recon_pairs_dicom.csv"))
    ver = pd.concat([nv.assign(dataset="NLST")[["dataset", "pid", "same_acquisition", "Manufacturer_A"]],
                     rv.assign(dataset="RG", pid=rv.alt_id, Manufacturer_A=rv.manufacturer)[
                         ["dataset", "pid", "same_acquisition", "Manufacturer_A"]]])
    M = M.merge(ver, on=["dataset", "pid"], how="left")
    flow = dict(kernel_discordant=M.groupby("dataset").size().to_dict())
    M = M[M.same_acquisition == True]
    flow["same_acquisition"] = M.groupby("dataset").size().to_dict()
    M = M[M.mask_ratio >= 0.95]
    flow["mask_retained_95pct"] = M.groupby("dataset").size().to_dict()
    M["vendor"] = M.Manufacturer_A.str.split().str[0].str.upper().replace({"GE": "GE"})
    M["equal_thickness"] = np.isclose(M.slice_A, M.slice_B)
    keep = set(zip(M.dataset, M.pid))
    L = L[[k in keep for k in zip(L.dataset, L.pid)]]

    # per pair, identify sharp and soft roles
    nat = pick(L, "native")
    pp = []
    for (ds, pid), g in nat.groupby(level=[0, 1]):
        g = g.reset_index()
        sh, so = g[g.sharp == 1].iloc[0], g[g.sharp == 0].iloc[0]
        pp.append(dict(dataset=ds, pid=pid, role_sharp=sh.role, role_soft=so.role, kernel_sharp=sh.kernel,
                       kernel_soft=so.kernel, noise_sharp=sh.noise_muscle, noise_soft=so.noise_muscle))
    P = pd.DataFrame(pp).merge(M[["dataset", "pid", "vendor", "equal_thickness", "slice_A", "slice_B", "same_grid"]],
                               on=["dataset", "pid"])
    fixed_target = P.groupby("dataset").noise_soft.median().to_dict()
    G = L[L.condition == "gauss_only"]

    def val(ds, pid, role, cond, sigma, m):
        s = L[(L.dataset == ds) & (L.pid == pid) & (L.role == role) & (L.condition == cond)]
        if sigma is not None:
            s = s[np.isclose(s.sigma, sigma)]
        return float(s[m].iloc[0]) if len(s) else np.nan

    rows = []
    for _, r in P.iterrows():
        g = G[(G.dataset == r.dataset) & (G.pid == r.pid) & (G.role == r.role_sharp)].sort_values("sigma")
        out = dict(dataset=r.dataset, pid=r.pid)
        for tgt_name, tgt in [("pair", r.noise_soft), ("fixed", fixed_target[r.dataset])]:
            ok = g[g.noise_muscle <= tgt]
            gs = ok.iloc[0] if len(ok) else g.iloc[-1]
            out[f"sigma_{tgt_name}"] = gs.sigma
            for m in MEAS:
                out[f"{m}_sharp_adapt_{tgt_name}"] = gs[m]
        for m in MEAS:
            out[f"{m}_soft_native"] = val(r.dataset, r.pid, r.role_soft, "native", None, m)
            out[f"{m}_sharp_native"] = val(r.dataset, r.pid, r.role_sharp, "native", None, m)
            out[f"{m}_soft_fixed"] = val(r.dataset, r.pid, r.role_soft, "slab_gauss", 0.75, m)
            out[f"{m}_sharp_fixed"] = val(r.dataset, r.pid, r.role_sharp, "slab_gauss", 0.75, m)
            for e in (1, 2):
                out[f"{m}_soft_matched_erode{e}"] = val(r.dataset, r.pid, r.role_soft, f"matched_erode{e}", None, m)
                out[f"{m}_sharp_matched_erode{e}"] = val(r.dataset, r.pid, r.role_sharp, f"matched_erode{e}", None, m)
        rows.append(out)
    W = P.merge(pd.DataFrame(rows), on=["dataset", "pid"])
    W.to_csv(os.path.join(OUT, "pairs_perpair.csv"), index=False)

    summ = []
    strata = [("all", lambda d: d)] + [(f"vendor={v}", (lambda v: lambda d: d[d.vendor == v])(v)) for v in sorted(W.vendor.unique())] \
        + [("equal thickness", lambda d: d[d.equal_thickness])]
    for ds in ["NLST", "RG"]:
        for sname, f in strata:
            d0 = f(W[W.dataset == ds])
            if len(d0) < 6:
                continue
            for m in MEAS:
                ests = {
                    "native": d0[f"{m}_sharp_native"] - d0[f"{m}_soft_native"],
                    "fixed matching": d0[f"{m}_sharp_fixed"] - d0[f"{m}_soft_fixed"],
                    "sharp-only, pairwise target": d0[f"{m}_sharp_adapt_pair"] - d0[f"{m}_soft_native"],
                    "sharp-only, fixed target": d0[f"{m}_sharp_adapt_fixed"] - d0[f"{m}_soft_native"],
                    "soft shift under fixed matching": d0[f"{m}_soft_fixed"] - d0[f"{m}_soft_native"],
                }
                for k, v in ests.items():
                    hl, lo, hi, n = hodges_lehmann(v)
                    p = wilcoxon(v.dropna()).pvalue if v.notna().sum() >= 6 and (v.dropna() != 0).any() else np.nan
                    summ.append(dict(dataset=ds, stratum=sname, measure=m, estimand=k, n=n, hl=hl, lo=lo, hi=hi,
                                     median_abs=float(v.abs().median()), p=p))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUT, "pairs_summary.csv"), index=False)

    # sigma sweep: fixed filter applied to both reconstructions
    sw = []
    SG = L[L.condition == "slab_gauss"]
    for ds in ["NLST", "RG"]:
        Pd = P[P.dataset == ds]
        for s in sorted(SG.sigma.unique()):
            for m in ["imat_pct", "smd_window_hu", "smd_all_hu"]:
                dd, sh = [], []
                for _, r in Pd.iterrows():
                    a = SG[(SG.dataset == ds) & (SG.pid == r.pid) & np.isclose(SG.sigma, s)].set_index("role")[m]
                    dd.append(a[r.role_sharp] - a[r.role_soft])
                    sh.append(a[r.role_soft] - float(L[(L.dataset == ds) & (L.pid == r.pid) & (L.role == r.role_soft)
                                                     & (L.condition == "native")][m].iloc[0]))
                hl, lo, hi, n = hodges_lehmann(dd)
                hl2, lo2, hi2, _ = hodges_lehmann(sh)
                sw.append(dict(dataset=ds, sigma=s, measure=m, n=n, kernel_diff=hl, kernel_lo=lo, kernel_hi=hi,
                               soft_shift=hl2, soft_lo=lo2, soft_hi=hi2))
    pd.DataFrame(sw).to_csv(os.path.join(OUT, "pairs_sigma_sweep.csv"), index=False)

    # noise-convolution decomposition
    H = np.load(os.path.join(OUT, "pairs_softhist.npz"))
    bins = H["bins"]
    v = (bins[:-1] + bins[1:]) / 2
    dec = []
    for _, r in W.iterrows():
        key = f"{r.dataset}|{r.pid}"
        if key not in H.files:
            continue
        c = H[key].astype(float)
        sa2 = r.noise_sharp ** 2 - r.noise_soft ** 2
        if not np.isfinite(sa2) or sa2 <= 0:
            continue
        sa = np.sqrt(sa2)
        F = lambda x: norm.cdf((x - v) / sa)
        pf = F(-30) - F(-190)
        pm = F(150) - F(-29)
        a_, b_ = (-29 - v) / sa, (150 - v) / sa
        ex = v * (norm.cdf(b_) - norm.cdf(a_)) + sa * (norm.pdf(a_) - norm.pdf(b_))
        pred_imat = 100 * (c * pf).sum() / (c * (pf + pm)).sum()
        pred_win = (c * ex).sum() / (c * pm).sum()
        dec.append(dict(dataset=r.dataset, pid=r.pid, vendor=r.vendor, sigma_add=sa, noise_sharp=r.noise_sharp,
                        noise_soft=r.noise_soft,
                        imat_soft=r.imat_pct_soft_native, imat_sharp=r.imat_pct_sharp_native, imat_pred=pred_imat,
                        win_soft=r.smd_window_hu_soft_native, win_sharp=r.smd_window_hu_sharp_native, win_pred=pred_win))
    D = pd.DataFrame(dec)
    D["imat_frac_explained"] = (D.imat_pred - D.imat_soft) / (D.imat_sharp - D.imat_soft)
    D["win_frac_explained"] = (D.win_pred - D.win_soft) / (D.win_sharp - D.win_soft)
    D.to_csv(os.path.join(OUT, "pairs_decomposition.csv"), index=False)

    print("flow:", flow)
    print("fixed noise targets (muscle residual SD, HU):", {k: round(v, 1) for k, v in fixed_target.items()})
    print(P.groupby(["dataset", "vendor"]).size().to_dict(), "equal thickness:", P.groupby("dataset").equal_thickness.sum().to_dict())
    print(S[(S.stratum == "all")].round(2).to_string(index=False))
    for ds in ["NLST", "RG"]:
        d = D[D.dataset == ds]
        if len(d):
            from scipy.stats import pearsonr
            print(ds, "decomposition n", len(d), "IMAT: soft %.1f sharp %.1f predicted %.1f; frac explained median %.2f; r(pred, obs) %.2f" % (
                d.imat_soft.median(), d.imat_sharp.median(), d.imat_pred.median(), d.imat_frac_explained.median(),
                pearsonr(d.imat_pred, d.imat_sharp)[0]),
                "| window: soft %.1f sharp %.1f pred %.1f frac %.2f" % (d.win_soft.median(), d.win_sharp.median(),
                                                                        d.win_pred.median(), d.win_frac_explained.median()))


if __name__ == "__main__":
    main()
