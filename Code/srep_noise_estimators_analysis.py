"""Sci Rep revision 2: does real intramuscular fat inflate the muscle noise estimate, and does it matter?

Inputs: results/srep/noise_est_pairs.csv and noise_est_cohort.csv (pipeline/srep_noise_estimators.py).
1. Test (review round 2, point 1): the ratio of erector spinae noise to liver noise, and the excess variance
   es_sd^2 - liv_sd^2, against intramuscular fat measured after fixed filtering (a fat measure that noise does not drive),
   in native images by dataset and kernel class (Spearman rho; medians by tertile of intramuscular fat).
2. Noise-referenced filtering with four noise estimators (es_sd = main analysis; es_mad; liv_sd; liv_mad), reference 25 HU,
   in-plane (as the main analysis) and with a 5-mm slab first (thin-slice variant): sharp - soft differences, change of
   the soft image, share of images filtered; clinical-CT exposures for the survival sensitivity analysis.
3. Back-convolution of the soft-image change (review round 2, point 2): under fixed filtering and noise-referenced
   filtering, the filtered soft histogram plus the removed noise predicts the native value; the change splits into a
   noise part (predicted native - filtered) and a remainder (observed native - predicted native).
Outputs: results/srep/noise_est_test.csv, noise_ref_alt_summary.csv, noise_ref_alt_exposures.csv,
         soft_change_decomposition.csv, noise_ref_alt_perpair.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from er_pairs_analysis import hodges_lehmann  # noqa: E402

ROOT = os.path.dirname(HERE)
R = lambda *p: os.path.join(ROOT, *p)
OUT = R("results", "srep")
REF = 25.0
EST = ["es_sd", "es_mad", "liv_sd", "liv_mad"]
MEAS = ["imat_pct", "smd_window_hu", "smd_all_hu", "sma_cm2"]


def pick(g, est, ref=REF):
    """Smallest sigma with est <= ref (native if already <= ref); NaN row if the estimator is unavailable."""
    g = g.sort_values("sigma")
    if g[est].isna().all():
        return None
    ok = g[g[est] <= ref]
    return ok.iloc[0] if len(ok) else g.iloc[-1]


def main():
    P = pd.read_csv(os.path.join(OUT, "noise_est_pairs.csv"))
    assert "error" not in P or P["error"].isna().all(), P[P["error"].notna()][["pid", "error"]]
    P["pid"] = P.pid.astype(str)
    C = pd.read_csv(os.path.join(OUT, "noise_est_cohort.csv"))
    C = C[C["error"].isna()] if "error" in C else C
    W = pd.read_csv(R("results", "er", "pairs_perpair.csv")); W["pid"] = W.pid.astype(str)
    K = pd.read_csv(R("results", "er", "cohort_kernel_percase.csv"))
    AE = pd.read_csv(R("results", "er", "adaptive_exposures.csv"))[["case_id", "kernel_type"]]

    # 1. test
    rows = []
    nat = P[(P.condition == "gauss_only") & (P.sigma == 0)].merge(W[["dataset", "pid", "vendor", "imat_pct_soft_fixed"]], on=["dataset", "pid"])
    nat["fat_ref"] = nat.imat_pct_soft_fixed  # same anatomy: soft image, fixed filter
    cn = C[C.sigma == 0].merge(AE, on="case_id").merge(K[["case_id", "imat_fixed"]], on="case_id", how="left")
    cn = cn[cn.imat_fixed.notna()].rename(columns={"imat_fixed": "fat_ref"})
    groups = [("NLST", "soft", nat[(nat.dataset == "NLST") & (nat.soft == 1)]), ("NLST", "sharp", nat[(nat.dataset == "NLST") & (nat.soft == 0)]),
              ("RG pairs", "soft", nat[(nat.dataset == "RG") & (nat.soft == 1)]),
              ("Clinical", "soft", cn[cn.kernel_type == "soft"]), ("Clinical", "sharp", cn[cn.kernel_type == "sharp"])]
    test_rows = []
    for ds, k, g in groups:
        g = g[g.liv_sd.notna()].copy()
        g["ratio"] = g.es_sd / g.liv_sd
        g["ratio_mad"] = g.es_mad / g.liv_mad
        g["excess_var"] = g.es_sd ** 2 - g.liv_sd ** 2
        rho, p = spearmanr(g.ratio, g.fat_ref)
        rho_m, p_m = spearmanr(g.ratio_mad, g.fat_ref)
        q = pd.qcut(g.fat_ref, 3, labels=["low", "middle", "high"])
        med = g.groupby(q, observed=True)[["fat_ref", "ratio", "es_sd", "liv_sd"]].median()
        test_rows.append(dict(dataset=ds, kernel=k, n=len(g), n_no_liver=0, ratio_median=g.ratio.median(), rho=rho, p=p,
                              rho_mad=rho_m, p_mad=p_m, es_sd_median=g.es_sd.median(), liv_sd_median=g.liv_sd.median(),
                              **{f"ratio_{t}": med.loc[t, "ratio"] for t in med.index}, **{f"fat_{t}": med.loc[t, "fat_ref"] for t in med.index}))
        for _, r in g.iterrows():
            rows.append(dict(dataset=ds, kernel=k, id=r.get("pid", r.get("case_id")), es_sd=r.es_sd, liv_sd=r.liv_sd, es_mad=r.es_mad,
                             liv_mad=r.liv_mad, ratio=r.ratio, fat_ref=r.fat_ref))
    T = pd.DataFrame(test_rows)
    T.to_csv(os.path.join(OUT, "noise_est_test.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "noise_est_test_perimage.csv"), index=False)

    # 2. noise-referenced filtering with each estimator, in-plane and with a 5-mm slab
    pp, summ = [], []
    for (ds, pid), g in P.groupby(["dataset", "pid"]):
        w = W[(W.dataset == ds) & (W.pid == pid)].iloc[0]
        sh, so = g[g.soft == 0], g[g.soft == 1]
        nso = so[(so.condition == "gauss_only") & (so.sigma == 0)].iloc[0]
        out = dict(dataset=ds, pid=pid, vendor=w.vendor, equal_thickness=w.equal_thickness, imat_soft_native=nso.imat_pct,
                   liver_available=int(min(sh.liver_px.min(), so.liver_px.min()) >= 300))
        for m in MEAS:
            out[f"{m}_soft_native"] = nso[m]
        for est in EST:
            for cond, lab in [("gauss_only", "inplane"), ("slab_gauss", "slab")]:
                a, b = pick(sh[sh.condition == cond], est), pick(so[so.condition == cond], est)
                if a is None or b is None:
                    continue
                t = f"{est}_{lab}"
                out[f"sigma_sharp_{t}"], out[f"sigma_soft_{t}"] = a.sigma, b.sigma
                out[f"filtered_soft_{t}"] = int(not (cond == "gauss_only" and b.sigma == 0))
                for m in MEAS:
                    out[f"{m}_sharp_{t}"], out[f"{m}_soft_{t}"] = a[m], b[m]
                out[f"pred_native_{t}"] = b.pred_native_imat
                out[f"pred_native_liv_{t}"] = b.pred_native_imat_liv
        pp.append(out)
    D = pd.DataFrame(pp)
    # fixed filter (5-mm slab + Gaussian 0.75 mm) of the soft image and its back-convolution
    fxp = P[(P.soft == 1) & (P.condition == "slab_gauss") & np.isclose(P.sigma, 0.75)][["dataset", "pid", "pred_native_imat", "pred_native_imat_liv", "imat_pct", "es_sd"]]
    fxp.columns = ["dataset", "pid", "pred_native_fixed", "pred_native_fixed_liv", "imat_soft_fixed_grid", "es_sd_fixed_grid"]
    D = D.merge(fxp, on=["dataset", "pid"], how="left").merge(W[["dataset", "pid", "imat_pct_soft_fixed"]], on=["dataset", "pid"])
    D.to_csv(os.path.join(OUT, "noise_ref_alt_perpair.csv"), index=False)
    for ds in ["NLST", "RG"]:
        d = D[D.dataset == ds]
        for est in EST:
            for lab in ["inplane", "slab"]:
                t = f"{est}_{lab}"
                if f"imat_pct_sharp_{t}" not in d:
                    continue
                dd = d[d[f"imat_pct_sharp_{t}"].notna()]
                for m in MEAS:
                    for estd, v in [("sharp - soft", dd[f"{m}_sharp_{t}"] - dd[f"{m}_soft_{t}"]),
                                    ("soft change", dd[f"{m}_soft_{t}"] - dd[f"{m}_soft_native"])]:
                        hl, lo, hi, n = hodges_lehmann(v)
                        summ.append(dict(dataset=ds, estimator=est, variant=lab, measure=m, estimand=estd, n=n, hl=hl, lo=lo, hi=hi,
                                         within2=float((v.abs() <= 2).mean()) if (m == "imat_pct" and estd == "sharp - soft") else np.nan,
                                         soft_filtered=float(dd[f"filtered_soft_{t}"].mean()) if lab == "inplane" else np.nan,
                                         sigma_soft_median=float(dd[f"sigma_soft_{t}"].median()), sigma_sharp_median=float(dd[f"sigma_sharp_{t}"].median())))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "noise_ref_alt_summary.csv"), index=False)

    # 3. decomposition of the soft-image change
    dec = []
    for ds in ["NLST", "RG"]:
        d = D[D.dataset == ds]
        for lab, filt, pred in [("fixed, liver noise", d.imat_soft_fixed_grid, d.pred_native_fixed_liv),
                                ("noise-referenced 25 HU, liver noise", d.imat_pct_soft_liv_mad_inplane, d.pred_native_liv_liv_mad_inplane),
                                ("fixed, muscle noise", d.imat_soft_fixed_grid, d.pred_native_fixed),
                                ("noise-referenced 25 HU, muscle noise", d.imat_pct_soft_es_sd_inplane, d.pred_native_es_sd_inplane)]:
            ok = pred.notna()
            parts = {"total": (filt - d.imat_soft_native)[ok], "noise part": (filt - pred)[ok], "remainder": (pred - d.imat_soft_native)[ok]}
            for k, v in parts.items():
                hl, lo, hi, n = hodges_lehmann(v)
                dec.append(dict(dataset=ds, processing=lab, part=k, n=n, hl=hl, lo=lo, hi=hi, median=float(v.median())))
    pd.DataFrame(dec).to_csv(os.path.join(OUT, "soft_change_decomposition.csv"), index=False)

    # clinical exposures with each estimator (in-plane)
    ex = []
    for cid, g in C.groupby("case_id"):
        out = dict(case_id=cid, liver_px=int(g.liver_px.iloc[0]))
        for est in EST:
            s = pick(g, est)
            if s is None:
                continue
            out.update({f"sigma_{est}": s.sigma, f"imat_{est}": s.imat_pct, f"window_{est}": s.smd_window_hu,
                        f"whole_{est}": s.smd_all_hu, f"area_{est}": s.sma_cm2})
        ex.append(out)
    pd.DataFrame(ex).to_csv(os.path.join(OUT, "noise_ref_alt_exposures.csv"), index=False)

    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    S = pd.DataFrame(summ)
    print(S[(S.measure == "imat_pct")].round(2).to_string(index=False))
    print(pd.DataFrame(dec).round(2).to_string(index=False))
    print("liver available in pairs:", D.groupby("dataset").liver_available.mean().round(3).to_dict(),
          "clinical:", (pd.DataFrame(ex).liver_px >= 300).mean().round(3))


if __name__ == "__main__":
    main()
