"""Sci Rep: noise-referenced filtering (revision 1 plan: docs/srep_revision/R1_analysis_plan.md, section 1; revision 2:
noise measured in the liver).

Every image, sharp or soft, whose image noise exceeds an absolute reference R is smoothed with the smallest in-plane
Gaussian sigma of the stored grid (0.1-mm steps) that brings its noise to <= R; images at or below R keep their native
values. Masks are unchanged. Primary R = 25 HU; sensitivity 20 and 30 HU; descriptive sweep 15-45 HU.
Noise (revision 2): robust SD (1.4826 x MAD) of the residual after a 3-mm local mean in the liver at least 5 mm from its
boundary on the same slice (NOISE = "liv_mad"; results/srep/noise_est_*.csv from pipeline/srep_noise_estimators.py),
because the muscle residual also contains real intramuscular fat (stats/srep_noise_estimators_analysis.py); where the
slice has no liver, the robust SD in the erector spinae (es_mad). The muscle-noise version of revision 1 is reported as
a sensitivity analysis (noise_ref_alt_summary.csv).

Pairs (results/srep/noise_est_pairs.csv, gauss_only grid 0-2 mm for both reconstructions):
  sharp - soft after noise-referenced filtering, and the change of the soft image itself, HL estimate with 95% CI,
  overall and by vendor; the fixed-filter soft shift is split into the part expected from noise alone for 40-HU
  tissue (normal model) and the remainder.
Clinical CTs (results/srep/noise_est_cohort.csv, grid 0-3 mm): noise-referenced exposures for the survival models.
Outputs: results/srep/noise_ref_perpair.csv, noise_ref_summary.csv, noise_ref_sweep.csv, noise_ref_exposures.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from er_pairs_analysis import hodges_lehmann, MEAS  # noqa: E402

ROOT = os.path.dirname(HERE)
R = lambda *p: os.path.join(ROOT, *p)
OUT = R("results", "srep")
R_PRIMARY, R_SENS, R_SWEEP = 25.0, (20.0, 30.0), [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
FAT, MUS = (-190.0, -30.0), (-29.0, 150.0)
MU_REF = 40.0  # tissue mean for the noise-only expectation (HU)
NOISE, FALLBACK = "liv_mad", "es_mad"


def noise_only_imat(sd, mu=MU_REF):
    """Intramuscular fat (%) that noise of SD `sd` alone produces in uniform tissue of mean mu (normal model)."""
    sd = np.maximum(np.asarray(sd, float), 1e-6)
    f = norm.cdf((FAT[1] - mu) / sd) - norm.cdf((FAT[0] - mu) / sd)
    m = norm.cdf((MUS[1] - mu) / sd) - norm.cdf((MUS[0] - mu) / sd)
    return 100 * f / (f + m)


def pick_ref(g, ref):
    """Row of a sigma grid (sorted) with the smallest sigma whose noise is <= ref (native if already <= ref)."""
    col = NOISE if g[NOISE].notna().all() else FALLBACK
    ok = g[g[col] <= ref]
    r = (ok.iloc[0] if len(ok) else g.iloc[-1]).copy()
    r["noise_muscle"] = r[col]
    return r


def pairs():
    L = pd.read_csv(R("results", "srep", "noise_est_pairs.csv"))
    L["pid"] = L.pid.astype(str)
    P = pd.read_csv(R("results", "er", "pairs_perpair.csv"))
    P["pid"] = P.pid.astype(str)
    G = L[L.condition == "gauss_only"].sort_values("sigma")
    fx = L[(L.condition == "slab_gauss") & np.isclose(L.sigma, 0.75)].set_index(["dataset", "pid", "role"])
    grid = {k: g for k, g in G.groupby(["dataset", "pid", "role"])}
    rows, sweep = [], []
    for _, r in P.iterrows():
        gs, gf = grid[(r.dataset, r.pid, r.role_sharp)], grid[(r.dataset, r.pid, r.role_soft)]
        nl = lambda g_: g_.sort_values("sigma").iloc[0][NOISE]
        out = dict(dataset=r.dataset, pid=r.pid, vendor=r.vendor, equal_thickness=r.equal_thickness,
                   noise_soft_es=r.noise_soft, noise_sharp_es=r.noise_sharp, noise_soft=nl(gf), noise_sharp=nl(gs),
                   noise_soft_fixed=fx.loc[(r.dataset, r.pid, r.role_soft), NOISE])
        for ref in sorted(set(R_SWEEP) | {R_PRIMARY, *R_SENS}):
            a, b = pick_ref(gs, ref), pick_ref(gf, ref)
            tag = f"r{int(ref)}"
            out[f"sigma_sharp_{tag}"], out[f"sigma_soft_{tag}"] = a.sigma, b.sigma
            out[f"noise_sharp_{tag}"], out[f"noise_soft_{tag}"] = a.noise_muscle, b.noise_muscle
            for m in MEAS:
                out[f"{m}_sharp_{tag}"], out[f"{m}_soft_{tag}"] = a[m], b[m]
        for m in MEAS:
            out[f"{m}_soft_native"], out[f"{m}_sharp_native"] = r[f"{m}_soft_native"], r[f"{m}_sharp_native"]
            out[f"{m}_soft_fixed"], out[f"{m}_sharp_fixed"] = r[f"{m}_soft_fixed"], r[f"{m}_sharp_fixed"]
            out[f"{m}_sharp_adapt_pair"] = r[f"{m}_sharp_adapt_pair"]
        # noise-only expectation for the soft image (40-HU tissue): native soft noise -> noise after processing
        out["soft_noiseonly_fixed"] = noise_only_imat(out["noise_soft_fixed"]) - noise_only_imat(out["noise_soft"])
        out["soft_noiseonly_r25"] = noise_only_imat(out["noise_soft_r25"]) - noise_only_imat(out["noise_soft"])
        out["soft_noiseonly_native"] = noise_only_imat(out["noise_soft"])
        rows.append(out)
    W = pd.DataFrame(rows)
    W.to_csv(os.path.join(OUT, "noise_ref_perpair.csv"), index=False)

    est = []
    strata = [("all", lambda d: d)] + [(f"vendor={v}", (lambda v: lambda d: d[d.vendor == v])(v))
                                       for v in sorted(W.vendor.unique())] + [("equal thickness", lambda d: d[d.equal_thickness])]
    for ds in ["NLST", "RG"]:
        for sname, f in strata:
            d = f(W[W.dataset == ds])
            if len(d) < 6:
                continue
            for m in MEAS:
                cand = {"native": d[f"{m}_sharp_native"] - d[f"{m}_soft_native"],
                        "fixed": d[f"{m}_sharp_fixed"] - d[f"{m}_soft_fixed"],
                        "sharp-only": d[f"{m}_sharp_adapt_pair"] - d[f"{m}_soft_native"],
                        "soft shift, fixed": d[f"{m}_soft_fixed"] - d[f"{m}_soft_native"]}
                for ref in sorted(set(R_SWEEP) | {R_PRIMARY, *R_SENS}):
                    t = f"r{int(ref)}"
                    cand[f"noise-referenced {int(ref)} HU"] = d[f"{m}_sharp_{t}"] - d[f"{m}_soft_{t}"]
                    cand[f"soft shift, noise-referenced {int(ref)} HU"] = d[f"{m}_soft_{t}"] - d[f"{m}_soft_native"]
                    cand[f"sharp vs native soft, noise-referenced {int(ref)} HU"] = d[f"{m}_sharp_{t}"] - d[f"{m}_soft_native"]
                if m == "imat_pct":
                    cand["soft shift, fixed, noise-only expectation"] = d["soft_noiseonly_fixed"]
                    cand["soft shift, fixed, remainder"] = (d[f"{m}_soft_fixed"] - d[f"{m}_soft_native"]) - d["soft_noiseonly_fixed"]
                    cand["soft shift, noise-referenced 25 HU, noise-only expectation"] = d["soft_noiseonly_r25"]
                    cand["soft native, noise-only expectation"] = d["soft_noiseonly_native"]
                for k, v in cand.items():
                    hl, lo, hi, n = hodges_lehmann(v)
                    est.append(dict(dataset=ds, stratum=sname, measure=m, estimand=k, n=n, hl=hl, lo=lo, hi=hi,
                                    median=float(np.nanmedian(v)), within2=float(np.mean(np.abs(v) <= 2)) if m == "imat_pct" else np.nan))
    S = pd.DataFrame(est)
    S.to_csv(os.path.join(OUT, "noise_ref_summary.csv"), index=False)
    # share of images changed and sigma used
    sw = []
    for ds in ["NLST", "RG"]:
        d = W[W.dataset == ds]
        for ref in R_SWEEP:
            t = f"r{int(ref)}"
            sw.append(dict(dataset=ds, ref=ref, soft_filtered=float((d[f"sigma_soft_{t}"] > 0).mean()),
                           sharp_filtered=float((d[f"sigma_sharp_{t}"] > 0).mean()),
                           sigma_soft_median=float(d[f"sigma_soft_{t}"].median()), sigma_sharp_median=float(d[f"sigma_sharp_{t}"].median()),
                           soft_unreached=int((d[f"noise_soft_{t}"] > ref).sum()), sharp_unreached=int((d[f"noise_sharp_{t}"] > ref).sum())))
    pd.DataFrame(sw).to_csv(os.path.join(OUT, "noise_ref_sweep.csv"), index=False)
    return W, S


def cohort():
    """Noise-referenced exposures (R = 25, 20, 30 HU) for every clinical CT, irrespective of kernel."""
    a = pd.read_csv(R("results", "srep", "noise_est_cohort.csv"))
    a = a[a["error"].isna()] if "error" in a else a
    rows = []
    for cid, g in a.groupby("case_id"):
        g = g.sort_values("sigma")
        out = dict(case_id=cid, noise_native=g.iloc[0][NOISE if g[NOISE].notna().all() else FALLBACK],
                   noise_estimator=NOISE if g[NOISE].notna().all() else FALLBACK)
        for ref in (R_PRIMARY, *R_SENS):
            s = pick_ref(g, ref)
            t = f"r{int(ref)}"
            out.update({f"sigma_{t}": s.sigma, f"noise_{t}": s.noise_muscle, f"imat_{t}": s.imat_pct,
                        f"window_{t}": s.smd_window_hu, f"whole_{t}": s.smd_all_hu, f"area_{t}": s.sma_cm2})
        rows.append(out)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "noise_ref_exposures.csv"), index=False)
    return d


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    W, S = pairs()
    C = cohort()
    show = S[(S.stratum == "all") & (S.measure.isin(["imat_pct", "smd_window_hu", "smd_all_hu", "sma_cm2"]))
             & S.estimand.isin(["native", "fixed", "sharp-only", "soft shift, fixed", "noise-referenced 25 HU",
                                "soft shift, noise-referenced 25 HU", "noise-referenced 20 HU", "noise-referenced 30 HU",
                                "soft shift, fixed, noise-only expectation", "soft shift, fixed, remainder",
                                "soft native, noise-only expectation"])]
    pd.set_option("display.width", 200)
    print(show[["dataset", "measure", "estimand", "n", "hl", "lo", "hi", "within2"]].round(2).to_string(index=False))
    print(pd.read_csv(os.path.join(OUT, "noise_ref_sweep.csv")).round(2).to_string(index=False))
    print("clinical CTs:", len(C), "filtered at 25 HU:", int((C.sigma_r25 > 0).sum()))
