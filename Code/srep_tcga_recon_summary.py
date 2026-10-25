"""Sci Rep revision 2: summary of the routine-dose same-acquisition pairs in TCGA and CPTAC.

Input: results/srep/tcga_recon_dicom.csv (DICOM verification) and tcga_recon_pairs.csv (pipeline/srep_tcga_recon_pairs.py,
grid measurements as pipeline/srep_noise_estimators.one_pair). Eligibility as for the NLST pairs: same acquisition, T12
measurable in the analysed series and inside the partner, and >=95% of the mask area inside the partner's reconstructed
field of view; one pair per patient (equal slice thickness preferred, then the first). Per pair: native, fixed filter
(5-mm slab + Gaussian 0.75 mm), sharp-only (sharp image to the soft image's muscle noise) and noise-referenced filtering
(25 HU, liver noise; muscle robust SD where the slice has no liver). Outputs: results/srep/tcga_recon_summary_perpair.csv,
tcga_recon_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from er_pairs_analysis import hodges_lehmann  # noqa: E402

R = lambda *p: os.path.join(ROOT, *p)
REF = 25.0
MEAS = ["imat_pct", "smd_window_hu", "smd_all_hu", "sma_cm2"]


def mask_ratio(case_id, alt_id):
    """Share of the analysed series' T12 muscle-mask area that lies inside the partner's reconstructed field of view,
    and whether the T12 slice lies inside the partner series (as in pipeline/er_pairs_measure.py)."""
    from bodycomp_metrics import Case, vertebra_info
    from er_pairs_measure import canon, map_mask, PAD_HU
    a_nii, a_seg = R("data", "nifti", "tcga_cptac", f"{case_id}.nii.gz"), R("data", "seg", "tcga_cptac", case_id)
    b_nii = R("data", "nifti", "tcga_cptac_recon", f"{alt_id}.nii.gz")
    ca = Case(a_nii, a_seg, harmonize=False)
    zc, ok = vertebra_info(ca.segs["total"], "T12")
    if zc is None or not ok:
        return dict(alt_id=alt_id, t12_ok=0)
    zA = int(round(zc))
    A, affA, zmA = canon(a_nii)
    B, affB, zmB = canon(b_nii)
    zB = int(np.rint((np.linalg.inv(affB) @ (affA @ np.array([A.shape[0] / 2, A.shape[1] / 2, zA, 1])))[2]))
    if not (2 <= zB < B.shape[2] - 2):
        return dict(alt_id=alt_id, t12_ok=1, inside_partner=0)
    musA = ca.muscle[:, :, zA]
    musB = map_mask(ca.muscle, affA, B.shape, affB, zB) & (B[:, :, zB] > PAD_HU)
    return dict(alt_id=alt_id, t12_ok=1, inside_partner=1,
                mask_valid_area_ratio=float(musB.sum() * zmB[0] * zmB[1] / max(musA.sum() * zmA[0] * zmA[1], 1e-9)))


def pick(g, col, ref):
    g = g.sort_values("sigma")
    ok = g[g[col] <= ref]
    return ok.iloc[0] if len(ok) else g.iloc[-1]


def main():
    from concurrent.futures import ProcessPoolExecutor
    V = pd.read_csv(R("results", "srep", "tcga_recon_dicom.csv"))
    M = pd.read_csv(R("results", "srep", "tcga_recon_pairs.csv"))
    err = M[M["error"].notna()] if "error" in M else M.iloc[:0]
    M = M[M["error"].isna()] if "error" in M else M
    V = V[V.same_acquisition]
    with ProcessPoolExecutor(12) as ex:
        mr = pd.DataFrame(list(ex.map(mask_ratio, V.case_id, V.alt_id)))
    V = V.merge(mr, on="alt_id", how="left")
    V["eligible"] = (V.t12_ok == 1) & (V.inside_partner == 1) & (V.mask_valid_area_ratio >= 0.95) & V.alt_id.isin(M.pid)
    V["equal_thickness"] = pd.to_numeric(V.thickness_primary, errors="coerce").round(2) == pd.to_numeric(V.thickness_alt, errors="coerce").round(2)
    E = V[V.eligible].sort_values(["case_id", "equal_thickness", "alt_id"], ascending=[True, False, True]).groupby("case_id").head(1)
    rows = []
    for _, v in E.iterrows():
        g = M[M.pid == v.alt_id]
        sh, so = g[g.soft == 0], g[g.soft == 1]
        gi = lambda x: x[x.condition == "gauss_only"]
        nat = lambda x: gi(x)[gi(x).sigma == 0].iloc[0]
        fx = lambda x: x[(x.condition == "slab_gauss") & np.isclose(x.sigma, 0.75)].iloc[0]
        ncol = lambda x: "liv_mad" if gi(x).liv_mad.notna().all() else "es_mad"
        out = dict(case_id=v.case_id, alt_id=v.alt_id, collection=v.collection, manufacturer=v.manufacturer,
                   kernel_primary=v.kernel_primary, kernel_alt=v.kernel_alt, equal_thickness=bool(v.equal_thickness),
                   noise_soft=nat(so)[ncol(so)], noise_sharp=nat(sh)[ncol(sh)], noise_soft_es=nat(so).es_sd, noise_sharp_es=nat(sh).es_sd)
        sel = {"native": (nat(so), nat(sh)), "fixed": (fx(so), fx(sh)),
               "sharp_only": (nat(so), pick(gi(sh), "es_sd", nat(so).es_sd)),
               f"r{int(REF)}": (pick(gi(so), ncol(so), REF), pick(gi(sh), ncol(sh), REF))}
        for k, (a, b) in sel.items():
            for m in MEAS:
                out[f"{m.split('_')[0] if m == 'imat_pct' else m}_soft_{k}"] = a[m]
                out[f"{m.split('_')[0] if m == 'imat_pct' else m}_sharp_{k}"] = b[m]
        out["sigma_soft_r25"], out["sigma_sharp_r25"] = sel["r25"][0].sigma, sel["r25"][1].sigma
        rows.append(out)
    P = pd.DataFrame(rows)
    P.to_csv(R("results", "srep", "tcga_recon_summary_perpair.csv"), index=False)
    S = []
    for k in ["native", "fixed", "sharp_only", "r25"]:
        for m in MEAS:
            mm = m.split("_")[0] if m == "imat_pct" else m
            d = P[f"{mm}_sharp_{k}"] - P[f"{mm}_soft_{k}"]
            hl, lo, hi, n = hodges_lehmann(d)
            S.append(dict(strategy=k, measure=m, n=n, hl=hl, lo=lo, hi=hi, within2=float((d.abs() <= 2).mean()) if m == "imat_pct" else np.nan))
        if k != "native":
            for m in ["imat_pct"]:
                d = P[f"imat_soft_{k}"] - P["imat_soft_native"]
                hl, lo, hi, n = hodges_lehmann(d)
                S.append(dict(strategy=k, measure="soft change imat_pct", n=n, hl=hl, lo=lo, hi=hi, within2=np.nan))
    S = pd.DataFrame(S)
    S.to_csv(R("results", "srep", "tcga_recon_summary.csv"), index=False)
    info = dict(candidates=int(len(pd.read_csv(R("results", "srep", "tcga_recon_dicom.csv")))), same_acquisition=int(len(V)),
                eligible_series=int(V.eligible.sum()), pairs=len(P), patients=P.case_id.nunique(),
                measure_errors=len(err), no_t12=int((V.t12_ok != 1).sum()), outside=int(((V.t12_ok == 1) & (V.inside_partner != 1)).sum()),
                fov=int(((V.inside_partner == 1) & (V.mask_valid_area_ratio < 0.95)).sum()),
                equal_thickness=int(P.equal_thickness.sum()),
                soft_noise_median=float(P.noise_soft.median()), sharp_noise_median=float(P.noise_sharp.median()),
                soft_filtered=float((P.sigma_soft_r25 > 0).mean()), vendors="; ".join(f"{k} {v}" for k, v in P.manufacturer.value_counts().items()))
    pd.Series(info).to_csv(R("results", "srep", "tcga_recon_info.csv"))
    pd.set_option("display.width", 200)
    print(pd.Series(info).to_string())
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
