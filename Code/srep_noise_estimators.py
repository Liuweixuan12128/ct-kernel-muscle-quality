"""Sci Rep revision 2: noise estimators that real intramuscular fat cannot inflate, back-convolution of the soft-image
change, and a slab variant of noise-referenced filtering.

Per image, at T12 with the masks of the main analysis (pairs: the soft image's masks mapped to the partner by world
coordinates; clinical CTs: their own masks), on every step of an in-plane Gaussian sigma grid (pairs 0-2 mm, clinical
CTs 0-3 mm; 0.1 mm; no slab, as in sharp-only and noise-referenced filtering) and, for pairs, of a 5-mm slab + in-plane
sigma grid (0-1.5 mm, plus 0.75 mm, the fixed filter):
  noise   es_sd    residual SD in the eroded erector spinae after a 3-mm local mean (the estimator of the main analysis)
          es_mad   1.4826 x median absolute deviation of the same residual (robust to sparse fat streaks)
          liv_sd, liv_mad   the same in the liver at least 5 mm from its boundary (no muscle fat); NaN if < 300 pixels
  measures (intramuscular fat, muscle-window density, whole-mask density, muscle area)
Soft images of pairs only, for each processed version (in-plane grid and the fixed filter): the closed-form noise-
convolution model applied backwards, i.e. the filtered image's muscle histogram convolved with Gaussian noise of variance
(native noise^2 - filtered noise^2), predicts the native intramuscular fat and muscle-window mean (noise variance from the muscle estimator es_sd and,
as pred_native_imat_liv, from the liver estimator liv_mad). Observed native minus
predicted native is the part of the change that adding the removed noise back does not restore (smoothing).
Output: results/srep/noise_est_pairs.csv, results/srep/noise_est_cohort.csv
Usage: env/bin/python pipeline/srep_noise_estimators.py --workers 24
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from er_pairs_measure import canon, slab_gauss, measures, muscle_noise, map_mask, PAD_HU, BINS  # noqa: E402

R = lambda *p: os.path.join(ROOT, *p)
GRID_P = [round(x, 1) for x in np.arange(0, 2.01, 0.1)]
GRID_S = sorted({round(x, 1) for x in np.arange(0, 1.51, 0.1)} | {0.75})  # 0.75 = the fixed filter
GRID_C = [round(x, 1) for x in np.arange(0, 3.01, 0.1)]
V = (BINS[:-1] + BINS[1:]) / 2


def resid(sl, region, px, erode_px=2):
    """Residual after a 3-mm Gaussian local mean inside region (as muscle_noise), on the eroded core."""
    core = ndi.binary_erosion(region, iterations=erode_px) if erode_px else region
    if core.sum() < 100:
        return None
    filled = np.where(region, sl, np.nan)
    filled = np.where(np.isnan(filled), np.nanmean(filled), filled)
    return (filled - ndi.gaussian_filter(filled, sigma=3.0 / px[0]))[core]


def noise_set(sl, esm, liv, px):
    out = {}
    for name, reg, e in [("es", esm, 2), ("liv", liv, 0)]:
        r = resid(sl, reg, px, e) if reg is not None and reg.sum() >= 300 else None
        out[f"{name}_sd"] = float(r.std()) if r is not None else np.nan
        out[f"{name}_mad"] = float(1.4826 * np.median(np.abs(r - np.median(r)))) if r is not None else np.nan
    return out


def liver_core(liv2d, px, mm=5.0):
    if liv2d is None or liv2d.sum() < 300:
        return None
    return ndi.distance_transform_edt(liv2d, sampling=px) > mm


def back_convolve(hist, s_add):
    """Native intramuscular fat and muscle-window mean predicted from a filtered-image histogram plus noise s_add."""
    c = hist.astype(float)
    if not np.isfinite(s_add) or s_add <= 0 or c.sum() == 0:
        return np.nan, np.nan
    F = lambda x: norm.cdf((x - V) / s_add)
    pf, pm = F(-30) - F(-190), F(150) - F(-29)
    a_, b_ = (-29 - V) / s_add, (150 - V) / s_add
    ex = V * (norm.cdf(b_) - norm.cdf(a_)) + s_add * (norm.pdf(a_) - norm.pdf(b_))
    return 100 * (c * pf).sum() / (c * (pf + pm)).sum(), (c * ex).sum() / (c * pm).sum()


def one_pair(job):
    from bodycomp_metrics import Case, vertebra_info, INV
    ds, pid, a_nii, a_seg, b_nii, role_soft = job
    try:
        case = Case(a_nii, a_seg, harmonize=False)
        zA = int(round(vertebra_info(case.segs["total"], "T12")[0]))
        A, affA, zmA = canon(a_nii)
        B, affB, zmB = canon(b_nii)
        wz = affA @ np.array([A.shape[0] / 2, A.shape[1] / 2, zA, 1])
        zB = int(np.rint((np.linalg.inv(affB) @ wz)[2]))
        liver3d = case.segs["total"] == INV["total"]["liver"]
        vA, vB = A[:, :, zA] > PAD_HU, B[:, :, zB] > PAD_HU
        img = {"A": (A, zA, zmA, case.muscle[:, :, zA] & vA, case.esm[:, :, zA] & vA, liver3d[:, :, zA] & vA),
               "B": (B, zB, zmB, map_mask(case.muscle, affA, B.shape, affB, zB) & vB,
                     map_mask(case.esm, affA, B.shape, affB, zB) & vB, map_mask(liver3d, affA, B.shape, affB, zB) & vB)}
        rows = []
        for role, (Vol, z, zm, mus, esm, liv) in img.items():
            px, pc = zm[:2], zm[0] * zm[1] / 100
            lc = liver_core(liv, px)
            soft = role == role_soft
            base = dict(dataset=ds, pid=pid, role=role, soft=int(soft), liver_px=int(lc.sum()) if lc is not None else 0)
            nat = slab_gauss(Vol, z, zm[2], px, 0.0, slab=False)
            n_nat = muscle_noise(nat, esm, px)
            l_nat = noise_set(nat, esm, lc, px)["liv_mad"]
            for cond, grid, slab in [("gauss_only", GRID_P, False), ("slab_gauss", GRID_S, True)]:
                for s in grid:
                    sl = slab_gauss(Vol, z, zm[2], px, s, slab=slab)
                    r = dict(base, condition=cond, sigma=s, **noise_set(sl, esm, lc, px), **measures(sl, mus, esm, pc))
                    if soft:
                        h = np.histogram(sl[mus], bins=BINS)[0]
                        r["pred_native_imat"], r["pred_native_window"] = back_convolve(h, np.sqrt(max(n_nat ** 2 - r["es_sd"] ** 2, 0)))
                        r["pred_native_imat_liv"], _ = back_convolve(h, np.sqrt(max(l_nat ** 2 - r["liv_mad"] ** 2, 0)))
                    rows.append(r)
        return rows
    except Exception as e:
        return [dict(dataset=ds, pid=pid, error=repr(e))]


def one_ct(args):
    from bodycomp_metrics import Case, vertebra_info, INV
    cohort, cid = args
    try:
        c = Case(R("data", "nifti", cohort, f"{cid}.nii.gz"), R("data", "seg", cohort, cid), harmonize=False)
        zc, ok = vertebra_info(c.segs["total"], "T12")
        if zc is None or not ok:
            return []
        z = int(round(zc))
        sl0 = c.raw[:, :, z]
        mus, esm = c.muscle[:, :, z], c.esm[:, :, z]
        lc = liver_core(c.segs["total"][:, :, z] == INV["total"]["liver"], c.zooms[:2])
        rows = []
        for s in GRID_C:
            sl = ndi.gaussian_filter(sl0, sigma=(s / c.zooms[0], s / c.zooms[1])) if s > 0 else sl0
            rows.append(dict(cohort=cohort, case_id=cid, sigma=s, liver_px=int(lc.sum()) if lc is not None else 0,
                             **noise_set(sl, esm, lc, c.zooms[:2]), **measures(sl, mus, esm, c.px_cm2)))
        return rows
    except Exception as e:
        return [dict(cohort=cohort, case_id=cid, error=repr(e))]


def pair_jobs():
    pw = pd.read_csv(R("results", "er", "pairs_perpair.csv"))
    sel = pd.read_csv(R("data", "nlst", "nlst_selected_series_last.csv"))[["pid", "round"]]
    ver = pd.read_csv(R("results", "r1005", "recon_pairs_dicom.csv")).set_index("alt_id")
    out = []
    for _, r in pw.iterrows():
        if r.dataset == "NLST":
            rnd = int(sel[sel.pid.astype(str) == str(r.pid)]["round"].iloc[0])
            name = f"{r.pid}_T{rnd}"
            out.append(("NLST", str(r.pid), R("data", "nlst", "nifti", f"{name}.nii.gz"), R("data", "seg", "nlst", name),
                        R("data", "nlst", "pairs", "nifti", f"{r.pid}_alt.nii.gz"), r.role_soft))
        else:
            cid = ver.loc[r.pid, "case_id"]
            out.append(("RG", r.pid, R("data", "nifti", "public_rg", f"{cid}.nii.gz"), R("data", "seg", "public_rg", cid),
                        R("data", "nifti", "public_rg_recon", f"{r.pid}.nii.gz"), r.role_soft))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(R("results", "srep"), exist_ok=True)
    pj = pair_jobs()
    cj = []
    for cohort in ("public_rg", "tcga_cptac"):
        bc = pd.read_csv(R("results", f"{cohort}_bodycomp.csv"))
        cj += [(cohort, i) for i in bc[bc.T12_complete == 1].case_id.astype(str)]
    if a.limit:
        pj, cj = pj[:a.limit], cj[:a.limit]
    tag = "_test" if a.limit else ""
    with Pool(a.workers) as p:
        rows = [r for rr in p.imap_unordered(one_pair, pj, chunksize=1) for r in rr]
        pd.DataFrame(rows).to_csv(R("results", "srep", f"noise_est_pairs{tag}.csv"), index=False)
        print("pairs", len(pj), "rows", len(rows), flush=True)
        rows = [r for rr in p.imap_unordered(one_ct, cj, chunksize=1) for r in rr]
        pd.DataFrame(rows).to_csv(R("results", "srep", f"noise_est_cohort{tag}.csv"), index=False)
        print("clinical CTs", len(cj), "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
