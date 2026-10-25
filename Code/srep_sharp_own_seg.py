"""Sci Rep revision 1 (plan section 3): sharp reconstructions segmented on their own.

For every analysed pair, the sharp reconstruction was segmented separately with TotalSegmentator (NLST:
data/seg/nlst_pairs_sharp; RG: data/seg/public_rg_recon). The soft image keeps its own masks and T12 level, as in the
shared-mask analysis; the sharp image is now measured with its own masks at its own T12 level, so the sharp - soft
difference contains segmentation and threshold effects together (as when each series is analysed independently).

Per pair:
  level      world-z distance between the two T12 centroids (mm)
  masks      Dice of the sharp image's own T12 muscle mask with the soft mask mapped to the sharp grid, on the same
             slice (the shared-mask slice), so that segmentation is compared without the level choice
  measures   soft (own masks): native, fixed, noise-referenced 25 HU; sharp (own masks, own level): native, fixed,
             sharp-only to the paired soft noise, noise-referenced 25 HU (liver noise; r25es: muscle noise); sharp on the shared-mask slice with its own
             masks (segmentation effect at fixed level)
  texture    noise correlation length (lag at which the in-plane autocorrelation of the noise residual falls to 0.5;
             eroded erector spinae, 3-mm local mean removed, mask-normalised) for soft native, sharp native, sharp
             filtered to the paired soft noise and both images at 25 HU
Output: results/srep/sharp_own_seg.csv
Usage: env/bin/python pipeline/srep_sharp_own_seg.py --workers 16
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from er_pairs_measure import canon, slab_gauss, fill_padding, measures, muscle_noise, map_mask, PAD_HU  # noqa: E402,F401

R = lambda *p: os.path.join(ROOT, *p)
GRID = [round(x, 1) for x in np.arange(0, 2.01, 0.1)]
REF = 25.0
MEAS = ["imat_pct", "smd_window_hu", "smd_all_hu", "sma_cm2"]


def corr_length(sl, esm, px, max_lag=12):
    """Lag (mm) at which the mask-normalised in-plane autocorrelation of the noise residual falls to 0.5."""
    core = ndi.binary_erosion(esm, iterations=2)
    if core.sum() < 100:
        return np.nan
    filled = np.where(esm, sl, np.nan)
    filled = np.where(np.isnan(filled), np.nanmean(filled), filled)
    res = filled - ndi.gaussian_filter(filled, sigma=3.0 / px[0])
    m = core.astype(float)
    r = np.where(core, res - res[core].mean(), 0.0)
    s = [2 * n for n in r.shape]
    num = np.fft.irfft2(np.abs(np.fft.rfft2(r, s)) ** 2, s)
    den = np.fft.irfft2(np.abs(np.fft.rfft2(m, s)) ** 2, s)
    ac = np.where(den > 50, num / np.maximum(den, 1e-9), np.nan)
    ac = ac / ac[0, 0]
    prof = np.array([np.nanmean([ac[lag, 0], ac[-lag, 0], ac[0, lag], ac[0, -lag]]) if lag else 1.0
                     for lag in range(max_lag + 1)])
    below = np.nonzero(prof <= 0.5)[0]
    if len(below) == 0:
        return max_lag * px[0]
    i = below[0]
    frac = (prof[i - 1] - 0.5) / max(prof[i - 1] - prof[i], 1e-9)
    return float((i - 1 + frac) * px[0])


def grid_pick(V, z, zm, mus, esm, target, liv=None):
    """Smallest in-plane Gaussian sigma of GRID bringing the noise to <= target: (sigma, slice, noise). Noise is the
    robust SD in the liver core when liv is given (revision 2, primary), otherwise the muscle residual SD."""
    from srep_noise_estimators import noise_set
    for s in GRID:
        sl = slab_gauss(V, z, zm[2], zm[:2], s, slab=False)
        nz = noise_set(sl, esm, liv, zm[:2])["liv_mad"] if liv is not None else muscle_noise(sl, esm, zm[:2])
        if nz <= target or s == GRID[-1]:
            return s, sl, nz


def one(job):
    from bodycomp_metrics import Case, vertebra_info, INV
    from srep_noise_estimators import liver_core, noise_set
    ds, pid, a_nii, a_seg, b_nii, b_seg = job
    try:
        ca, cb = Case(a_nii, a_seg), Case(b_nii, b_seg)
        za_c, oka = vertebra_info(ca.segs["total"], "T12")
        zb_c, okb = vertebra_info(cb.segs["total"], "T12")
        out = dict(dataset=ds, pid=pid)
        if zb_c is None or not okb:
            out["error"] = "no complete T12 in the sharp segmentation"
            return out
        zA, zBo = int(round(za_c)), int(round(zb_c))
        A, affA, zmA = canon(a_nii)
        B, affB, zmB = canon(b_nii)
        wA = affA @ np.array([A.shape[0] / 2, A.shape[1] / 2, za_c, 1])
        wB = affB @ np.array([B.shape[0] / 2, B.shape[1] / 2, zb_c, 1])
        out["level_diff_mm"] = float(wB[2] - wA[2])
        zBm = int(np.rint((np.linalg.inv(affB) @ wA)[2]))  # shared-mask slice of the sharp image
        vA = A[:, :, zA] > PAD_HU
        musA, esmA = ca.muscle[:, :, zA] & vA, ca.esm[:, :, zA] & vA
        vBo = B[:, :, zBo] > PAD_HU
        musBo, esmBo = cb.muscle[:, :, zBo] & vBo, cb.esm[:, :, zBo] & vBo
        pcA, pcB = zmA[0] * zmA[1] / 100, zmB[0] * zmB[1] / 100
        # mask agreement on the shared-mask slice
        if 2 <= zBm < B.shape[2] - 2:
            vBm = B[:, :, zBm] > PAD_HU
            mapped = map_mask(ca.muscle, affA, B.shape, affB, zBm) & vBm
            own = cb.muscle[:, :, zBm] & vBm
            out["dice_muscle"] = float(2 * (mapped & own).sum() / max(mapped.sum() + own.sum(), 1))
            nat_m = B[:, :, zBm]
            for k, v in measures(nat_m, own, cb.esm[:, :, zBm] & vBm, pcB).items():
                out[f"sharp_ownmask_sharedslice_native_{k}"] = v
            for k, v in measures(nat_m, mapped, map_mask(ca.esm, affA, B.shape, affB, zBm) & vBm, pcB).items():
                out[f"sharp_sharedmask_native_{k}"] = v
        # soft image, own masks
        natA = A[:, :, zA]
        nA = muscle_noise(natA, esmA, zmA[:2])
        out["noise_soft"] = nA
        for k, v in measures(natA, musA, esmA, pcA).items():
            out[f"soft_native_{k}"] = v
        for k, v in measures(slab_gauss(A, zA, zmA[2], zmA[:2], 0.75, slab=True), musA, esmA, pcA).items():
            out[f"soft_fixed_{k}"] = v
        livA = liver_core((ca.segs["total"][:, :, zA] == INV["total"]["liver"]) & vA, zmA[:2])
        livB = liver_core((cb.segs["total"][:, :, zBo] == INV["total"]["liver"]) & vBo, zmB[:2])
        out["liver_soft"], out["liver_sharp"] = int(livA is not None), int(livB is not None)
        out["noise_soft_liver"] = noise_set(natA, esmA, livA, zmA[:2])["liv_mad"]
        sA, slA, nA25 = grid_pick(A, zA, zmA, musA, esmA, REF, livA if livA is not None else None)
        out["sigma_soft_r25"] = sA
        for k, v in measures(slA, musA, esmA, pcA).items():
            out[f"soft_r25_{k}"] = v
        sAe, slAe, _ = grid_pick(A, zA, zmA, musA, esmA, REF)  # muscle-noise version (sensitivity)
        for k, v in measures(slAe, musA, esmA, pcA).items():
            out[f"soft_r25es_{k}"] = v
        # sharp image, own masks and own level
        natB = B[:, :, zBo]
        out["noise_sharp"] = muscle_noise(natB, esmBo, zmB[:2])
        for k, v in measures(natB, musBo, esmBo, pcB).items():
            out[f"sharp_native_{k}"] = v
        for k, v in measures(slab_gauss(B, zBo, zmB[2], zmB[:2], 0.75, slab=True), musBo, esmBo, pcB).items():
            out[f"sharp_fixed_{k}"] = v
        sP, slP, _ = grid_pick(B, zBo, zmB, musBo, esmBo, nA)
        out["sigma_sharp_pair"] = sP
        for k, v in measures(slP, musBo, esmBo, pcB).items():
            out[f"sharp_pair_{k}"] = v
        sB, slB, _ = grid_pick(B, zBo, zmB, musBo, esmBo, REF, livB if livB is not None else None)
        out["sigma_sharp_r25"] = sB
        for k, v in measures(slB, musBo, esmBo, pcB).items():
            out[f"sharp_r25_{k}"] = v
        sBe, slBe, _ = grid_pick(B, zBo, zmB, musBo, esmBo, REF)
        for k, v in measures(slBe, musBo, esmBo, pcB).items():
            out[f"sharp_r25es_{k}"] = v
        # noise texture
        out["corr_soft_native"] = corr_length(natA, esmA, zmA[:2])
        out["corr_soft_r25"] = corr_length(slA, esmA, zmA[:2])
        out["corr_sharp_native"] = corr_length(natB, esmBo, zmB[:2])
        out["corr_sharp_pair"] = corr_length(slP, esmBo, zmB[:2])
        out["corr_sharp_r25"] = corr_length(slB, esmBo, zmB[:2])
        out["px_soft"], out["px_sharp"] = zmA[0], zmB[0]
        return out
    except Exception as e:
        return dict(dataset=ds, pid=pid, error=repr(e))


def jobs():
    pw = pd.read_csv(R("results", "er", "pairs_perpair.csv"))
    sel = pd.read_csv(R("data", "nlst", "nlst_selected_series_last.csv"))[["pid", "round"]]
    out = []
    for _, r in pw[pw.dataset == "NLST"].iterrows():
        rnd = int(sel[sel.pid.astype(str) == str(r.pid)]["round"].iloc[0])
        name = f"{r.pid}_T{rnd}"
        out.append(("NLST", str(r.pid), R("data", "nlst", "nifti", f"{name}.nii.gz"), R("data", "seg", "nlst", name),
                    R("data", "nifti", "nlst_pairs_sharp", f"{r.pid}_alt.nii.gz"), R("data", "seg", "nlst_pairs_sharp", f"{r.pid}_alt")))
    ver = pd.read_csv(R("results", "r1005", "recon_pairs_dicom.csv")).set_index("alt_id")
    for _, r in pw[pw.dataset == "RG"].iterrows():
        cid = ver.loc[r.pid, "case_id"]
        prim = (R("data", "nifti", "public_rg", f"{cid}.nii.gz"), R("data", "seg", "public_rg", cid))
        alt = (R("data", "nifti", "public_rg_recon", f"{r.pid}.nii.gz"), R("data", "seg", "public_rg_recon", r.pid))
        soft, sharp = (prim, alt) if r.role_soft == "A" else (alt, prim)
        out.append(("RG", r.pid, soft[0], soft[1], sharp[0], sharp[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    js = jobs()
    js = [j for j in js if os.path.exists(os.path.join(j[5], "body.nii.gz"))]
    if a.limit:
        js = js[:a.limit]
    print(len(js), "pairs with a finished sharp segmentation", flush=True)
    with Pool(a.workers) as p:
        res = p.map(one, js, chunksize=1)
    d = pd.DataFrame(res)
    os.makedirs(R("results", "srep"), exist_ok=True)
    d.to_csv(R("results", "srep", "sharp_own_seg.csv" if not a.limit else "sharp_own_seg_test.csv"), index=False)
    print(d.drop(columns=[c for c in d.columns if c.startswith("sharp_ownmask") or c.startswith("sharp_sharedmask")]).describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
