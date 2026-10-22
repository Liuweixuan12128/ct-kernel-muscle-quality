"""ER revision: kernel effect in reconstructions of one acquisition, measured with ONE anatomical mask.

Pairs
  NLST          analysed screening CT (segmented) + the kernel-discordant partner series of the same study
  RG            NSCLC-Radiogenomics analysed staging CT (segmented) + alternative reconstruction (results/audit/e3)
Masks come from the segmented series and are mapped to the partner by world coordinates (nearest neighbour), so both
reconstructions are measured inside the same anatomy; only the image intensities differ. Pixels outside the reconstructed
field of view (padding <= -1100 HU) are excluded from the masks and replaced by the nearest valid pixel before filtering.

Per image, at the T12 slice (vertebral-body centroid of the segmented series, mapped by world z):
  * native; published noise matching (5-mm slab + in-plane Gaussian 0.75 mm); slab + Gaussian sigma sweep
  * in-plane Gaussian only, sigma grid 0..2 mm (for sharp-only adaptive matching, chosen post hoc by noise target)
  * muscle mask eroded by 1 and 2 pixels (native and noise-matched)
  * image noise measured in muscle: residual SD in the eroded erector spinae after a 3-mm Gaussian local mean
  * a noise-convolution prediction: the soft image's own voxel values plus Gaussian noise with variance
    (sharp noise^2 - soft noise^2) predict the sharp intramuscular fat % and muscle-window mean (computed later
    from the stored soft-voxel histogram)
Output: results/er/pairs_long.csv, results/er/pairs_meta.csv, results/er/pairs_softhist.npz
Usage: env/bin/python pipeline/er_pairs_measure.py --workers 16
"""
import argparse
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
R = lambda *p: os.path.join(ROOT, *p)
OUT = R("results", "er")
SHARP = "LUNG|BONE|B4|B5|B6|B7|B8|I70|SHARP|YB|YA|FC5|FC3|^D$|^E$|FC8"
HU_MUSCLE, HU_FAT = (-29, 150), (-190, -30)
SLAB_SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
GAUSS_ONLY = [round(x, 2) for x in np.arange(0, 2.01, 0.1)]
BINS = np.arange(-400, 401, 1.0)
PAD_HU = -1100  # values at or below this are padding outside the reconstructed field of view (-2048/-3024), not tissue


def is_sharp(k):
    import re
    return bool(re.search(SHARP, str(k).upper()))


def canon(path):
    img = nib.as_closest_canonical(nib.load(path))
    a = np.asanyarray(img.dataobj).astype(np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    return a, img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


def slab_gauss(vol, z, slice_mm, px, sigma_mm, slab=True, slab_mm=5.0):
    """Processed T12 slice computed on a sub-volume (identical to filtering the whole volume at slice z)."""
    k = max(1, int(round(slab_mm / slice_mm))) if slab else 1
    lo, hi = max(0, z - k - 2), min(vol.shape[2], z + k + 3)
    sub = fill_padding(vol[:, :, lo:hi].copy())
    if k > 1:
        sub = ndi.uniform_filter1d(sub, k, axis=2, mode="nearest")
    sl = sub[:, :, z - lo]
    if sigma_mm > 0:
        sl = ndi.gaussian_filter(sl, sigma=(sigma_mm / px[0], sigma_mm / px[1]))
    return sl


def fill_padding(sub):
    """Replace out-of-field padding by the nearest valid pixel before filtering, so padding is not blurred into tissue."""
    for k in range(sub.shape[2]):
        inv = sub[:, :, k] <= PAD_HU
        if inv.any() and (~inv).any():
            ii, jj = ndi.distance_transform_edt(inv, return_distances=False, return_indices=True)
            sub[:, :, k] = sub[:, :, k][ii, jj]
    return sub


def measures(sl, mus, esm, px_cm2):
    mw = mus & (sl >= HU_MUSCLE[0]) & (sl <= HU_MUSCLE[1])
    fw = mus & (sl >= HU_FAT[0]) & (sl <= HU_FAT[1])
    sma, imat = mw.sum() * px_cm2, fw.sum() * px_cm2
    ew = esm & (sl >= HU_MUSCLE[0]) & (sl <= HU_MUSCLE[1])
    return dict(sma_cm2=sma, imat_pct=100 * imat / (sma + imat) if sma + imat > 0 else np.nan,
                smd_window_hu=float(sl[mw].mean()) if mw.sum() > 50 else np.nan,
                smd_all_hu=float(sl[mus].mean()) if mus.sum() > 50 else np.nan,
                esm_hu=float(sl[ew].mean()) if ew.sum() > 50 else np.nan, esm_cm2=ew.sum() * px_cm2)


def muscle_noise(sl, esm, px):
    core = ndi.binary_erosion(esm, iterations=2)
    if core.sum() < 100:
        return np.nan
    filled = np.where(esm, sl, np.nan)
    filled = np.where(np.isnan(filled), np.nanmean(filled), filled)
    res = filled - ndi.gaussian_filter(filled, sigma=3.0 / px[0])
    return float(res[core].std())


def map_mask(maskA, affA, shapeB, affB, zB):
    """Nearest-neighbour lookup of A-space 3D mask at every pixel of B's slice zB (world coordinates)."""
    ii, jj = np.meshgrid(np.arange(shapeB[0]), np.arange(shapeB[1]), indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, zB), np.ones(ii.size)])
    wa = np.linalg.inv(affA) @ affB @ vox
    ia, ja, ka = [np.rint(wa[d]).astype(int) for d in range(3)]
    ok = (ia >= 0) & (ia < maskA.shape[0]) & (ja >= 0) & (ja < maskA.shape[1]) & (ka >= 0) & (ka < maskA.shape[2])
    out = np.zeros(ii.size, bool)
    out[ok] = maskA[ia[ok], ja[ok], ka[ok]]
    return out.reshape(shapeB[:2])


def one(job):
    from bodycomp_metrics import Case, vertebra_info, labels
    ds, pid, a_nii, a_seg, b_nii, k_a, k_b = job
    try:
        case = Case(a_nii, a_seg)
        zc, ok = vertebra_info(case.segs["total"], "T12")
        if zc is None or not ok:
            return None
        zA = int(round(zc))
        A, affA, zoomA = canon(a_nii)
        B, affB, zoomB = canon(b_nii)
        # T12 plane of A in B by world z
        wz = (affA @ np.array([A.shape[0] / 2, A.shape[1] / 2, zA, 1]))
        vb = np.linalg.inv(affB) @ wz
        zB = int(np.rint(vb[2]))
        if not (2 <= zB < B.shape[2] - 2):
            return dict(dataset=ds, pid=pid, error="T12 outside partner")
        validA, validB = A[:, :, zA] > PAD_HU, B[:, :, zB] > PAD_HU
        musA_full = case.muscle[:, :, zA]
        musB_full = map_mask(case.muscle, affA, B.shape, affB, zB)
        musA, esmA = musA_full & validA, case.esm[:, :, zA] & validA
        musB, esmB = musB_full & validB, map_mask(case.esm, affA, B.shape, affB, zB) & validB
        same_grid = A.shape == B.shape and np.allclose(affA, affB, atol=1e-3)
        rows, hist = [], None
        roles = {"A": (A, zA, zoomA, musA, esmA, k_a), "B": (B, zB, zoomB, musB, esmB, k_b)}
        for role, (V, z, zm, mus, esm, kern) in roles.items():
            px = zm[:2]
            pc = zm[0] * zm[1] / 100
            base = dict(dataset=ds, pid=pid, role=role, kernel=kern, sharp=int(is_sharp(kern)), slice_mm=zm[2],
                        px_mm=zm[0])
            nat = V[:, :, z]
            rows.append(dict(base, condition="native", sigma=0.0, noise_muscle=muscle_noise(nat, esm, px),
                             **measures(nat, mus, esm, pc)))
            for s in SLAB_SIGMAS:
                sl = slab_gauss(V, z, zm[2], px, s, slab=True)
                rows.append(dict(base, condition="slab_gauss", sigma=s, noise_muscle=muscle_noise(sl, esm, px),
                                 **measures(sl, mus, esm, pc)))
            for s in GAUSS_ONLY:
                sl = slab_gauss(V, z, zm[2], px, s, slab=False)
                rows.append(dict(base, condition="gauss_only", sigma=s, noise_muscle=muscle_noise(sl, esm, px),
                                 **measures(sl, mus, esm, pc)))
            nm = slab_gauss(V, z, zm[2], px, 0.75, slab=True)
            for e in (1, 2):
                me = ndi.binary_erosion(mus, iterations=e)
                rows.append(dict(base, condition=f"native_erode{e}", sigma=0.0, **measures(nat, me, esm, pc)))
                rows.append(dict(base, condition=f"matched_erode{e}", sigma=0.75, **measures(nm, me, esm, pc)))
            if not is_sharp(kern):
                hist = np.histogram(nat[mus], bins=BINS)[0]
        meta = dict(dataset=ds, pid=pid, kernel_A=k_a, kernel_B=k_b, shape_A=str(A.shape), shape_B=str(B.shape),
                    same_grid=int(same_grid), zA=zA, zB=zB, slice_A=zoomA[2], slice_B=zoomB[2],
                    mask_overlap_px=int(musB_full.sum()), mask_px_A=int(musA_full.sum()),
                    mask_valid_area_ratio=float(musB.sum() * zoomB[0] * zoomB[1] / (musA_full.sum() * zoomA[0] * zoomA[1])),
                    mask_valid_frac_A=float(musA.sum() / musA_full.sum()))
        return dict(rows=rows, meta=meta, hist=hist)
    except Exception as e:
        return dict(dataset=ds, pid=pid, error=str(e))


def jobs():
    out = []
    cand = pd.read_csv(R("data", "nlst", "pairs", "candidates.csv"))
    sel = pd.read_csv(R("data", "nlst", "nlst_selected_series_last.csv"))[["pid", "round"]]
    cand = cand.merge(sel, on="pid")
    bc = pd.read_csv(R("results", "nlst_bodycomp.csv"))
    okc = set(bc[bc.T12_complete == 1].case_id)
    for _, r in cand.iterrows():
        name = f"{r.pid}_T{r['round']}"
        b = R("data", "nlst", "pairs", "nifti", f"{r.pid}_alt.nii.gz")
        if name in okc and os.path.exists(b):
            out.append(("NLST", str(r.pid), R("data", "nlst", "nifti", f"{name}.nii.gz"), R("data", "seg", "nlst", name),
                        b, r.kernel_sel, r.kernel_alt))
    ver = pd.read_csv(R("results", "r1005", "recon_pairs_dicom.csv"))
    for _, r in ver.iterrows():
        out.append(("RG", r.alt_id, R("data", "nifti", "public_rg", f"{r.case_id}.nii.gz"),
                    R("data", "seg", "public_rg", r.case_id), R("data", "nifti", "public_rg_recon", f"{r.alt_id}.nii.gz"),
                    r.kernel_primary, r.kernel_alt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    js = jobs()
    if a.limit:
        js = js[:a.limit] + [j for j in js if j[0] == "RG"][:2]
    print("pairs:", len(js), flush=True)
    rows, meta, errs, hists = [], [], [], {}
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(one, js)):
            if r is None:
                continue
            if "error" in r:
                errs.append(r)
                continue
            rows += r["rows"]
            meta.append(r["meta"])
            if r["hist"] is not None:
                hists[f"{r['meta']['dataset']}|{r['meta']['pid']}"] = r["hist"]
            if (i + 1) % 25 == 0:
                print("done", i + 1, flush=True)
    sfx = "_test" if a.limit else ""
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"pairs_long{sfx}.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, f"pairs_meta{sfx}.csv"), index=False)
    pd.DataFrame(errs).to_csv(os.path.join(OUT, f"pairs_errors{sfx}.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, f"pairs_softhist{sfx}.npz"), bins=BINS, **hists)
    print("pairs measured:", len(meta), "errors:", len(errs), flush=True)


if __name__ == "__main__":
    main()
