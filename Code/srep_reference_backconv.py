"""Sci Rep revision 3: is the further fall of soft-kernel intramuscular fat between the 25-HU and 15-HU references noise
or real fat? (Review round 3, point 1: the uniform 40-HU tissue model understates noise effects in real tissue.)

For every NLST and RG soft image, the image filtered to 15 HU (liver noise, the sigma of noise-referenced filtering) is
convolved with Gaussian noise of variance (muscle noise at 25 HU)^2 - (muscle noise at 15 HU)^2, which predicts the
intramuscular fat of the image at 25 HU (noise-convolution model, no assumption about the tissue). Predicted fall
(filtered 15 - predicted 25) against observed fall (filtered 15 - observed 25); the remainder is the part the noise
removed does not explain. The same is done for each reference step 30->25, 25->20 and 20->15.
Output: results/srep/reference_backconv.csv, reference_backconv_summary.csv
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
STEPS = [(30, 25), (25, 20), (20, 15), (25, 15)]


def one(job):
    from bodycomp_metrics import Case, vertebra_info
    from er_pairs_measure import canon, slab_gauss, measures, muscle_noise, map_mask, PAD_HU, BINS
    from srep_noise_estimators import back_convolve
    ds, pid, a_nii, a_seg, b_nii, role_soft, sig = job
    case = Case(a_nii, a_seg, harmonize=False)
    zA = int(round(vertebra_info(case.segs["total"], "T12")[0]))
    A, affA, zmA = canon(a_nii)
    if role_soft == "A":
        V, z, zm = A, zA, zmA
        valid = A[:, :, zA] > PAD_HU
        mus, esm = case.muscle[:, :, zA] & valid, case.esm[:, :, zA] & valid
    else:
        B, affB, zmB = canon(b_nii)
        z = int(np.rint((np.linalg.inv(affB) @ (affA @ np.array([A.shape[0] / 2, A.shape[1] / 2, zA, 1])))[2]))
        V, zm = B, zmB
        valid = B[:, :, z] > PAD_HU
        mus, esm = map_mask(case.muscle, affA, B.shape, affB, z) & valid, map_mask(case.esm, affA, B.shape, affB, z) & valid
    px, pc = zm[:2], zm[0] * zm[1] / 100
    img = {r: slab_gauss(V, z, zm[2], px, s, slab=False) for r, s in sig.items()}
    nz = {r: muscle_noise(im, esm, px) for r, im in img.items()}
    imat = {r: measures(im, mus, esm, pc)["imat_pct"] for r, im in img.items()}
    out = dict(dataset=ds, pid=pid, **{f"sigma_r{r}": s for r, s in sig.items()}, **{f"es_r{r}": nz[r] for r in sig},
               **{f"imat_r{r}": imat[r] for r in sig})
    for hi, lo in STEPS:
        if sig[lo] == sig[hi]:
            out[f"pred_{hi}_{lo}"] = imat[hi]  # same image: nothing removed
            continue
        h = np.histogram(img[lo][mus], bins=BINS)[0]
        out[f"pred_{hi}_{lo}"] = back_convolve(h, np.sqrt(max(nz[hi] ** 2 - nz[lo] ** 2, 0)))[0]
    return out


def main():
    from multiprocessing import Pool
    from srep_noise_estimators import pair_jobs
    NP = pd.read_csv(R("results", "srep", "noise_ref_perpair.csv"))
    NP["pid"] = NP.pid.astype(str)
    jobs = []
    for j in pair_jobs():
        ds, pid = j[0], str(j[1])
        r = NP[(NP.dataset == ds) & (NP.pid == pid)].iloc[0]
        sig = {k: float(r[f"sigma_soft_r{k}"]) for k in (15, 20, 25, 30)}
        jobs.append((*j, sig))
    with Pool(24) as p:
        rows = p.map(one, jobs, chunksize=1)
    D = pd.DataFrame(rows)
    D.to_csv(R("results", "srep", "reference_backconv.csv"), index=False)
    S = []
    for ds in ["NLST", "RG"]:
        d = D[D.dataset == ds]
        for hi, lo in STEPS:
            filt = d[f"sigma_r{lo}"] > d[f"sigma_r{hi}"]
            x = d[filt]
            obs, pred = x[f"imat_r{lo}"] - x[f"imat_r{hi}"], x[f"imat_r{lo}"] - x[f"pred_{hi}_{lo}"]
            for k, v in (("observed", obs), ("predicted from noise", pred), ("remainder", obs - pred)):
                hl, l_, h_, n = hodges_lehmann(v)
                S.append(dict(dataset=ds, step=f"{hi}->{lo}", part=k, n=n, hl=hl, lo=l_, hi=h_, median=float(np.median(v)) if len(v) else np.nan))
    S = pd.DataFrame(S)
    S.to_csv(R("results", "srep", "reference_backconv_summary.csv"), index=False)
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
