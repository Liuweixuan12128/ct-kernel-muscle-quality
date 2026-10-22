"""ER revision: sharp-only adaptive noise matching for the survival cohorts.

For every analysed CT (NSCLC-Radiogenomics, TCGA/CPTAC), at T12 with the CT's own masks:
  native measurements and image noise measured in muscle (eroded erector spinae, residual after a 3-mm local mean);
  an in-plane Gaussian sigma grid (0-3 mm, no slab) with noise and measurements at each sigma.
The analysis script then leaves soft-kernel CTs unchanged and filters each sharp-kernel CT with the smallest sigma
that brings its muscle noise down to the median native muscle noise of soft-kernel CTs (one fixed target).
Output: results/er/adaptive_<cohort>.csv
Usage: env/bin/python pipeline/er_adaptive_cohort.py --cohort public_rg --workers 16
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
from er_pairs_measure import measures, muscle_noise  # noqa: E402

GRID = [round(x, 2) for x in np.arange(0, 3.01, 0.1)]


def one(args):
    cohort, cid = args
    from bodycomp_metrics import Case, vertebra_info
    try:
        c = Case(os.path.join(ROOT, "data", "nifti", cohort, f"{cid}.nii.gz"), os.path.join(ROOT, "data", "seg", cohort, cid))
        zc, ok = vertebra_info(c.segs["total"], "T12")
        if zc is None or not ok:
            return []
        z = int(round(zc))
        sl0 = c.raw[:, :, z]
        mus, esm = c.muscle[:, :, z], c.esm[:, :, z]
        rows = []
        for s in GRID:
            sl = ndi.gaussian_filter(sl0, sigma=(s / c.zooms[0], s / c.zooms[1])) if s > 0 else sl0
            rows.append(dict(cohort=cohort, case_id=cid, sigma=s, slice_mm=c.zooms[2],
                             noise_muscle=muscle_noise(sl, esm, c.zooms[:2]), **measures(sl, mus, esm, c.px_cm2)))
        return rows
    except Exception as e:
        return [dict(cohort=cohort, case_id=cid, error=str(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    bc = pd.read_csv(os.path.join(ROOT, "results", f"{a.cohort}_bodycomp.csv"))
    ids = bc[bc.T12_complete == 1].case_id.astype(str).tolist()
    rows = []
    with Pool(a.workers) as p:
        for r in p.imap_unordered(one, [(a.cohort, i) for i in ids]):
            rows += r
    out = os.path.join(ROOT, "results", "er", f"adaptive_{a.cohort}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len({r["case_id"] for r in rows}))


if __name__ == "__main__":
    main()
