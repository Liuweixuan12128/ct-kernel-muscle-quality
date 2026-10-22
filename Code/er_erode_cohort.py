"""ER revision: how much of the residual intramuscular fat after noise matching sits at the mask boundary.
T12 intramuscular fat % and muscle-window density with the muscle mask eroded by 0, 1 and 2 pixels, on native and
noise-matched (published) images, for every NSCLC-Radiogenomics staging CT.
Output: results/er/erode_public_rg.csv
"""
import os
import sys
from multiprocessing import Pool

import pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from er_pairs_measure import measures  # noqa: E402


def one(cid):
    from bodycomp_metrics import Case, vertebra_info
    try:
        c = Case(os.path.join(ROOT, "data", "nifti", "public_rg", f"{cid}.nii.gz"), os.path.join(ROOT, "data", "seg", "public_rg", cid))
        zc, ok = vertebra_info(c.segs["total"], "T12")
        if zc is None or not ok:
            return []
        z = int(round(zc))
        mus, esm = c.muscle[:, :, z], c.esm[:, :, z]
        rows = []
        for e in (0, 1, 2):
            m = ndi.binary_erosion(mus, iterations=e) if e else mus
            for cond, sl in [("native", c.raw[:, :, z]), ("matched", c.ct[:, :, z])]:
                rows.append(dict(case_id=cid, erode_px=e, erode_mm=e * c.zooms[0], condition=cond,
                                 mask_px=int(m.sum()), **measures(sl, m, esm, c.px_cm2)))
        return rows
    except Exception as ex:
        return [dict(case_id=cid, error=str(ex))]


if __name__ == "__main__":
    bc = pd.read_csv(os.path.join(ROOT, "results", "public_rg_bodycomp.csv"))
    ids = bc[bc.T12_complete == 1].case_id.tolist()
    rows = []
    with Pool(16) as p:
        for r in p.imap_unordered(one, ids):
            rows += r
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "er", "erode_public_rg.csv"), index=False)
    print("cases", len({r["case_id"] for r in rows}))
