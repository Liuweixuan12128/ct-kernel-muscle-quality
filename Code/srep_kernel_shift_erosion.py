"""Sci Rep revision 3: are the kernel shifts of mean attenuation (GE STANDARD/LUNG, Philips C/D) confined to the mask
boundary? Whole-mask density sharp - soft (HL estimate) in NLST pairs with the shared muscle mask native and eroded by
one and two pixels, native and fixed-filtered (results/er/pairs_long.csv). Output: results/srep/kernel_shift_erosion.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from er_pairs_analysis import hodges_lehmann  # noqa: E402

ROOT = os.path.dirname(HERE)
R = lambda *p: os.path.join(ROOT, *p)


def main():
    L = pd.read_csv(R("results", "er", "pairs_long.csv")); L["pid"] = L.pid.astype(str)
    W = pd.read_csv(R("results", "er", "pairs_perpair.csv")); W["pid"] = W.pid.astype(str)
    rows = []
    for cond in ["native", "native_erode1", "native_erode2", "matched_erode1", "matched_erode2"]:
        x = L[L.condition == cond]
        d = (x[x.sharp == 1].set_index(["dataset", "pid"]).smd_all_hu - x[x.sharp == 0].set_index(["dataset", "pid"]).smd_all_hu)
        d = d.rename("d").reset_index().merge(W[["dataset", "pid", "kernel_soft", "kernel_sharp"]], on=["dataset", "pid"])
        d["grp"] = np.where((d.kernel_soft == "STANDARD") & (d.kernel_sharp == "LUNG"), "GE STANDARD/LUNG",
                            np.where((d.kernel_soft == "C") & (d.kernel_sharp == "D"), "Philips C/D", "other"))
        for g, y in d[d.dataset == "NLST"].groupby("grp"):
            hl, lo, hi, n = hodges_lehmann(y.d)
            rows.append(dict(cond=cond, grp=g, n=n, hl=hl, lo=lo, hi=hi))
    D = pd.DataFrame(rows)
    D.to_csv(R("results", "srep", "kernel_shift_erosion.csv"), index=False)
    print(D.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
