"""Sci Rep revision 1 (plan section 3): summary of the separately segmented sharp reconstructions.

Sharp - soft differences per strategy with the shared mask (threshold effect only; results/er/pairs_perpair.csv and
results/srep/noise_ref_perpair.csv) and with each image's own masks and T12 level (segmentation + threshold;
results/srep/sharp_own_seg.csv), HL estimates with 95% CI, on the pairs available in both; Dice and T12 level agreement;
noise correlation length. Output: results/srep/sharp_own_seg_summary.csv
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
MEAS = ["imat_pct", "smd_window_hu", "smd_all_hu", "sma_cm2"]


def main():
    O = pd.read_csv(R("results", "srep", "sharp_own_seg.csv"))
    O["pid"] = O.pid.astype(str)
    fail = O[O["error"].notna()] if "error" in O else O.iloc[:0]
    O = O[O["error"].isna()] if "error" in O else O
    N = pd.read_csv(R("results", "srep", "noise_ref_perpair.csv"))
    N["pid"] = N.pid.astype(str)
    D = N.merge(O, on=["dataset", "pid"], how="inner")
    rows = []
    for ds in ["NLST", "RG"]:
        d = D[D.dataset == ds]
        n_all = int((N.dataset == ds).sum())
        rows.append(dict(dataset=ds, item="pairs", value=len(d), n=n_all))
        rows.append(dict(dataset=ds, item="failed", value=int((fail.dataset == ds).sum()) if len(fail) else 0, n=n_all))
        lvl = d.level_diff_mm.abs()
        rows += [dict(dataset=ds, item="level_same_within_1mm", value=float((lvl <= 1).mean()), n=len(d)),
                 dict(dataset=ds, item="level_abs_median_mm", value=float(lvl.median()), n=len(d)),
                 dict(dataset=ds, item="level_abs_max_mm", value=float(lvl.max()), n=len(d)),
                 dict(dataset=ds, item="dice_median", value=float(d.dice_muscle.median()), n=int(d.dice_muscle.notna().sum())),
                 dict(dataset=ds, item="dice_q1", value=float(d.dice_muscle.quantile(.25)), n=len(d)),
                 dict(dataset=ds, item="dice_q3", value=float(d.dice_muscle.quantile(.75)), n=len(d)),
                 dict(dataset=ds, item="dice_min", value=float(d.dice_muscle.min()), n=len(d))]
        for c in ["corr_soft_native", "corr_sharp_native", "corr_sharp_pair", "corr_soft_r25", "corr_sharp_r25"]:
            rows.append(dict(dataset=ds, item=f"{c}_median", value=float(d[c].median()), n=int(d[c].notna().sum())))
        # segmentation alone on the shared-mask slice of the sharp image (own masks - mapped soft masks), native
        for m in MEAS:
            v = d[f"sharp_ownmask_sharedslice_native_{m}"] - d[f"sharp_sharedmask_native_{m}"]
            hl, lo, hi, n = hodges_lehmann(v)
            rows.append(dict(dataset=ds, item=f"segmentation_only_native_{m}", value=hl, lo=lo, hi=hi, n=n))
        strategies = {"native": (f"{{}}_sharp_native", f"{{}}_soft_native", "sharp_native_{}", "soft_native_{}"),
                      "fixed": (f"{{}}_sharp_fixed", f"{{}}_soft_fixed", "sharp_fixed_{}", "soft_fixed_{}"),
                      "sharp-only": (f"{{}}_sharp_adapt_pair", f"{{}}_soft_native", "sharp_pair_{}", "soft_native_{}"),
                      "noise-referenced 25 HU": (f"{{}}_sharp_r25", f"{{}}_soft_r25", "sharp_r25_{}", "soft_r25_{}")}
        for st, (sh_shared, so_shared, sh_own, so_own) in strategies.items():
            for m in MEAS:
                shared = d[sh_shared.format(m)] - d[so_shared.format(m)]
                own = d[sh_own.format(m)] - d[so_own.format(m)]
                for lab, v in [("shared mask", shared), ("own masks", own)]:
                    hl, lo, hi, n = hodges_lehmann(v)
                    rows.append(dict(dataset=ds, item=f"{st} | {lab} | {m}", value=hl, lo=lo, hi=hi, n=n,
                                     within2=float((v.abs() <= 2).mean()) if m == "imat_pct" else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(R("results", "srep", "sharp_own_seg_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False))
    if len(fail):
        print(fail[["dataset", "pid", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
