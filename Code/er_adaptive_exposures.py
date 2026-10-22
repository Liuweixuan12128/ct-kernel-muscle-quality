"""ER revision: survival exposures under sharp-only adaptive noise matching.
Soft-kernel CTs keep native values; each sharp-kernel CT is filtered (in-plane Gaussian, grid 0-3 mm) with the
smallest sigma that brings its muscle noise to the median native muscle noise of all soft-kernel CTs (one fixed
target across NSCLC-Radiogenomics and TCGA/CPTAC). Also: noise-matched intramuscular fat with the mask eroded by
one pixel. Output: results/er/adaptive_exposures.csv"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)
SHARP = "LUNG|BONE|B4|B5|B6|B7|I70|SHARP|YB|YA"

a = pd.concat([pd.read_csv(R("results", "er", f"adaptive_{c}.csv")) for c in ["public_rg", "tcga_cptac"]])
a = a[a.get("error").isna()] if "error" in a else a
ser = pd.concat([pd.read_csv(R("data", "public_rg", "series_candidates.csv")), pd.read_csv(R("data", "tcga_cptac", "series_candidates.csv"))])
ser = ser[ser.chosen.astype(str).isin(["True", "TRUE", "1"])]
kt = pd.DataFrame({"case_id": ser.PatientID.astype(str),
                   "kernel_type": np.where(ser.kernel.fillna("").str.contains(SHARP, case=False), "sharp", "soft")}).drop_duplicates("case_id")
a = a.merge(kt, on="case_id")
nat = a[a.sigma == 0]
target = float(nat[nat.kernel_type == "soft"].noise_muscle.median())
rows = []
for cid, g in a.groupby("case_id"):
    g = g.sort_values("sigma")
    k = g.kernel_type.iloc[0]
    if k == "soft":
        s = g.iloc[0]
    else:
        ok = g[g.noise_muscle <= target]
        s = ok.iloc[0] if len(ok) else g.iloc[-1]
    rows.append(dict(case_id=cid, kernel_type=k, sigma_used=s.sigma, noise_native=g.iloc[0].noise_muscle,
                     noise_after=s.noise_muscle, imat_adapt=s.imat_pct, window_adapt=s.smd_window_hu,
                     whole_adapt=s.smd_all_hu, area_adapt=s.sma_cm2, esm_hu_adapt=s.esm_hu))
d = pd.DataFrame(rows)
e = pd.read_csv(R("results", "er", "erode_public_rg.csv"))
e = e[(e.condition == "matched") & (e.erode_px == 1)][["case_id", "imat_pct", "smd_window_hu"]].rename(
    columns={"imat_pct": "imat_matched_erode1", "smd_window_hu": "window_matched_erode1"})
d = d.merge(e, on="case_id", how="left")
d.to_csv(R("results", "er", "adaptive_exposures.csv"), index=False)
print("target muscle noise (HU):", round(target, 1), "| CTs:", len(d), d.kernel_type.value_counts().to_dict())
print(d.groupby("kernel_type")[["sigma_used", "noise_native", "noise_after", "imat_adapt"]].median().round(2))
