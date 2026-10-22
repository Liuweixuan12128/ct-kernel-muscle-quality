"""ER revision: digital reference for the filter-strength trade-off. Recovery of small fat inclusions and false fat
in a fat-free block under 5-mm slab averaging + in-plane Gaussian of increasing SD (geometry, HU values and noise
model of stats/e4_phantom.py; 50 realizations; white noise SD 25 HU and 1.0-mm correlated noise SD 75 HU).
Output: results/er/phantom_sweep.csv"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e4_phantom as e4  # noqa: E402

SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def main():
    vol, truth, control, _ = e4.build()
    k = max(1, int(round(e4.SLAB_MM / e4.SLICE)))
    zs = range(k, vol.shape[2] - k)
    lab, n = ndi.label(truth[:, :, 0])
    order = np.argsort([ndi.center_of_mass(lab == i + 1)[1] for i in range(n)])
    incl = {d: lab == (order[i] + 1) for i, d in enumerate(e4.DIAMS)}
    rng = np.random.default_rng(20261014)
    rows = []
    for sd, corr in [(25, 0.0), (75, 1.0)]:
        for rep in range(50):
            noisy = vol + e4.noise(vol.shape, sd, corr, rng)
            slab = ndi.uniform_filter1d(noisy, k, axis=2, mode="nearest")
            for s in SIGMAS:
                img = ndi.gaussian_filter(slab, sigma=(s / e4.PX, s / e4.PX, 0)) if s > 0 else slab
                fat = (img >= e4.HU_FAT[0]) & (img <= e4.HU_FAT[1])
                r = dict(noise_sd=sd, corr_mm=corr, rep=rep, sigma=s,
                         false_fat_pct=float(100 * np.mean([(fat[:, :, z] & control).sum() / control.sum() for z in zs])))
                for d, tm in incl.items():
                    r[f"recovery_{d}mm"] = float(np.mean([fat[:, :, z][tm].sum() / tm.sum() for z in zs]))
                rows.append(r)
    d = pd.DataFrame(rows)
    g = d.groupby(["noise_sd", "corr_mm", "sigma"])
    out = g.mean().drop(columns="rep").add_suffix("_mean").join((1.96 * g.std() / np.sqrt(50)).drop(columns="rep").add_suffix("_ci"))
    out.reset_index().to_csv(os.path.join(os.path.dirname(HERE), "results", "er", "phantom_sweep.csv"), index=False)
    print(out[["false_fat_pct_mean", "recovery_2mm_mean", "recovery_3mm_mean", "recovery_5mm_mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
