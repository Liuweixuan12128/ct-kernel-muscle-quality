"""Sci Rep revision 1 (plan section 2): digital phantom under noise-referenced filtering.

Same phantom as stats/er_phantom_sweep.py (stats/e4_phantom.py geometry: 40-HU block with -100-HU fat inclusions of
1-10 mm and a fat-free control block; 0.7-mm pixels, 1.25-mm slices). For each noise condition, every realization is
filtered in-plane only (no slab) with the smallest Gaussian sigma (0.1-mm grid, up to 3 mm) that brings the residual
noise of the control block (3-mm Gaussian local mean removed, as in the images) to <= 25 HU, i.e. the processing of
noise-referenced (and sharp-only) filtering, and (revision 2) with a 5-mm slab first and then the smallest in-plane
sigma reaching 25 HU. The fixed filter (5-mm slab + Gaussian 0.75 mm) is run on the same realizations for comparison. Noise conditions span the images of this study: white 25, 45 and 120 HU, and
1.0-mm correlated 75 HU. 50 realizations each.
Output: results/srep/phantom_noise_ref.csv (per realization) and phantom_noise_ref_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e4_phantom as e4  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "srep")
REF = 25.0
GRID = [round(x, 1) for x in np.arange(0, 3.01, 0.1)]
CONDS = [(25, 0.0), (45, 0.0), (120, 0.0), (75, 1.0)]


def resid_sd(img2d, control):
    r = img2d - ndi.gaussian_filter(img2d, sigma=3.0 / e4.PX)
    core = ndi.binary_erosion(control, iterations=int(round(4 / e4.PX)))
    return float(r[core].std())


def score(img, incl, control, zs):
    fat = (img >= e4.HU_FAT[0]) & (img <= e4.HU_FAT[1])
    r = dict(false_fat_pct=float(100 * np.mean([(fat[:, :, z] & control).sum() / control.sum() for z in zs])))
    for d, tm in incl.items():
        r[f"recovery_{d}mm"] = float(np.mean([fat[:, :, z][tm].sum() / tm.sum() for z in zs]))
    return r


def main():
    vol, truth, control, _ = e4.build()
    k = max(1, int(round(e4.SLAB_MM / e4.SLICE)))
    zs = list(range(k, vol.shape[2] - k))
    zm = zs[len(zs) // 2]
    lab, n = ndi.label(truth[:, :, 0])
    order = np.argsort([ndi.center_of_mass(lab == i + 1)[1] for i in range(n)])
    incl = {d: lab == (order[i] + 1) for i, d in enumerate(e4.DIAMS)}
    rng = np.random.default_rng(20261024)
    rows = []
    for sd, corr in CONDS:
        for rep in range(50):
            noisy = vol + e4.noise(vol.shape, sd, corr, rng)
            base = dict(noise_sd=sd, corr_mm=corr, rep=rep)
            rows.append(dict(base, method="native", sigma=0.0, resid_sd=resid_sd(noisy[:, :, zm], control),
                             **score(noisy, incl, control, zs)))
            for s in GRID:
                img = ndi.gaussian_filter(noisy, sigma=(s / e4.PX, s / e4.PX, 0)) if s > 0 else noisy
                rs = resid_sd(img[:, :, zm], control)
                if rs <= REF or s == GRID[-1]:
                    rows.append(dict(base, method="noise-referenced", sigma=s, resid_sd=rs, **score(img, incl, control, zs)))
                    break
            slab = ndi.uniform_filter1d(noisy, k, axis=2, mode="nearest")
            for s in GRID:  # revision 2: 5-mm slab first, then the smallest in-plane sigma reaching the reference
                img = ndi.gaussian_filter(slab, sigma=(s / e4.PX, s / e4.PX, 0)) if s > 0 else slab
                rs = resid_sd(img[:, :, zm], control)
                if rs <= REF or s == GRID[-1]:
                    rows.append(dict(base, method="slab + noise-referenced", sigma=s, resid_sd=rs, **score(img, incl, control, zs)))
                    break
            fx = ndi.gaussian_filter(slab, sigma=(e4.SMOOTH_MM / e4.PX, e4.SMOOTH_MM / e4.PX, 0))
            rows.append(dict(base, method="fixed", sigma=e4.SMOOTH_MM, resid_sd=resid_sd(fx[:, :, zm], control),
                             **score(fx, incl, control, zs)))
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "phantom_noise_ref.csv"), index=False)
    g = d.drop(columns="rep").groupby(["noise_sd", "corr_mm", "method"])
    s = g.mean().add_suffix("_mean").join((1.96 * g.std() / np.sqrt(50)).add_suffix("_ci")).reset_index()
    s.to_csv(os.path.join(OUT, "phantom_noise_ref_summary.csv"), index=False)
    cols = ["noise_sd", "corr_mm", "method", "sigma_mean", "resid_sd_mean", "false_fat_pct_mean"] + \
        [f"recovery_{x}mm_mean" for x in e4.DIAMS]
    print(s[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
