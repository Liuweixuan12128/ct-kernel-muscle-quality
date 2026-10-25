"""Revision work package E4 (digital reference): what the HU thresholds do to known fat, under white and
spatially correlated noise, with and without the harmonisation steps.

The phantom is a uniform muscle block (40 HU) containing fat inclusions (-100 HU) of known diameter, plus a
fat-free control block. Noise is added either as white Gaussian noise or as spatially correlated noise
(white noise smoothed to a target in-plane correlation length, matched to what sharp-kernel CT shows).
Every condition is then processed exactly like a patient CT (5-mm slab average, in-plane Gaussian) and the
fat window (-190..-30 HU) is applied inside the known muscle boundary.

Reported per condition:
  fp_fat_pct        fat detected in the fat-free block (false positive), % of its area
  recovery_<d>mm    detected fat area / true fat area for an inclusion of diameter d
  hu_bias           mean HU of the muscle block minus its true value

Simulation is not equivalent to a real artefact; it isolates the behaviour of the thresholds under a known truth.

Usage: env/bin/python stats/e4_phantom.py
Output: results/audit/e4_phantom.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "audit")
MUSCLE_HU, FAT_HU = 40.0, -100.0
HU_FAT = (-190, -30)
PX, SLICE = 0.7, 1.25          # mm, typical of the thin sharp-kernel series
SLAB_MM, SMOOTH_MM = 5.0, 0.75
DIAMS = [1, 2, 3, 5, 10]       # mm
SD_LEVELS = [10, 25, 50, 75, 120]
CORR_MM = [0.0, 0.6, 1.0, 1.5]  # 0 = white noise
SEED = 20260920


def build():
    """Muscle block with fat inclusions (left) and a fat-free control block (right)."""
    ny = int(round(90 / PX))
    nx = int(round(140 / PX))
    nz = 24
    vol = np.full((ny, nx, nz), MUSCLE_HU, np.float32)
    truth = np.zeros((ny, nx, nz), bool)
    yy, xx = np.ogrid[:ny, :nx]
    cy = ny // 2
    x0 = int(round(20 / PX))
    for d in DIAMS:                                   # one column of inclusions per diameter, through all slices
        r_px = (d / 2) / PX
        m = ((yy - cy) ** 2 + (xx - x0) ** 2) <= r_px ** 2
        vol[m] = FAT_HU
        truth[m] = True
        x0 += int(round(max(d * 2.2, 8) / PX))
    control = np.zeros((ny, nx), bool)
    control[:, int(round(95 / PX)):] = True           # fat-free block
    muscle_mask = np.ones((ny, nx), bool)
    return vol, truth, control, muscle_mask


def noise(shape, sd, corr_mm, rng):
    n = rng.normal(0, 1, shape).astype(np.float32)
    if corr_mm > 0:
        sig = corr_mm / PX / 2.355 * 2               # smoothing that yields ~corr_mm correlation length
        n = ndi.gaussian_filter(n, sigma=(sig, sig, 0))
        n /= n.std()
    return n * sd


def process(vol, cond):
    k = max(1, int(round(SLAB_MM / SLICE)))
    out = vol
    if cond in ("slab", "both"):
        out = ndi.uniform_filter1d(out, k, axis=2, mode="nearest")
    if cond in ("gauss", "both"):
        out = ndi.gaussian_filter(out, sigma=(SMOOTH_MM / PX, SMOOTH_MM / PX, 0))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    vol, truth, control, muscle = build()
    rng = np.random.default_rng(SEED)
    z = vol.shape[2] // 2
    rows = []
    for sd in SD_LEVELS:
        for corr in CORR_MM:
            for rep in range(5):
                noisy = vol + noise(vol.shape, sd, corr, rng)
                for cond in ["native", "slab", "gauss", "both"]:
                    img = process(noisy, cond)[:, :, z]
                    fat = (img >= HU_FAT[0]) & (img <= HU_FAT[1]) & muscle
                    r = dict(sd_hu=sd, corr_mm=corr, rep=rep, condition=cond,
                             fp_fat_pct=float(100 * (fat & control).sum() / control.sum()),
                             hu_bias=float(img[muscle & ~truth[:, :, z] & ~control].mean() - MUSCLE_HU))
                    for d in DIAMS:                   # recovery per inclusion, in its own neighbourhood
                        t = truth[:, :, z]
                        lab, n = ndi.label(t)
                        sizes = ndi.sum(t, lab, index=np.arange(1, n + 1))
                        order = np.argsort([ndi.center_of_mass(lab == i + 1)[1] for i in range(n)])
                        idx = order[DIAMS.index(d)] + 1
                        tm = lab == idx
                        r[f"recovery_{d}mm"] = float(fat[tm].sum() / max(tm.sum(), 1))
                    rows.append(r)
    d = pd.DataFrame(rows)
    f = os.path.join(OUT, "e4_phantom.csv")
    d.to_csv(f, index=False)
    print("wrote", f, len(d), "runs")
    agg = d.groupby(["condition", "corr_mm", "sd_hu"]).agg(
        fp_fat_pct=("fp_fat_pct", "mean"), hu_bias=("hu_bias", "mean"),
        rec_2mm=("recovery_2mm", "mean"), rec_5mm=("recovery_5mm", "mean"), rec_10mm=("recovery_10mm", "mean")).round(2)
    print(agg.to_string())


if __name__ == "__main__":
    main()
