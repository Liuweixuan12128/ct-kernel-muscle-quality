"""ER revision: DICOM verification that each NLST pair is two reconstructions of one acquisition
(same FrameOfReferenceUID and AcquisitionNumber; matching slice acquisition times at the same table position),
plus the acquisition and reconstruction parameters of both series.
Output: results/er/nlst_pairs_dicom.csv
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)
TAGS = ["FrameOfReferenceUID", "AcquisitionNumber", "ConvolutionKernel", "SliceThickness", "KVP", "XRayTubeCurrent",
        "Exposure", "Manufacturer", "ManufacturerModelName", "ReconstructionDiameter", "PixelSpacing", "StudyInstanceUID"]


def t2s(t):
    t = str(t)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:]) if len(t) >= 6 else np.nan


def read_series(d):
    meta, rows = {}, []
    for f in sorted(glob.glob(os.path.join(d, "*")))[:: 1]:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if "SOPInstanceUID" not in ds:
            continue
        if not meta:
            meta = {k: str(ds.get(k, "")) for k in TAGS}
        if "ImagePositionPatient" in ds:
            rows.append((float(ds.ImagePositionPatient[2]), t2s(ds.get("AcquisitionTime", "")), str(ds.get("AcquisitionNumber", ""))))
    return meta, pd.DataFrame(rows, columns=["z", "t", "acq"])


def one(r):
    a_dir = R("data", "nlst", "dicom", str(r.pid), r.SeriesInstanceUID)
    b_dir = R("data", "nlst", "pairs", "dicom", str(r.pid), r.alt_uid)
    ma, a = read_series(a_dir)
    mb, b = read_series(b_dir)
    if a.empty or b.empty:
        return dict(pid=r.pid, error="no headers")
    lo, hi = max(a.z.min(), b.z.min()), min(a.z.max(), b.z.max())
    bb = b[(b.z >= lo) & (b.z <= hi)]
    if len(bb) and a.t.notna().any() and bb.t.notna().any():
        idx = np.abs(a.z.values[None, :] - bb.z.values[:, None]).argmin(1)
        dt = np.abs(a.t.values[idx] - bb.t.values)
        med_dt = float(np.nanmedian(dt))
    else:
        med_dt = np.nan
    same_for = ma.get("FrameOfReferenceUID") == mb.get("FrameOfReferenceUID") != ""
    same_acq = bool(set(a.acq) & set(b.acq))
    return dict(pid=r.pid, same_study=ma.get("StudyInstanceUID") == mb.get("StudyInstanceUID"),
                same_frame_of_reference=same_for, shared_acquisition_number=same_acq, median_time_diff_s=med_dt,
                time_available=bool(np.isfinite(med_dt)), z_overlap_mm=hi - lo,
                same_acquisition=bool(same_for and same_acq and (not np.isfinite(med_dt) or med_dt <= 1.0)),
                **{f"{k}_A": ma.get(k) for k in TAGS if k != "StudyInstanceUID"},
                **{f"{k}_B": mb.get(k) for k in TAGS if k != "StudyInstanceUID"})


def main():
    cand = pd.read_csv(R("data", "nlst", "pairs", "candidates.csv"))
    sel = pd.read_csv(R("data", "nlst", "nlst_selected_series_last.csv"))[["pid", "SeriesInstanceUID"]]
    cand = cand.merge(sel, on="pid")
    with Pool(24) as p:
        rows = p.map(one, [r for _, r in cand.iterrows()])
    d = pd.DataFrame(rows)
    d.to_csv(R("results", "er", "nlst_pairs_dicom.csv"), index=False)
    print(len(d), "pairs; same FoR:", int(d.same_frame_of_reference.sum()), "shared AcquisitionNumber:",
          int(d.shared_acquisition_number.sum()), "time available:", int(d.time_available.sum()),
          "same acquisition:", int(d.same_acquisition.sum()))
    print(d.groupby("Manufacturer_A").size().to_dict())
    print("mA/kVp identical:", int(((d.XRayTubeCurrent_A == d.XRayTubeCurrent_B) & (d.KVP_A == d.KVP_B)).sum()))


if __name__ == "__main__":
    main()
