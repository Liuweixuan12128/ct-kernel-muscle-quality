"""Sci Rep revision 2: routine-dose same-acquisition reconstruction pairs in TCGA-LUAD/LUSC and CPTAC-LUAD/LSCC.

Same procedure as the NSCLC-Radiogenomics pairs (pipeline/e3_recon_pairs.py, stats/r1005_technical.py): in the study of
each analysed diagnostic CT, every other original axial series of the other kernel class is downloaded from The Cancer
Imaging Archive and accepted as the same acquisition when FrameOfReferenceUID and AcquisitionNumber match and slice
acquisition times at matched table positions differ by <1 s (median). Accepted pairs are measured with the analysed
series' masks mapped by world coordinates (pipeline/srep_noise_estimators.one_pair: native, in-plane and slab grids,
muscle and liver noise).

Steps (each skips finished work):
  list      candidates -> results/srep/tcga_recon_candidates.csv
  download  -> data/dicom/tcga_cptac_recon/<alt_id>.zip, data/nifti/tcga_cptac_recon/<alt_id>.nii.gz
  verify    DICOM checks -> results/srep/tcga_recon_dicom.csv
  measure   -> results/srep/tcga_recon_pairs.csv
Usage: env/bin/python pipeline/srep_tcga_recon_pairs.py --all --workers 6
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "stats"))
R = lambda *p: os.path.join(ROOT, *p)
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DCM2NIIX = os.path.join(os.path.dirname(sys.executable), "dcm2niix")
NII, ZIPD = R("data", "nifti", "tcga_cptac_recon"), R("data", "dicom", "tcga_cptac_recon")
SHARP = "LUNG|BONE|B4|B5|B6|B7|B8|I70|SHARP|YB|YA|FC5|FC3|^D$|^E$|FC8|DETAIL|EDGE"


def is_sharp(k):
    import re
    return bool(re.search(SHARP, str(k).upper()))


def candidates():
    s = pd.read_csv(R("data", "tcga_cptac", "series_candidates.csv"))
    s = s[s.axial.astype(str).isin(["True", "TRUE", "1"]) & s.image_type.astype(str).str.upper().str.startswith("ORIGINAL")]
    s["chosen"] = s.chosen.astype(str).isin(["True", "TRUE", "1"])
    out = []
    for _, c in s[s.chosen].iterrows():
        same = s[(s.StudyInstanceUID == c.StudyInstanceUID) & (s.SeriesInstanceUID != c.SeriesInstanceUID)
                 & (s.kernel.map(is_sharp) != is_sharp(c.kernel))]
        for k, (_, r) in enumerate(same.iterrows(), start=1):
            out.append(dict(case_id=c.PatientID, alt_id=f"{c.PatientID}__alt{k}", collection=c.Collection,
                            chosen_series=c.SeriesInstanceUID, alt_series=r.SeriesInstanceUID, chosen_kernel=c.kernel,
                            alt_kernel=r.kernel, chosen_mm=c.slice_thickness, alt_mm=r.slice_thickness,
                            chosen_images=c.ImageCount, alt_images=r.ImageCount))
    return pd.DataFrame(out)


def fetch(row):
    out = os.path.join(NII, f"{row.alt_id}.nii.gz")
    if os.path.exists(out):
        return row.alt_id, "exists"
    z = os.path.join(ZIPD, f"{row.alt_id}.zip")
    try:
        if not os.path.exists(z) or not zipfile.is_zipfile(z):
            with requests.get(f"{API}/getImage", params={"SeriesInstanceUID": row.alt_series}, stream=True, timeout=900) as r:
                r.raise_for_status()
                with open(z + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(z + ".part", z)
        with tempfile.TemporaryDirectory(dir=NII) as tmp:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
            subprocess.run([DCM2NIIX, "-z", "y", "-f", row.alt_id, "-o", tmp, tmp], check=True, capture_output=True)
            got = sorted([f for f in os.listdir(tmp) if f.endswith(".nii.gz")], key=lambda f: os.path.getsize(os.path.join(tmp, f)), reverse=True)
            if not got:
                return row.alt_id, "no nifti"
            shutil.move(os.path.join(tmp, got[0]), out)
        return row.alt_id, "ok"
    except Exception as e:
        return row.alt_id, f"error: {e}"


def t2s(t):
    t = str(t)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:]) if len(t) >= 6 else np.nan


def dicom_series(zpath):
    """Slice positions, acquisition times and numbers, and series metadata (as stats/r1005_technical.py)."""
    import pydicom
    z = zipfile.ZipFile(zpath)
    out, meta = [], {}
    for n in z.namelist():
        if n.endswith("/"):
            continue
        with z.open(n) as f:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except Exception:
                continue
        if not meta and "SOPInstanceUID" in ds and "FrameOfReferenceUID" in ds:
            meta = {k: str(ds.get(k, "")) for k in ["FrameOfReferenceUID", "AcquisitionNumber", "ConvolutionKernel",
                                                    "SliceThickness", "KVP", "Manufacturer", "StudyInstanceUID"]}
        if "ImagePositionPatient" in ds and ds.get("AcquisitionTime"):
            out.append((float(ds.ImagePositionPatient[2]), t2s(ds.AcquisitionTime), str(ds.get("AcquisitionNumber", ""))))
    return pd.DataFrame(out, columns=["z_mm", "acq_time_s", "acq_no"]), meta


def verify(cands):
    ver = []
    for _, r in cands.iterrows():
        za = os.path.join(ZIPD, f"{r.alt_id}.zip")
        if not os.path.exists(za):
            continue
        p, mp = dicom_series(R("data", "raw", "tcga_cptac", f"{r.case_id}.zip"))
        a, ma = dicom_series(za)
        if not len(p) or not len(a):
            ver.append(dict(case_id=r.case_id, alt_id=r.alt_id, same_acquisition=False, note="no slice times"))
            continue
        lo, hi = max(p.z_mm.min(), a.z_mm.min()), min(p.z_mm.max(), a.z_mm.max())
        aa = a[(a.z_mm >= lo) & (a.z_mm <= hi)]
        if len(aa):
            idx = np.abs(p.z_mm.values[None, :] - aa.z_mm.values[:, None]).argmin(1)
            dt = np.abs(p.acq_time_s.values[idx] - aa.acq_time_s.values)
        else:
            dt = np.array([np.nan])
        same_for = mp.get("FrameOfReferenceUID") == ma.get("FrameOfReferenceUID") != ""
        same_acqno = bool(set(p.acq_no) & set(a.acq_no))
        ver.append(dict(case_id=r.case_id, alt_id=r.alt_id, collection=r.collection, manufacturer=mp.get("Manufacturer"),
                        same_frame_of_reference=same_for, shared_acquisition_number=same_acqno,
                        kernel_primary=mp.get("ConvolutionKernel"), kernel_alt=ma.get("ConvolutionKernel"),
                        thickness_primary=mp.get("SliceThickness"), thickness_alt=ma.get("SliceThickness"),
                        kvp_primary=mp.get("KVP"), kvp_alt=ma.get("KVP"), z_overlap_mm=hi - lo, slices_compared=len(aa),
                        median_time_diff_s=float(np.nanmedian(dt)),
                        same_acquisition=bool(same_for and same_acqno and len(aa) > 0 and np.nanmedian(dt) <= 1.0)))
    return pd.DataFrame(ver)


def measure(ver, workers):
    from srep_noise_estimators import one_pair
    jobs = []
    for _, r in ver[ver.same_acquisition].iterrows():
        a_nii, a_seg = R("data", "nifti", "tcga_cptac", f"{r.case_id}.nii.gz"), R("data", "seg", "tcga_cptac", r.case_id)
        b_nii = os.path.join(NII, f"{r.alt_id}.nii.gz")
        role_soft = "B" if is_sharp(r.kernel_primary) else "A"
        jobs.append(("TCGA/CPTAC", r.alt_id, a_nii, a_seg, b_nii, role_soft))
    with Pool(workers) as p:
        rows = [x for rr in p.imap_unordered(one_pair, jobs, chunksize=1) for x in rr]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(NII, exist_ok=True); os.makedirs(ZIPD, exist_ok=True); os.makedirs(R("results", "srep"), exist_ok=True)
    c = candidates()
    c.to_csv(R("results", "srep", "tcga_recon_candidates.csv"), index=False)
    print(len(c), "kernel-discordant candidate series in", c.case_id.nunique(), "studies", flush=True)
    if not a.all:
        return
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(fetch, [r for _, r in c.iterrows()]))
    print("download:", pd.Series([s_.split(":")[0] for _, s_ in res]).value_counts().to_dict(), flush=True)
    v = verify(c)
    v.to_csv(R("results", "srep", "tcga_recon_dicom.csv"), index=False)
    print("same acquisition:", int(v.same_acquisition.sum()), "of", len(v), flush=True)
    m = measure(v, 16)
    m.to_csv(R("results", "srep", "tcga_recon_pairs.csv"), index=False)
    print("measured rows", len(m), flush=True)


if __name__ == "__main__":
    main()
