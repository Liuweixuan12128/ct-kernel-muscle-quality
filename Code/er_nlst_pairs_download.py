"""ER revision: download the kernel-discordant partner series of each analysed NLST screening CT (same
StudyInstanceUID, original primary axial, opposite kernel class, closest slice thickness) from IDC, then convert
to NIfTI. Candidates: data/nlst/pairs/candidates.csv (built from the IDC index; see stats/er_nlst_pairs.py).
Usage: env/bin/python pipeline/er_nlst_pairs_download.py
"""
import glob
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "nlst", "pairs")


def main():
    from idc_index import IDCClient
    c = IDCClient()
    cand = pd.read_csv(os.path.join(D, "candidates.csv"))
    dcm = os.path.join(D, "dicom")
    os.makedirs(dcm, exist_ok=True)
    todo = [u for p, u in zip(cand.pid, cand.alt_uid) if not os.path.isdir(os.path.join(dcm, str(p), u))]
    print("to download:", len(todo), flush=True)
    for i in range(0, len(todo), 25):
        c.download_from_selection(seriesInstanceUID=todo[i:i + 25], downloadDir=dcm,
                                  dirTemplate="%PatientID/%SeriesInstanceUID", quiet=True)
        print("downloaded", min(i + 25, len(todo)), "of", len(todo), flush=True)
    nii = os.path.join(D, "nifti")
    os.makedirs(nii, exist_ok=True)
    exe = os.path.join(os.path.dirname(sys.executable), "dcm2niix")
    for _, r in cand.iterrows():
        name = f"{r.pid}_alt"
        if os.path.exists(os.path.join(nii, f"{name}.nii.gz")):
            continue
        src = os.path.join(dcm, str(r.pid), r.alt_uid)
        subprocess.run([exe, "-z", "y", "-f", name, "-o", nii, src], capture_output=True)
        extra = sorted(glob.glob(os.path.join(nii, f"{name}*.nii.gz")), key=os.path.getsize)
        for e in extra[:-1]:
            os.remove(e)
        if extra and extra[-1] != os.path.join(nii, f"{name}.nii.gz"):
            os.replace(extra[-1], os.path.join(nii, f"{name}.nii.gz"))
    print("conversion finished:", len(glob.glob(os.path.join(nii, "*_alt.nii.gz"))), flush=True)


if __name__ == "__main__":
    main()
