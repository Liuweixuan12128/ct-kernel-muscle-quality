"""ER revision: demographics of the same-acquisition pairs (Table 1).
NLST: age at the analysed screening CT = age at diagnosis - (days from screen to diagnosis) / 365.25 (results/nlst/analysis_table.csv
or, if absent there, NLST participant file); RG: age at histological diagnosis and sex from the NSCLC-Radiogenomics clinical file.
Output: results/er/pairs_demographics.csv"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)


def nlst(pids):
    t = pd.read_csv(R("results", "nlst", "analysis_table.csv"))
    t["pid"] = t.iloc[:, 0].astype(str) if "pid" not in t else t.pid.astype(str)
    t["age_ct"] = t.age_at_dx - t.days_screen_to_dx / 365.25
    d = pd.DataFrame({"pid": pids}).merge(t[["pid", "age_ct", "sex"]], on="pid", how="left")
    miss = d.age_ct.isna() | d.sex.isna()
    if miss.any():  # fallback: NLST participant file (age at randomization + screening round; gender 1 = male, 2 = female)
        pr = pd.read_csv(R("data", "nlst", "nlst_prsn_public.csv"), usecols=["pid", "age", "gender"]).assign(pid=lambda x: x.pid.astype(str))
        sel = pd.read_csv(R("data", "nlst", "nlst_selected_series_last.csv"))[["pid", "round"]].assign(pid=lambda x: x.pid.astype(str))
        f = d[miss][["pid"]].merge(pr, on="pid", how="left").merge(sel, on="pid", how="left")
        f["age_ct"] = f.age + f["round"]
        f["sex"] = f.gender.map({1: "M", 2: "F"})
        d = d.set_index("pid")
        d.loc[f.pid, "age_ct"] = f.set_index("pid").age_ct
        d.loc[f.pid, "sex"] = f.set_index("pid").sex
        d = d.reset_index()
    return d


def main(perpair=R("results", "er", "pairs_perpair.csv"), out=R("results", "er", "pairs_demographics.csv")):
    P = pd.read_csv(perpair)
    n = nlst(P[P.dataset == "NLST"].pid.astype(str).tolist())
    rg = P[P.dataset == "RG"].assign(case_id=lambda d: d.pid.astype(str).str.split("__").str[0])
    cl = pd.read_csv(R("data", "public_rg", "clinical_rg.csv")).rename(columns={"Case ID": "case_id"})
    rg = rg[["case_id"]].merge(cl[["case_id", "Age at Histological Diagnosis", "Gender"]], on="case_id", how="left")
    rows = [dict(dataset="NLST", pairs=len(n), age_missing=int(n.age_ct.isna().sum()), age_mean=n.age_ct.mean(),
                 age_sd=n.age_ct.std(), men=int(n.sex.astype(str).str.upper().str[0].eq("M").sum()),
                 sex_missing=int(n.sex.isna().sum())),
            dict(dataset="RG", pairs=len(rg), age_missing=int(rg["Age at Histological Diagnosis"].isna().sum()),
                 age_mean=rg["Age at Histological Diagnosis"].mean(), age_sd=rg["Age at Histological Diagnosis"].std(),
                 men=int((rg.Gender == "Male").sum()), sex_missing=int(rg.Gender.isna().sum()))]
    d = pd.DataFrame(rows)
    if out:
        d.to_csv(out, index=False)
    return d


if __name__ == "__main__":
    print(main().round(2).to_string())
