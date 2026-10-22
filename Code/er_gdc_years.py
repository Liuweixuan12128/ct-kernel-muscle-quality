"""ER revision: year of diagnosis of the TCGA and CPTAC patients in the clinical cohort (Genomic Data Commons API),
for the date ranges required in Methods. Output: results/er/gdc_diagnosis_years.csv"""
import json
import os
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = pd.read_csv(os.path.join(ROOT, "results", "public_pooled", "analysis_population.csv"))
ids = ap[~ap.cohort.str.contains("RG")].case_id.tolist()
q = {"filters": json.dumps({"op": "in", "content": {"field": "submitter_id", "value": ids}}),
     "fields": "submitter_id,project.project_id,diagnoses.year_of_diagnosis", "format": "JSON", "size": "1000"}
req = urllib.request.Request("https://api.gdc.cancer.gov/cases", data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
rows = []
for h in hits:
    ys = [d.get("year_of_diagnosis") for d in h.get("diagnoses", []) if d.get("year_of_diagnosis")]
    rows.append(dict(case_id=h["submitter_id"], project=h["project"]["project_id"], year_of_diagnosis=min(ys) if ys else None))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "results", "er", "gdc_diagnosis_years.csv"), index=False)
print(len(ids), len(d))
print(d.groupby(d.project.str.split("-").str[0]).year_of_diagnosis.agg(["min", "max", "count"]))
