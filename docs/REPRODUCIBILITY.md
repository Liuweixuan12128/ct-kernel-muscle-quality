# Reproducibility status

## Included and missing materials

Included are the available v8 analysis/figure scripts, 58 exported panel-data files, the original S1–S16 workbook and displayed-value exports of S17–S22. These are not substitutes for the full model-fitting inputs.

Known missing upstream code includes:

- `bodycomp_metrics.py`
- `fig_cohort.py`
- `c2c_t12.py`
- `pipeline/p2_segment_worker.py`
- `stats/r1005_hr_stability.R`

Other original project dependencies and intermediate paths may also be needed. Original DICOM/NIfTI volumes, segmentation masks, full clinical preparation tables, original cohort selection/QC records and intermediate `results/er`, `results/srep`, `results/r1005`, `results/c2c` inputs are outside the supplied package.

## Environment and paths

The manuscript reports Python 3.11, R 4.5.2, survival 3.8.3 and TotalSegmentator 2.18.0. These are author-reported study versions, not versions used to validate this package. `requirements.txt` records observed direct imports without inventing a complete environment lockfile. dcm2niix, segmentation models, GPU dependencies and all upstream R packages must be installed separately.

Most analysis scripts infer the original project root from their location; plotting scripts use `style.ROOT` and the historical figure directories. Some figure scripts retain absolute interpreter paths from the original environment. Adapt these configuration paths in a working copy after restoring the full project. The two SLURM examples have portable working/log paths, but still depend on the missing segmentation worker and the original data layout.

Suggested dependency order is public-data preparation and QC, acquisition-pair verification, shared-mask measurements, clinical-cohort measurements, noise-reference and estimator analyses, independent sharp-image segmentation, phantom and routine-dose checks, survival models, and figure composition. This order has not been executed from the released folder.

## Checks performed

Python AST parsing; rectangular CSV structure; non-empty tables; unchanged-source SHA-256 comparisons; manifest completeness; workbook sheet-name inspection; and comparison against the previous repository snapshot. The eight supplementary-table exports preserve every displayed cell from the supplied Word tables. No new statistical estimates or synthetic data were created. No GPU analysis, R model or full pipeline was run.
