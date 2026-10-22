# Figure source data

This directory contains 46 CSV files copied unchanged from the Scientific Reports manuscript package: 39 files in `main/` for Figures 1–7, and 7 files in `supplementary/` for Supplementary Figures S1–S4.

[data_catalog.csv](data_catalog.csv) lists each file, its figure/panel label, row count, number of columns, exact column names, and a brief description. Paths in that catalogue are relative to the repository root. Row counts exclude the CSV header. Some files contain multiple rows for one participant, processing condition or measurement, so row counts are not necessarily sample sizes.

## File naming

`Figure2_c_d_e_f.csv` contains data for main Figure 2 panels c–f. `SupplementaryFigS1_b_pairs.csv` contains the pair-level data for Supplementary Figure S1b. Names ending in `summary`, `vendor`, `hr` or `icc` contain exported summaries or estimates as specified by their columns.

The files preserve their original labels, numeric precision, identifiers and missing values. CT images, segmentation masks and the complete original clinical or survival modelling inputs are outside this dataset collection.

## Common fields and units

| Field / pattern | Meaning in the included scripts |
| --- | --- |
| `dataset`, `cohort` | Source cohort label; labels can differ between analysis stages. |
| `pid`, `case_id`, `alt_id` | Participant, case or alternative-series identifier retained from the supplied source file. |
| `kernel`, `kernel_type`, `kernel_sharp`, `kernel_soft` | Reconstruction-kernel label or sharp/soft classification. |
| `role_sharp`, `role_soft` | A/B image roles assigned to the sharp and soft reconstructions. |
| `native` | Measurement on the original reconstructed image. |
| `fixed`, `matched`, `both`, `noise_matched` | Labels used for the combined slab/Gaussian filtering strategy; consult the corresponding script for each exported table. |
| `adapt`, `adaptive`, `sharp-only` | Adaptive processing of sharp-kernel images using a muscle-noise target. |
| `sigma`, `sigma_pair`, `sigma_fixed`, `sigma_used` | In-plane Gaussian standard deviation in millimetres in the filter-selection/sweep tables. |
| `slice_mm`, `px_mm`, `corr_mm`, `erode_mm` | Slice spacing/thickness, pixel size, correlation scale or erosion distance in millimetres, as named. |
| `erode_px` | Number of mask-erosion iterations/pixels. |
| `sma_cm2`, `muscle_cm2` | Muscle-window area in square centimetres in the corresponding exported table. |
| `imat_pct` | Intramuscular fat percentage; the paired-measurement script uses 100 × fat-window area / (muscle-window area + fat-window area). |
| `smd_window_hu`, `window_*` | Mean muscle-window attenuation in Hounsfield units (HU). |
| `smd_all_hu`, `whole_*` | Mean attenuation across the anatomical muscle mask, in HU. |
| `noise_*`, `residual_sd` | Noise/residual standard deviation; HU where the underlying quantity is CT attenuation. |
| `hl` | Hodges–Lehmann estimate of a paired difference. |
| `HR` | Exported Cox-model hazard ratio. |
| `icc` | Exported intraclass correlation coefficient. |
| `lo`, `hi`, `ci_lo`, `ci_hi` | Lower/upper uncertainty limits as defined by the corresponding analysis. |
| `p`, `P`, `q` | Exported significance or adjusted-significance values; consult the originating analysis for the test. |
| `time_years`, `n_risk`, `surv` | Kaplan–Meier time, number at risk and survival estimate. |

The paired measurement script uses HU windows of −29 to 150 for muscle and −190 to −30 for fat. Percentage differences are expressed in percentage points where the source table compares percentages. Some panel-specific columns use different names or definitions; the exact CSV headers and corresponding script take precedence over this general glossary.

For Supplementary Tables S1–S16, see the workbook at the repository root. Tables S17–S18 are contained in the original supplementary manuscript document rather than in this workbook.
