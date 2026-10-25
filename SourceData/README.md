# Figure source data

The current v8 dataset contains 58 byte-identical source CSV files: 36 for main Figures 1–7 and 22 for Supplementary Figures S1–S8. The generated [catalogue](data_catalog.csv) lists the exact columns and row counts. Row counts are not necessarily sample sizes because participants or estimates can occupy several rows.

Use current figure numbers with the current manuscript. Historical retired filenames are under `archive/2026-10-20` and are excluded from this catalogue. Blank values retain their source meaning and were not filled with zero.

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


## Absolute-noise processing

The v8 noise-referenced strategy uses liver residual noise and a stated absolute reference (conventionally 25 HU). Muscle-noise reference estimates are sensitivity analyses. The relevant script and column names determine which estimator and filter variant a row represents. Fixed-unit hazard ratios must be interpreted in the stated direction; they must not be combined with per-standard-deviation estimates.

Tables S1–S16 are in the root workbook. Tables S17–S22 have displayed-value exports in `SupplementaryTables/`.
