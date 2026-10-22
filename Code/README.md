# Analysis and figure scripts

These 18 files were copied without changes from `Manuscript_SciRep/Code`. The original filenames and scientific calculations have been preserved. `er_*` names refer to the earlier manuscript version; several of those scripts supply calculations or figure components used by the Scientific Reports version.

## Script roles

| Script | Role / principal output in the original project |
| --- | --- |
| `er_nlst_pairs_download.py` | Download selected NLST partner series from IDC and convert DICOM to NIfTI; requires the original candidate table and `dcm2niix`. |
| `er_nlst_verify.py` | Verify acquisition identity using DICOM headers; writes `results/er/nlst_pairs_dicom.csv`. |
| `er_pairs_measure.py` | Measure paired sharp/soft reconstructions with a shared mask, filtering grids and mask erosion; writes paired measurements, metadata and soft-voxel histograms. |
| `er_pairs_analysis.py` | Pair eligibility, paired contrasts, adaptive targets, filter sweep and noise-convolution decomposition. |
| `er_pairs_demographics.py` | Summarise paired-cohort demographics using original clinical inputs. |
| `er_adaptive_cohort.py` | Measure the Gaussian-filter grid in each clinical cohort. |
| `er_erode_cohort.py` | Measure the effect of mask erosion in NSCLC-Radiogenomics. |
| `er_adaptive_exposures.py` | Select sharp-only adaptive exposures and combine them with the erosion results. |
| `er_survival.R` | Fit survival associations across processing strategies and study populations. |
| `er_cohort_kernel_medians.R` | Summarise clinical CT measurements by reconstruction-kernel class. |
| `srep_cohort_kernel_percase.R` | Export the per-CT table used for Figure 3c–d. |
| `er_gdc_years.py` | Query diagnosis years through the GDC API for a pre-existing analysis population. |
| `er_phantom_sweep.py` | Digital-phantom filter-strength sweep; depends on the original phantom implementation. |
| `er_figures.py` | Earlier main-figure components reused by the Scientific Reports plotting script. |
| `er_supp_figures.py` | Measurement examples, noise analyses, repeat scans, survival and phantom figure components. |
| `er_supp_figS_c2c.py` | Comp2Comp comparison panels using original comparison results and image examples. |
| `srep_figures.py` | Render parts, compose Figures 1–7 and supplementary figures, and export panel source data. |
| `srep_compose.py` | Stack and trim rendered PDF parts using a layout JSON, producing PDF and 600-dpi PNG files. |

## Missing upstream files

The following imported modules or sourced scripts are referenced by the included code but were not present in the supplied directory or ZIP archive:

| Missing file | Referenced by |
| --- | --- |
| `bodycomp_metrics.py` | Image measurement and several figure scripts; supplies `Case` and `vertebra_info`. |
| `e4_phantom.py` | `er_phantom_sweep.py`; supplies phantom geometry and noise generation. |
| `style.py` | Plotting scripts; supplies the original project root, style and export functions. |
| `fig1_design_pipeline.py` | CT example and measurement-pipeline panels. |
| `fig_cohort.py` | `er_supp_figS_c2c.py`; supplies the forest-plot helper. |
| `c2c_t12.py` | `er_supp_figS_c2c.py`; supplies Comp2Comp helpers. |
| `fig_revision.py` | `er_supp_figures.py`; supplies the measurement-slice offset panel. |
| `stats/r1005_hr_stability.R` | All three included R scripts; supplies data preparation and modelling functions. |

There are also required data inputs outside `SourceData/`, including original `data/` tables and image volumes, segmentation outputs, `results/er/`, `results/r1005/`, `results/audit/`, `results/c2c/`, cohort body-composition tables, and the pooled survival inputs. The list above identifies missing code dependencies; it is not a complete inventory of all missing data files.

## Original layout and paths

`Code/` is the collection supplied with the manuscript. The script comments refer to the original project's `pipeline/`, `stats/` and `figures/` directories. Recover that project structure, or deliberately adapt the paths in a separate working copy, before executing the scripts.

- Most measurement and analysis Python scripts derive the project root from their parent directory and expect `data/` and `results/` below that root.
- The R scripts use the current working directory as their root and read `stats/r1005_hr_stability.R` from there.
- Figure scripts use the missing `style.ROOT` and import helpers from the original figure/pipeline directories.
- `srep_figures.py` sets `COMPOSE_PY` to an absolute interpreter path in the author's original environment. Set it to an interpreter with PyMuPDF in a working copy after restoring the dependencies.
- The NLST download script looks for `dcm2niix` beside its Python executable.
- `srep_compose.py` expects a layout JSON argument and existing rendered PDF parts. The full plotting script normally generates that layout.

## Analysis order after restoring the original project

1. Restore original candidate/clinical tables, CT volumes, segmentation masks and the upstream preprocessing results.
2. Download or locate NLST partner images and perform DICOM verification; retain the original RG verification results.
3. Run paired measurements, then paired analysis and demographics.
4. Measure adaptive filtering in each clinical cohort and measure RG mask erosion, then assemble adaptive exposures.
5. Run the R survival and cohort-summary exports using the restored upstream R preparation script.
6. Run the phantom sweep and restore the other upstream phantom, noise, repeat-scan and Comp2Comp results.
7. Render the figure components and compose the Scientific Reports figures.

This sequence describes dependencies found in the scripts. It has not been executed or validated end to end from this upload folder. The supplied figure CSVs support inspection of the exported results, but cannot substitute for all upstream inputs.
