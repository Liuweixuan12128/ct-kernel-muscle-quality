# Analysis and figure scripts

The files retain their original `er_*` and `srep_*` names because the scripts import one another and refer to historical project directories. These names do not imply that the current submission is to Scientific Reports.

The v8 additions include:

- `srep_noise_reference.py`, `srep_reference_backconv.py` and `srep_noise_estimators*.py`: absolute-noise filtering, soft-image back-convolution and noise-estimator comparisons.
- `srep_phantom_noise_ref.py` and `e4_phantom.py`: digital-phantom reference-noise and filtering analyses.
- `srep_sharp_own_seg*.py` and `srep_seg_sharp*.sbatch`: separate sharp-image segmentation and shared/own-mask comparisons.
- `srep_tcga_recon_pairs.py`, `srep_tcga_recon_summary.py` and `srep_kernel_shift_erosion.py`: routine-dose reconstruction pairs and attenuation-shift checks.
- `srep_survival_units.R`: fixed-unit survival associations.
- `style.py`, `fig1_design_pipeline.py` and `fig_revision.py`: figure helpers absent from the earlier repository package.

Earlier paired-analysis, clinical-cohort and plotting scripts are retained. The `.py` and `.R` files are byte-identical to the supplied v8 code. The two SLURM examples replace the author's absolute cluster working/log paths with `CT_KERNEL_PROJECT_ROOT` / `SLURM_SUBMIT_DIR` and relative `logs/` paths. Create `logs/` before invoking `sbatch`; adjust hardware requests to the target cluster.

`Code/` is a source collection. The original scripts expect a project with separate `pipeline/`, `stats/`, `figures/`, `data/` and `results/` directories. Restore that structure and its inputs before execution. Installing Python dependencies does not restore these missing project modules.

See [reproducibility notes](../docs/REPRODUCIBILITY.md) for the known missing modules and verification limits.
