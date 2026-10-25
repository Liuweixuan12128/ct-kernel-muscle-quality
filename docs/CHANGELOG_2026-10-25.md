# Update on 25 October 2026

This update aligns the repository with the supplied v8 study materials used for the Frontiers in Endocrinology formatting pass.

- Expanded code from 18 files to 35 analysis/helper/job files, including absolute-noise references, liver noise, back-convolution, separate segmentation, routine-dose pairs and fixed-unit survival estimates.
- Replaced the current source-data set with the 58 v8 CSVs, covering main Figures 1–7 and Supplementary Figures S1–S8.
- Retained the supplied S1–S16 workbook and added the eight displayed table blocks for S17–S22.
- Retired 16 obsolete source-data filenames to `archive/2026-10-20` so that their former panel numbering is not mistaken for current numbering.
- Updated README, dependency notes, source-data catalogue and integrity manifest. Scientific code/data are unchanged, except for configuration-only working/log paths in two SLURM examples.

Final supplementary-file alignment: the eight S17–S22 CSV blocks were checked cell by cell against `Supplementary_Information_Frontiers.docx`, and the S1–S16 workbook was verified byte for byte. The document formatting and figure placement do not require new calculations or changes to the exported research data.

The manuscript and cover letter are maintained outside the public repository. The journal-formatting changes did not rerun or alter the reported study results.
