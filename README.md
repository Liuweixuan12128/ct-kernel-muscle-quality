# Reconstruction kernel noise and CT muscle quality

Analysis code and exported results accompanying **Reconstruction kernel noise biases threshold-based measurements of muscle quality on chest computed tomography**.

Updated on 25 October 2026 from the supplied v8 research package while preparing the manuscript for Frontiers in Endocrinology. The package has also been checked against the final formatted `Supplementary_Information_Frontiers.docx`: all eight displayed table blocks for S17–S22 and the S1–S16 workbook agree exactly. Journal formatting does not change the scientific calculations. This repository contains the available analysis materials; full numerical reproduction requires additional upstream modules and study inputs.

## Contents

| Path | Contents |
| --- | --- |
| [Code/](Code/) | 35 analysis, figure, helper and job files: 29 Python, 4 R and 2 SLURM examples |
| [SourceData/main/](SourceData/main/) | 36 CSV files for Figures 1–7 |
| [SourceData/supplementary/](SourceData/supplementary/) | 22 CSV files for Supplementary Figures S1–S8 |
| [SourceData/data_catalog.csv](SourceData/data_catalog.csv) | Exact columns and row counts for all 58 current figure-data files |
| [Supplementary_Tables.xlsx](Supplementary_Tables.xlsx) | Unchanged source workbook for Tables S1–S16 |
| [SupplementaryTables/](SupplementaryTables/) | Eight displayed-value CSV blocks for Tables S17–S22, with captions and notes |
| [docs/CHANGELOG_2026-10-25.md](docs/CHANGELOG_2026-10-25.md) | Changes since the repository snapshot of 20 October |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Required inputs, path assumptions and verification limits |
| [FILE_MANIFEST.csv](FILE_MANIFEST.csv) | Sizes, SHA-256 hashes and provenance for all package files except the manifest itself |
| [archive/2026-10-20/](archive/2026-10-20/) | Retired figure-data filenames retained as historical records |

## Processing conventions

The research uses NLST, NSCLC-Radiogenomics, TCGA-LUAD/LUSC and CPTAC-LUAD/LSCC images and clinical data. Native means the original reconstructed image. Fixed filtering uses a 5-mm slab and an in-plane Gaussian standard deviation of 0.75 mm. Sharp-only filtering matches muscle noise to a soft-image or cohort target. Noise-referenced filtering smooths every image with liver noise above the stated absolute reference, conventionally 25 HU, and leaves quieter images unchanged.

The 25-HU reference is a convention; it is not a validated optimal cutoff or a noise-free truth. Some kernels also shift mean attenuation, which this noise filter does not correct. Survival associations are exploratory. See the manuscript for the full definitions and limitations.

## Reproduction

The CSV files and supplementary tables are directly inspectable. The supplied package does not contain a complete executable project. Original images, masks, clinical preparation tables, several upstream modules and intermediate results must be recovered before a full run. Figure source data do not replace the complete subject-level survival dataset. See [Code/README.md](Code/README.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Python files were parsed for syntax and copied inputs were verified using SHA-256. No image-processing pipeline, R survival model or end-to-end figure regeneration was executed during this packaging task. Package versions reported by the manuscript are distinguished from the packaging environment in the reproducibility notes.

No software or data licence has been assigned during packaging. Source-collection terms continue to apply to the original public datasets.

## 中文说明

本版本对应 2026-10-25 提供的 v8 研究材料及 Frontiers in Endocrinology 投稿整理工作。当前作图数据为主图 1–7、补充图 S1–S8，共 58 个 CSV；补充表扩展至 S22。旧版已经退出当前图号体系的文件移至 archive，不能与本版数据混用。

分析脚本保持原始科学计算；仅两个 SLURM 示例替换了个人集群工作目录和日志路径。投稿正文、投稿信、作者待填信息、账号凭据和排版校样不在此仓库包中。
