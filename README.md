# Reconstruction kernel noise and CT muscle quality

Analysis code and figure source data accompanying the manuscript:

**Reconstruction kernel noise biases threshold-based measurements of muscle quality on chest computed tomography**

This repository was assembled from the supplied Scientific Reports manuscript package on 20 October 2026. The scientific scripts, source-data CSV files and supplementary workbook are unchanged copies of the supplied files.

## Contents

| Path | Contents |
| --- | --- |
| [Code/](Code/) | 18 analysis and figure scripts: 15 Python files and 3 R files. |
| [SourceData/main/](SourceData/main/) | 39 CSV files for main Figures 1–7. |
| [SourceData/supplementary/](SourceData/supplementary/) | 7 CSV files for Supplementary Figures S1–S4. |
| [SourceData/data_catalog.csv](SourceData/data_catalog.csv) | Figure/panel labels, row counts and column names for all 46 source-data files. |
| [Supplementary_Tables.xlsx](Supplementary_Tables.xlsx) | Supplementary Tables S1–S16. |
| [requirements.txt](requirements.txt) | Python packages identified from the included scripts. |
| [FILE_MANIFEST.csv](FILE_MANIFEST.csv) | File sizes, SHA-256 hashes and original locations for the copied files; generated documentation is also listed. |

## Reproducibility status

The source-data files and supplementary workbook can be inspected directly. This is a collection of the analysis materials available in the manuscript package; **the supplied materials do not contain the full executable study pipeline**.

Several scripts require original CT images, segmentation masks, clinical input tables, intermediate `results/` files and project modules that were not included in the supplied package or its ZIP archive. The panel CSV files are exported figure data and do not replace every original analysis input. In particular, they do not constitute the full subject-level survival modelling dataset.

[Code/README.md](Code/README.md) lists the script roles, missing modules and path assumptions. The copied code has been checked for file integrity and Python syntax; full image processing, statistical analyses and figure regeneration have not been executed for this repository.

## Environment

The original package versions were not supplied. `requirements.txt` lists observed third-party Python dependencies without inventing version pins:

```sh
python -m pip install -r requirements.txt
```

Installing these packages does not supply the missing project modules or study inputs. The R scripts load `stats/r1005_hr_stability.R`, which is absent from the supplied material. They use `data.table` functions; the complete R dependency list and version information must be recovered with that upstream script. The download script also expects a `dcm2niix` executable.

## Data and processing conventions

The study uses NLST, NSCLC-Radiogenomics, TCGA-LUAD/LUSC and CPTAC-LUAD/LSCC data. The included CSV files contain derived measurements or summaries exported for the manuscript figures. Original CT/DICOM images, segmentation volumes, model weights and full clinical input tables are not included.

`native` denotes the original reconstruction. The fixed processing strategy applies a 5-mm slab average and an in-plane Gaussian filter with a 0.75-mm standard deviation. Sharp-only adaptive processing retains the soft-kernel values and selects a Gaussian filter for sharp-kernel images using a muscle-noise target. See [SourceData/README.md](SourceData/README.md) and the script comments for field definitions and panel-specific context.

## Scope of this upload folder

The manuscript, cover letter, author submission instructions, Word temporary files, manuscript-build/reference-fetch scripts and rendered figure files are outside this folder. The retained `er_*` analysis and plotting scripts are also used by the Scientific Reports figure scripts, so their original names have been preserved.

The workbook contains Tables S1–S16. Tables S17–S18 remain in the original `Supplementary_Information` document and are not separately supplied here. The source file `supp_tables_srep.md` contains an unresolved table placeholder and has therefore not been included as a finished results file.

No software or data licence was specified in the supplied material, and none has been assigned during packaging.

## 中文说明

本目录为可单独上传至 GitHub 的文件副本，包含 18 个分析/绘图脚本、46 份作图源数据 CSV、补充表 S1–S16，以及仓库说明、Python 依赖清单和文件校验清单。原始代码、数据和工作簿未作修改。

请将本目录内的全部内容上传到仓库根目录，并保留文件夹结构与隐藏文件 `.gitignore`。本次仅整理本地文件，尚未执行 GitHub 上传。

当前投稿包缺少部分上游模块、影像和中间结果，因此本目录尚不能独立完成全流程复现；具体缺项及原始路径要求见 `Code/README.md`。未代作者选择开源许可证。
