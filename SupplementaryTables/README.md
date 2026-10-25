# Supplementary Tables S17–S22

These UTF-8 CSV files were extracted from the tables in the supplied v8 `Supplementary_Information.docx`. They contain displayed values, including confidence intervals and the original rounding. No estimates were recalculated. Empty cells preserve the original table grouping and do not mean zero. The three consecutive blocks of Table S22 are named S22A, S22B and S22C.

The separate root workbook is an unchanged copy containing Tables S1–S16. All eight CSV blocks were checked cell by cell against the final formatted `Supplementary_Information_Frontiers.docx`. Table S21 spans two pages in that document but remains one CSV here. The empty final column in the original S22C export is retained for compatibility; it is omitted from the formatted Word table. These layout changes do not change any values.

Supplementary Table S17. Characteristics of the same-acquisition reconstruction pairs

Values are n (%) unless stated otherwise. HU, Hounsfield unit; NLST, National Lung Screening Trial (low-dose screening CT); RG, NSCLC-Radiogenomics (diagnostic staging CT); SD, standard deviation. Muscle noise is the residual standard deviation in the eroded erector spinae.

Supplementary Table S18. Hazard ratios for overall survival under each processing strategy

HRs (95% confidence intervals) per standard deviation (within cohort and sex) from Cox models adjusted for age, sex, coarse stage, histology and kernel class; the pooled model is stratified by cohort; the sharp-kernel model is adjusted for age, sex and stage. Primary: NSCLC-Radiogenomics patients aged ≥65 years with date-concordant CT 0–180 days before surgery. Whole-mask density was evaluated post hoc. HR, hazard ratio.

Supplementary Table S19. Noise-referenced filtering: remaining sharp − soft difference in intramuscular fat by dataset, vendor and reference level

Hodges–Lehmann estimates (percentage points) with Wilcoxon signed-rank 95% confidence intervals. Every image, sharp or soft, whose image noise (robust standard deviation of the residual in the liver) exceeded the reference was smoothed (in-plane Gaussian, smallest σ of a 0.1-mm grid) to the reference; images at or below it were unchanged. The soft value shift is soft (filtered) − soft (native). NLST, National Lung Screening Trial; RG, NSCLC-Radiogenomics.

Supplementary Table S20. Sharp − soft differences with a shared mask and with each reconstruction segmented on its own

Hodges–Lehmann estimates with Wilcoxon signed-rank 95% confidence intervals for pairs with a completed separate segmentation of the sharp reconstruction. Shared mask: the soft-image mask mapped to the sharp image (threshold effect only). Own masks: each image measured with its own masks at its own T12 level (segmentation and threshold effects together). Strategies as in Table 1 of the main article; noise-referenced, 25 HU of image noise measured in the liver of each image.

Supplementary Table S21. Hazard ratios for overall survival per fixed unit under each processing strategy

HRs (95% confidence intervals) from Cox models adjusted for age, sex, coarse stage, histology and kernel class (pooled model stratified by cohort; kernel-specific models adjusted for age, sex and stage). Masks were fixed under every strategy. Noise-referenced filtering used image noise measured in the liver, or muscle noise in the last column. Whole-mask density was evaluated post hoc. HR, hazard ratio.

Supplementary Table S22. Noise-referenced filtering with each noise estimator and with a 5-mm slab, and the routine-dose same-acquisition pairs from TCGA and CPTAC

Hodges–Lehmann estimates with Wilcoxon signed-rank 95% confidence intervals; reference 25 HU. Noise estimators: standard deviation (SD) or robust SD (1.4826 × median absolute deviation) of the residual after a 3-mm local mean, in the eroded erector spinae (muscle) or in the liver at least 5 mm from its boundary. Soft value change: soft (filtered) − soft (native). TCGA/CPTAC pairs: sharp − soft differences; noise-referenced filtering with liver noise. Lowering the reference: change of soft-kernel intramuscular fat when the reference is lowered, observed and predicted by convolving the more strongly filtered image back to the muscle noise of the less filtered one (noise-convolution model); images unchanged by the step are excluded. NLST, National Lung Screening Trial; RG, NSCLC-Radiogenomics.
