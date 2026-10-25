# Sci Rep revision 1 (plan section 4): survival associations of the T12 measures under every processing strategy,
# per fixed unit in the stated direction, so that estimates share one scale across strategies:
#   intramuscular fat per 5 percentage points higher; muscle-window density and whole-mask density per 10 HU lower;
#   muscle area per 10 cm2 lower. Per-SD estimates (z within cohort x sex) are kept for the Supplementary Information.
# Strategies: native, fixed (5-mm slab + Gaussian 0.75 mm), sharp-only (results/er/adaptive_exposures.csv),
# noise-referenced 25 HU with image noise measured in the liver (sensitivity 20 and 30 HU; results/srep/
# noise_ref_exposures.csv) and with muscle noise (noise_ref_25_muscle; noise_ref_alt_exposures.csv). Masks fixed throughout.
# Data preparation and model as stats/er_survival.R (reuses stats/r1005_hr_stability.R up to the model formula).
# Output: results/srep/survival_units.csv
root <- normalizePath(".")
src <- readLines(file.path(root, "stats/r1005_hr_stability.R"))
eval(parse(text = src[1:grep("^FULL <- ", src)]))
ad <- fread(file.path(root, "results/er/adaptive_exposures.csv"))
nr <- fread(file.path(root, "results/srep/noise_ref_exposures.csv"))
alt <- fread(file.path(root, "results/srep/noise_ref_alt_exposures.csv"))[, .(case_id, imat_es_sd, window_es_sd, whole_es_sd, area_es_sd)]
d <- merge(d, ad[, .(case_id, imat_adapt, window_adapt, whole_adapt, area_adapt)], by = "case_id", all.x = TRUE)
d <- merge(d, nr[, !c("noise_native", "noise_estimator")], by = "case_id", all.x = TRUE)
d <- merge(d, alt, by = "case_id", all.x = TRUE)
ex <- list(
  imat = list(dir = 1, unit = 5, cols = c(native = "native_imat_pct", fixed = "both_imat_pct", sharp_only = "imat_adapt",
                                          noise_ref_25 = "imat_r25", noise_ref_20 = "imat_r20", noise_ref_30 = "imat_r30", noise_ref_25_muscle = "imat_es_sd")),
  window = list(dir = -1, unit = 10, cols = c(native = "native_smd_window_hu", fixed = "both_smd_window_hu", sharp_only = "window_adapt",
                                              noise_ref_25 = "window_r25", noise_ref_20 = "window_r20", noise_ref_30 = "window_r30", noise_ref_25_muscle = "window_es_sd")),
  whole = list(dir = -1, unit = 10, cols = c(native = "native_smd_all_hu", fixed = "both_smd_all_hu", sharp_only = "whole_adapt",
                                             noise_ref_25 = "whole_r25", noise_ref_20 = "whole_r20", noise_ref_30 = "whole_r30", noise_ref_25_muscle = "whole_es_sd")),
  area = list(dir = -1, unit = 10, cols = c(native = "native_sma_cm2", fixed = "both_sma_cm2", sharp_only = "area_adapt",
                                            noise_ref_25 = "area_r25", noise_ref_20 = "area_r20", noise_ref_30 = "area_r30", noise_ref_25_muscle = "area_es_sd")))
for (e in names(ex)) for (cn in names(ex[[e]]$cols)) {
  col <- ex[[e]]$cols[[cn]]
  d[, paste0("u_", e, "_", cn) := get(col) / ex[[e]]$unit]
  d[, paste0("z_", e, "_", cn) := zs(get(col)), by = .(cohort, sex)]
}
pops <- list(primary_121 = list(d[date_ok == 1 & age >= 65], FALSE, FULL),
             interval_only_130 = list(d[timing_verified == 1 & age >= 65], FALSE, FULL),
             all_ages_160 = list(d[date_ok == 1], FALSE, FULL),
             pooled_195 = list(d[age >= 65], TRUE, FULL),
             primary_sharp = list(d[date_ok == 1 & age >= 65 & kernel_type == "sharp"], FALSE, "age10 + sex + stage_am"),
             primary_soft = list(d[date_ok == 1 & age >= 65 & kernel_type == "soft"], FALSE, "age10 + sex + stage_am"))
res <- rbindlist(lapply(names(pops), function(p) rbindlist(lapply(names(ex), function(e) rbindlist(lapply(names(ex[[e]]$cols), function(cn) {
  dat <- pops[[p]][[1]]
  rbind(cbind(population = p, exposure = e, processing = cn, scale = "unit",
              fit1(dat, paste0("u_", e, "_", cn), ex[[e]]$dir, pops[[p]][[3]], pops[[p]][[2]])),
        cbind(population = p, exposure = e, processing = cn, scale = "sd",
              fit1(dat, paste0("z_", e, "_", cn), ex[[e]]$dir, pops[[p]][[3]], pops[[p]][[2]])))
}))))))
# distribution of each exposure in the primary population (median, IQR, SD), to show how the scale changes
desc <- rbindlist(lapply(names(ex), function(e) rbindlist(lapply(names(ex[[e]]$cols), function(cn) {
  x <- pops$primary_121[[1]][[ex[[e]]$cols[[cn]]]]
  data.table(exposure = e, processing = cn, median = median(x, na.rm = TRUE), q1 = quantile(x, 0.25, na.rm = TRUE),
             q3 = quantile(x, 0.75, na.rm = TRUE), sd = sd(x, na.rm = TRUE))
}))))
fwrite(res, file.path(root, "results/srep/survival_units.csv"))
fwrite(desc, file.path(root, "results/srep/survival_units_exposure_scale.csv"))
res[, txt := sprintf("%.2f (%.2f-%.2f)", HR, lo, hi)]
print(dcast(res[population %in% c("primary_121", "pooled_195", "primary_sharp")], population + scale + exposure ~ processing, value.var = "txt"))
print(desc)
print(unique(res[note != "", .(population, exposure, processing, note)]))
