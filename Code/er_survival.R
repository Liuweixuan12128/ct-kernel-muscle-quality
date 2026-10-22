# ER revision: survival associations of T12 measurements under each processing strategy, masks fixed.
#   native | fixed noise matching (published; both kernels filtered) | sharp-only adaptive matching (soft unchanged,
#   sharp filtered to the soft-kernel muscle-noise level) | fixed matching with the muscle mask eroded by 1 pixel.
# Populations: primary (date-concordant, >=65 y), date-concordant all ages, pooled older; kernel strata in primary.
# Reuses the data preparation of stats/r1005_hr_stability.R. Output: results/er/survival_processing.csv
root <- normalizePath(".")
src <- readLines(file.path(root, "stats/r1005_hr_stability.R"))
eval(parse(text = src[1:grep("^FULL <- ", src)]))
ad <- fread(file.path(root, "results/er/adaptive_exposures.csv"))
d <- merge(d, ad[, .(case_id, imat_adapt, window_adapt, whole_adapt, area_adapt, imat_matched_erode1)], by = "case_id", all.x = TRUE)
ex <- list(
  imat = list(dir = 1, cols = c(native = "native_imat_pct", fixed = "both_imat_pct", adaptive = "imat_adapt", fixed_eroded = "imat_matched_erode1")),
  window = list(dir = -1, cols = c(native = "native_smd_window_hu", fixed = "both_smd_window_hu", adaptive = "window_adapt")),
  whole = list(dir = -1, cols = c(native = "native_smd_all_hu", fixed = "both_smd_all_hu", adaptive = "whole_adapt")),
  area = list(dir = -1, cols = c(native = "native_sma_cm2", fixed = "both_sma_cm2", adaptive = "area_adapt")))
for (e in names(ex)) for (cn in names(ex[[e]]$cols)) {
  col <- ex[[e]]$cols[[cn]]; d[, paste0("z_", e, "_", cn) := zs(get(col)), by = .(cohort, sex)]
}
d[, imat_adapt_per5 := imat_adapt / 5]
pops <- list(primary_121 = list(d[date_ok == 1 & age >= 65], FALSE, FULL),
             all_ages_160 = list(d[date_ok == 1], FALSE, FULL),
             pooled_195 = list(d[age >= 65], TRUE, FULL),
             primary_sharp = list(d[date_ok == 1 & age >= 65 & kernel_type == "sharp"], FALSE, "age10 + sex + stage_am"),
             primary_soft = list(d[date_ok == 1 & age >= 65 & kernel_type == "soft"], FALSE, "age10 + sex + stage_am"))
res <- rbindlist(lapply(names(pops), function(p) rbindlist(lapply(names(ex), function(e) rbindlist(lapply(names(ex[[e]]$cols), function(cn) {
  z <- paste0("z_", e, "_", cn)
  dat <- pops[[p]][[1]]
  if (sum(!is.na(dat[[z]])) < 30) return(NULL)
  cbind(population = p, exposure = e, processing = cn, fit1(dat, z, ex[[e]]$dir, pops[[p]][[3]], pops[[p]][[2]]))
}))))))
res <- rbind(res, cbind(population = "primary_121", exposure = "imat", processing = "adaptive, per 5 percentage points",
                        fit1(pops$primary_121[[1]], "imat_adapt_per5", 1, FULL, FALSE)))
fwrite(res, file.path(root, "results/er/survival_processing.csv"))
res[, txt := sprintf("%.2f (%.2f-%.2f)", HR, lo, hi)]
print(dcast(res, population + exposure ~ processing, value.var = "txt"))
print(unique(res[note != "", .(population, note)]))
