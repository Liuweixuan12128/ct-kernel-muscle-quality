# Sci Rep revision: per-CT native, fixed-filtered and sharp-only T12 measurements in the 274 clinical CTs (unpaired),
# for Figure 3c, d. Same data as stats/er_cohort_kernel_medians.R, one row per CT.
# Output: results/er/cohort_kernel_percase.csv
root <- normalizePath(".")
src <- readLines(file.path(root, "stats/r1005_hr_stability.R"))
eval(parse(text = src[1:grep("^FULL <- ", src)]))
ad <- fread(file.path(root, "results/er/adaptive_exposures.csv"))
d <- merge(d, ad[, .(case_id, imat_adapt, window_adapt, whole_adapt, sigma_used, noise_native, noise_after)], by = "case_id", all.x = TRUE)
out <- d[, .(case_id, cohort, kernel_type, imat_native = native_imat_pct, imat_fixed = both_imat_pct, imat_adapt,
             window_native = native_smd_window_hu, window_fixed = both_smd_window_hu, window_adapt,
             sigma_used, noise_native, noise_after)]
fwrite(out, file.path(root, "results/er/cohort_kernel_percase.csv"))
print(out[, .(n = .N, imat_native = median(imat_native), imat_fixed = median(imat_fixed), imat_adapt = median(imat_adapt, na.rm = TRUE),
              noise_native = median(noise_native, na.rm = TRUE), noise_after = median(noise_after, na.rm = TRUE)), by = kernel_type])
