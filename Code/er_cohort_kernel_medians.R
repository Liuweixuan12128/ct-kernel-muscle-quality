# ER revision: native and filtered T12 measurements by kernel class in the 274 clinical CTs (unpaired).
# Output: results/er/cohort_kernel_medians.csv
root <- normalizePath(".")
src <- readLines(file.path(root, "stats/r1005_hr_stability.R"))
eval(parse(text = src[1:grep("^FULL <- ", src)]))
ad <- fread(file.path(root, "results/er/adaptive_exposures.csv"))
d <- merge(d, ad[, .(case_id, imat_adapt, window_adapt, whole_adapt, sigma_used, noise_native, noise_after)], by = "case_id", all.x = TRUE)
q <- function(x, p) as.numeric(quantile(x, p, na.rm = TRUE))
out <- d[, .(n = .N, imat_native = median(native_imat_pct, na.rm = TRUE), imat_fixed = median(both_imat_pct, na.rm = TRUE),
             imat_adapt = median(imat_adapt, na.rm = TRUE), window_native = median(native_smd_window_hu, na.rm = TRUE),
             window_adapt = median(window_adapt, na.rm = TRUE), sigma = median(sigma_used, na.rm = TRUE),
             sigma_q1 = q(sigma_used, .25), sigma_q3 = q(sigma_used, .75),
             noise_native = median(noise_native, na.rm = TRUE), noise_after = median(noise_after, na.rm = TRUE)), by = kernel_type]
fwrite(out, file.path(root, "results/er/cohort_kernel_medians.csv"))
print(out)
