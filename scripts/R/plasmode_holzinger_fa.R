# Plasmode FA params from lavaan::HolzingerSwineford1939.
#
# Board steps 1–2: load data, standardize x1..x9, fit 3-factor CFA with
# factor variances fixed at 1 (Phi is a correlation matrix), export
# Lambda / Phi / Omega as CSV for Python.
#
# Usage (from repo root):
#   Rscript scripts/R/plasmode_holzinger_fa.R [out_dir]
# Default out_dir: params/plasmode_holzinger

args <- commandArgs(trailingOnly = TRUE)

find_repo_root <- function() {
  # Walk up from this script / cwd until we see scripts/R + (params|dgp).
  candidates <- c(
    normalizePath(".", winslash = "/", mustWork = FALSE),
    normalizePath(file.path(".."), winslash = "/", mustWork = FALSE),
    normalizePath(file.path("..", ".."), winslash = "/", mustWork = FALSE)
  )
  script <- tryCatch(
    normalizePath(sys.frame(1)$ofile, winslash = "/", mustWork = FALSE),
    error = function(e) NA_character_
  )
  if (!is.na(script)) {
    candidates <- c(
      normalizePath(file.path(dirname(script), "..", ".."), winslash = "/", mustWork = FALSE),
      candidates
    )
  }
  for (cand in unique(candidates)) {
    if (dir.exists(file.path(cand, "scripts", "R"))) {
      return(cand)
    }
  }
  normalizePath(".", winslash = "/", mustWork = TRUE)
}

root <- find_repo_root()
out_dir <- if (length(args) >= 1) {
  args[[1]]
} else {
  file.path(root, "params", "plasmode_holzinger")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    stop("Package 'lavaan' is required. Install with install.packages('lavaan').")
  }
  library(lavaan)
})

# =============================================================================
# Data + model
# =============================================================================

data(HolzingerSwineford1939)
vars <- paste0("x", 1:9)
X <- scale(as.matrix(HolzingerSwineford1939[, vars]), center = TRUE, scale = TRUE)
colnames(X) <- vars

model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

# std.lv=TRUE => factor variances fixed at 1; Phi is a correlation matrix.
fit <- cfa(model, data = as.data.frame(X), std.lv = TRUE, meanstructure = FALSE)
est <- parameterEstimates(fit)

# =============================================================================
# Extract Lambda (9x3), Phi (3x3), Omega (diag uniquenesses)
# =============================================================================

factors <- c("visual", "textual", "speed")
p <- length(vars)
k <- length(factors)

Lambda <- matrix(0, nrow = p, ncol = k, dimnames = list(vars, factors))
for (i in seq_len(nrow(est))) {
  if (est$op[i] == "=~" && est$rhs[i] %in% factors && est$lhs[i] %in% vars) {
    Lambda[est$lhs[i], est$rhs[i]] <- est$est[i]
  }
}

Phi <- diag(k)
dimnames(Phi) <- list(factors, factors)
for (i in seq_len(nrow(est))) {
  if (est$op[i] == "~~" && est$lhs[i] %in% factors && est$rhs[i] %in% factors) {
    Phi[est$lhs[i], est$rhs[i]] <- est$est[i]
    Phi[est$rhs[i], est$lhs[i]] <- est$est[i]
  }
}

uniq <- setNames(rep(NA_real_, p), vars)
for (i in seq_len(nrow(est))) {
  if (est$op[i] == "~~" && est$lhs[i] %in% vars && est$lhs[i] == est$rhs[i]) {
    uniq[est$lhs[i]] <- est$est[i]
  }
}
if (anyNA(uniq)) {
  stop("Failed to extract all uniquenesses from lavaan fit.")
}
Omega <- diag(as.numeric(uniq))
dimnames(Omega) <- list(vars, vars)

Sigma_X <- Lambda %*% Phi %*% t(Lambda) + Omega

# =============================================================================
# Write CSVs + meta (CSV interchange; Python reads these directly)
# =============================================================================

write.csv(Lambda, file.path(out_dir, "lambda.csv"), row.names = TRUE)
write.csv(Phi, file.path(out_dir, "phi.csv"), row.names = TRUE)
write.csv(Omega, file.path(out_dir, "omega.csv"), row.names = TRUE)
write.csv(Sigma_X, file.path(out_dir, "sigma_x.csv"), row.names = TRUE)
write.csv(
  data.frame(variable = vars, uniqueness = as.numeric(uniq)),
  file.path(out_dir, "uniqueness.csv"),
  row.names = FALSE
)

json_escape <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub("\"", "\\\"", x, fixed = TRUE)
  x <- gsub("\n", "\\n", x, fixed = TRUE)
  x <- gsub("\r", "", x, fixed = TRUE)
  x
}
json_string_vec <- function(v) {
  paste0("[", paste(sprintf("\"%s\"", vapply(v, json_escape, character(1))), collapse = ", "), "]")
}

meta_json <- paste0(
  "{\n",
  "  \"source\": \"lavaan::HolzingerSwineford1939\",\n",
  sprintf("  \"n_obs_fit\": %d,\n", nrow(X)),
  sprintf("  \"variables\": %s,\n", json_string_vec(vars)),
  sprintf("  \"factors\": %s,\n", json_string_vec(factors)),
  "  \"std_lv\": true,\n",
  sprintf("  \"model\": \"%s\",\n", json_escape(trimws(model))),
  sprintf(
    "  \"lavaan_converged\": %s,\n",
    if (isTRUE(lavInspect(fit, "converged"))) "true" else "false"
  ),
  sprintf("  \"created\": \"%s\"\n", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  "}\n"
)
writeLines(meta_json, file.path(out_dir, "meta.json"))

cat("Wrote plasmode FA params to:\n  ", normalizePath(out_dir, winslash = "/"), "\n", sep = "")
cat("  lambda.csv  phi.csv  omega.csv  sigma_x.csv  uniqueness.csv  meta.json\n")
