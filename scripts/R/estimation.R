aipw_z <- function(X, Z, pi) {
  
  # Preliminaries
  R <- !is.na(X)
  n <- nrow(X)
  p <- ncol(X)
  XR <- X; XR[R == 0] <- 0
  Zd <- cbind(1, Z)
  d <- ncol(Zd)
  
  # Linear Models
  beta <- matrix(NA_real_, d, p)
  E <- matrix(NA_real_, n, p)
  for(j in 1:p) {
    ind <- R[, j] == 1
    mod <- qr(Zd[ind, ])
    beta[, j] <- qr.coef(mod, X[ind, j])
    E[ind, j] <- qr.resid(mod, X[ind, j])
  }
  sig_e <- cov(E, use = "pairwise.complete.obs")
  
  # Predictions
  M1 <- Zd %*% beta
  M2 <- sweep(outer_array(M1), 1:2, sig_e, "+")
  
  # Augmentations
  W1 <- 1 / pi
  W2 <- 1 / outer_array(pi)
  for(i in 1:n) diag(W2[, , i]) <- W1[i, ]
  XRXR <- outer_array(XR)
  RR <- outer_array(R)
  A1 <- M1 + (W1 * R) * (XR - M1)
  A2 <- M2 + (W2 * RR) * (XRXR - M2)
  
  # Estimates
  E1 <- apply(A1, 2, mean)
  E2 <- apply(A2, 1:2, mean)
  sig_X <- E2 - tcrossprod(E1)
  return(sig_X)
  
}

# n x p to p x p n
outer_array <- function(X) {
  p <- ncol(X)
  n <- nrow(X)
  output <- array(NA_real_, dim = c(p, p, n))
  for(i in 1:n) output[, , i] <- tcrossprod(X[i, ])
  return(output)
}

psd_project <- function(A) {
  e <- eigen(A, symmetric = TRUE)
  e$values <- pmax(e$values, 0.0001)
  output <- e$vectors %*% diag(e$values) %*% t(e$vectors)
  dimnames(output) <- dimnames(A)
  return(output)
}

fiml <- function(X) {
  
  vars <- colnames(X)
  var_lines <- paste0("  ", vars, " ~~ ", vars, collapse = "\n")
  cov_pairs <- combn(vars, 2, simplify = FALSE)
  cov_lines <- sapply(cov_pairs, function(p) paste0("  ", p[1], " ~~ ", p[2]))
  form <- paste(var_lines, paste(cov_lines, collapse = "\n"), sep = "\n")
  
  mod <- lavaan::sem(form, data = X, missing = "ML") |>
    suppressWarnings()
  pe <- lavaan::parameterEstimates(mod, remove.nonfree = TRUE)
  
  # Keep only variance/covariance parameters
  vcov_rows <- pe[pe$op == "~~", ]
  vars <- unique(c(vcov_rows$lhs, vcov_rows$rhs))
  vars <- vars[order(vars)]  # consistent order
  
  # Initialize matrix
  p <- length(vars)
  S <- matrix(0, p, p, dimnames = list(vars, vars))
  
  # Fill in estimates (symmetric)
  for (i in seq_len(nrow(vcov_rows))) {
    r <- vcov_rows$lhs[i]
    c <- vcov_rows$rhs[i]
    S[r, c] <- S[c, r] <- vcov_rows$est[i]
  }
  return(S)
   
}