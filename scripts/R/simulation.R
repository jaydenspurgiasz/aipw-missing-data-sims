# Input:
# - (sim x method x parameter) matrix of parameter estimates,
# - vector of true parameters
# Output: Table of estimator performance

result_table <- function(est, theta0) {
  
  # Preliminaries
  k <- ncol(est)
  d <- dim(est)[3]
  
  dn <- list(
    Method = dimnames(est)[[2]],
    Outcome = c("Var", "Bias", "Rel. Bias", "MSE", "F", "pval")
  )
  output <- array(NA_real_, dim = sapply(dn, length), dimnames = dn)
  for(i in 1:k) {
    
    # if(i == 6) browser()
    
    est_i <- est[, i, ]
    est_i <- est_i[complete.cases(est[, i, ]), ]
    r <- nrow(est_i)
    
    m <- apply(est_i, 2, mean)
    S <- cov(est_i)
    b <- m - theta0
    df1 <- r - d
    df2 <- d * (r - 1)
    
    output[i, "Var"] <- sum(diag(S))
    output[i, "Bias"] <- sqrt(sum(b^2))
    output[i, "Rel. Bias"] <- sqrt(sum(b^2)) / sqrt(sum(theta0^2))
    output[i, "MSE"] <- output[i, "Var"] + output[i, "Bias"]^2
    
    if(is_invertible(S)) {
      output[i, "F"] <- (df1/df2) * r * t(b) %*% solve(S) %*% b
      output[i, "pval"] <- pf(
        output[i, "F"], df1 = d, df2 = r - d, lower.tail = FALSE
      )
    } else {
      output[i, "F"] <- NA
      output[i, "pval"] <- NA
    }
  }
  return(output)
  
}

is_invertible <- function(A) {
  test <- tryCatch(
    solve(A),
    error = function(e) {
      NULL
    }
  )
  if(is.null(test)) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}

lavaan_pass <- function(fit) {
  c(
    converged = lavaan::lavInspect(fit, "converged"),
    optim_warn = lavaan::lavTech(fit, "optim")$warn == "",
    negative_variances = nrow(subset(
      lavaan::parameterEstimates(fit),
      op == "~~" & lhs == rhs & est < 0
    )) == 0
  ) |> all()
}
