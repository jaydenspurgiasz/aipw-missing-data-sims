# Working Demonstration

# Initialization ####
rm(list = ls())
list.files("functions", full.names = TRUE) |> lapply(source) |> invisible()

# Data Generation
set.seed(23423)

n <- 1000
dgp <- dgp_two_fac$new(n)
p <- 6
lt <- lower.tri(diag(p), diag = TRUE)
r <- 100
methods <- c(
  "Full", "Cases", "Pairs", "AIPW_pi_hat", "AIPW_pi", "AIPW_pi_mis", "AIPW_z2",
  "FIML", "FIML_Z2", "FIML_Z"
)

# Covariance Container
dn <- list(r = 1:r, Method = methods, theta = 1:sum(lt))
est <- array(NA_real_, dim = sapply(dn, length), dimnames = dn)

# Factor Analysis Containers
dn <- list(r = 1:r, Method = methods, theta = 1:13)
est_fa <- array(NA_real_, dim = sapply(dn, length), dimnames = dn)
form <- "l1 =~ x1 + x2 + x3 \n l2 =~ x4 + x5 + x6"

for(iter in 1:r) {
  cat("Iteration:", iter, "\r")
  
  dgp$gen()
  
  pi_hat <- glm(
    cbind(apply(dgp$R, 1, sum), p - apply(dgp$R, 1, sum)) ~ dgp$Z,
    family = "binomial"
  ) |> predict(type = "response")
  pi_hat <- pi_hat %*% t(rep(1, p))
  
  pi_mis <- glm(
    cbind(apply(dgp$R, 1, sum), p - apply(dgp$R, 1, sum)) ~ dgp$Z2,
    family = "binomial"
  ) |> predict(type = "response")
  pi_mis <- pi_mis %*% t(rep(1, p))
  
  pi_hat2 <- glm(
    cbind(apply(dgp$R2, 1, sum), p - apply(dgp$R2, 1, sum)) ~ dgp$Z2,
    family = "binomial"
  ) |> predict(type = "response")
  pi_hat2 <- pi_hat2 %*% t(rep(1, p))
  
  # Estimate Covariances
  dn <- list(paste0("x", 1:p), paste0("x", 1:p), dimnames(est)[[2]])
  est_cov <- array(NA_real_, dim = sapply(dn, length), dimnames = dn)
  est_cov[, , 1] <- cov(dgp$X_true)
  est_cov[, , 2] <- cov(dgp$X_o, use = "complete.obs")
  est_cov[, , 3] <- cov(dgp$X_o, use = "pairwise.complete.obs")
  est_cov[, , 4] <- aipw_z(dgp$X_o, dgp$Z, pi_hat)
  est_cov[, , 5] <- aipw_z(dgp$X_o, dgp$Z, dgp$pi %*% t(rep(1, p)))
  est_cov[, , 6] <- aipw_z(dgp$X_o, dgp$Z, pi_mis)
  est_cov[, , 7] <- aipw_z(dgp$X_o2, dgp$Z2, pi_hat2)
  est_cov[, , 8] <- fiml(dgp$X_o)
  est_cov[, , 9] <- fiml(cbind(dgp$X_o2, dgp$Z2))[1:p, 1:p]
  est_cov[, , 10] <- fiml(cbind(dgp$X_o, dgp$Z))[1:p, 1:p]
  for(j in 1:dim(est)[2]) {
    est[iter, j, ] <- est_cov[, , j][lt]
    if(is_invertible(est_cov[, , j])) {
      mod <- lavaan::cfa(
        form, sample.cov = est_cov[, , j] |> psd_project(), sample.nobs = n,
        std.lv = TRUE
      )
      if(lavaan_pass(mod)) {
        est_fa[iter, j, ] <- lavaan::parameterEstimates(
          mod, remove.nonfree = TRUE
        )[, "est"]
      } else {
        est_fa[iter, j, ] <- NA
      }
    } else {
      est_fa[iter, j, ] <- NA
    }
  }
  est_fa <- abs(est_fa)
}
cat("\n")
result <- result_table(est, dgp$sig_X[lt])
result

theta_fa <- c(dgp$Lambda[dgp$Lambda != 0], diag(dgp$Omega), dgp$Phi[1, 2])
result_fa <- result_table(est_fa, theta_fa)
result_fa

apply(est_fa, 2, function(x) sum(complete.cases(x)))

while(dev.cur() > 1) dev.off()
par(mfrow = c(2, 3))
barplot(result[, "Rel. Bias"], las = 2, main = "Cov Rel. Bias")
barplot(result[, "Var"], las = 2, main = "Cov Var")
barplot(result[, "MSE"], las = 2, main = "Cov MSE")
barplot(result_fa[, "Rel. Bias"], las = 2, main = "FA Rel. Bias")
barplot(result_fa[, "Var"], las = 2, main = "FA Var")
barplot(result_fa[, "MSE"], las = 2, main = "FA MSE")