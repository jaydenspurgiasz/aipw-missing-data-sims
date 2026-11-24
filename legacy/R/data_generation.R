dgp_base <- R6::R6Class(
  classname = "base_dgp",
  public = list(
    n = NULL,
    X_true = NULL,
    X_o = NULL,
    X_o2 = NULL,
    Z = NULL,
    Z2 = NULL,
    R = NULL,
    R2 = NULL,
    seed = NULL,
    sig_X = NULL,
    pi = NULL,
    pi2 = NULL,
    
    initialize = function(warn = TRUE) {
      if(warn) warning("dgp should not be instantiated.")
    },
    
    set_seed = function(seed = NULL) {
      if(is.null(seed)) {
        self$seed <- sample(-999999:999999, 1)
      } else {
        self$seed <- seed
      }
      set.seed(self$seed)
    }
  )
)

dgp_two_fac <- R6::R6Class(
  inherit = dgp_base,
  classname = "missingess_template",
  public = list(
    
    Lambda = NULL,
    Phi = NULL,
    Omega = NULL,
    
    initialize = function(n) {
      self$n <- n
    },
    
    gen = function(seed = NULL) {
      self$set_seed(seed)
      self$Lambda <- rbind(diag(2) %x% rep(.8, 3))
      p <- nrow(self$Lambda)
      self$Phi <- matrix(c(1, .3, .3, 1), 2, 2)
      self$Omega <- diag(1 - diag(self$Lambda %*% self$Phi %*% t(self$Lambda)))
      self$sig_X <- self$Lambda %*% self$Phi %*% t(self$Lambda) + self$Omega
      mu_X <- rep(2, p); names(mu_X) <- paste0("x", 1:p)
      sig_U <- rbind(cbind(self$sig_X, .64), .64)
      sig_U[p + 1, p + 1] <- 1
      mu_U <- c(mu_X, Z = 2)
      U <- MASS::mvrnorm(self$n, mu_U, sig_U)
      self$X_true <- U[, 1:p]
      self$Z <- U[, p + 1, drop = FALSE]
      self$pi <- plogis(self$Z - 2 + 1.4)
      self$pi2 <- plogis(2 * scale(self$Z^2) + 2)
      self$Z2 <- self$Z^2
      self$R <- matrix(NA_real_, self$n, p)
      for(i in 1:p) self$R[, i] <- rbinom(self$n, 1, self$pi) == 1
      self$R2 <- matrix(NA_real_, self$n, p)
      for(i in 1:p) self$R2[, i] <- rbinom(self$n, 1, self$pi2) == 1
      self$X_o <- self$X_true
      self$X_o2 <- self$X_true
      self$X_o[self$R == 0] <- NA
      self$X_o2[self$R2 == 0] <- NA
    }
    
  )
)
