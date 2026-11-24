# Exact input-covariance replay: no covariance repair, no hidden N/(N-1) scaling.
args <- commandArgs(trailingOnly=TRUE)
out <- if(length(args)) args[1] else 'runs/lavaan_replay_n250'
input_root <- out
.libPaths(c(file.path(out,'r_library'),.libPaths()))
suppressPackageStartupMessages(library(lavaan))
stopifnot(requireNamespace('numDeriv',quietly=TRUE))
input <- read.csv(file.path(out,'covariances.csv'),check.names=FALSE)
# Binary64 is the authoritative transport; CSV remains a readable export.
rawcov <- readBin(file.path(out,'covariance_input.bin'),'double',n=nrow(input)*36,size=8,endian='little')
stopifnot(length(rawcov)==nrow(input)*36)
input[paste0('s',1:36)] <- matrix(rawcov,ncol=36,byrow=TRUE)
vtable <- read.csv(file.path(out,'covariance_vcov.csv'))
stopifnot(identical(vtable$replication,input$replication),identical(vtable$method,input$method))
vinput <- as.matrix(vtable[paste0('v',1:441)])
rawvcov <- readBin(file.path(out,'covariance_vcov_input.bin'),'double',n=nrow(input)*441,size=8,endian='little')
stopifnot(length(rawvcov)==nrow(input)*441)
vinput <- matrix(rawvcov,ncol=441,byrow=TRUE)
original <- read.csv(file.path(out,'original_status.csv'))
oldpars <- read.csv(file.path(out,'original_parameters.csv'))
if(length(args)>=3) {
  shard <- as.integer(args[2]); shards <- as.integer(args[3])
  keep <- ((seq_len(nrow(input))-1) %% shards)==shard
  input <- input[keep,,drop=FALSE]; vinput <- vinput[keep,,drop=FALSE]
  out <- file.path(out,paste0('shard_',shard)); dir.create(out,showWarnings=FALSE)
}
# Emit exactly the doubles R parsed, in Python's row-major order for byte comparison.
binary <- file(file.path(out,'R_parsed_covariances.bin'),'wb')
writeBin(as.double(t(as.matrix(input[paste0('s',1:36)]))),binary,size=8,endian='little')
close(binary)
model <- 'f1 =~ x1 + x2 + x3
          f2 =~ x4 + x5 + x6'
truth <- c(rep(.8,6),rep(.36,6),.3)
nm <- c(paste0('loading_',1:6),paste0('uniqueness_',1:6),'factor_correlation')
# Python lower triangle uses row-major order, explicitly retain it here.
pairs <- do.call(rbind,lapply(1:6,function(j)cbind(j,1:j)))
sigma <- function(theta) {
  L <- matrix(0,6,2); L[1:3,1] <- theta[1:3]; L[4:6,2] <- theta[4:6]
  L %*% matrix(c(1,theta[13],theta[13],1),2,2) %*% t(L)+diag(theta[7:12])
}
score <- function(theta,S) {
  Sig <- sigma(theta); inv <- solve(Sig)
  derivatives <- numDeriv::jacobian(function(t)as.vector(sigma(t)),theta)
  as.vector(.5 * t(derivatives) %*% as.vector(inv-inv%*%S%*%inv))
}
robust_variance <- function(theta,S,V) {
  objective <- function(t) {
    Sig <- sigma(t)
    .5*(as.numeric(determinant(Sig,logarithm=TRUE)$modulus)+sum(diag(solve(Sig,S))))
  }
  H <- numDeriv::hessian(objective,theta)
  inv <- solve(sigma(theta))
  ds <- numDeriv::jacobian(function(t)as.vector(sigma(t)),theta)
  B <- matrix(0,13,21)
  for(a in 1:21) {
    E <- matrix(0,6,6); j <- pairs[a,1]; k <- pairs[a,2]; E[j,k] <- 1; E[k,j] <- 1
    B[,a] <- -.5*t(ds)%*%as.vector(inv%*%E%*%inv)
  }
  deriv <- -solve(H,B)
  variance <- deriv%*%V%*%t(deriv)
  list(V=(variance+t(variance))/2,condition=kappa(H),score=max(abs(numDeriv::grad(objective,theta))))
}
rows <- list(); params <- list(); processed <- 0
for(i in seq_len(nrow(input))) {
  item <- input[i,]; S <- matrix(as.numeric(item[paste0('s',1:36)]),6,6,byrow=TRUE)
  data_values <- readBin(file.path(input_root,'data',sprintf('rep_%04d.bin',item$replication)),
    'double',n=250*15,size=8,endian='little')
  stopifnot(length(data_values)==250*15)
  original_data <- matrix(data_values,ncol=15,byrow=TRUE)
  if(item$method=='oracle_full_data') stopifnot(max(abs(cov(original_data[,1:6])-S))<1e-12)
  if(item$method=='listwise_deletion') stopifnot(max(abs(cov(na.omit(original_data[,7:12]))-S))<1e-12)
  dimnames(S) <- list(paste0('x',1:6),paste0('x',1:6))
  ans <- list(replication=item$replication,seed=item$seed,method=item$method,
    fit_status='unexpected_error',converged=FALSE,min_uniqueness=NA_real_,
    negative_uniqueness=FALSE,near_boundary=FALSE,min_latent_eigenvalue=NA_real_,
    max_abs_gradient=NA_real_,input_max_abs_difference=NA_real_,
    objective=NA_real_,chisq=NA_real_,df=NA_real_,rmsea=NA_real_,cfi=NA_real_,srmr=NA_real_,
    robust_inference_ok=FALSE,robust_hessian_condition=NA_real_,robust_score=NA_real_,
    observed_fraction=mean(!is.na(original_data[,7:12])),complete_rows=sum(complete.cases(original_data[,7:12])),
    error='',warnings='')
  warnings <- character()
  fit <- if(item$min_eigenvalue<=0) NULL else tryCatch(withCallingHandlers(cfa(model,sample.cov=S,sample.nobs=item$n_obs,
    sample.cov.rescale=FALSE,std.lv=TRUE,estimator='ML',meanstructure=FALSE,
    control=list(iter.max=2000)),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
    error=function(e){ans$error<<-conditionMessage(e);NULL})
  ans$warnings <- paste(unique(warnings),collapse=' | ')
  if(item$min_eigenvalue<=0) {
    ans$fit_status <- 'invalid_input_covariance'
    ans$error <- 'Original covariance is not SPD; not passed to lavaan or repaired.'
  } else if(!is.null(fit)) {
    ans$converged <- lavInspect(fit,'converged')
    used <- lavInspect(fit,'sampstat')$cov
    ans$input_max_abs_difference <- max(abs(used-S))
    stopifnot(ans$input_max_abs_difference<1e-14)
    pt <- parameterEstimates(fit)
    load <- pt$est[pt$op=='=~']; L <- matrix(0,6,2); L[1:3,1]<-load[1:3]; L[4:6,2]<-load[4:6]
    phi <- lavInspect(fit,'cov.lv'); stopifnot(max(abs(diag(phi)-1))<1e-12)
    signs <- sign(c(L[1,1],L[4,2])); signs[signs==0]<-1
    load <- c(L[1:3,1]*signs[1],L[4:6,2]*signs[2])
    unique <- diag(lavInspect(fit,'theta'))
    theta <- c(load,unique,phi[1,2]*prod(signs))
    ans$min_uniqueness <- min(unique); ans$negative_uniqueness <- any(unique<0)
    ans$near_boundary <- any(unique<=1e-4)
    ans$min_latent_eigenvalue <- min(eigen(phi,symmetric=TRUE,only.values=TRUE)$values)
    ans$max_abs_gradient <- max(abs(lavInspect(fit,'gradient')))
    Sig <- fitted(fit)$cov
    ans$objective <- .5*(as.numeric(determinant(Sig,logarithm=TRUE)$modulus)+sum(diag(solve(Sig,S))))
    ans$fit_status <- if(!ans$converged)'nonconvergence' else if(ans$near_boundary)'heywood_boundary' else if(ans$min_latent_eigenvalue<=0 || abs(theta[13])>=.995)'inadmissible_solution' else 'success'
    fm <- tryCatch(suppressWarnings(fitMeasures(fit,c('chisq','df','rmsea','cfi','srmr'))),error=function(e)NULL)
    if(!is.null(fm)) for(key in names(fm)) ans[[key]] <- unname(fm[key])
    se <- rep(NA_real_,13)
    if(ans$fit_status=='success' && startsWith(item$method,'drfa_')) {
      V <- matrix(vinput[i,],21,21,byrow=TRUE)
      rob <- tryCatch(robust_variance(theta,S,V),error=function(e){ans$error<<-paste(ans$error,conditionMessage(e));NULL})
      if(!is.null(rob)) {
        ans$robust_hessian_condition <- rob$condition; ans$robust_score <- rob$score
        ans$robust_inference_ok <- all(is.finite(rob$V)) && all(diag(rob$V)>0) && min(eigen((rob$V+t(rob$V))/2,symmetric=TRUE)$values)> -1e-9 && rob$score<1e-5
        if(ans$robust_inference_ok)se <- sqrt(diag(rob$V))
      }
    }
    params[[length(params)+1]] <- data.frame(replication=item$replication,method=item$method,
        parameter=nm,estimate=theta,truth=truth,robust_se=se,
        covered=abs(theta-truth)<=qnorm(.975)*se,ci_width=2*qnorm(.975)*se)
  }
  rows[[length(rows)+1]] <- as.data.frame(ans,stringsAsFactors=FALSE)
  processed <- processed+1
  if(processed%%100==0 || i==nrow(input)) {
    write.csv(do.call(rbind,rows),file.path(out,'lavaan_fits.csv'),row.names=FALSE,na='NA')
    if(length(params))write.csv(do.call(rbind,params),file.path(out,'lavaan_parameters.csv'),row.names=FALSE,na='NA')
    cat(format(Sys.time()),processed,'/',nrow(input),'fits complete\n');flush.console()
  }
}
fits <- do.call(rbind,rows)
paired <- merge(fits,original[,c('replication','method','fit_status')],by=c('replication','method'),suffixes=c('_lavaan','_torch'))
write.csv(paired,file.path(out,'paired_status.csv'),row.names=FALSE)
write.csv(as.data.frame(with(paired,table(method,fit_status_torch,fit_status_lavaan))),file.path(out,'paired_status_counts.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'))
writeLines('Complete. Native lavaan ML standard errors and fit-index reference distributions are not validated for AIPW covariance inputs. AIPW intervals use exported moment-estimator vcov propagated through the observed CFA score Hessian.',file.path(out,'R_COMPLETE.txt'))
