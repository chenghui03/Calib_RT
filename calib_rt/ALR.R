# source from: https://github.com/olrodrig/ALR

#
# Program to compute automated loess regressions
#
#
suppressPackageStartupMessages(library(R.utils, quietly=TRUE))
options(width=100)
args = commandArgs(trailingOnly=TRUE)
library('matrixStats')
options(warn=-1) #=-1: does not print warnings. =0: pring warnings. =1: print warnings and details

#------------------------------------------------------------------------

# Automated_Loess_Regression class
Automated_Loess_Regression <- function(x, y, err_y=0, deg=2, alpha=0, outliers_det=FALSE, n_sims=1000, average=TRUE, verbose=FALSE){

  if (length(err_y) > 1){
    with_y_errors <- TRUE
    if (0 %in% err_y){
      n_zeros <- sum(err_y == 0)
      if (n_zeros == length(err_y)){
        err_y <- 0
        with_y_errors <- FALSE
      } else {
        cat('ERROR (in Automated_Loess_Regression): some of your error measurements are zero. Chech your input data.\n')
        stop()
      }
    }
  } else {
    if (err_y == 0){
      with_y_errors <- FALSE
    } else {
      with_y_errors <- TRUE
      err_y <- rep(err_y, length(y))
    }
  }

  #if alpha is a vector (with two elements), then the optimum alpha value to use is the alpha value computed
  #from data with x values in the range [alpha[1], alpha[2]], rescaled to the total number of data
  if (length(alpha) == 2){
  
    #selects data in the range [alpha[1], alpha[2]]
    i_bool    <- indices.in.range(x, alpha[1], alpha[2])
    x_loc     <-     x[i_bool]
    y_loc     <-     y[i_bool]
    err_y_loc <- err_y[i_bool]
    ALR_loc   <- Automated_Loess_Regression(x_loc, y_loc, err_y=err_y_loc, deg=deg, alpha=0, outliers_det=outliers_det, average=average, verbose=FALSE)
    alpha_loc <- ALR_loc$alpha
    alpha <- alpha_loc*length(x_loc)/length(x)
  }
  
  N.input.data <- length(x)
  k_tukey      <- 1.5    #parameter to define the inner fences of the Tukey's rule
  
  #Identification of possible outliers using the Tukey's rule
  data  <- identify.possible.outliers(x, y, err_y=err_y, average=average, deg=deg, alpha=alpha, k_tukey=k_tukey, outliers_det=outliers_det)
  x     <- data[[1]]; y     <- data[[2]]; err_y     <- data[[3]]
  x_out <- data[[4]]; y_out <- data[[5]]; err_y_out <- data[[6]]
  loess.solution <- data[[7]]
  N.out <- length(x_out)
  
  #x-coordinates for data interpolation
  num      <- 20
  x_sorted <- sort(unique(x))
  x_intpol <- min(x_sorted)
  for (i in 1:length(x_sorted)-1){
    dx <- (x_sorted[i+1]-x_sorted[i])/num
    for (j in 1:num){
      x_intpol <- c(x_intpol, x_sorted[i]+dx*j)
    }
  }

  if(loess.solution == TRUE){
  
    #compute the model, including intrinsic error
    data  <- model.with.intrinsic.error(x, y, err_y=err_y, deg=deg, alpha=alpha)
    model <- data[[1]]; optimum.alpha <- data[[2]]; err_0_tot <- data[[3]] 
    
    #check whether the possible outliers are real outliers
    check_outliers <- TRUE
    while(outliers_det == TRUE && check_outliers == TRUE && length(x_out) != 0){
        
      data    <- tukey.fences(model$residuals, k_tukey)
      fence_l <- data[1]; fence_u <- data[2]
      
      residuals_out <- y_out - predict(model, x_out)
      #check outliers, one by one
      i_outlier       <- -1
      lowest_residual <- 1000.0
      for (i in 1:length(residuals_out)){
        if(residuals_out[i] > fence_l && residuals_out[i] < fence_u){
          if(abs(residuals_out[i]) < lowest_residual){
            i_outlier       <- i
            lowest_residual <- abs(residuals_out[i])
          }
        }
      }           
      if(i_outlier != -1){
        i_bool <- rep(TRUE,length(x_out))
        i_bool[i_outlier] <- FALSE
        x <- c(x, x_out[i_outlier]); y <- c(y, y_out[i_outlier]); err_y <- c(err_y, err_y_out[i_outlier])
        x_out <- x_out[i_bool]; y_out <- y_out[i_bool]; err_y_out <- err_y_out[i_bool]
      } else {
        check_outliers <- FALSE
      }
      #compute the model with intrinsic error again
      data <- model.with.intrinsic.error(x, y, err_y=err_y, deg=deg, alpha=alpha)
      model <- data[[1]]; optimum.alpha <- data[[2]]; err_0_tot <- data[[3]] 
    }
    
    y_loess    <- predict(model, x_intpol) #loess fit
    enp        <- model$trace.hat + 1      #equivalent number of parameters
    residuals  <- model$residuals
    ssd        <- sqrt(sum(residuals^2)/(length(residuals)-enp)) #sample standard deviation
    N.out      <- 0
    if (outliers_det == TRUE) { N.out <- length(x_out) }
    if (with_y_errors == FALSE) { err_0_tot <- ssd}
    
#     if (N.out/N.input.data>0.2){
#       cat(paste0('WARNING: ',N.out,' of ',N.input.data,' (',round(N.out/N.input.data*100,0),'%) of the input data were detected as outliers.\n'))
#       cat('         Check the ALR fit to consider disabling the outliers detection.\n')
#     }
    
    
    #CHARACTERISTICS OF THE ALR FIT
    
    if (verbose==TRUE){
      #input parameters
      print('### inputs ###')
      
      #deg
      description <- "order of the local polynomial    "
      par_value   <- toString(deg)
      print(paste(description, par_value))
      
      #alpha (if it is given as an input)
      if (alpha != 0.0) {
        description <- 'input alpha (AIC disabled)       '
        par_value   <- toString(round(optimum.alpha, digits=3))
        print(paste(description, par_value))
      }
      
      #outliers detection
      description <- 'outliers detection               '
      if (outliers_det == TRUE){
        print(paste(description,'TRUE'))
      }
      if (outliers_det == FALSE){
        print(paste(description,'FALSE'))
      }
      
      #average data
      description <- 'average y values with the same x '
      if (average == TRUE){
        print(paste(description,'TRUE'))
      }
      if (average == FALSE){
        print(paste(description,'FALSE'))
      }
      
      #number of input data
      description <- "number of input data             "
      par_value   <- toString(N.input.data)
      print(paste(description, par_value))
      print('')
      
      #output parameters
      print('### outputs ###')
      
      #number of outliers
      if (outliers_det == TRUE){
        description <- "number of outliers (Tukey's rule)"
        par_value   <- toString(N.out)
        print(paste(description, par_value)) 
      }
       
      #alpha
      if (alpha == 0.0) {
        description <- 'optimum alpha (using AIC)        '
        par_value   <- toString(round(optimum.alpha, digits=3))
        print(paste(description, par_value))
      }
      
      #equivalent number of parameters
      description <- 'equivalent number of parameters  '
      par_value   <- toString(round(enp,digits=1))
      print(paste(description, par_value))
      
      #sample standard deviation
      description <- 'sample standard deviation        '
      par_value   <- toString(sprintf('%.3E', ssd))
      print(paste(description, par_value))
      
      #intrinsic error
      description <- 'intrinsic error (Gaussian)       '
      par_value   <- toString(sprintf('%.3E', err_0_tot))
      print(paste(description, par_value))
      print('')
    }
  
    #TO INCLUDE THE EFFECTS OF OBSERVED ERRORS, WE HAVE TO PERFORM SIMULATIONS
    if (with_y_errors == TRUE){
      models <- loess.simulations(x, y, err_y, optimum.alpha, deg, n_sims, positive_only=FALSE)
    } else {
      models <- loess.simulations(x, y, rep(ssd, length(x)), optimum.alpha, deg, n_sims, positive_only=FALSE)
    }
        
    simulations  <- loess.data(models, n_sims, x_intpol)
    
    #compute the noise around the loess fit
    y_loess_sims <- simulations[[1]]
    res_all      <- sweep(y_loess_sims,1,y_loess)
    err_y_loess  <- numeric(0)
    for(i in 1:length(x_intpol)){
      res  <- res_all[i,]
      
      #remove possible outliers in the residuals
      data    <- tukey.fences(res, k_tukey)
      fence_l <- data[1]; fence_u <- data[2]
      res     <- res[res > fence_l]
      res     <- res[res < fence_u]
      
      ssd_i <- sqrt(sum(res^2)/(length(res)-enp))
      err_y_loess <- c(err_y_loess, ssd_i)
    }
    
    err_y_loess <- sqrt(err_y_loess^2) 
    
  } else { #linear interpolation between points
    cat('WARNING: Unable to perform a loess fit.\n')
    cat('         Performing linear interpolation between points.\n')
    if (deg==1 && N.out != 0) { 
      cat('         Consider using outliers_det=False\n')
    }
    if (deg==2) { 
      if (N.out == 0) {
        if (N.input.data > 2) { cat('         Consider using deg=1\n')}
      } else {
        cat('         Consider using outliers_det=False or deg=1\n')
      }
    }
    optimum.alpha <- 0.0
    enp           <- 0.0
    ssd           <- 0.0
    err_0_tot     <- 0.0
    N.out         <- 0
    if (verbose == TRUE) {
      print(paste("order of the local polynomial  =",deg))
      print(paste("# of input data                =",N.input.data))
      print(paste("# of outliers (Tukey's rule)   =",length(x_out)))
      print('')
    }
    
    residuals <- rep(0.0, length(x))
    model     <- approx(x, y, x_intpol)
    y_loess   <- model$y
    
    #compute loess error
    if (with_y_errors == TRUE){
      n_data <- length(x)
      y_sims <- matrix(1, nrow=n_data, ncol=n_sims)
      for (i in 1:n_data){
        y_sims[i,] <- rnorm(n_sims, y[i], err_y[i])
      }
      
      y_loess_sims <- matrix(1, nrow=length(x_intpol), ncol=n_sims)
      for (i in 1:n_sims){
        model <- approx(x, y_sims[,i], x_intpol)
        y_loess_sims[,i] <- model$y
      }
      err_y_loess  <- sqrt(rowSums(sweep(y_loess_sims,1,y_loess)^2)/(n_sims-1))
    } else {
      err_y_loess <- rep(0.0, length(x_intpol))
    }
  }
  
  interp <- function(x){
    if (min(x) < min(x_intpol) || max(x) > max(x_intpol)){
      cat(paste0('ERROR (in interp): x has values out of the valid range (i.e., [',min(x_intpol),', ',max(x_intpol),']).\n'))
    }
    else {
      y_interp      <- approx(x_intpol, y_loess, x)$y
      err_y_interp  <- approx(x_intpol, err_y_loess, x)$y
      return(list(y_interp, err_y_interp))
    }
  }
  
  Automated_Loess_Regression <- list(deg=deg, outliers_det=outliers_det, n_data=N.input.data, x=x, y=y, err_y=err_y, with_y_errors=with_y_errors, alpha=optimum.alpha, enp=enp, ssd=ssd, err_0=err_0_tot, n_outliers=N.out, n_fit=N.input.data-N.out, x_outliers=x_out, y_outliers=y_out, err_y_outliers=err_y_out, x_ALR=x_intpol, y_ALR=y_loess, err_y_ALR=err_y_loess, interp=interp, y_ALR_sims=t(y_loess_sims))
  attr(Automated_Loess_Regression, "class") <- "ALR"
  Automated_Loess_Regression
}

#
# Set of functions used by Automated_Loess_Regression
#
# References:
#
# Akaike 1974, IEEE Trans. Autom. Control, 19, 716
# Burnham & Anderson 2002, "Model Selection and Multimodel Inference". 2nd edition, Springer-Verlag, New York
# Hurvich & Tsai 1989, Biometrika, 76, 297
# Hurvich et al. 1998, J. Royal Stat. Soc. Series B, 60, 271
# Izevic et al. 2014, "Statistics, Data Mining, and Machine Learning in Astronomy", Princeton University Press, Princeton, NJ
# Tukey 1977, "Exploratory Data Analysis", Addison-Wesley, Reading


#Perform a weighted average, including a possible intrinsic error (e.g., Izevic et al. 2014)
weighted_average <- function(x, err_x, with_intrinsic_error=TRUE){

  if (with_intrinsic_error==TRUE){
    if (0 %in% err_x){cat('ERROR (in weighted_average): attempt to compute a weighted average with zero errors.\n')}
    residuals <- x - mean(x)
    ssd       <- sqrt(sum(residuals^2)/(length(residuals)-1))
    err_0s    <- seq(0.0, ssd, length.out=11)
  } else {
    err_0s    <- seq(0.0, 0.0, length.out=1)
  }
      
  m2lnL_min <- 1.e90
  for (err_0 in err_0s){
      
    Var   <- err_x^2 + err_0^2             #variance, including the intrinsic error
    w_ave <- sum(x/Var)/sum(1.0/Var)       #weighted average
    m2lnL <- sum(log(Var)+(x-w_ave)^2/Var) #-2ln(Likelihood) in the Gaussian case
    
    if (m2lnL < m2lnL_min){
      m2lnL_min <- m2lnL
      x_ave     <- w_ave
      err_x_ave <- sqrt(1.0/sum(1.0/Var))  #weighted average error
    }
  }
  return(list(x_ave, err_x_ave))
}


#Average y values with the same x value
average_y_values <- function(x, y, err_y=0){

  with_y_error <- TRUE
  if (length(err_y) == 1){
    with_y_error <- FALSE
    err_y <- y*0.0 + 1.0
  }
  x_new     <- numeric(0)
  y_new     <- numeric(0)
  err_y_new <- numeric(0)
  n         <- length(x)
  k         <- 1
  
  while (k <= n){
    ind <- 1
    xs     <- seq(x[k]    , x[k]    , length.out=1)
    ys     <- seq(y[k]    , y[k]    , length.out=1)
    err_ys <- seq(err_y[k], err_y[k], length.out=1)
    while (k+ind <= n && abs(x[k]-x[k+ind]) <= 0.0){
      xs      <- c(xs, x[k+ind])
      ys      <- c(ys, y[k+ind])
      err_ys  <- c(err_ys, err_y[k+ind])
      ind <- ind + 1
    }
  
    x_new <- c(x_new, xs[1])
    if (ind == 1) {
      y_new     <- c(y_new    , ys[1])
      err_y_new <- c(err_y_new, err_ys[1])
    } else {
      data      <- weighted_average(ys, err_ys)
      y_new     <- c(y_new    , data[[1]])
      err_y_new <- c(err_y_new, data[[2]])
    }
    k <- k + ind
  }
  if (with_y_error == TRUE){
    return(list(x_new, y_new, err_y_new))
  } else {
    return(list(x_new, y_new))
  }
}


#-2ln(Likelihood) in the Gaussian case
#In the case of observations without errors, we use the normal residuals approximation
gaussian.minus.two.log.likelihood <- function(residuals, err_y=0, err_i=0){

  if (length(err_y) == 1){
    N             <- length(residuals)
    sigma.squared <- sum(residuals^2)/N     #based on the definition of Hurvich & Tsai (1989)
    m2lnL         <- N*log(sigma.squared)+N #Burnham & Anderson 2002, page 17
  } else {
    var   <- err_y^2+err_i^2
    m2lnL <- sum(log(var)+residuals^2/var)
  }
  return(m2lnL)
}


#create a sample of alpha values to explore for loess
alpha.values.to.explore <- function(x, y, err_y=0, deg=2, family='gaussian'){

  all_alphas <- seq(1.00,  0.05, length.out=20)

  #compute the approximate order of each alpha
  orders.fit <- numeric(0)
  alphas.fit <- numeric(0)
  for (alpha in all_alphas){
    if (length(err_y) == 1)  model <- try(loess(y~x, span=alpha, degree=deg, family=family), silent=TRUE)
    if (length(err_y) >  1)  model <- try(loess(y~x, span=alpha, degree=deg, family=family, weights=1.0/err_y^2), silent=TRUE)
    if (inherits(model, "try-error")==FALSE){
      trL <- model$trace.hat  #tr(L)
      if (is.nan(trL)==FALSE){
        order      <- trL-1.0
        if (order >= 0.0 && !(order %in% orders.fit)){
          orders.fit <- c(orders.fit,order)
          alphas.fit <- c(alphas.fit,alpha)
        }
      }
    }
  }
  if (length(alphas.fit)==0 || length(alphas.fit)==1){
    alphas <- alphas.fit
  } else {
    #define the minimum and maximum order
    order.min <- ceiling(min(orders.fit)) #nearest upper integer
    order.max <- floor(max(orders.fit))   #nearest lower integer
    orders    <- seq(order.min, min(length(x)-3,order.max, 15), length.out=order.max-order.min+1)
    
    #linear interpolation bewteen points
    orders.alphas <- approx(orders.fit, alphas.fit, orders)
    alphas        <- orders.alphas$y 
  }
  return(alphas)
}


#Compute the optimum alpha value using the "an" information criterion (AIC; Akaike 1974)
loess.IC.alpha <- function(x, y, err_y=0, alphas=0, deg=2, family='gaussian'){

  n         <- length(x)    
  ICs       <- numeric(0)
  alps      <- numeric(0)
  orders    <- numeric(0)
  IC.min    <- 1.e100
  optimum.alpha <- -1.0
  for (alpha in alphas){
    if (length(err_y) == 1)  model <- try(loess(y~x, span=alpha, degree=deg, family=family), silent=TRUE)
    if (length(err_y) >  1)  model <- try(loess(y~x, span=alpha, degree=deg, family=family, weights=1.0/err_y^2), silent=TRUE)
    if (inherits(model, "try-error")==FALSE){

      trL <- model$trace.hat  #tr(L)
      if (is.nan(trL)==FALSE){
        k       <- trL + 1 #equivalent number of parameters
        k.max   <- n-1.0
        penalty <- 2.0*n*k/(n-k-1.0) #Hurvich et al. (1998)
        
        if (k<k.max){ #otherwise (n-k-1.0)<=0
          residuals <- model$residuals
          m2lnL     <- gaussian.minus.two.log.likelihood(residuals, err_y) #-2*ln(Likelihood)

          IC <- m2lnL + penalty
          if (is.finite(IC)){
            ICs    <- c(ICs,IC)
            alps   <- c(alps,alpha)
            orders <- c(orders,k-2.0)
            if (IC < IC.min){
              optimum.alpha <- alpha
              IC.min        <- IC
            }
          }
        }
      }
    }
  }
  return(optimum.alpha)
}


#Returns the optimum model given an optimum alpha
loess.optimum.model <- function(x, y, err_y=0, alpha=0.2, deg=2, family='gaussian'){
  if (length(err_y) == 1){
    optimum.model <- try(loess(y~x, span=alpha, degree=deg, family=family), silent=TRUE)
  } else {
    optimum.model <- try(loess(y~x, span=alpha, degree=deg, family=family, weights=1.0/err_y^2), silent=TRUE)
  }
  return(optimum.model)
}


#Computes n_sims loess fits to n_sims realization of the data
loess.simulations <- function(x, y, err_y, optimum.alpha, deg, n_sims, positive_only){
  family <- 'gaussian' 
  n_max  <- 3
  n_data <- length(x)
  y_sims <- matrix(1, nrow=n_data, ncol=n_max*n_sims)
  for (i in 1:n_data){
    y_sims[i,] <- rnorm(n_sims, y[i], err_y[i])
  }
  
  i_max  <- 1
  i      <- 1
  models <- list()
  
  while (i <= n_sims){
    model <- loess.optimum.model(x, y_sims[,i_max], err_y=err_y, alpha=optimum.alpha, deg=deg, family=family)
    if (inherits(model, "try-error")==FALSE){
      if((positive_only == TRUE && min(predict(model, x)) > 0.0) || positive_only == FALSE){
        models <- c(models, list(model))
        i <- i+1
      }
    }
    i_max <- i_max + 1
    if (i_max>(n_max-1)*n_sims) {print('problems in loess.simulation!')}
  }
  return(models)
}


#Return the loess fit if the n_sims data realizations
loess.data <- function(models, n_sims, x){

  n_data       <- length(x)
  y_loess_sims <- matrix(1, nrow=n_data, ncol=n_sims)
  enp_sims     <- numeric(0)
  for (i in 1:n_sims){
    model            <- models[[i]]
    y_loess_sims[,i] <- predict(model, x)
    enp_sims       <- c(enp_sims, model$trace.hat)
  }
  return(list(y_loess_sims, enp_sims))
}


#Compute the sample standard deviation and estimate the intrinsic error (err_0) through log-likelihood maximization
ssd.intrinsic.dispersion <- function(residuals, err_y=0){

  n_data       <- length(residuals)
  residuals.sq <- residuals^2
  ssd          <- sqrt(sum(residuals.sq)/n_data)
  err_0        <- 0.0
  if (length(err_y) > 1){
    errs     <- seq(0.0, ssd, length.out=11)
    lnL_max <- -1.e90
    for (err in errs){
      Var <- err_y^2+err^2
      lnL <- -0.5*sum(log(Var)+residuals.sq/Var)
      if (lnL > lnL_max){
        lnL_max <- lnL
        err_0   <- err
      }
    }
  }
  return(list(ssd, err_0))
}


#Compute the optimum model, including a possible intrinsic error
model.with.intrinsic.error <- function(x, y, err_y=0, deg=2, alpha=0.2){
  family    <- 'gaussian'
  err_0_tot <- 0.0
  compute_intrinsic_error <- TRUE #DO NOT CHANGE
  while (compute_intrinsic_error){
  
    err_y_tot <- sqrt(err_y^2 + err_0_tot^2)

    if (alpha == 0.0){
      #generate alpha values to explore
      alphas   <- alpha.values.to.explore(x, y, err_y=err_y_tot, deg=deg, family=family)
      #compute the optimum alpha value based on an information criterion
      optimum.alpha <- loess.IC.alpha(x, y, err_y=err_y_tot, alphas=alphas, deg=deg, family=family)
    } else {
      optimum.alpha <- alpha
    }
    
    model     <- loess.optimum.model(x, y, err_y=err_y_tot, alpha=optimum.alpha, deg=deg, family=family)
    residuals <- model$residuals
    
    #check whether it is necessary to include an intrinsic error
    ssd.err0 <- ssd.intrinsic.dispersion(residuals, err_y=err_y_tot)
    ssd      <- ssd.err0[[1]]; err_0 <- ssd.err0[[2]]
    if (err_0==0){
      compute_intrinsic_error <- FALSE
    }
    else {
      err_0_tot <- sqrt(err_0_tot^2 + err_0^2)
    }
  }
  return(list(model, optimum.alpha, err_0_tot))
}


#Tukey (1977) fences
tukey.fences <- function(x, k_tukey){
  Q1      <- unname(quantile(x,probs=c(0.25),na.rm=TRUE))
  Q3      <- unname(quantile(x,probs=c(0.75),na.rm=TRUE))
  IQR     <- Q3-Q1
  fence_l <- Q1-k_tukey*IQR
  fence_u <- Q3+k_tukey*IQR
  return(list(fence_l, fence_u))
}

#returns the indices (booleans) of the elements in x within the range [xmin, xmax]
indices.in.range <- function(x, xmin, xmax){
  i_bool <- rep(TRUE,length(x))
  for (i in 1:length(x)){
    if (x[i]<xmin || x[i]>xmax){ i_bool[i] <- FALSE}
  }
  return(i_bool)
}


#from an (x, y, err_y) dataset, it identifies outliers in the distribution 
identify.possible.outliers <- function(x, y, err_y=0, average=average, deg=deg, alpha=alpha, k_tukey=k_tukey, outliers_det=outliers_det){

  with_y_error <- TRUE
  if (length(err_y) == 1){ with_y_error <- FALSE }

  family   <- 'symmetric'  #more robust against outliers than 'gaussian' family
  x_out    <- numeric(0); y_out <- numeric(0); err_y_out <- numeric(0)
  outliers <- TRUE  #DO NOT CHANGE
  while (outliers){
    #weighted average of y values
    if (average == TRUE){
      data  <- average_y_values(x, y, err_y=err_y)
      x_ave <- data[[1]]; y_ave <- data[[2]]; err_y_ave <- 0
      if (with_y_error == TRUE) { err_y_ave <- data[[3]] }
    } else {
      x_ave <- x; y_ave <- y; err_y_ave <- 0
      if (with_y_error == TRUE){ err_y_ave <- err_y }
    }

    #generate alpha values to explore
    if (alpha == 0){
      alphas <- alpha.values.to.explore(x_ave, y_ave, err_y=err_y_ave, deg=deg, family=family)
      #compute the optimum alpha value based on an information criterion
      optimum.alpha <- loess.IC.alpha(x_ave, y_ave, err_y=err_y_ave, alphas=alphas, deg=deg, family=family)
    } else {
      #test whether the input alpha has sense
      n <- length(x)
      optimum.alpha <- -1.0
      if (length(err_y) == 1)  model <- try(loess(y~x, span=alpha, degree=deg, family=family), silent=TRUE)
      if (length(err_y) >  1)  model <- try(loess(y~x, span=alpha, degree=deg, family=family, weights=1.0/err_y^2), silent=TRUE)
      if (inherits(model, "try-error")==FALSE){
      
        trL <- model$trace.hat  #tr(L)
        if (is.nan(trL)==FALSE){
          k       <- trL + 1 #equivalent number of parameters
          k.max   <- n-1.0
          penalty <- 2.0*n*k/(n-k-1.0) #Hurvich et al. (1998)
          
          if (k<k.max){ #otherwise (n-k-1.0)<=0
            optimum.alpha <- alpha
          }
        }
      }
      if (optimum.alpha == -1.0){
        print('WARNING (in identify.possible.outliers): input alpha value is not valid.')
      }
      
    }

    if(optimum.alpha == -1.0){
      loess.solution <- FALSE
      outliers       <- FALSE
    } else {
      loess.solution <- TRUE
      model          <- loess.optimum.model(x_ave, y_ave, err_y=err_y_ave, alpha=optimum.alpha, deg=deg, family=family)
  
      #check if there are outliers
      if (outliers_det == FALSE){
        outliers <- FALSE
      } else {
      
        #fences defined in Tukey (1977)
        data    <- tukey.fences(model$residuals, k_tukey)
        fence_l <- data[1]; fence_u <- data[2]
        
        residuals <- y - predict(model, x)
        
        #delete outliers, one by one
        i_outlier        <- -1
        highest_residual <- 0.0
        for (i in 1:length(residuals)){
          if(residuals[i] < fence_l || residuals[i] > fence_u){
            if (x[i] != min(x) && x[i] != max(x)){  #we MUST NOT delete the extreme x values
              if(abs(residuals[i]) > highest_residual){
                i_outlier        <- i
                highest_residual <- abs(residuals[i])
              }
            }
          }
        }
      
        if(i_outlier != -1){
          i_bool <- rep(TRUE,length(x))
          i_bool[i_outlier] <- FALSE
          x_out <- c(x_out, x[i_outlier]); y_out <- c(y_out, y[i_outlier])
          x     <- x[i_bool]; y <- y[i_bool]
          if (with_y_error == TRUE) { 
            err_y_out <- c(err_y_out, err_y[i_outlier]) 
            err_y <- err_y[i_bool] 
          }
        } else {
          outliers <- FALSE
        }
      }
    }
  }
  
  #weighted average for y values
  if (average == TRUE){
    data <- average_y_values(x, y, err_y=err_y)
    x    <- data[[1]]; y <- data[[2]]; err_y <- 0
    if (with_y_error == TRUE) { err_y <- data[[3]] }
  }
  return(list(x, y, err_y, x_out, y_out, err_y_out, loess.solution))
}
