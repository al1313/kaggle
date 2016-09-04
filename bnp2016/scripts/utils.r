###########################################################################################################
# utility functions
###########################################################################################################

#==========================================================================================================
# reliability plot
#==========================================================================================================

reliability.plot <- function(obs, pred, bins=10, scale=T) {
  #  Plots a reliability chart and histogram of a set of predicitons from a classifier
  #
  # Args:
  #   obs: Vector of true labels. Should be binary (0 or 1)
  #   pred: Vector of predictions of each observation from the classifier. Should be real
  #       number
  #   bins: The number of bins to use in the reliability plot
  #   scale: Scale the pred to be between 0 and 1 before creating reliability plot
  require(plyr)
  library(Hmisc)
  
  min.pred <- min(pred)
  max.pred <- max(pred)
  min.max.diff <- max.pred - min.pred
  
  if (scale) {
    pred <- (pred - min.pred) / min.max.diff 
  }
  
  bin.pred <- cut(pred, bins)
  
  k <- ldply(levels(bin.pred), function(x) {
    idx <- x == bin.pred
    c(sum(obs[idx]) / length(obs[idx]), mean(pred[idx]))
  })
  
  is.nan.idx <- !is.nan(k$V2)
  k <- k[is.nan.idx,]  
  plot(k$V2, k$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="red", type="o", main="Reliability Plot")
  lines(c(0,1),c(0,1), col="grey")
  subplot(hist(pred, xlab="", ylab="", main="", xlim=c(0,1), col="blue"), grconvertX(c(.8, 1), "npc"), grconvertY(c(0.08, .25), "npc"))
}

#==========================================================================================================
# Using variance inflation factor to take out correlated variables in a step-wise process
#==========================================================================================================

vif_func<-function(in_frame,thresh=10,trace=T,...){
  
  require(fmsb)
  
  if(class(in_frame) != 'data.frame') in_frame<-data.frame(in_frame)
  
  #get initial vif value for all comparisons of variables
  vif_init<-NULL
  var_names <- names(in_frame)
  for(val in var_names){
    regressors <- var_names[-which(var_names == val)]
    form <- paste(regressors, collapse = '+')
    form_in <- formula(paste(val, '~', form))
    vif_init<-rbind(vif_init, c(val, VIF(lm(form_in, data = in_frame, ...))))
  }
  vif_max<-max(as.numeric(vif_init[,2]))
  
  if(vif_max < thresh){
    if(trace==T){ #print output of each iteration
      prmatrix(vif_init,collab=c('var','vif'),rowlab=rep('',nrow(vif_init)),quote=F)
      cat('\n')
      cat(paste('All variables have VIF < ', thresh,', max VIF ',round(vif_max,2), sep=''),'\n\n')
    }
    return(var_names)
  }
  else{
    in_dat<-in_frame
    
    #backwards selection of explanatory variables, stops when all VIF values are below 'thresh'
    while(vif_max >= thresh){
      vif_vals<-NULL
      var_names <- names(in_dat)
      
      for(val in var_names){
        regressors <- var_names[-which(var_names == val)]
        form <- paste(regressors, collapse = '+')
        form_in <- formula(paste(val, '~', form))
        vif_add<-VIF(lm(form_in, data = in_dat, ...))
        vif_vals<-rbind(vif_vals,c(val,vif_add))
      }
      max_row<-which(vif_vals[,2] == max(as.numeric(vif_vals[,2])))[1]
      
      vif_max<-as.numeric(vif_vals[max_row,2])
      
      if(vif_max<thresh) break
      
      if(trace==T){ #print output of each iteration
        prmatrix(vif_vals,collab=c('var','vif'),rowlab=rep('',nrow(vif_vals)),quote=F)
        cat('\n')
        cat('removed: ',vif_vals[max_row,1],vif_max,'\n\n')
        flush.console()
      }
      in_dat<-in_dat[,!names(in_dat) %in% vif_vals[max_row,1]]
    }
    return(names(in_dat))
  }
}

#==========================================================================================================
# Get a list of variable names for a selected type
#==========================================================================================================

varlist <- function (df=NULL,type=c("numeric","factor","character"), pattern="", exclude=NULL) {
  vars <- character(0)
  if (any(type %in% "numeric")) {
    vars <- c(vars,names(df)[sapply(df,is.numeric)])
  }
  if (any(type %in% "factor")) {
    vars <- c(vars,names(df)[sapply(df,is.factor)])
  }  
  if (any(type %in% "character")) {
    vars <- c(vars,names(df)[sapply(df,is.character)])
  }  
  vars[(!vars %in% exclude) & grepl(vars,pattern=pattern)]
}
