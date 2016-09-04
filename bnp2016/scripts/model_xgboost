###########################################################################################################
# xgboost linear regression model
###########################################################################################################

#==========================================================================================================
# load libraries and set working directory
#==========================================================================================================

require(xgboost)
require(data.table)
require(magrittr)
require(CORElearn)
require(FeatureHashing)
setwd("E:/Models/bnp/")

source(file="./script/utils.r")

#==========================================================================================================
# Read in file
#==========================================================================================================

train <- fread("./input/train.csv", stringsAsFactors = T)
test <- fread("./input/test.csv", stringsAsFactors = T)

#==========================================================================================================
# Pre process
#==========================================================================================================

# Extract ID and target from train data
target <- train[,c('ID','target'),with=FALSE]

# delete target from train data
train[,'target' := NULL, with=F]

# combine train and test data together for preprocessing
train[,type:="train"]
test[,type:="test"]
combined <- rbind(train,test)

preprocess <- function(input){
  
  # place to add other preprocesses 
  
  # add mean to each variable
  # num.col <- sapply(input, is.numeric)
  # num.data <- input[,num.col,with=FALSE]
  # 
  # des.stats.mean <- t(sapply(num.data, mean, na.rm=T))
  # colnames(des.stats.mean) <- paste(colnames(des.stats.mean), "mean", sep = "_")
  
  #input <- cbind(input,des.stats.mean)
  
  # Deal with NA
  input[is.na(input)]   <- -99
  
  return(input)
}

#combined<-preprocess(combined)

##### Deal with high level categorical factor v22 (23,420 levels)

## Encode as 4 variables each taking values 1-26
v22a <- match(sapply(strsplit(as.character(combined$v22), ""), "[", 1),LETTERS)
v22b <- match(sapply(strsplit(as.character(combined$v22), ""), "[", 2),LETTERS)
v22c <- match(sapply(strsplit(as.character(combined$v22), ""), "[", 3),LETTERS)
v22d <- match(sapply(strsplit(as.character(combined$v22), ""), "[", 4),LETTERS)

## Encode as 4 variables, but keep as letter
# v22a <- sapply(strsplit(as.character(combined$v22), ""), "[", 1)
# v22b <- sapply(strsplit(as.character(combined$v22), ""), "[", 2)
# v22c <- sapply(strsplit(as.character(combined$v22), ""), "[", 3)
# v22d <- sapply(strsplit(as.character(combined$v22), ""), "[", 4)

# Did not improve model
# v56a <- match(sapply(strsplit(as.character(combined$v56), ""), "[", 1),LETTERS)
# v56b <- match(sapply(strsplit(as.character(combined$v56), ""), "[", 2),LETTERS)
# 
# v125a <- match(sapply(strsplit(as.character(combined$v125), ""), "[", 1),LETTERS)
# v125b <- match(sapply(strsplit(as.character(combined$v125), ""), "[", 2),LETTERS)

var_split <- data.frame(v22a=v22a,v22b=v22b,v22c=v22c,v22d=v22d)
var_split[is.na(var_split)]   <- -99

# map factors with multiple letters to numbers (A-Z, AA-AZ, etc.)  
az_to_int <- function(az) {
  xx <- strsplit(tolower(az), "")[[1]]
  pos <- match(xx, letters[(1:26)])
  result <- sum( pos* 26^rev(seq_along(xx)-1))
  return(result)
}

combined$v22<-sapply(combined$v22, az_to_int)
combined$v56<-sapply(combined$v56, az_to_int)
combined$v125<-sapply(combined$v125, az_to_int)

# turn factor variables into 0,1 indicator variables
factor_list <- varlist(combined,type="factor",exclude=c(
  # mapped to numbers from above
  "v22", "v56", "v125",
  # these are variables that are perfect mappings of others
  "v31","v71", "v107"
))
mat_formula <- as.formula(paste("~",paste(factor_list, collapse="+"),"-1"))
factor_dummy <- model.matrix(mat_formula, data=combined)

# From step-wise VIF selection 
vif_keep <- c("v12",  "v16",  "v21",  "v23",  "v28",  "v39",  "v50",  "v58",  "v62",  "v72",  "v81",  "v82",  "v89",  "v109",
              "v114", "v119", "v124", "v127", "v129", "v131")

# Try adding some interaction terms
options(na.action='na.pass')
int_list <- c("v50","v114")
mat_formula <- as.formula(paste("~(",paste(int_list, collapse="+"),")^2-1-", paste(int_list, collapse="-") ))
int_set <- model.matrix(mat_formula, data=combined)

combined_prep<- cbind(combined[,c(vif_keep,"type","v22", "v56", "v125",factor_list),with=FALSE],factor_dummy, var_split,int_set)
# for (f in factor_list){
#   combined_prep[[f]] <- match(combined_prep[[f]],LETTERS)
# }
combined_prep[is.na(combined_prep)]   <- -99

col_delete <- c("ID", "type")
features<-colnames(combined_prep)[!(colnames(combined_prep) %in% col_delete)]

# Turn any factor variables into levels in numbers (for xbg)
for (f in features) {
  if (class(combined_prep[[f]])=="factor") {
    levels <- unique(combined_prep[[f]])
    combined_prep[[f]] <- as.integer(factor(combined_prep[[f]], levels=levels))
  }
}

train_features <- combined_prep[type=="train",]
train_features[,"type" := NULL, with=F]
# train_features[,drop_list := NULL, with=F]

test_features <- combined_prep[type=="test",]
test_features[,"type" := NULL, with=F]
# test_features[,drop_list := NULL, with=F]

#==========================================================================================================
# xgb
#==========================================================================================================

set.seed(25)
h<-sample(nrow(train_features),50000)

dstack1<-xgb.DMatrix(data=data.matrix(train_features[-h,]),label=target$target[-h])
dstack2<-xgb.DMatrix(data=data.matrix(train_features[h,]),label=target$target[h])
dfull<-xgb.DMatrix(data=data.matrix(train_features),label=target$target)

# watchlist<-list(val=dval,train=dtrain)

param <- list(  
  objective= "binary:logistic", 
  eval_metric = "logloss",
  booster = "gbtree",
  eta = 0.01, #0.01, 0.02,
  max_depth = 11, #11, 10
  subsample = 0.96, # 0.96, 0.9
  colsample_bytree = 0.4, # 0.4, 0.7
  #nthread = 4,
  set.seed = 25,
  #num_parallel_tree   =1,
  min_child_weight    =1
  
)

set.seed(18)

train_xgb_stack1 <- xgb.train(
  params = param, 
  data = dstack1, 
  nrounds = 1900, #2000
  # verbose = 1,
  # print.every.n = 10,
  # early.stop.round = 100, #50
  # watchlist = watchlist,
  maximize = FALSE
)

train_xgb_stack2 <- xgb.train(
  params = param, 
  data = dstack2, 
  nrounds = 1900, #2000
  # verbose = 1,
  # print.every.n = 10,
  # early.stop.round = 100, #50
  # watchlist = watchlist,
  maximize = FALSE
)

full_xgb <- xgb.train(
  params = param, 
  data = dfull, 
  nrounds = 1900, #This is from running 10 fold CV to finding the optimal number of trees
  verbose = 1,
  print.every.n = 10,
  maximize = FALSE
)

pred_stack1 <- predict(train_xgb_stack2, dstack1)
pred_stack2 <- predict(train_xgb_stack1, dstack2)

stack1_scored <- data.frame(ID = target$ID[-h], target=target$target[-h], PredictedProb=pred_stack1)
stack2_scored <- data.frame(ID = target$ID[h], target=target$target[h], PredictedProb=pred_stack2)

logloss_calc<-function(actuals,predictions){return(  -mean(actuals*log(predictions) + (1-actuals)*log(1-predictions))  )}
logloss_calc(stack1_scored[,2],stack1_scored[,3])
logloss_calc(stack2_scored[,2],stack2_scored[,3])

# reliability.plot(val_scored[,1],val_scored[,2])
stack_scored = rbind(stack1_scored,stack2_scored)
write.csv(stack_scored, "./ensemble/level_0/train_scored/xgb.csv", row.names=F)

# Get importance plot
# names <- dimnames(train_features)[[2]]
# 
# importance_matrix <- xgb.importance(names, model = train_xgb)
# xgb.plot.importance(importance_matrix[1:20,])
# 
# write.csv(importance_matrix, "./importance1.csv", row.names=F)

#==========================================================================================================
# Create submission file
#==========================================================================================================

pred <- predict(full_xgb, data.matrix(test_features))
submission <- data.frame(ID=test$ID, PredictedProb=pred)
write.csv(submission, "./ensemble/level_0/test_scored/xgb.csv", row.names=F)

###########################################################
# ver0 - val-logloss:0.457271 train-logloss:0.325930
# ver1 - val-logloss:0.455925 train-logloss:0.270132
# ver2 - val-logloss:0.454952	train-logloss:0.303999
# ver3 - val-logloss:0.454220	train-logloss:0.266167
#	ver5 - val-logloss:0.453739	train-logloss:0.292085
# ver6 - val-logloss:0.452137	train-logloss:0.252591
# ver7 - val-logloss:0.451972	train-logloss:0.246354
# ver9 - val-logloss:0.451775	train-logloss:0.253311
# ver10 -val-logloss:0.452393	train-logloss:0.256893
# ver12 - val-logloss:0.452143	train-logloss:0.259822
