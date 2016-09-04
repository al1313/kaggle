###########################################################################################################
# random forest using H2O
###########################################################################################################

#==========================================================================================================
# load libraries and set working directory
#==========================================================================================================

require(data.table)
require(h2o)
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
train_features[,c("type","v3") := NULL, with=F]
# train_features[,drop_list := NULL, with=F]

test_features <- combined_prep[type=="test",]
test_features[,c("type","v3") := NULL, with=F]
# test_features[,drop_list := NULL, with=F]


#==========================================================================================================
# RF
#==========================================================================================================

train_features$target <- as.factor(target$target)

h2o.init(nthreads=-1, max_mem_size='8G')
trainHex<-as.h2o(train_features)

# Fit model using entire train data

train_rf<- h2o.randomForest(x=features, y="target", training_frame=trainHex,
                            # model parameters
                            ntrees = 2000,
                            max_depth = 20,
                            sample_rate= 0.5,
                            nbins = 50
                            # stopping_metric = "logloss",
                            # stopping_rounds = 50
)

#==========================================================================================================
# Create submission file
#==========================================================================================================

testHex<-as.h2o(test_features)
pred <- predict(train_rf, testHex)
submission <- data.frame(ID=test$ID, PredictedProb=as.data.frame(pred)[,3])
write.csv(submission, "./submission/rf_v1.csv", row.names=F)
