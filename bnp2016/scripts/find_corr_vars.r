###########################################################################################################
# remove correlated variables
###########################################################################################################

#==========================================================================================================
# load libraries and set working directory
#==========================================================================================================

require(data.table)

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

# combine train and test data together
train[,type:="train"]
test[,type:="test"]
combined <- rbind(train,test)

# get names of numeric variables
num_list <- c(varlist(combined,type="numeric",exclude=c("ID","target")))

combined[is.na(combined)]   <- -1
num_vars <-combined[,num_list,with=FALSE]

# run step-wise VIF function
vif_keep <- vif_func(in_frame=num_vars,thresh=5,trace=T)

#vif_keep <- c("v12",  "v16",  "v21",  "v23",  "v28",  "v39",  "v50",  "v58",  "v62",  "v72",  "v81",  "v82",  "v89",  "v109",
#"v114", "v119", "v124", "v127", "v129", "v131")

num_vars_filtered <- num_vars[,vif_keep,with=FALSE]

kappa(num_vars)
kappa(num_vars_filtered)

# get names of factor variables
factor_list <- c(varlist(combined,type="factor",exclude=c("v22")))
factor_vars <-combined[,factor_list,with=FALSE]
mat_formula <- as.formula(paste("~",paste(factor_list, collapse="+"),"-1"))
factor_dummy <- model.matrix(mat_formula, data=factor_vars)

# run step-wise VIF function
vif_factor_keep <- vif_func(in_frame=factor_vars,thresh=5,trace=T)

######################################################################
# numeric list
# "v12"  "v16"  "v21"  "v23"  "v28"  "v39"  "v50"  "v58"  "v62"  "v72"  "v81"  "v82"  "v89"  "v109" "v114" "v119" "v124" "v127" "v129" "v131"
