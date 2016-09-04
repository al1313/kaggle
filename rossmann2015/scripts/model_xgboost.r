###########################################################################################################
# xgboost linear regression model
###########################################################################################################

#==========================================================================================================
# load libraries and set working directory
#==========================================================================================================

require(xgboost)
require(data.table)
#require(methods)
#require(magrittr))

setwd("E:/VboxShare/rossmann/")

#==========================================================================================================
# Read in file
#==========================================================================================================

train <- fread("./input/train.csv", stringsAsFactors = T)
test <- fread("./input/test.csv", stringsAsFactors = T)
store <- fread("./input/store.csv", stringsAsFactors = T)

#==========================================================================================================
# Pre process
#==========================================================================================================

# Take out stores that have no sales
train <- train[Sales> 0,] 

preprocess <- function(input){
	#merge store information
	temp <- merge(input,store, by="Store")
	# turn date into date format
	temp[,Date:=as.Date(Date)] 

	# extract month year
	temp[,day:=as.integer(format(Date, "%d"))]
	temp[,month:=as.integer(format(Date, "%m"))]
	temp[,year:=as.integer(format(Date, "%y"))]
	temp[,week:=as.integer(format(Date, "%W"))]

	# Add promo2 duration week counter
	temp $Promo2SinceDate <- as.Date(paste(temp$Promo2SinceYear, temp$Promo2SinceWeek, 1, sep=" "), format = "%Y %U %u")
	temp[, c("temp1", "temp2", "temp3", "temp4") := tstrsplit(PromoInterval, ",", fixed = TRUE)]
	temp[,"Promo2Interval_Wk1" := as.integer(format(as.Date(paste(year, temp1, 1, sep=" "), format = "%y %b %d"),"%W"))]
	temp[,"Promo2Interval_Wk2" := as.integer(format(as.Date(paste(year, temp2, 1, sep=" "), format = "%y %b %d"),"%W"))]
	temp[,"Promo2Interval_Wk3" := as.integer(format(as.Date(paste(year, temp3, 1, sep=" "), format = "%y %b %d"),"%W"))]
	temp[,"Promo2Interval_Wk4" := as.integer(format(as.Date(paste(year, temp4, 1, sep=" "), format = "%y %b %d"),"%W"))]

	temp[,"Promo2Duration_Wk1" := week  - Promo2Interval_Wk1 + 1]
	temp[,"Promo2Duration_Wk2" := week  - Promo2Interval_Wk2 + 1]
	temp[,"Promo2Duration_Wk3" := week  - Promo2Interval_Wk3 + 1]
	temp[,"Promo2Duration_Wk4" := week  - Promo2Interval_Wk4 + 1]
	temp[,"Promo2Duration_Wk" := ifelse(Promo2SinceDate >= Date, min(Promo2Duration_Wk1,Promo2Duration_Wk2,Promo2Duration_Wk3,Promo2Duration_Wk4),0), by="Date"]
	temp[,"Promo2Duration_Wk" := min(Promo2Duration_Wk1,Promo2Duration_Wk2,Promo2Duration_Wk3,Promo2Duration_Wk4), by="Date"]

	temp[, c("temp1", "temp2", "temp3", "temp4", "Promo2Interval_Wk1", "Promo2Interval_Wk2", "Promo2Interval_Wk3", "Promo2Interval_Wk4",
	"Promo2Duration_Wk1", "Promo2Duration_Wk2", "Promo2Duration_Wk3", "Promo2Duration_Wk4", "Promo2SinceDate") := NULL]

	# Deal with NA
	temp[is.na(temp)]   <- 0

	return(temp)
}

train<-preprocess(train)
train[,logSales:=log1p(Sales)]
test<-preprocess(test)

col_delete <- c("Date","Sales","logSales","Customers", "StateHoliday", "week","PromoInterval")

features<-colnames(train)[!(colnames(train) %in% col_delete)]
test_col_delete <-colnames(test)[!(colnames(test) %in% features)]

for (f in features) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

train_only <- train[Date < as.Date("2015-4-1"),]
holdout <- train[Date >= as.Date("2015-4-1"),]

train_features <-data.table(train)
train_features[,(col_delete) := NULL, with=F]

train_only_features <-data.table(train_only)
train_only_features[,(col_delete) := NULL, with=F]

holdout_features <-data.table(holdout)
holdout_features[,(col_delete) := NULL, with=F]

test_features <- data.table(test)
test_features[,(test_col_delete) := NULL, with=F]

RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

set.seed(25)
h<-sample(nrow(train_features),10000)

dtrain<-xgb.DMatrix(data=data.matrix(train_features[-h,]),label=train$logSales[-h])
dval<-xgb.DMatrix(data=data.matrix(train_features[h,]),label=train$logSales[h])

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.02, #0.25,
                max_depth           = 10, #8
                subsample           = 0.9, # 0.7
                colsample_bytree    = 0.7, # 0.7
                nthread = 4,
		    set.seed = 25
                # alpha = 0.0001, 
                # lambda = 1
)

set.seed(18)

train_xgb <- xgb.train(params              = param, 
                    data                = dtrain, 
                    nrounds             = 5000, #2000
                    verbose             = 1,
                    early.stop.round    = 100, #50
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)

pred <- exp(predict(train_xgb, data.matrix(test_features))) -1
submission <- data.frame(Id=test$Id, Sales=pred)
write.csv(submission, "./submission/xgboost.csv", row.names=F)


set.seed(25)
h<-sample(nrow(train_only_features),10000)

dtrain<-xgb.DMatrix(data=data.matrix(train_only_features[-h,]),label=train_only$logSales[-h])
dval<-xgb.DMatrix(data=data.matrix(train_only_features[h,]),label=train_only$logSales[h])

watchlist<-list(val=dval,train=dtrain)

set.seed(18)
train_holdout_xgb <- xgb.train( params              = param, 
                    	data                = dtrain, 
                    	nrounds             = 2000, #2000
                    	verbose             = 1,
                    	early.stop.round    = 100, #50
                    	watchlist           = watchlist,
                    	maximize            = FALSE,
                    	feval=RMPSE,
				nfold=5,
)

pred <- exp(predict(train_holdout_xgb, data.matrix(holdout_features))) -1
submission <- data.frame(Store=holdout$Store,Date=holdout$Date,Sales=pred)
write.csv(submission, "./holdout_pred/xgboost.csv", row.names=F)

########################################################

submission V3
[1055]  val-RMPSE:0.116813772833176     train-RMPSE:0.090648618463039

Using 10000 validation set - submission V4
[1068]  val-RMPSE:0.109925012543011     train-RMPSE:0.0946679841350541

Bad result - don't add sales mean and sd to data too much overfit - submission V5
[827]   val-RMPSE:0.112054738779926     train-RMPSE:0.0840330672177471

Drop sales mean and SD - submission V6
[1122]  val-RMPSE:0.109760155021841     train-RMPSE:0.0825029421482006

Submission V7
Drop "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"
next step is probably to properly deal with promo2interval
[1282]  val-RMPSE:0.107186819427557     train-RMPSE:0.0838527985247638

Not a submission - try adding week number but no improvement in validation
val-RMPSE:0.108667256549275     train-RMPSE:0.0831889533552838

Submission V9
[1468]  val-RMPSE:0.102047855382712     train-RMPSE:0.0808548516070833

v10 New param - best score
[2999]  val-RMPSE:0.0973653163482132    train-RMPSE:0.0950686355187876

V11 new param with more trees
[4999]  val-RMPSE:0.094698324811906     train-RMPSE:0.0778234116886031
