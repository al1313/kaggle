###########################################################################################################
# Random forest  with H2o
###########################################################################################################

#==========================================================================================================
# load libraries and set working directory
#==========================================================================================================

library(h2o)
library(data.table)

setwd("E:/VboxShare/rossmann")

#==========================================================================================================
# Read in file
#==========================================================================================================

train <- fread("./input/train.csv", stringsAsFactors = T)
test <- fread("./input/test.csv", stringsAsFactors = T)
store <- fread("./input/store.csv", stringsAsFactors = T)

#==========================================================================================================
# Preprocess
#==========================================================================================================

# Take out stores that have no sales
train <- train[Sales> 0,] 

preprocess <- function(input){
	#merge store information
	temp <- merge(input,store, by="Store")
	
	# turn date into date format
	temp[,Date:=as.Date(Date)] 

	# extract month year
	temp[,month:=as.integer(format(Date, "%m"))]
	temp[,year:=as.integer(format(Date, "%y"))]

	#turn store into factor
	temp[,Store:=as.factor(as.numeric(Store))]

	return(temp)
}

train<-preprocess(train)
test<-preprocess(test)

train[,logSales:=log1p(Sales)]
train_only <- train[Date < as.Date("2015-4-1"),]
holdout <- train[Date >= as.Date("2015-4-1"),]

#==========================================================================================================
# Define feature list
#==========================================================================================================

# features are all columns excluding those in this list

features<-colnames(train)[!(colnames(train) %in% c("Id","Date","Sales","logSales","Customers"))]

# Evaluation metric - Root Mean Square Percentage Error
rmspe<-function(actuals,predictions){return(mean(((actuals[actuals>0]-predictions[actuals>0])/actuals[actuals>0])^2)^0.5)}

#==========================================================================================================
# Use H2O's random forest (requires Java JDK)
#==========================================================================================================

h2o.init(nthreads=-1, max_mem_size='6G')

# Load data into cluster from R
trainHex<-as.h2o(train)
train_onlyHex<-as.h2o(train_only)
holdoutHex<-as.h2o(holdout)
testHex <- as.h2o(test)

# Fit model using entire train data
train_rf<- h2o.randomForest(x=features, y="logSales", training_frame=trainHex,
	# model parameters
	ntrees = 100,
    max_depth = 30,
    nbins_cats = 1115, ## allow it to fit store ID
	)

judge<-as.data.frame(cbind(as.data.frame(trainHex$Sales),as.data.frame(h2o.predict(train_rf,trainHex))[,1]))
rmspe(judge[,1],expm1(judge[,2]))

# Score and export test predictions
predictions <- as.data.frame(h2o.predict(train_rf,testHex))
pred <- expm1(predictions[,1])
submission <- data.frame(Id=test$Id,Sales=pred)
write.csv(submission, './submission/rf.csv', row.names=F)

# Fit model using sampled train data (for blending)
train_only_rf<- h2o.randomForest(x=features, y="logSales", training_frame=train_onlyHex,
	# model parameters
	ntrees = 100,
	max_depth = 30,
	nbins_cats = 1115, ## allow it to fit store ID
	)	

# Score and export holdout predictions

predictions <- as.data.frame(h2o.predict(train_only_rf,holdoutHex))
pred <- expm1(predictions[,1])
submission <- data.frame(Store=holdout$Store,Date=holdout$Date,Sales=pred)
write.csv(submission, './holdout_pred/rf.csv', row.names=F)

#==========================================================================================================
# Modelling log
#==========================================================================================================

0 - Original Model - 0.1305143
1 - Add sales mean and SD - 0.1286997
2 - Added daysOpened - 0.1343778
MSE:  0.01662112
R2 :  0.9034348
Mean Residual Deviance :  0.01662112

3 - Replaced daysOpened with monthsOpened - 0.131901
MSE:  0.01637788
R2 :  0.904848
Mean Residual Deviance :  0.01637788

4 - Removed year, competitionSinceMonth and competitionSinceYear - 0.1288798
MSE: 0.01589676
R2 :  0.9076432
Mean Residual Deviance :  0.01589676

5 - add refurbished indicator - 0.1292041
MSE:  0.0158802
R2 :  0.9077395
Mean Residual Deviance :  0.0158802

6 - take out promosinceweek and promosinceyear - 0.1292188


