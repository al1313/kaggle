###########################################################################################################
# Time series models
###########################################################################################################

#==========================================================================================================
# load libraries and set working directory
#==========================================================================================================

library(dplyr)
library(plyr)
library(data.table)
library(forecast)

setwd("E:/VboxShare/rossmann")

#==========================================================================================================
# Read in file
#==========================================================================================================

train <- fread("./input/train.csv", stringsAsFactors = T)
test <- fread("./input/test.csv", stringsAsFactors = T)
store <- fread("./input/store.csv", stringsAsFactors = T)

#==========================================================================================================
# Pre process
#==========================================================================================================

preprocess <- function(input){
	#merge store information
	#temp <- merge(input,store, by="Store")
	temp <- input
	# turn date into date format
	temp[,Date:=as.Date(Date)] 

	# extract month year
	#temp[,month:=as.integer(format(Date, "%m"))]
	#temp[,year:=as.integer(format(Date, "%Y"))]

	#turn store into factor
	#temp[,Store:=as.factor(as.numeric(Store))]

	return(temp)
}

train<-preprocess(train)
train[,logSales:=log1p(Sales)]
train <- train[order(Store,Date)]

test<-preprocess(test)
test <- test[order(Store,Date)]
test <- test[,model:=0]

#==========================================================================================================
# Fixing stores with missing data 
# These stores have a 6 month gap due to renovation
#==========================================================================================================

# Identifying missing stores
all_stores <- unique(train$Store)
stores_reporting <- train$Store[train$Date == as.Date("2014-7-1")]
missing_stores <- all_stores[!(all_stores %in% stores_reporting)]

# Create empty dataframe for inputation of missing data
gap <- seq(as.Date("2014-7-1"),as.Date("2014-12-31"),by="day")
n_missing <- length(gap)*length(missing_stores)
missing_df <- data.frame(Store = integer(n_missing),
                         DayOfWeek = integer(n_missing),
                         Date = rep(gap,length(missing_stores)),
                         Sales = integer(n_missing),
                         Customers = integer(n_missing),
                         Open = integer(n_missing),
                         Promo = integer(n_missing),
                         StateHoliday = character(n_missing),
                         SchoolHoliday = integer(n_missing),
                         logSales = numeric(n_missing),
                         stringsAsFactors=FALSE)

# Using majority rules to determine factor variables - state_holiday, promo, school holiday, open etc.

for (date in gap) {
  missing_df$Store[missing_df$Date == date] <- missing_stores
  
  day_of_week <- unique(train$DayOfWeek[train$Date == date])
  missing_df$DayOfWeek[missing_df$Date == date] <- rep(day_of_week, length(missing_stores))
  
  missing_df$Sales[missing_df$Date == date] <- rep(NA, length(missing_stores))

  missing_df$Customers[missing_df$Date == date] <- rep(NA, length(missing_stores))
  
  open <- as.numeric(names(which.max(table(train$Open[train$Date == date]))))
  missing_df$Open[missing_df$Date == date] <- rep(open, length(missing_stores))
  
  promo <- as.numeric(names(which.max(table(train$Promo[train$Date == date]))))
  missing_df$Promo[missing_df$Date == date] <- rep(promo, length(missing_stores))

  state_holiday <- names(which.max(table(train$StateHoliday[train$Date == date])))
  missing_df$StateHoliday[missing_df$Date == date] <- rep(state_holiday, length(missing_stores))

  school_holiday <- as.numeric(names(which.max(table(train$SchoolHoliday[train$Date == date]))))
  missing_df$SchoolHoliday[missing_df$Date == date] <- rep(school_holiday, length(missing_stores))
  
  missing_df$logSales[missing_df$Date == date] <- rep(NA, length(missing_stores))

}

head(missing_df)

# Combining imputation data frame with original train

train_filled_gap <- rbind(train,missing_df)
train_filled_gap <- train_filled_gap[order(train_filled_gap$Date),]

train_filled_gap <- as.data.table( train_filled_gap %>% 
                      group_by(Store, DayOfWeek, Open, Promo) %>%
                      mutate(Sales = as.integer(ifelse(is.na(Sales), 
                                                       ifelse(Open == 0, 
                                                              0,
                                                              median(Sales, na.rm=T)), 
                                                       Sales))) %>%
                      mutate(Customers = as.integer(ifelse(is.na(Customers),
                                                           ifelse(Open == 0, 
                                                              0,
                                                              median(Customers, na.rm=T)),
                                                           Customers))) %>%
                      mutate(logSales = ifelse(is.na(logSales),
                                               ifelse(Open == 0,
                                                      0,
                                                      mean(logSales, na.rm=T)), 
                                               logSales)) )

train_filled_gap <- as.data.table(train_filled_gap)

train_filled_gap[,logSales:=log1p(Sales)]
train_filled_gap[,model:= 1 * 1]

anything_missed <- subset(train_filled_gap, is.na(Sales) | is.na(logSales))
anything_missed

head(combined)

combined <- as.data.table(rbind.fill(train_filled_gap,test))
combined<-combined[Open==1,]

train_only <- combined[model ==1,]
train_only[,model := ifelse(Date >= as.Date("2015-4-1"),0,1)]
train_only <- train_only[order(Store,Date)]

#==========================================================================================================
# Auto.arima with xreg
#==========================================================================================================

ts_arima_fit = function(x) {
	# Split train and test data
	temp_m <- subset(x, model == 1)
	temp_v <- subset(x, model == 0)

	# Set time series for input store data x
	logSales<- ts(temp_m$logSales, frequency = 7)
	
	# Create regressor matrix
	temp_xreg <- cbind(DayOfWeek =temp_m$DayOfWeek, SchoolHoliday =temp_m$SchoolHoliday , Promo =temp_m$Promo)
	temp_newxreg <- cbind(DayOfWeek =temp_v$DayOfWeek, SchoolHoliday =temp_v$SchoolHoliday , Promo =temp_v$Promo)

	# Remove intercept
	temp_xreg <- temp_xreg[,-1]
	temp_newxreg <- temp_newxreg[,-1]

	# Calculate number of periods to forecast
	horizon <- nrow(temp_v)

	#Some stores are on train set but not on test set
	if (horizon > 0){
		temp_fc <- stlf(logSales, 
			h=horizon, 
			s.window='periodic', 
			method='arima',
			ic='bic',
			xreg = temp_xreg,
			newxreg = temp_newxreg,
			approximation=FALSE,
			trace=FALSE 
		)
		logPred <- as.numeric(temp_fc$mean * ifelse( is.na(temp_v$Open), 0, temp_v$Open) )
		predictions <- data.frame(pred = expm1(logPred), Store=temp_v$Store, Id = temp_v$Id, Date=temp_v$Date)
		return(predictions)
	}
}

ts_arima <- ddply(combined, .(Store), ts_arima_fit)
foo <- merge(x=test, y=ts_arima, by="Id",all.x=TRUE)
submission <- data.frame(Id=foo$Id,Sales=foo$pred)
submission [is.na(submission )]   <- 0
write.csv(submission, './submission/ts_arima.csv', row.names=F)

ts_arima_holdout <- ddply(train_only, .(Store), ts_arima_fit)
submission <- data.frame(Store=ts_arima_holdout$Store,Date=ts_arima_holdout$Date,Sales=ts_arima_holdout$pred)
write.csv(submission, './holdout_pred/ts_arima.csv', row.names=F)


#==========================================================================================================
# STLF
#==========================================================================================================

ts_stlf_fit = function(x) {
	# Split train and test data
	temp_m <- subset(x, model == 1)
	temp_v <- subset(x, model == 0)

	# Set time series for input store data x
	logSales<- ts(temp_m$logSales, frequency = 7)
	
	# Calculate number of periods to forecast
	horizon <- nrow(temp_v)

	#Some stores are on train set but not on test set
	if (horizon > 0){
		temp_fc <- stlf(logSales, 
			h=horizon, 
			s.window='periodic', 
			method='ets',
			ic='bic',
			opt.crit='mae'
		)
		logPred <- as.numeric(temp_fc$mean * ifelse( is.na(temp_v$Open), 0, temp_v$Open) )
		predictions <- data.frame(pred = expm1(logPred), Store=temp_v$Store, Id = temp_v$Id, Date=temp_v$Date)
		return(predictions)
	}
}

ts_stlf <- ddply(combined, .(Store), ts_stlf_fit)
foo <- merge(x=test, y=ts_stlf, by="Id",all.x=TRUE)
submission <- data.frame(Id=foo$Id,Sales=foo$pred)
submission [is.na(submission )]   <- 0
write.csv(submission, './submission/ts_stlf.csv', row.names=F)

ts_stlf_holdout <- ddply(train_only, .(Store), ts_stlf_fit)
submission <- data.frame(Store=ts_stlf_holdout$Store,Date=ts_stlf_holdout$Date,Sales=ts_stlf_holdout$pred)
write.csv(submission, './holdout_pred/ts_stlf.csv', row.names=F)

#==========================================================================================================
# TSLM
#==========================================================================================================

ts_tslm_fit = function(x) {
	temp_m <- subset(x, model == 1)
	temp_v <- subset(x, model == 0)

	# Set time series for input store data x
	logSales<- ts(temp_m$logSales, frequency = 7)

	# Set regressor 
	DayOfWeek <- temp_m$DayOfWeek
	SchoolHoliday <- temp_m$SchoolHoliday
 	Promo <- temp_m$Promo

	tslm_fit<- tslm(logSales~ trend + season + DayOfWeek + SchoolHoliday  + Promo)

	horizon <- nrow(temp_v)

	#Some stores are on train set but not on test set
	if (horizon > 0){
		temp_fc <- data.frame(forecast(tslm_fit, newdata = data.frame(
					DayOfWeek = temp_v$DayOfWeek,
					SchoolHoliday = temp_v$SchoolHoliday,
 					Promo = temp_v$Promo)))
	
		logPred <- as.numeric(temp_fc$Point.Forecast * ifelse( is.na(temp_v$Open), 0, temp_v$Open) )
		predictions <- data.frame(pred = expm1(logPred), Store=temp_v$Store, Id = temp_v$Id, Date=temp_v$Date)
		return(predictions )
	}
}

ts_tslm <- ddply(combined, .(Store), ts_tslm_fit)
foo <- merge(x=test, y=ts_tslm, by="Id",all.x=TRUE)
submission <- data.frame(Id=foo$Id,Sales=foo$pred)
submission [is.na(submission )]   <- 0
write.csv(submission, './submission/ts_tslm.csv', row.names=F)

ts_tslm_holdout <- ddply(train_only, .(Store), ts_tslm_fit)
submission <- data.frame(Store=ts_tslm_holdout$Store,Date=ts_tslm_holdout$Date,Sales=ts_tslm_holdout$pred)
write.csv(submission, './holdout_pred/ts_tslm.csv', row.names=F)

