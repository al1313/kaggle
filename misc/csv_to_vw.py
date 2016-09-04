#! /usr/bin/python

####################################################################################################################################
# Code: Converts CSV to VW format required for running Vowpal Wabbit
# Note:
#   Input should be CSV
####################################################################################################################################  

from datetime import datetime
from csv import DictReader

cat_var_list = ["_id", "site","device_type","ad_position"]

def csv_to_vw(loc_csv, loc_output, train=True):
  start = datetime.now()
  print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
  
  with open(loc_output,"wb") as outfile:
    for e, row in enumerate( DictReader(open(loc_csv)) ):
	
	  #Creating the features
      numerical_features = ""
      categorical_features = ""
      for k,v in row.items():
        if k not in ["action","id"]:
          if "hour" in k: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if any(x in k for x in cat_var_list): # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about the labels in the train set
        if row['action'] == "1":
          label = 1
        else:
          label = -1 #Note for vowpal wabbit only, the non-positive label is set to -1, rather than 0
        outfile.write( "%s '%s |i%s |c%s\n" % (label,row['id'],numerical_features,categorical_features) )
		
      else: #we dont care about labels in the test set
        outfile.write( "1 '%s |i%s |c%s\n" % (row['id'],numerical_features,categorical_features) )
      
	  #Reporting progress
      if e % 1000000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

csv_to_vw("./ggl_click_train.csv", "./ggl_click_train.vw",train=True)
csv_to_vw("./ggl_click_test.csv", "./ggl_click_test.vw",train=False)
