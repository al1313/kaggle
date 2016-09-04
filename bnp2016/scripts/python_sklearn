########################################################################################################################
# Models built using sklearn
# Extratrees, Adaboost, KNeighbourClassifier
########################################################################################################################

import pandas as pd
import numpy as np
import os

#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
#from sklearn import ensemble
from sklearn.cross_validation import train_test_split

os.chdir('E:/models/bnp')
os.getcwd()

#np.random.seed(13)

print('Load data...')

train = pd.read_csv('./input/train.csv')
ID = pd.DataFrame({"ID": train['ID'].values})
target = train['target'].values
train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51',
'v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107',
'v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

test = pd.read_csv("./input/test.csv")
id_test = test['ID'].values
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54',
'v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109',
'v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

train = pd.concat([ID, train], axis=1)

X_stack_1, X_stack_2, y_stack_1, y_stack_2 = train_test_split(train, target, test_size=0.5, random_state=13)

id_X_stack_1 = X_stack_1['ID'].values
id_X_stack_2 = X_stack_2['ID'].values
X_stack_1 = X_stack_1.drop(['ID'],axis=1)
X_stack_2 = X_stack_2.drop(['ID'],axis=1)

train = train.drop(['ID'],axis=1)

########################################################################################################################
# Extra trees (entropy)
########################################################################################################################

print('Extra Trees Training...')
extce_stack_1 = ExtraTreesClassifier(n_estimators=852,max_features= 60,criterion= 'entropy',min_samples_split= 3,
                            max_depth= 45, min_samples_leaf= 2, n_jobs = -1)    
extce_stack_2 = ExtraTreesClassifier(n_estimators=852,max_features= 60,criterion= 'entropy',min_samples_split= 3,
                            max_depth= 45, min_samples_leaf= 2, n_jobs = -1)    
extce_train = ExtraTreesClassifier(n_estimators=852,max_features= 60,criterion= 'entropy',min_samples_split= 3,
                            max_depth= 45, min_samples_leaf= 2, n_jobs = -1)      
                            
extce_stack_1.fit(X_stack_1,y_stack_1) 
extce_stack_2.fit(X_stack_2,y_stack_2) 
extce_train.fit(train,target) 

print('Predict...')
pred_stack_1_train = extce_stack_1.predict_proba(X_stack_1)
pred_stack_1_val = extce_stack_1.predict_proba(X_stack_2)

print('stack 1 train loss is', log_loss(y_stack_1, pred_stack_1_train[:,1]))
print('stack 1 val loss is', log_loss(y_stack_2, pred_stack_1_val[:,1]))

pred_stack_2_train = extce_stack_2.predict_proba(X_stack_2)
pred_stack_2_val = extce_stack_2.predict_proba(X_stack_1)

print('stack 2 train loss is', log_loss(y_stack_2, pred_stack_2_train[:,1]))
print('stack 2 val loss is', log_loss(y_stack_1, pred_stack_2_val[:,1]))

extce_pred_test = extce_train.predict_proba(test)

print('Export...')

foo = pd.DataFrame({"ID": id_X_stack_1, "PredictedProb": pred_stack_2_val[:,1]})
goo = pd.DataFrame({"ID": id_X_stack_2, "PredictedProb": pred_stack_1_val[:,1]})

extce_train_pred = pd.concat([foo,goo])
 
extce_train_pred.to_csv('./ensemble/level_0/train_scored/et_entropy.csv',index=False)
pd.DataFrame({"ID": id_test, "PredictedProb": extce_pred_test[:,1]}).to_csv('./ensemble/level_0/test_scored/et_entropy.csv',index=False)

########################################################################################################################
# Extra trees (gini)
########################################################################################################################

print('Extra Trees Training...')
extcg_stack_1 = ExtraTreesClassifier(n_estimators=1200,max_features= 60,criterion= 'gini',min_samples_split= 3,
                            max_depth= 45, min_samples_leaf= 2, n_jobs = -1)    
extcg_stack_2 = ExtraTreesClassifier(n_estimators=1200,max_features= 60,criterion= 'gini',min_samples_split= 3,
                            max_depth= 45, min_samples_leaf= 2, n_jobs = -1)  
extcg_train = ExtraTreesClassifier(n_estimators=1200,max_features= 60,criterion= 'gini',min_samples_split= 3,
                            max_depth= 45, min_samples_leaf= 2, n_jobs = -1)      
                            
extcg_stack_1.fit(X_stack_1,y_stack_1) 
extcg_stack_2.fit(X_stack_2,y_stack_2) 
extcg_train.fit(train,target) 

print('Predict...')
pred_stack_1_train = extcg_stack_1.predict_proba(X_stack_1)
pred_stack_1_val = extcg_stack_1.predict_proba(X_stack_2)

print('stack 1 train loss is', log_loss(y_stack_1, pred_stack_1_train[:,1]))
print('stack 1 val loss is', log_loss(y_stack_2, pred_stack_1_val[:,1]))

pred_stack_2_train = extcg_stack_2.predict_proba(X_stack_2)
pred_stack_2_val = extcg_stack_2.predict_proba(X_stack_1)

print('stack 2 train loss is', log_loss(y_stack_2, pred_stack_2_train[:,1]))
print('stack 2 val loss is', log_loss(y_stack_1, pred_stack_2_val[:,1]))

extcg_pred_test = extcg_train.predict_proba(test)

print('Export...')

foo = pd.DataFrame({"ID": id_X_stack_1, "PredictedProb": pred_stack_2_val[:,1]})
goo = pd.DataFrame({"ID": id_X_stack_2, "PredictedProb": pred_stack_1_val[:,1]})

extcg_train_pred = pd.concat([foo,goo])
 
extcg_train_pred.to_csv('./ensemble/level_0/train_scored/et_gini.csv',index=False)
pd.DataFrame({"ID": id_test, "PredictedProb": extcg_pred_test[:,1]}).to_csv('./ensemble/level_0/test_scored/et_gini.csv',index=False)

########################################################################################################################
# Adaboost
########################################################################################################################

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

print('Adaboost Training...') 
adac_stack_1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=50, learning_rate=0.01)      
adac_stack_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=50, learning_rate=0.01)      
adac_train = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=50, learning_rate=0.01)    
 
adac_stack_1.fit(X_stack_1,y_stack_1) 
adac_stack_2.fit(X_stack_2,y_stack_2) 
adac_train.fit(train,target) 

print('Predict...')
pred_stack_1_train = adac_stack_1.predict_proba(X_stack_1)
pred_stack_1_val = adac_stack_1.predict_proba(X_stack_2)

print('stack 1 train loss is', log_loss(y_stack_1, pred_stack_1_train[:,1]))
print('stack 1 val loss is', log_loss(y_stack_2, pred_stack_1_val[:,1]))

pred_stack_2_train = adac_stack_2.predict_proba(X_stack_2)
pred_stack_2_val = adac_stack_2.predict_proba(X_stack_1)

print('stack 2 train loss is', log_loss(y_stack_2, pred_stack_2_train[:,1]))
print('stack 2 val loss is', log_loss(y_stack_1, pred_stack_2_val[:,1]))

adac_pred_test = adac_train.predict_proba(test)

print('Export...')
foo = pd.DataFrame({"ID": id_X_stack_1, "PredictedProb": pred_stack_2_val[:,1]})
goo = pd.DataFrame({"ID": id_X_stack_2, "PredictedProb": pred_stack_1_val[:,1]})

adac_train_pred = pd.concat([foo,goo])
 
adac_train_pred.to_csv('./ensemble/level_0/train_scored/adaboost.csv',index=False)
pd.DataFrame({"ID": id_test, "PredictedProb": adac_pred_test[:,1]}).to_csv('./ensemble/level_0/test_scored/adaboost.csv',index=False)

########################################################################################################################
# KNeighborsClassifier
########################################################################################################################

from sklearn.neighbors import KNeighborsClassifier

print('KNeighborsClassifier Uniform Training...') 
knnc_uniform_stack_1 = KNeighborsClassifier(n_neighbors=60, weights='uniform')      
knnc_uniform_stack_2 = KNeighborsClassifier(n_neighbors=60, weights='uniform')      
knnc_uniform_train = KNeighborsClassifier(n_neighbors=60, weights='uniform')      

knnc_uniform_stack_1.fit(X_stack_1,y_stack_1)
knnc_uniform_stack_2.fit(X_stack_2,y_stack_2)
knnc_uniform_train.fit(train,target)
 
print('Predict...') 
pred_stack_1_train = knnc_uniform_stack_1.predict_proba(X_stack_1)
pred_stack_1_val = knnc_uniform_stack_1.predict_proba(X_stack_2)

print('stack 1 train loss is', log_loss(y_stack_1, pred_stack_1_train[:,1]))
print('stack 1 val loss is', log_loss(y_stack_2, pred_stack_1_val[:,1]))

pred_stack_2_train = knnc_uniform_stack_2.predict_proba(X_stack_2)
pred_stack_2_val = knnc_uniform_stack_2.predict_proba(X_stack_1)

print('stack 2 train loss is', log_loss(y_stack_2, pred_stack_2_train[:,1]))
print('stack 2 val loss is', log_loss(y_stack_1, pred_stack_2_val[:,1]))

knnc_uniform_pred_test = knnc_uniform_train.predict_proba(test)

print('Export...')
foo = pd.DataFrame({"ID": id_X_stack_1, "PredictedProb": pred_stack_2_val[:,1]})
goo = pd.DataFrame({"ID": id_X_stack_2, "PredictedProb": pred_stack_1_val[:,1]})

knnc_uniform_train_pred = pd.concat([foo,goo])
 
knnc_uniform_train_pred.to_csv('./ensemble/level_0/train_scored/knnc_uniform.csv',index=False)
pd.DataFrame({"ID": id_test, "PredictedProb": knnc_uniform_pred_test[:,1]}).to_csv('./ensemble/level_0/test_scored/knnc_uniform.csv',index=False)

########################################################################################################################

print('KNeighborsClassifier Distance Training...') 
knnc_distance_stack_1 = KNeighborsClassifier(n_neighbors=50, weights='distance')      
knnc_distance_stack_2 = KNeighborsClassifier(n_neighbors=50, weights='distance')      
knnc_distance_train = KNeighborsClassifier(n_neighbors=50, weights='distance')      

knnc_distance_stack_1.fit(X_stack_1,y_stack_1)
knnc_distance_stack_2.fit(X_stack_2,y_stack_2)
knnc_distance_train.fit(train,target)
 
print('Predict...') 
pred_stack_1_train = knnc_distance_stack_1.predict_proba(X_stack_1)
pred_stack_1_val = knnc_distance_stack_1.predict_proba(X_stack_2)

print('stack 1 train loss is', log_loss(y_stack_1, pred_stack_1_train[:,1]))
print('stack 1 val loss is', log_loss(y_stack_2, pred_stack_1_val[:,1]))

pred_stack_2_train = knnc_distance_stack_2.predict_proba(X_stack_2)
pred_stack_2_val = knnc_distance_stack_2.predict_proba(X_stack_1)

print('stack 2 train loss is', log_loss(y_stack_2, pred_stack_2_train[:,1]))
print('stack 2 val loss is', log_loss(y_stack_1, pred_stack_2_val[:,1]))

knnc_distance_pred_test = knnc_distance_train.predict_proba(test)

print('Export...')
foo = pd.DataFrame({"ID": id_X_stack_1, "PredictedProb": pred_stack_2_val[:,1]})
goo = pd.DataFrame({"ID": id_X_stack_2, "PredictedProb": pred_stack_1_val[:,1]})

knnc_distance_train_pred = pd.concat([foo,goo])
 
knnc_distance_train_pred.to_csv('./ensemble/level_0/train_scored/knnc_distance.csv',index=False)
pd.DataFrame({"ID": id_test, "PredictedProb": knnc_distance_pred_test[:,1]}).to_csv('./ensemble/level_0/test_scored/knnc_distance.csv',index=False)
