#! /usr/bin/python

####################################################################################################################################
# Code: Fast Implementation of SGD on a binary classification
# Note:
#   Input should be CSV
#   Code does a single pass of SGD
#   Has two way interaction capability 
####################################################################################################################################  

####################################################################################################################################
# Required Libraries 
# Note: 
#   mmh3 is not built in and should be installed manually
#   it is used for hashing feature|value	
####################################################################################################################################

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import mmh3

####################################################################################################################################
# Files
####################################################################################################################################

train = './ggl_click_train.csv'  # path to training file
test = './ggl_click_test.csv'  # path to testing file
output = './scored.csv'# path to output file

####################################################################################################################################
# Parameters
####################################################################################################################################

target_variable = 'action' # name of target variable (case sensitive)
drop_variable = ['id'] # variables you want to drop (case sensitive)
two_way = 'yes' # Specify one way or two ways
weight_num = 2 ** 25   # number of weights use for learning
learn_rate = 0.5    # learning rate for sgd optimization, require tweaking each time for best performance

####################################################################################################################################
# Define functions
####################################################################################################################################

# A. Bounded logloss ####################################
# INPUT:
#     p: prediction
#     y: target
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)

# B. Hash each input csv row ####################################
# Note:
#   Current capability treats both integer and categorical features as categorical for simplicity
# INPUT:
#     csv_row: a csv dictionary, ex: {'Label': '1', 'variable1': '10', 'variable2': 'sydney', ...}
#     weight_num: the max index that we can hash to
# OUTPUT:
#     x: a list of hashed features
def get_x(csv_row, weight_num):
    x = [] 
    # Main features
    for key, value in csv_row.items():
        index = mmh3.hash(value + key[1:]) % weight_num  # Using the package mmh3 for 32-bit hash, up to weight_num
        x.append(index)
    # Two-way Interactions
    if two_way == 'yes':
        length = len(x)
        for i in range(length):
    	   for j in range(i+1,length):
    	       index = (x[i] * x[j]) % weight_num
    	       x.append(index)
    # Bias
    x.append(0) # 0 is the index of the bias term
    return x  

# C. Get probability estimation on x ####################################
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model ####################################
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, n, x, p, y):
    for i in x:
        # learn_rate / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * learn_rate / (sqrt(n[i]) + 1.)
        n[i] += 1.

    return w, n


####################################################################################################################################
# Training
####################################################################################################################################

# initialize model ####################################
w = [0.] * weight_num  # weights
n = [0.] * weight_num  # number of times we've encountered a feature

drop_variable.append(target_variable)

# Single pass SGD on a linear regression model
# It loops through each row in the data table and update the weights of each hashed feature
loss = 0.
for t, row in enumerate(DictReader(open(train))):
    y = 1. if row[target_variable] == '1' else 0.
    
    # Delete target variable and any other variables that you do not wish to train the model on 
    for a in drop_variable:
        del row[a] 

    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, weight_num)

    # step 2, get prediction
    p = get_p(x, w)

    # calcualte log loss for progress validation
    loss += logloss(p, y)
    if t % 1000000 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (
        datetime.now(), t, loss/t))

    # step 3, update weights in the model base on each row of data
    w, n = update_w(w, n, x, p, y)

####################################################################################################################################
# Scoring test set
####################################################################################################################################

with open(output, 'w') as submission:
    submission.write('id,actual,predicted\n')
    for t, row in enumerate(DictReader(open(test))):
        id = row['id']
        del row['id']
        actual = row['action']
        del row['action']
        x = get_x(row, weight_num)
        p = get_p(x, w)
        submission.write('%s,%s,%f\n' % (id,actual,p))
