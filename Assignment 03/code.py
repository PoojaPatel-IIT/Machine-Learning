# Programming Assignment 3
import pandas as pd
import pickle
import numpy as np
# %matplotlib inline
from matplotlib import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import sklearn
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

# Model complexity. Rows are datasets, columns are differrent C values ranging from 10^-7 to 10^7
l2_model_complexity = np.zeros((10,15))

# CLL on the train set. Rows are datasets, columns are differrent C values ranging from 10^-7 to 10^7
l2_train_cll = np.zeros((10, 15))

# CLL on the test set. Rows are datasets, columns are differrent C values ranging from 10^-7 to 10^7
l2_test_cll = np.zeros((10, 15))


# Number of zero weighths in L2. Rows correspond to datasets, and columns correspond to C values ranging from 10^-7 to 10^7
l2_num_zero_weights = np.zeros((10, 15))

# Number of zero weighths in L1. Rows correspond to datasets, and columns correspond to C values ranging from 10^-7 to 10^7
l1_num_zero_weights = np.zeros((10, 15))


CValues = []
c = -7
for i in range(0,15):
    f = pow(10,c)
    CValues.append(f)
    c+=1
#     print("C {0} : {1}" .format(i,f) )
    
# print("C LIST : " ,CValues )
# For 10 datasets 
for e in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(Xs[e], ys[e], test_size= 1./3, random_state=6099) # A20396099
    # For 15 alpha values
    for cval in range(0,15):
        # LR classifier and fitting 
        clf = LogisticRegression(penalty="l2", C=CValues[cval],  random_state=42)
        # fitting model on train data
        clf.fit(X_train, y_train)
        #w0
        intercept = clf.intercept_
        #w1
        weight1 = clf.coef_
#         l2_model_complexity[e][cval] = np.sum(np.array(clf.intercept_)*np.array(clf.intercept_) + np.array(clf.coef_)*np.array(clf.coef_))
        l2_model_complexity[e][cval] = np.sum(intercept**2 + np.sum(weight1**2))
        
        # CLL for training set and test set
        log_train,log_test=0,0
        predict_train = clf.predict_log_proba(X_train)
        predict_test = clf.predict_log_proba(X_test)
#         predict_train = np.log(predict_train)
#         predict_test = np.log(predict_test)
        for test in range(len(predict_test)):
            if y_test[test] == True:
                log_test += predict_test[test][1]
            else:
                log_test += predict_test[test][0]
#         summing train predections            
        for train in range(len(predict_train)):
            if y_train[train] == True:
                log_train += predict_train[train][1]
            else:
                log_train += predict_train[train][0]             
        # storing the result
        l2_train_cll[e][cval] = log_train
        l2_test_cll[e][cval] = log_test

# Number of zero weights
def zerosL1AndL2(reg):
    num_zero_weights = np.zeros((10,15))
    # Number of zero weights
    Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
    ys = pickle.load(open('binarized_ys.pkl', 'rb'))
    for e in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(Xs[e], ys[e], test_size= 1./3, random_state=6099) # A20396099
        # For 15 alpha values
        for cval in range(0,15):
            # LR classifier and fitting 
            clf = LogisticRegression(penalty=reg, C=CValues[cval],  random_state=42)
            # fitting model on train data
            clf.fit(X_train, y_train)
            #w0
            intercept = clf.intercept_
            count_w0 = 0
            for i in intercept:
    #             print(i)
                if i == 0.0:
                    count_w0 +=1

            #w1
            weight1 = clf.coef_
    #         print(weight1)
            count_w1 = 0
            for j in weight1[0]:
                if j == 0.0:
                    count_w1 +=1
            num_zero_weights[e][cval] = count_w0+count_w1
    return num_zero_weights

# L1 - Number of zero weights
l1_num_zero_weights = zerosL1AndL2("l1")
# L2 - Number of zero weights
l2_num_zero_weights = zerosL1AndL2("l2")


# 10 Plots for model_complexity vs train_cll and test_cll
for i in range(0,10):
    _, ax = plt.subplots()
    ax.plot(l2_model_complexity[i],l2_train_cll[i] , linewidth=2, label='train_cll')
    ax.plot(l2_model_complexity[i],l2_test_cll[i] , linewidth=2, label='test_cll')
#     ax.axhline(y=0.5, color='k')
#     ax.axvline(x=0, color='k')
    ax.legend()
    _.suptitle('Dataset {0}'.format(i+1))
    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('L2 CLL')
    # As of now saves the last graph
#     plt.savefig('favorite_complexity_vs_overfit.png')
    
    
#  10 Plots for exponent of C vs Number of zero weights for L1 and L2

for i in range(0,10):
    xs = np.linspace(-7, 7, 15)
    _, ax = plt.subplots()
    ax.plot(xs,l2_num_zero_weights[i] , linewidth=3, label='l2_num_zero')
    ax.plot(xs,l1_num_zero_weights[i] , linewidth=3, label='l1_num_zero')
#     ax.axhline(y=0.5, color='k')
#     ax.axvline(x=0, color='k')
    ax.legend()
    plt.xticks(np.arange(min(xs), max(xs)+1, 1.0))
    _.suptitle('Dataset {0}'.format(i+1))
    ax.set_xlabel('Exponent of C')
    ax.set_ylabel('Number of zero weights')
    # As of now saves the last graph
#     plt.savefig('favorite_feature_selection.png')

# DO NOT MODIFY BELOW THIS LINE.

print("\nModel Complexity")
for i in range(10):
    print(" ".join("{0:.4f}".format(n) for n in l2_model_complexity[i]))

print("\nTrain set cll")
for i in range(10):
    print(" ".join("{0:.4f}".format(n) for n in l2_train_cll[i]))
    

print("\nTest set cll")
for i in range(10):
    print(" ".join("{0:.4f}".format(n) for n in l2_test_cll[i]))

print("\n L2 - Number of zero weights")
for i in range(10):
    print(" ".join("{0:.4f}".format(n) for n in l2_num_zero_weights[i]))
    
print("\n L1 - Number of zero weights")
for i in range(10):
    print(" ".join("{0:.4f}".format(n) for n in l1_num_zero_weights[i]))
    

# Once you run the code, it will generate a 'result.pkl' file. Do not modify the following code.
pickle.dump((l2_model_complexity, l2_train_cll,l2_test_cll,l2_num_zero_weights,l1_num_zero_weights), open('result.pkl', 'wb'))    
      