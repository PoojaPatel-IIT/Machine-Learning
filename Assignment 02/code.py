# Programming Assignment 2
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

# jll on the train set. Rows are datasets, columns are aplha values with alpha ranging from 10^-7 to 10^7
train_jll = np.zeros((10, 15))

# jll on the test set. Rows are datasets, columns are aplha values with alpha ranging from 10^-7 to 10^7
test_jll = np.zeros((10, 15))

alphaVal = []
pa = -7
for i in range(0,15):
    f = pow(10,pa)
    alphaVal.append(f)
    pa+=1
#     print("Alpha {0} : {1}" .format(i,f) )
    
#print("Alpha LIST : " ,alphaVal )
# For 10 datasets 
for e in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(Xs[e], ys[e], test_size= 1./3, random_state=6099) # A20396099
    # For 15 alpha values
    for alp in range(0,15):
        # BernoulliNB classifier
        clf = BernoulliNB(alpha=alphaVal[alp], binarize=0.0, fit_prior=True, class_prior=None)
        # fitting model on train data
        clf.fit(X_train, y_train)
        # prediction for train data using jll
        predict_train = clf._joint_log_likelihood(X_train)
        # prediction for test data using jll
        predict_test = clf._joint_log_likelihood(X_test)
        log_train,log_test=0,0
#         print("Train : ", log_train)
#         print("Train : ", log_train)
#         print("Predict Train jll: ", predict_train)
#         print("Predict Test jll: ", predict_test)
#         summing test predections
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
        train_jll[e][alp] = log_train
        test_jll[e][alp] = log_test

       
## DO NOT MODIFY BELOW THIS LINE.

# print("Train set jll")
# for i in range(10):
#     print(" ".join("{0:.4f}".format(n) for n in train_jll[i]))
    

# print("\nTest set jll")
# for i in range(10):
#     print(" ".join("{0:.4f}".format(n) for n in test_jll[i]))


#plt.plot(train_jll)
# plt.plot(test_jll)
# # Once you run the code, it will generate a 'results.pkl' file. Do not modify the following code.
pickle.dump((train_jll, test_jll), open('result.pkl', 'wb'))