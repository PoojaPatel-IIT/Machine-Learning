# Programming assignment 1
# Prereqs: all previous prereqs, plus pandas
# Implement what is asked at the TODO section.
# You can import additional methods/classes if you need them.
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split


# Load Datasets
# The “Xs” has 10 entries, each has the feature matrix of a dataset, and the “ys” has 10 entries, each has the target array.
# The datasets are downloaded from scikit-learn or UCI machine learning repository.
# We downloaded the datasets and "pickle"d them for you; you should be able to un"pickle" them
# using the following code, but if pickle fails, please use the attached load_datasets code.

print("Loading datasets...")
Xs = pickle.load(open('datasets_x.pkl', 'rb'))
ys = pickle.load(open('datasets_y.pkl', 'rb'))
print("Done.")

# Accuracies on the train set. Rows are datasets, columns are decision trees with depths ranging from 1 to 15.
# Note that the arrays start with a zero index; not 1. So, the ith column should have the result of the tree that has depth (i+1).
train_ac = np.zeros((10, 15))

# Accuracies on the test set.
test_ac = np.zeros((10, 15))



############ TODO ############
# Your task is to do the following, per dataset; tip: you might find it easier loop over datasets
# 1. 	Create a train_test split of your dataset Xi, yi. Use train_test_split() method from scikit-learn.
#		Test size should be set to 1./3, and random_state should set to the 4th, 5th, and 6th digits of your A# (AXXX123XX).
# 2. 	For a depth d, ranging from 1 to 15, inclusive,
# 2.a		Create a decision tree of maximum depth d, and random_state set to the last 3 digits of your A# (AXXXXX123).  
# 2.b 		Fit the classifier to your train split.
# 2.c		Compute the score (accuracy) of the classifier on your train split; record the result in the correct position in the train_ac array
# 2.d		Compute the score (accuracy) of the classifier on your test split; record the result in the correct position in the test_ac array

# For your reference
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

# Write your code below this line.

seed = int("099")  # random_state for DecisionTreeClassifier 

# to loop for all 10 datasets given and which are stored in Xs and ys
for e in range(0,10):
    #A20396099
    X_train, X_test, y_train, y_test = train_test_split(Xs[e], ys[e], test_size= 1./3, random_state=960)#960
    for d in range(1,16):
        dt_clf = DecisionTreeClassifier( criterion = "gini",random_state=seed , max_depth=d )
        dt_clf.fit(X_train, y_train)
       
        # prediction on train data
        y_pred_traindataset  = dt_clf.predict(X_train)
         # prediction on test data
        y_pred_testdataset = dt_clf.predict(X_test)
        
        # accuracy score for train dataset
        train_ac[e][d-1] =  sklearn.metrics.accuracy_score(y_train, y_pred_traindataset)
        # accuracy score for test dataset
        test_ac[e][d-1] =  sklearn.metrics.accuracy_score(y_test, y_pred_testdataset)

## DO NOT MODIFY BELOW THIS LINE.

print("Train set accuracies")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in train_ac[i]))
	

print("\nTest set accuracies")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in test_ac[i]))

 

# Once you run the code, it will generate a 'results.pkl' file. Do not modify the following code.
pickle.dump((train_ac, test_ac), open('results.pkl', 'wb'))