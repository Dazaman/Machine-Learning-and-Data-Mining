import numpy as np
import pandas as pd
import nltk
import sklearn
import csv
import os,sys
import time
import matplotlib
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from numpy import genfromtxt
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import IncrementalPCA
import scipy as sp
import scipy.optimize

#====================  Function Declerations   =======================#

# Used to insert a new entry into a dictionary structure
def update_dict(d, key, value):
    if key in d.keys(): 
        if value not in d[key]:
            print('dict = ', d)
            d[key].append(value)
    else:
        d[key] = [value]

#****************************************************************************************
# We have used functions with for likelihood and probabilty calculations
# which are essentially just equations of their respective mathematical formulae.
#Title : Logistic Regression mathematical equation
# Author : tchakravarty
# Date : 11-May-2014
# Code version : V 1.0
# Availability : http://stats.stackexchange.com/questions/17436/logistic-regression-with-lbfgs-solver
#*****************************************************************************************

# Returns the value of the logistic function/ the probability of an instance
def logit(mX, vBeta):
	return((np.exp(np.dot(mX, vBeta))/(1.0 + np.exp(np.dot(mX, vBeta)))))

# Built up Likelihood function to use for stable parametrisation of the cost function
def logLikelihoodLogitStable(vBeta, mX, vY):
	return(-(np.sum(vY*(np.dot(mX, vBeta) -
    np.log((1.0 + np.exp(np.dot(mX, vBeta))))) +
                    (1-vY)*(-np.log((1.0 + np.exp(np.dot(mX, vBeta))))))))

# Returns the respective score
def likelihoodScore(vBeta, mX, vY):
    return(np.dot(mX.T,
                  (logit(mX, vBeta) - vY)))

#******************************************************************************************

# Function to write an array to a csv file
def csv_matrix_writer(file_obj, Matrix):
    writer = csv.writer(file_obj, delimiter=',', quoting = csv.QUOTE_ALL)
    for row in Matrix:
        writer.writerow(row)
#===============================================================#

# Start the execution time
start_time = time.time()

# Reading the input training, test and label files into data frames using Pandas
traindata_df = pd.read_csv('PCA_train_result.csv', header = None)
testdata_df = pd.read_csv('PCA_test_result.csv', header = None)
testappnames_df = pd.read_csv('test_data.csv', header = None, usecols=[0] )

# Converting the data frames to numpy arrays 
# X = Train, Z = Test, N = TestAppName
X = traindata_df.as_matrix()
Z = testdata_df.as_matrix()
N = testappnames_df.as_matrix()


#******************************************************************
# Taking labels in order to app names reference
trainingnames_df = pd.read_csv('training_labels.csv', header = None, usecols=[0])
traininglabels_df = pd.read_csv('training_labels.csv', header = None, usecols=[1])
trainingapp_df = pd.read_csv('training_data.csv',header = None, usecols = [0], nrows = 15068)
YN = trainingnames_df.as_matrix()
YL = traininglabels_df.as_matrix()
YT = trainingapp_df.as_matrix()

dict_app_label = {}
for i,value in enumerate(YL):
    key = YN[i]
    key = str(key)
    update_dict(dict_app_label, key, value)

Ytemp = ["" for x in range(X.shape[0])]

for i,key in enumerate(YT):
	#print(value)
	key = str(key)
	labels = dict_app_label[key]
	Ytemp[i] = labels
	#print(type(Y))

Y = np.array(Ytemp)
#******************************************************************

print("Shape of Training Matrix loaded = ", X.shape)
print("Shape of Test Matrix loaded = ", Z.shape)
print("Please wait while Logistic Regression is being applied...")

# Creating the intercept column and concatenating them to Test and Train
# to be used for calculation of Beta Values
train_intercept = np.ones(X.shape[0]).reshape(X.shape[0], 1)
test_intercept = np.ones(Z.shape[0]).reshape(Z.shape[0], 1)
X = np.concatenate((train_intercept, X), axis = 1) #intercept attached
Z = np.concatenate((test_intercept, Z), axis = 1) # to multiply b0 term

# Storing the unique labels in alphabetical order and computing the size of label list
setoflabels = np.unique(Y)
list_setof_labels= list(setoflabels)
num_labels = len(list_setof_labels)

# Inserting labels into dictionary for use in prediction using indeces
dict_labels = {}
for i,value in enumerate(list_setof_labels):
    key = i
    update_dict(dict_labels, key, value)

# X0 is random values for use of initializing the values in beta value estimation
x0 = np.random.rand(X.shape[1])

# Calcilating the dimensions of the Train and Test matrices
xrows = X.shape[0]
xcols = X.shape[1]
zrows = Z.shape[0]
zcols = Z.shape[1]


# Initializing sizes of arrays to store Labels, Betas and Probabilities
newY = np.zeros(shape=(xrows,num_labels))

betas = np.zeros(shape=(num_labels,xcols))

estimated_log_prob = np.zeros(shape = (zrows,num_labels))

# Calculating Label matrices for our 30 class Logistic regression functions.
# One vs All Approach.

for i in range(1,num_labels+1):
	label=list_setof_labels[i-1]
	for j in range(1,(xrows+1)):
		if label == Y[j-1]:
			newY[j-1,i-1]=1
		else:
			newY[j-1,i-1]=0

# Calculating the probabilty for each of our 30 class classifier
# Evaluates Betas using BFGS

for i in range(1,num_labels+1):
	optimLogitLBFGS = sp.optimize.fmin_l_bfgs_b(logLikelihoodLogitStable,x0 = x0,args = (X, newY[:,i-1]), fprime = likelihoodScore,pgtol =  1e-3, disp = False)
	#print("Optim Logit BFGS = ", optimLogitLBFGS)
	est_beta_matrix, value, rest_of_the_stuff = optimLogitLBFGS
	list_beta_matrix = list(est_beta_matrix)
	for j in range(1,xcols+1):
		betas[i-1,j-1] = list_beta_matrix[j-1]

# Using the betas to calculate probabilities using mathematical equations
for i in range(1,num_labels+1):
	for j in range(1,zrows+1):
		value = logit(Z,betas[i-1,:])
		estimated_log_prob[j-1,i-1] = value[j-1]

Prediction_MAX = np.zeros(shape = (Z.shape[0],1))
Prediction_MAX_index = ["" for x in range(Z.shape[0])]

# Finding the maximum probabilty among the 30 predicted probabilites
# to find which class is the sample most likely to be in.

for i in range(1,Z.shape[0]+1):
	max_Value = np.amax(estimated_log_prob[i-1], axis=0)
	Prediction_MAX[i-1] = max_Value

# Using dictionary to match index of variable to appropriate label

for i in range(1,Z.shape[0]+1):
	max_Value = np.amax(estimated_log_prob[i-1], axis=0)
	for j in range(1,num_labels+1):
		if estimated_log_prob[i-1,j-1] == max_Value:
			label = dict_labels[j-1]
			Prediction_MAX_index[i-1]= label

# Combining test app names to their predicted labels
Predictions = np.concatenate((N, Prediction_MAX_index), axis = 1)

#Writing Predictions to the file
with open("predicted_labels.csv", 'w', newline='') as csvtestoutfile:
    csv_matrix_writer(csvtestoutfile , Predictions)	

print("\n Successfuly written predicted labels to predictions_labels.csv!\n")


#print('The nltk version is {}.'.format(nltk.__version__))
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
#print('The numpy version is {}.'.format(np.__version__))
elapsed_time = time.time() - start_time
print("Time taken to run Logistic Regression = ", elapsed_time, "seconds")