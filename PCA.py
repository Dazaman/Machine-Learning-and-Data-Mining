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
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#=============================  I/0 functions  =====================================#

# Custom reader for our training file.
def csv_reader(file_obj,c_rows,row_counter,limit):
    reader = csv.reader(file_obj, delimiter="\t")
    for i, line in enumerate(reader):
        if row_counter<limit:
            linestring = line[0]
            floats = linestring.split(',')
            linerow=[]
            for i in floats[1:]:
                linerow.append(float(i))
            c_rows.append(linerow)
            row_counter+=1
            
# Used to write a numpy nd array to a csv file
def csv_matrix_writer(file_obj, Matrix):
    writer = csv.writer(file_obj, delimiter=',', quoting = csv.QUOTE_ALL)
    for row in Matrix:
        writer.writerow(row)

#=============================  Initialization  ====================================#

start_time = time.time()
trrow_counter=0;
terow_counter=0;
rowstrn = []
rowstst = []
csv_training_path = "training_data.csv"
csv_test_path = "test_data.csv"
csv_pca_train_out_path = "PCA_train_result.csv"
csv_pca_test_out_path = "PCA_test_result.csv"

#==============================  Data Loading  =====================================#

print('#================================================================#')
print('\nLoading Training Data')
with open(csv_training_path, "r") as f_obj_rd1:
    csv_reader(f_obj_rd1,rowstrn,trrow_counter,15068) #15068 is maximum that our RAM can work with
    Train_matrix = np.array(rowstrn)
print('\nTraining data loaded!\n')
print('#================================================================#')
print('\nLoading Test Data')
with open(csv_test_path, "r") as f_obj_rd2:
    csv_reader(f_obj_rd2,rowstst,terow_counter,5000)
    Test_matrix = np.array(rowstst)
print('\nTest data loaded!\n')
print('#================================================================#')

print('#================================================================#')
print('\nshape of Training Matrix = ', Train_matrix.shape)
print('shape of Test Matrix = ', Test_matrix.shape,'\n')
print('#================================================================#')


#=========================  Principal Component Analysis  ==========================#

print ('\nRunning Incrmental PCA with 200 Componenets and 5000 batch size')

pca = IncrementalPCA(n_components=200, batch_size = 5000)
pca.fit(Train_matrix)
Train_matrix = pca.transform(Train_matrix)
Test_matrix = pca.transform(Test_matrix)

parameters = pca.get_params()
variance = pca.explained_variance_ratio_
cumvariance = pca.explained_variance_ratio_.cumsum() 
#np.savetxt("pca_result_variance_200.csv", variance, delimiter=",")
#np.savetxt("pca_result_cum_variance_200.csv", variance, delimiter=",")

print ('\nPCA complete!\n')
print ('#================================================================#')
print('\nWriting transformed Train and Test matrices to CSV\n')
print('#================================================================#')

with open(csv_pca_train_out_path, 'w', newline='') as csvtrainoutfile:
    csv_matrix_writer(csvtrainoutfile , Train_matrix)

with open(csv_pca_test_out_path, 'w', newline='') as csvtestoutfile:
    csv_matrix_writer(csvtestoutfile , Test_matrix)

print('Successfully executed dimensionality reduction! New Train and Test Matrices created.')
print('Please run the LogisticReg.py file to predict test labels!')
#===============================================================#

#print('The nltk version is {}.'.format(nltk.__version__))
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
elapsed_time = time.time() - start_time
print("Time taken to run Principal Component Analysis (PCA) = ", elapsed_time, "seconds")