#! /usr/bin/env python2

# from phil import *
import argparse
import random
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge
import pylab as pl
import math
import itertools
import pandas as pd

def Help():
    print "test code for linear regression"
    print "Usage: <input sequences and phenotype (in this fold induction) in one file>"
    exit()

seq_len = 19 ## This number has to be changed for different Nmers ########

parser = argparse.ArgumentParser()
parser.add_argument("input_file",help="input file with sequences and phenotype(in this case fold induction)")
args = parser.parse_args()
print args.input_file

lines = map(lambda x: str.split(x),open(args.input_file,'r').readlines())

sequences = []
sequence_and_phenotype = []
promoters = []


for line in lines:
    if len(line[0]) == 19:
        sequence_and_phenotype.append((line[0],float(line[1])))
        sequences.append(line[0])


seq_len = 19
reference_seq = ''.join([random.choice('AGTC') for x in range(seq_len)])

print reference_seq
single_pos_terms = []
for pos in range(seq_len):
    for nt in 'AGCT':
        if nt != reference_seq[pos]:
            single_pos_terms.append((pos,nt))

print single_pos_terms

mutated_reference_seq = []
for i in range(seq_len):
    mutated_reference_seq.append((i,'AGCT'.replace(reference_seq[i],""))) #returns all bases except the reference_seq base

print mutated_reference_seq

#TESTING ITERTOOLS 
#mutations = list(itertools.product('AGCT','AGCT'))
base_pos = list(itertools.combinations(range(seq_len),2))
#print mutations
#print base_pos
#print list(itertools.product(mutations,base_pos))
#print len(list(itertools.product(mutations,base_pos)))


double_pos_terms = []
for items in base_pos:
    pos_one_mut_bases = [t[1] for t in mutated_reference_seq if t[0] == items[0]]
    pos_two_mut_bases = [t[1] for t in mutated_reference_seq if t[0] == items[1]]
    mutation_combinations = list(itertools.product(pos_one_mut_bases.pop(),pos_two_mut_bases.pop()))
    for elements in mutation_combinations:
        double_pos_terms.append((items,elements))

print double_pos_terms

def single_pos_binary_indicator(seq):
    x_single = [1 if seq[t[0]]==t[1] else 0 for t in single_pos_terms]
    return x_single

def double_pos_binary_indicator(seq):
    double_mut_this_seq = []
    for items in base_pos:
        double_mut_this_seq.append((items,(seq[items[0]],seq[items[1]])))
#    print double_mut_this_seq
    x_double = []
    for items1 in double_pos_terms:
        for items2 in double_mut_this_seq:
            if items1[0] == items2[0]:
                if items1[1] == items2[1]:
                    x_double.append(1)
                else:
                    x_double.append(0)
#    print x_double
#    print 'len x_double',len(x_double)
    return x_double
                
X = []

for seq in sequences:
    x = [1]
    x = x + single_pos_binary_indicator(seq)
    x = x + double_pos_binary_indicator(seq)
    X.append(x)
X = numpy.array(X)

print X
### Different linear regression models can be tried here #####
#ordinary least-sq linear regression
#linreg = LinearRegression()

#Lasso prefers least number of non-zero coefficients. This reduces the number of variables required to explain the phenotype
# linreg = Lasso(alpha=0.0001,precompute=True,max_iter=10000,
#             positive=False, random_state=9999, selection='random')
linreg = Lasso(alpha=1,precompute=True,max_iter=10000,
            positive=False, random_state=9999, selection='random')
# linreg = LinearRegression()

#Ridge imposes a penalty on the magnitude of the coefficients. One can get high coeff with OLS
#linreg = Ridge(fit_intercept=False,alpha=0.5)

Y = []
for items in sequence_and_phenotype:
    Y.append(math.log10(float(items[1])))  #Note the data is input in log
Y = numpy.array(Y)

# from sklearn.cross_validation import train_test_split  ## split data into training and test sets
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.7,random_state=0)
half_data_len = len(Y) / 2
X_train = X[:half_data_len]
X_test = X[half_data_len:]
Y_train = Y[:half_data_len]
Y_test = Y[half_data_len:]

linreg.fit(X_train,Y_train)
#linreg.fit(X,Y_true)
numpy.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)
print "Regression Coefficients\n", linreg.coef_
print "Regression Coefficients dimensions\n", linreg.coef_.shape

beta_hat = linreg.coef_
# estimate beta from data using the normal equation
#beta_hat = numpy.dot(numpy.linalg.inv((numpy.dot(X.T,X))),numpy.dot(X.T,Y))
Y_hat = numpy.dot(X_test,beta_hat)

print "Beta_hat **********"
count = 1
for items in beta_hat:
    count += 1
#    print "beta_hat", count, items
print "X_test.shape", X_test.shape
print "Y_test.shape ", Y_test.shape
print "Beta_hat.shape", beta_hat.shape


## plotting ###
import matplotlib.pyplot as plt
cross_validate = False
if cross_validate:
    from sklearn import svm
    from sklearn import preprocessing
    from sklearn import utils
    from sklearn.model_selection import KFold
    
    lab_enc = preprocessing.LabelEncoder()
    Y_transform = lab_enc.fit_transform(Y) # This important: have to transform full Y before splitting because it is internally scaled
    k_fold = KFold(n_splits=10)
    for train_indices, test_indices in k_fold.split(X):
        print ('Train: %s | test %s'%(train_indices, test_indices))
        Y_train_transform = Y_transform[train_indices]
        Y_test_transform = Y_transform[test_indices]
        X_train_ = X[train_indices]
        X_test_ = X[test_indices]
        
        clf = svm.SVC(C=50, kernel="linear")
        clf.fit(X_train_,Y_train_transform)
        print "clf.score ",clf.score(X_test_, Y_test_transform)  
    

    #insert K-fold loop here
#    Y_train_transform = Y_transform[:-400]
#    Y_test_transform = Y_transform[400:]
    #print encoded



plt.plot(Y_test,Y_hat,'.')
mx = max(numpy.hstack([Y_test,Y_hat]))
mn = min(numpy.hstack([Y_test,Y_hat]))
print 'mx mn', mx, mn
plt.plot([mn,mx],[mn,mx],'k')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.axis('image')
plt.text(mn,mx,'r = %0.2f'%numpy.corrcoef(Y_test,Y_hat)[0,1],verticalalignment='top', horizontalalignment='left')
plt.savefig('{0}.png'.format(args.input_file.split('/')[-1].split('.')[0]))

