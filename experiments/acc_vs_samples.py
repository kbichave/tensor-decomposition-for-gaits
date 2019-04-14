# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:04:45 2019

@author: kbich
"""
# sio is used to load .mat files
import scipy.io as sio
# Tensorly, the most buzzed, tensor library introduced by Dr. Anima Anandkumar and group
import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
import random
# Load tensors for 10%, 50% and 95% recovered data

class AccVsSamples:
    def __init__(self):
        recovered10 = sio.loadmat('Data/recovered10.mat')
        self.recovered10 = recovered10['recovered10']

        recovered50 = sio.loadmat('Data/recovered50.mat')
        self.recovered50 = recovered50['recovered50']

        recovered95 = sio.loadmat('Data/recovered95.mat')
        self.recovered95 = recovered95['recovered95_1']

        self.num_test = 4
        self.recovered10_test = self.recovered10[-self.num_test:,:,:]
        self.recovered10 = self.recovered10[:len(self.recovered10)-4,:,:]

        self.recovered50_test = self.recovered50[-self.num_test:,:,:]
        self.recovered50 = self.recovered50[:len(self.recovered50)-4,:,:]

        self.recovered95_test = self.recovered95[-self.num_test:,:,:]
        self.recovered95 = self.recovered95[:len(self.recovered95)-4:,:]


        self.acc_dict = []

    #function to decompose the matrix based on factors obtained
    def decomposed(self,factor,recovered,l):
        '''
        Input: 
            recovered: a tensor of the order (numberof samles, 3, 283)
            factor: factors obtained by tensor decomposition for each of the category
            l: reduced dimesion selected
        Output:
            matrix of the size (number of samples,l)

        '''
        unfolded =  np.dot(np.transpose(factor[1]) , tl.unfold(recovered, mode=1))
        decompose = tl.fold(unfolded, mode=1, shape=[recovered.shape[0],1,283])
        decompose = np.dot(decompose , factor[2])
        
        return np.reshape(decompose,(recovered.shape[0],l))
    
    def run_exp(self):
        # Boolean, as an switch to change the code between acc_vs_samples and acc_vs_reduced_dimension
        acc_vs_samples = True
        # Empty list to store the accuracies at each step
        acc_dict=[]
        # fixed to 122. l is the reduced_dimension variable
        l=18

        # loop over redced dimension or samples
        #for l in range(2,283,10):
        for samples in range (2,30):
            # Empty list declared to store accuraces, everytime the expriment is repeated
            acc=[]
            # Repeatation loop
            for repeat in range(1,500):
                _recovered10 = self.recovered10[random.sample(range(len(self.recovered10)), samples),:,:]
                _recovered50 = self.recovered50[random.sample(range(len(self.recovered50)), samples),:,:]
                _recovered95 = self.recovered95[random.sample(range(len(self.recovered95)), samples),:,:]
                
                # Tucker is applied on tensor of each category to obtain core and factors
                core10,factor10 = tucker(_recovered10, ranks = [_recovered10.shape[0],1,l])
                core50,factor50 = tucker(_recovered50, ranks =  [_recovered50.shape[0],1,l])
                core95,factor95 = tucker(_recovered95, ranks =  [_recovered95.shape[0],1,l])
                # Tensor of each category is decompsed based on the factors obtained earlier
                _decomposed10 = self.decomposed(factor10, _recovered10,l)
                _decomposed50 = self.decomposed(factor50, _recovered50,l)
                _decomposed95 = self.decomposed(factor95, _recovered95,l)

                test_decomposed10 = self.decomposed(factor10, self.recovered10_test,l)
                test_decomposed50 = self.decomposed(factor50, self.recovered50_test,l)
                test_decomposed95 = self.decomposed(factor95, self.recovered95_test,l)
                
                
                # Switch to execute acc_vs_samples branch here
                if acc_vs_samples:
                    # t=np.random.randint((self.recovered10.shape[0]+self.recovered50.shape[0]+self.recovered95.shape[0]),size= samples)
                    print('Sample: %i, Repeat: %i'%(samples,repeat))
                    _Y = np.ravel(np.array([[1]*samples + [2]*samples + [3]*samples]))
                    X = np.concatenate((_decomposed10, _decomposed50, _decomposed95))

                

                # The data into training-testing 
                xtrain = X
                xtest = np.concatenate((test_decomposed10,test_decomposed50,test_decomposed95))
                ytrain = _Y
                ytest = np.ravel(np.array([[1]*self.num_test + [2]*self.num_test + [3]*self.num_test]))
                # Classifiers are imported fsrom Sklearn, trained and tested. Accuracies are written


                #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
                clf = KNeighborsClassifier(n_neighbors=3)
                clf.fit(xtrain,ytrain)
                ypreds_knn = clf.predict(xtest)

                clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
                clf.fit(xtrain,ytrain)
                ypreds_mlp = clf.predict(xtest)

                clf = SVC(gamma='auto')
                clf.fit(xtrain,ytrain)
                ypreds_svm = clf.predict(xtest)

                clf = NearestCentroid()
                clf.fit(xtrain,ytrain)
                ypreds_nc = clf.predict(xtest)


                acc.append(np.array([accuracy_score(ytest, ypreds_knn) , accuracy_score(ytest, ypreds_mlp), accuracy_score(ytest, ypreds_svm), accuracy_score(ytest, ypreds_nc)]))
            acc_dict.append(np.mean(acc, axis=0))
            
        # To save accuracies as an checkpoint for later use
        with open('acc_dict.txt', 'w') as f:
            for item in np.array(acc_dict):
                f.write("%s\n" % item)
        acc_dict = np.array(acc_dict)

        _classifiers = ['kNN', 'MLP', 'SVM', 'Nearest Centroid']
        # Plotting
        if acc_vs_samples:
            fig = plt.figure()
            for _ in range(4):
                plt.plot([i for i in range(2,30)], acc_dict[:,_], label = _classifiers[_])
            fig.suptitle("Accuracy vs Samples")
            plt.xlabel("Samples")
            plt.ylabel("Accuracy")
            plt.legend( loc='lower right')
            plt.savefig('Figures/acc_vs_samples.png')
            plt.show()


            
            

            