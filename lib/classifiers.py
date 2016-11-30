
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.optimize import minimize


# In[ ]:

def sigmoid(value):
    #, value.shape()
    return (1.0 / (1.0 + np.exp(-1.0 * value)))

def cost_function(w, X, labels):
    #print "w func", w, w.shape
    #w.shape = (1, len(w))
    #pred = sigmoid(np.dot(w, np.transpose(X)))
    pred = np.dot(w, np.transpose(X))
    #print pred.shape, labels.shape
    #print "pred", pred
    #error = -(np.dot(labels,np.transpose(np.log(pred))) + (np.dot((1 - labels) , np.transpose(np.log(1 - pred))))) / len(labels)
    #multivariate
    error = -np.dot(labels,np.transpose(np.log(pred))) / len(labels)
    #binary
    #error = (-np.dot(labels,np.transpose(np.log(sigmoid(pred)))) - np.dot((1 - labels), np.log(1 - np.transpose(sigmoid(pred))))) / len(labels)
    #print "error =", error
    grad = np.dot((pred - labels), X) / len(labels)
    #print "grad =", grad
    return [error, grad]

def train_logistic_regression(X_train, y_train, no_classes):
    no_features = len(X_train[0])
    w  = np.zeros((no_classes, no_features))
    print "init",w[:,:]
    print len(w[0])
    for i in range(len(w)):
        print "\n\nclass", i+1, "\n\n\n"
        #multivariate
        w[i] = minimize(cost_function, w[i].reshape((1,no_features)), args=(X_train, np.equal(y_train,(i+1))), method="Newton-CG", jac=True).x
        #binary
        #w[i] = minimize(cost_function, w[i].reshape((1,no_features)), args=(X_train, y_train_thresholded), method="Newton-CG", jac=True).x
    #temp = np.matrix(temp)
    #w = np.matrix(temp)
    #print "this is temp", temp
    print w
    return w

