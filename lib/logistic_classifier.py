
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.optimize import minimize


# In[ ]:

def sigmoid(value):
    #, value.shape()
    return (1.0 / (1.0 + np.exp(-1.0 * value)))

def cost_function(w, X, labels):
    pred = np.dot(w, np.transpose(X))
    error = -np.dot(labels,np.transpose(np.log(pred))) / len(labels)
    grad = np.dot((pred - labels), X) / len(labels)
    return [error, grad]

def train_logistic_regression(X_train, y_train, no_classes):
    no_features = len(X_train[0])
    w  = np.zeros((no_classes, no_features))
    print "init",w[:,:]
    print len(w[0])
    for i in range(len(w)):
        print "\n\nclass", i+1, "\n\n\n"
        w[i] = minimize(cost_function, w[i].reshape((1,no_features)), args=(X_train, np.equal(y_train,(i+1))), method="Newton-CG", jac=True).x
    print w
    return w
	
	
#Predicts on all 5 classes - 1, 2, 3, 4, 5
def predict(w, X_test):
    prob = np.transpose(np.dot(w, np.transpose(X_test)))
	#a/a.sum(axis=1)[:,None]
    prob /= prob.sum(axis=1)[:, None]
    return [np.argmax(prob, axis=1)+1, np.max(prob, axis=1)]

#To be used when logistic regression is trained only on three classes
def predict_ternary(w, X_test):
    prob = np.transpose(np.dot(w, np.transpose(X_test)))
    pred = (np.argmax(prob, axis=1)+1)
    pred[pred < 3] = 0
    pred[pred == 3] = 1
    pred[pred > 3] = 2
    prob /= prob.sum(axis=1)[:, None]
    ter_prob = np.zeros((len(pred), 3))
    ter_prob[:,0] = prob[:,0] + prob[:,1]
    ter_prob[:,1] = prob[:,2] 
    ter_prob[:,2] = prob[:,3] + prob[:,4]
    return [pred, np.max(ter_prob, axis=1)]
	
    
#To be used when logistic regression is trained only on two classes
def predict_binary(w, X_test):
    prob = np.transpose(np.dot(w, np.transpose(X_test)))
    pred = (np.argmax(prob, axis=1)+1)
    pred[pred < 3] = 0
    pred[pred >= 3] = 1
    prob /= prob.sum(axis=1)[:, None]
    bin_prob = np.zeros((len(pred), 2))
    bin_prob[:,0] = prob[:,0] + prob[:,1]
    bin_prob[:,1] = prob[:,2] + prob[:,3] + prob[:,4]
    return [pred, np.max(bin_prob, axis=1)]
	


