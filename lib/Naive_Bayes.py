
# coding: utf-8

# In[1]:

import numpy as np


# In[ ]:

def nbtrain_w(X, y, no_classes, no_features):
   
    w = np.zeros((no_classes, no_features))
    #index = 0
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    index=0
    for i in y:
        if i==1:
            class1.append(index)
        elif i==2:
            class2.append(index)
        elif i==3:
            class3.append(index)
        elif i==4:
            class4.append(index)
        else:
            class5.append(index)
        index=index+1
            
    labels = np.array([class1,class2,class3,class4,class5])
    #print X[labels[0]]
    total_class1= sum(sum(X[labels[0]]))
    #print total_class1
    total_class2= sum(sum(X[labels[1]]))
    total_class3= sum(sum(X[labels[2]]))
    #print total_class2
    total_class4= sum(sum(X[labels[3]]))
    total_class5= sum(sum(X[labels[4]]))
  
    #print np.sum(X[labels[0],:])
    for i in range(no_features):
        sum_freq1 = np.sum(X[labels[0],i])
        sum_freq2 = np.sum(X[labels[1],i])
        sum_freq3 = np.sum(X[labels[2],i])
        sum_freq4 = np.sum(X[labels[3],i])
        sum_freq5 = np.sum(X[labels[4],i])
        w[0][i] = (sum_freq1 + 1.0) /   (total_class1+no_features+1)
        w[1][i] = (sum_freq2 + 1.0) /   (total_class2+no_features+1)
        w[2][i] = (sum_freq3 + 1.0) /   (total_class3+no_features+1)
        w[3][i] = (sum_freq4 + 1.0) /  (total_class4+no_features+1)
        w[4][i] = (sum_freq5 + 1.0) /  (total_class5+no_features+1)
    return w


# In[ ]:

no_classes = 5
no_features = 942

w = nbtrain_w(X_train, y_train, no_classes, no_features)
print w


# In[ ]:

def nbtest(X_test,w):
    prob =  np.ones((len(X_test),5))
#print len(X_test)
#print prob
    for index in range(5):#5 classes
        for review in range(len(X_test)): #no of test reviews
            for feature in range(no_features): #features
                if(X_test[review][feature] == 0.0):
                    #prob[review][index] = prob[review][index] * (1-w[index][feature])
                    continue
                else:
                    prob[review][index] = prob[review][index] * (w[index][feature] * X_test[review][feature] )
    
# prob has 6000 rows and 2 cols
# test_data_vector has 6000 rows and features cols
# w has 2 rows has 2000 cols
    #print prob 
    pclass1=float(sum(y_train==1))/len(y_train)
    #print pclass1
    pclass2=float(sum(y_train==2))/len(y_train) 
    #print pclass2
    pclass3=float(sum(y_train==3))/len(y_train) 
    #print pclass3
    pclass4=float(sum(y_train==4))/len(y_train) 
    #print pclass4
    pclass5=float(sum(y_train==5))/len(y_train)
    #print pclass5
    #print len(prob)

    for i in range(len(prob)):
        np.sum(prob[i][:])
        prob[i][0]= (prob[i][0]*pclass1)
        prob[i][1]= (prob[i][1]*pclass2)
        prob[i][2]= (prob[i][2]*pclass3)
        prob[i][3]= (prob[i][3]*pclass4)
        prob[i][4]= (prob[i][4]*pclass5)
        
    return prob


# In[ ]:

def nbpred(y_test,X_test,w):
    prob=nbtest(X_test,w)
    pred=np.zeros(len(y_test))
    for i in range(len(prob)):
        pred[i]= prob[i][:].argmax(axis=0)+1
        
    #import math
    prob_normal=np.zeros((len(prob),1))
    #print prob
    inf = float("inf")
    #print inf
    for i in range(len(prob)):
        normalizer=np.sum(prob[i][:])
        if prob[i][pred[i]-1]== inf or prob[i][pred[i]-1]== -inf :
            prob_normal[i]=1
        else:
            prob_normal[i]=prob[i][pred[i]-1]/normalizer
    
    for i in range(len(prob)):
        if math.isnan(prob_normal[i]):
            prob_normal[i]=0
    #print prob_normal[1:100]   
    return pred,prob_normal


# In[ ]:

from sklearn.metrics import confusion_matrix
pred,prob_normal=nbpred(y_test,X_test,w)
c = confusion_matrix(y_test,pred)
print c
accuracy=sum(pred==y_test) * 100.0 / len(y_test)
print accuracy


# In[ ]:

def nbpred_ternary(y_test):
    #import copy
    print y_test[0:5]
    y_test_thresholded = np.array(copy.deepcopy(y_test))
    #print type(y_test_thresholded)
    #print y_test_thresholded
    y_test_thresholded[y_test_thresholded < 3] = 0
    y_test_thresholded[y_test_thresholded == 3] = 1
    y_test_thresholded[y_test_thresholded > 3] = 2
    #print  y_test_thresholded
    #print id(y_test)
    #print(y_test_thresholded)
    pred=nbpred(y_test,X_test,w)
    pred[pred < 3] = 0
    pred[pred == 3] = 1
    pred[pred > 3] = 2
    accuracy= sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
    return accuracy,pred,y_test_thresholded


# In[ ]:

accuracy,pred,y_test_thresholded=nbpred_ternary(y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(pred, y_test_thresholded)
print accuracy


# In[ ]:

def nbpred_binary(y_test):
    import copy
    print y_test[0:5]
    y_test_thresholded = np.array(copy.deepcopy(y_test))
    #print type(y_test_thresholded)
    #print y_test_thresholded
    y_test_thresholded[y_test_thresholded < 3] = 0
    y_test_thresholded[y_test_thresholded == 3] = 1
    y_test_thresholded[y_test_thresholded > 3] = 1
    #print  y_test_thresholded
    #print id(y_test)
    #print(y_test_thresholded)
    pred,prob_normal=nbpred(y_test,X_test,w)
    pred[pred < 3] = 0
    pred[pred == 3] = 1
    pred[pred > 3] = 1
    accuracy= sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
    return accuracy,pred,y_test_thresholded


# In[ ]:

accuracy1,pred1,y_test_thresholded1=nbpred_binary(y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(pred1, y_test_thresholded1)
print accuracy1

