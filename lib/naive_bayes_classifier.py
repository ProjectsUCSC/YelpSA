import numpy as np
import copy
import math

def train(X, y, no_classes):
    no_features = len(X[0])
    w = np.zeros((no_classes, no_features))
    #index = 0
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    
    pclass1=float(sum(y==1))/len(y)
    pclass2=float(sum(y==2))/len(y) 
    pclass3=float(sum(y==3))/len(y) 
    pclass4=float(sum(y==4))/len(y) 
    pclass5=float(sum(y==5))/len(y)
    prior = [pclass1, pclass2, pclass3, pclass4, pclass5]
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
    total_class1= sum(sum(X[labels[0]]))
    total_class2= sum(sum(X[labels[1]]))
    total_class3= sum(sum(X[labels[2]]))
    total_class4= sum(sum(X[labels[3]]))
    total_class5= sum(sum(X[labels[4]]))
  
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
    return w, prior

def prob_calc(X_test, w, prior):
    no_features = len(X_test[0])
    prob =  np.ones((len(X_test),5))
    for index in range(5):#5 classes
        for review in range(len(X_test)): #no of test reviews
            for feature in range(no_features): #features
                if(X_test[review][feature] == 0.0):
                    prob[review][index] = prob[review][index] * (1-w[index][feature]) 
                else:
                    prob[review][index] = prob[review][index] * (w[index][feature] * X_test[review][feature] )

    for i in range(len(prob)):
        np.sum(prob[i][:])
        prob[i][0]= (prob[i][0]*prior[0])
        prob[i][1]= (prob[i][1]*prior[1])
        prob[i][2]= (prob[i][2]*prior[2])
        prob[i][3]= (prob[i][3]*prior[3])
        prob[i][4]= (prob[i][4]*prior[4])
        
    return prob

def predict(X_test, w, prior):
    prob = prob_calc(X_test,w, prior)
    pred = np.zeros(len(X_test))
    pred = (np.argmax(prob, axis=1)+1)
        
    prob_normal = np.zeros((len(prob),1))
    inf = float("inf")
    for i in range(len(prob)):
        normalizer=np.sum(prob[i][:])
        if prob[i][pred[i]-1]== inf or prob[i][pred[i]-1]== -inf :
            prob_normal[i]=1
        else:
            prob_normal[i]=prob[i][pred[i]-1]/normalizer
    
    for i in range(len(prob)):
        if math.isnan(prob_normal[i]):
            prob_normal[i]=0
    
    return pred, prob_normal

def pred_ternary(X_test, w, prior):
    pred, prob = predict(X_test, w, prior)
    pred[pred < 3] = 0
    pred[pred == 3] = 1
    pred[pred > 3] = 2
    return [pred, prob]

def pred_binary(X_test, w, prior):
    
    pred, prob = predict(X_test, w, prior)
    pred[pred < 3] = 0
    pred[pred >= 3] = 1
    return [pred, prob]