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
    
    prior = []
    for i in range(no_classes):
        prior.append(float(sum(y==i+1))/len(y)) 

    index = 0
    labels = []
    for i in range(no_classes):
        labels.append([])
    #labels = [[]] * no_classes
    classes = np.array(range(no_classes)) + 1
    
    y = np.array(y)
    for i in range(len(y)):
        #print "######### printing for i : " , i , "  #########################"
        #print y[i]
        #print labels[y[i] -1]
        #print labels
        labels[y[i] - 1].append(i)
    
    labels = np.array(labels)
    

    #for i in range(no_features):
    for j in range(no_classes):
            #sum_freq = np.sum(X[labels[j],i])
            #w[j][i] = (sum_freq1 + 1.0) /   (total_class1+no_features+1)
        w[j, :] = (np.sum(X[labels[j]],axis=0) + 1.0) / (sum(sum(X[labels[j]])) + no_features)#total_class[j]

    return w, prior
    
def prob_calc(X_test, w, prior):
    no_classes = len(w)
    no_features = len(X_test[0])
    prob =  np.ones((len(X_test), no_classes))
    for index in range(no_classes):#5 classes
        for review in range(len(X_test)): #no of test reviews
            for feature in range(no_features): #features
                if(X_test[review][feature] == 0.0):
                    #continue
                    prob[review][index] = prob[review][index] * (1-w[index][feature]) 
                else:
                    prob[review][index] = prob[review][index] * (w[index][feature] * X_test[review][feature] )

    prob = np.multiply(prob, prior)
   
        
    return prob

def predict(X_test, w, prior):
    prob = prob_calc(X_test,w, prior)
    pred = np.zeros(len(X_test))
    print prob.shape
    pred = (np.argmax(prob, axis=1)+1)
    #for i in range(len(prob)):
        #pred[i] = prob[i][:].argmax(axis=0)+1
        
    prob_normal = np.zeros((len(prob),1))
    inf = float("inf")
    prob /= prob.sum(axis=1)[:, None]
    
   
    return pred, prob

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