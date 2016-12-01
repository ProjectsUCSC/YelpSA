import numpy as np
import copy
def train(X, y, no_classes):
    no_features = len(X[0])
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
    return w

def prob_calc(X_test,w):
	no_features = len(X_test[0])
    prob =  np.ones((len(X_test),5))
    for index in range(5):#5 classes
        for review in range(len(X_test)): #no of test reviews
            for feature in range(no_features): #features
                if(X_test[review][feature] == 0.0):
                    continue
                else:
                    prob[review][index] = prob[review][index] * (w[index][feature] * X_test[review][feature] )
    
    pclass1=float(sum(y_train==1))/len(y_train)
    pclass2=float(sum(y_train==2))/len(y_train) 
    pclass3=float(sum(y_train==3))/len(y_train) 
    pclass4=float(sum(y_train==4))/len(y_train) 
    pclass5=float(sum(y_train==5))/len(y_train)

    for i in range(len(prob)):
        np.sum(prob[i][:])
        prob[i][0]= (prob[i][0]*pclass1)
        prob[i][1]= (prob[i][1]*pclass2)
        prob[i][2]= (prob[i][2]*pclass3)
        prob[i][3]= (prob[i][3]*pclass4)
        prob[i][4]= (prob[i][4]*pclass5)
        
    return prob

def predict(y_test,X_test,w):
    prob=prob_calc(X_test,w)
    pred=np.zeros(len(y_test))
    for i in range(len(prob)):
        pred[i]= prob[i][:].argmax(axis=0)+1
        
    prob_normal=np.zeros((len(prob),1))
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
    return pred,prob_normal

def pred_ternary(y_test):
    print y_test[0:5]
    y_test_thresholded = np.array(copy.deepcopy(y_test))
    y_test_thresholded[y_test_thresholded < 3] = 0
    y_test_thresholded[y_test_thresholded == 3] = 1
    y_test_thresholded[y_test_thresholded > 3] = 2
    pred=predict(y_test,X_test,w)
    pred[pred < 3] = 0
    pred[pred == 3] = 1
    pred[pred > 3] = 2
    accuracy= sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
    return accuracy,pred,y_test_thresholded

def pred_binary(y_test):
    
    print y_test[0:5]
    y_test_thresholded = np.array(copy.deepcopy(y_test))
    y_test_thresholded[y_test_thresholded < 3] = 0
    y_test_thresholded[y_test_thresholded == 3] = 1
    y_test_thresholded[y_test_thresholded > 3] = 1
    pred,prob_normal=predict(y_test,X_test,w)
    pred[pred < 3] = 0
    pred[pred == 3] = 1
    pred[pred > 3] = 1
    accuracy= sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
    return accuracy,pred,y_test_thresholded





