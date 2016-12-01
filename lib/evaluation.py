import pre_processing as pp
from sklearn import svm
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import datasets, linear_model

import copy
from wordcloud import WordCloud

[preprocessed_data_sample, labels_sample, vectorizer, no_features, samples, unvectorized_data_sample] = pp.preprocess()

#Dividing samples into train and test
train_plus_test = samples#20000#(train_length + test_length) * 3#20000
ratio = int(train_plus_test * 0.7)

X_train = preprocessed_data_sample[0:ratio]
y_train = labels_sample[0:ratio]
X_test = preprocessed_data_sample[ratio:train_plus_test]
y_test = labels_sample[ratio:train_plus_test]


no_classes = 5

y_test_thresholded = np.array(copy.deepcopy(y_test))
y_test_thresholded[y_test_thresholded < 3] = 0
y_test_thresholded[y_test_thresholded == 3] = 1
y_test_thresholded[y_test_thresholded > 3] = 2

y_test_thresholded_binary = np.array(copy.deepcopy(y_test))
y_test_thresholded_binary[y_test_thresholded_binary < 3] = 0
y_test_thresholded_binary[y_test_thresholded_binary >= 3] = 1


#Word Cloud
print "Word Cloud for training data"
wordcloud = WordCloud().generate(''.join(unvectorized_data_sample))
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

print "Linear Regression"
#Linear Regression
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
pred = regr.predict(X_test)
pred = pred + 0.5
pred = pred.astype('int32')
print sum(pred==y_test) * 100.0 / len(y_test)
print 'Ternary Linear Regression'
pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
print 'Binary'
pred[pred == 2] = 1
print sum(pred==y_test_thresholded_binary) * 100.0 / len(y_test_thresholded_binary)

#Logistic Regression
print "logistic Regression"
from sklearn.linear_model import LogisticRegression
logregr = LogisticRegression()
logregr.fit(X_train, y_train)
pred = logregr.predict(X_test)
pred = pred + 0.5
pred = pred.astype('int32')
print sum(pred==y_test) * 100.0 / len(y_test)
print"Ternary"
pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
print "Binary"
pred[pred == 2] = 1
print sum(pred==y_test_thresholded_binary) * 100.0 / len(y_test_thresholded_binary)

print "Naive bayes"
#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train, y_train)
# make predictions
pred = nb.predict(X_test)
print sum(pred==y_test) * 100.0 / len(y_test)
print"Ternary"
pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
print "Binary"
pred[pred == 2] = 1
print sum(pred==y_test_thresholded_binary) * 100.0 / len(y_test_thresholded_binary)


print "KNN"
#K-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# make predictions
pred = knn.predict(X_test)
print sum(pred==y_test) * 100.0 / len(y_test)
print"Ternary"
pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
print "Binary"
pred[pred == 2] = 1
print sum(pred==y_test_thresholded_binary) * 100.0 / len(y_test_thresholded_binary)

print "CART"
#Classification and Regression Trees(CART)
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
cart = DecisionTreeClassifier()
cart.fit(X_train, y_train)
# make predictions
pred = cart.predict(X_test)
print sum(pred==y_test) * 100.0 / len(y_test)
print"Ternary"
pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
print "Binary"
pred[pred == 2] = 1
print sum(pred==y_test_thresholded_binary) * 100.0 / len(y_test_thresholded_binary)

print "Support Vector Machines"
#Support Vector Machines
from sklearn.svm import SVC
# fit a SVM model to the data
svm = SVC()
svm.fit(X_train, y_train)
print(svm)
# make predictions
pred = svm.predict(X_test)
print pred[1:100]
print sum(pred==y_test) * 100.0 / len(y_test)
print"Ternary"
pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
print "Binary"
pred[pred == 2] = 1
print sum(pred==y_test_thresholded_binary) * 100.0 / len(y_test_thresholded_binary)
