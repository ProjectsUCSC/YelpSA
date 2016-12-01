from pre_processing import *
from sklearn import svm



from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
pred = regr.predict(X_test)