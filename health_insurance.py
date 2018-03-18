# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:49:10 2018

@author: Rinky
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('insurance.csv')

X = dataset.iloc[:,0:6].values
y = dataset.iloc[:,[6]].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:, 1])


# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,4] = labelencoder_X.fit_transform(X[:, 4])

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,5] = labelencoder_X.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#creating a column in the beginning with all ones i.e x0 values
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1338,1)).astype(int),values = X, axis=1) 

#Applying backward elimination
X_opt = X[:,[0,1,2,3,4,5,6,7,8]]
regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4,6,7,8]]
regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,2,3,4,6,7,8]]
regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,2,4,6,7,8]]
regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,4,6,7,8]]
regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()
regressor_OLS.summary()

#This gives us the most optimum model
X_opt = X[:,[0,2,3,4,6,7,8]]

#Applying the model again
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X_opt, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train,y_train)

y_pred2 = regressor1.predict(X_test)

