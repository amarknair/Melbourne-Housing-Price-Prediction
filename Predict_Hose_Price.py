#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: amar.kelunair
"""
import os
dirname = os.path.dirname(__file__)

import pandas as pd

melb_data = pd.read_csv(dirname+'/input/melbourne-housing-snapshot/panda_test.csv')
print(melb_data.describe())
print(melb_data.columns)
melb_data.dropna(axis=0)

#Selecting Predicting target and choosing random features
y =melb_data.Price
melb_features = ['Rooms','Bathroom','Landsize']#,'Lattitude','Longtitude']
X = melb_data[melb_features]
X.describe()
X.head()

#Building Model using scikit-learn
from sklearn.tree import DecisionTreeRegressor
#define model
melb_model = DecisionTreeRegressor(random_state=1)

#fit model
melb_model.fit(X,y)

#printing predictions for a small subset
print("Predictions for the following 5 houses")
print (X.head())
print("Predictions are ")
print(melb_model.predict(X.head()))

#In-Sample validation score
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melb_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

#Model Validation using absolute mean error
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y,test_size=0.30, random_state=0)
melb_model = DecisionTreeRegressor()
melb_model.fit(train_X, train_y)

val_predictions = melb_model.predict(val_X)

print "Mean Absolute Error = " + str(mean_absolute_error(val_predictions, val_y))