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

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)
melb_model = DecisionTreeRegressor()
melb_model.fit(train_X, train_y)

val_predictions = melb_model.predict(val_X)

print "Mean Absolute Error = " + str(mean_absolute_error(val_predictions, val_y))

#Comparing the accuracy of models built with different values for max_leaf_nodes.
def get_mae(max_leafnodes):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leafnodes, random_state=0)
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    mae = mean_absolute_error(pred, val_y)
    return mae
    
leaf = range(100, 2000, 50)
min_error = float('inf')
optimal_leaf = 0
MAE=[]
for max_leafnodes in leaf:
    mae = get_mae(max_leafnodes)
    MAE.append(mae)
    print("Max leaf nodes: %d \t\t MAE: %d"%(max_leafnodes, mae))
    if min_error>mae:
        min_error = mae
        optimal_leaf = max_leafnodes
print("Optimal leaf = %d forr minimum error = %d"%(optimal_leaf, min_error))

from matplotlib import pyplot

pyplot.plot(leaf,MAE)