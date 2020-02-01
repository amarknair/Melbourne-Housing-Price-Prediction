#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: amar.kelunair
"""
import os
dirname = os.path.dirname(__file__)

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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