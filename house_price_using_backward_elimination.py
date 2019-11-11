# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:58:38 2019

@author: vaseem
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing datasets using pandas
dataset = pd.read_csv('kc_house_data.csv')

#Droping the date and id column
dataset = dataset.drop(['id', 'date'], axis = 1)

#Seperating indepandent variable and depandant variable
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#spliting datasets into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting result
y_pred = regressor.predict(X_test)

#optimal model using Backward Elimination

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVars = max(regressor_OLS.pvalues).astype(float)
        if maxVars > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVars):
                    np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_Modeled = backwardElimination(X_opt, SL)
























