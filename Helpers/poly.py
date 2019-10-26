#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:05:42 2019

@author: konraduciechowski
"""

import pandas as pd
from framework.regressor import Regressor
from framework.visualisator import Visualisator

# Importing the dataset
dataset = pd.read_csv('datasets/Position_Salaries.csv')
XEl = dataset.iloc[:, 1:2].values #always Matrix
yEl = dataset.iloc[:, 2].values   #always Vector

regression = Regressor()
visualisation = Visualisator()

X_linear_pred_El = regression.linearRegressionModelCreator(XEl, yEl)
X_poly_pred_el_2 = regression.polynomialRegressionModelCreator(XEl, yEl, 2)
X_poly_pred_el_3 = regression.polynomialRegressionModelCreator(XEl, yEl, 3)
X_poly_pred_el_4 = regression.polynomialRegressionModelCreator(XEl, yEl, 4)
X_poly_pred_el_5 = regression.polynomialRegressionModelCreator(XEl, yEl, 5)

#Predict one Value
predictLinearValue = regression.linearRegressionModelCreator(XEl, yEl, 6.5)
predictPolynomialValue = regression.polynomialRegressionModelCreator(XEl, yEl, 4, 6.5)

dataLabel = "Data"

plots = [X_linear_pred_El, X_poly_pred_el_2, X_poly_pred_el_3, X_poly_pred_el_4, X_poly_pred_el_5]
colors = ['blue', 'red', 'cyan', 'purple', 'yellow']
labels = ['linearRegression', 'polynomialRegression degree = 2', 'polynomialRegression degree = 3', 'polynomialRegression degree = 4', 'polynomialRegression degree = 5']

visualisation.visualisePredictions(XEl, dataLabel, yEl, plots, labels, 'green', colors, 'Truth or Bluff (Polynomial Regression and Linear Regression)', 'Position level', 'Salary')

visualisation.visualisePrediction(XEl, dataLabel, yEl, X_linear_pred_El, 'linearRegression', ['green', 'blue'], 'Truth or Bluff (Linear Regression)', 'Position level', 'Salary')
visualisation.visualisePrediction(XEl, dataLabel, yEl, X_poly_pred_el_2, 'polynomialRegression degree = 2', ['green', 'red'], 'Truth or Bluff (Polynomial Regression degree == 2)', 'Position level', 'Salary')
visualisation.visualisePrediction(XEl, dataLabel, yEl, X_poly_pred_el_3, 'polynomialRegression degree = 3', ['green', 'cyan'], 'Truth or Bluff (Polynomial Regression degree == 3)', 'Position level', 'Salary')
visualisation.visualisePrediction(XEl, dataLabel, yEl, X_poly_pred_el_4, 'polynomialRegression degree = 4', ['green', 'purple'], 'Truth or Bluff (Polynomial Regression degree == 4)', 'Position level', 'Salary')
visualisation.visualisePrediction(XEl, dataLabel, yEl, X_poly_pred_el_5, 'polynomialRegression degree = 5', ['green', 'yellow'], 'Truth or Bluff (Polynomial Regression degree == 5)', 'Position level', 'Salary')
