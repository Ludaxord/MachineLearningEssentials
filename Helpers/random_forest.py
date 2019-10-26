# -*- coding: utf-8 -*-
import pandas as pd
from framework.regressor import Regressor
from framework.visualisator import Visualisator
from framework.data_converter import DataConverter
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('datasets/Position_Salaries.csv')
XEl = dataset.iloc[:, 1:2].values #always Matrix
yEl = dataset.iloc[:, 2].values   #always Vector

dataConverter = DataConverter()
regression = Regressor()
visualisation = Visualisator()

XPred = regression.RandomForestRegressionModelCreator(XEl, yEl, nEstimators = 300)

xxxxx = regression.RandomForestRegressionModelCreator(XEl, yEl, [[6.5]], nEstimators = 300)

XGrid = dataConverter.createGrid(XEl, dType = 0.01, columns = 1)

XXGrid = regression.RandomForestRegressionModelCreator(XEl, yEl, XGrid, nEstimators = 300)

dataLabel = "Data"

visualisation.visualisePrediction(XEl, dataLabel, yEl, XXGrid, 'Random Forest Regression', ['green', 'blue'], 'Truth or Bluff (Random Forest Regression)', 'Position level', 'Salary', anotherXPlot = XGrid)

