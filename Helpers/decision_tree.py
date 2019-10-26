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

XPred = regression.DecisionTreeRegressionModelCreator(XEl, yEl)

XGrid = dataConverter.createGrid(XEl, dType = 0.01, columns = 1)

XXGrid = regression.DecisionTreeRegressionModelCreator(XEl, yEl, XGrid)

dataLabel = "Data"

visualisation.visualisePrediction(XEl, dataLabel, yEl, XXGrid, 'Decistion Tree', ['green', 'blue'], 'Truth or Bluff (Decistion Tree Regression)', 'Position level', 'Salary', anotherXPlot = XGrid)

