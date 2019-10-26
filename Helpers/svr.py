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

yE = yEl.reshape((len(yEl), 1))

scalledValues = dataConverter.featureScalling([XEl], [yE])

scalledXValues = scalledValues[0]
scalledYValues = scalledValues[1]

scalledX = scalledXValues['x_0']
scalledY = scalledYValues['y_0']

xScaller = scalledXValues['x_scaller']
yScaller = scalledYValues['y_scaller']

regression = Regressor()
visualisation = Visualisator()

dataLabel = "Data"

predictSVRValues = regression.SVRModelCreator(scalledX, scalledY)

visualisation.visualisePrediction(XEl, dataLabel, scalledY, predictSVRValues, 'SVR', ['green', 'blue'], 'Truth or Bluff (SVR)', 'Position level', 'Salary')




