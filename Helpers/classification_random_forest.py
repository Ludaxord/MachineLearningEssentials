
import pandas as pd
from framework.regressor import Regressor
from framework.visualisator import Visualisator
from framework.data_converter import DataConverter

# Importing the dataset
dataset = pd.read_csv('datasets/Social_Network_Ads.csv')
XEl = dataset.iloc[:, [2,3]].values #always Matrix
yEl = dataset.iloc[:, 4].values   #always Vector

dataConverter = DataConverter()
regression = Regressor()
visualisation = Visualisator()

XTrain, XTest, yTrain, yTest = dataConverter.splittingDataSetToTrainingSetAndTestSet(XEl, yEl, testSize = 0.25)

scalling = dataConverter.featureScalling([XTrain, XTest], [])[0]

scaller = scalling['x_scaller']
XTrain = scalling['x_0']
XTest = scalling['x_1']

logistics = regression.RandomForestClassificationModelCreator(XTrain, yTrain, valueToPredict = XTest)
yPred = logistics[0]
classifier = logistics[1]

confusionMatrix = dataConverter.errorRateMatrix(yTest, yPred)

visualisation.visualiseColoredMap(XTrain, yTrain, ['red', 'green'], 'Random Forest Classifier (Training set)', 'Age', 'Estimated Salary', classifier = classifier)
visualisation.visualiseColoredMap(XTest, yTest, ['blue', 'gray'], 'Random Forest Classifier (Test set)', 'Age', 'Estimated Salary', classifier = classifier)


