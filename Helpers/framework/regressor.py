#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class Regressor: 
        
        
        # More elegant closed in function Decision Tree Regression model creator
    def RandomForestClassificationModelCreator(self, xMatr, yVector, valueToPredict = None, nEstimators = 10, criterion = "entropy", randomState = 0):
        regressor = RandomForestClassifier(n_estimators = nEstimators, criterion = criterion, random_state = randomState) 
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        l = list()
        l.append(predictVal)
        l.append(regressor)
        return l    
    
        # More elegant closed in function Decision Tree Regression model creator
    def DecisionTreeClassificationModelCreator(self, xMatr, yVector, valueToPredict = None, criterion = "entropy", randomState = 0):
        regressor = DecisionTreeClassifier(criterion = criterion ,random_state = randomState)
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        l = list()
        l.append(predictVal)
        l.append(regressor)
        return l
    
         # More elegant closed in function Naive Bayes model creator
    def NaiveBayesModelCreator(self, xMatr, yVector, valueToPredict = None):
        regressor = GaussianNB()
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        l = list()
        l.append(predictVal)
        l.append(regressor)
        return l
            
        # More elegant closed in function SVM model creator
    def SVMModelCreator(self, xMatr, yVector, valueToPredict = None, kernel = 'linear', randomState = 0):
        regressor = SVC(kernel = kernel, random_state = randomState)
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        l = list()
        l.append(predictVal)
        l.append(regressor)
        return l
    
        # More elegant closed in function KNearest Neighbors model creator
    def KNearestNeighborsModelCreator(self, xMatr, yVector, valueToPredict = None, nNeighbors = 5, metric = 'minkowski', p = 2):
        regressor = KNeighborsClassifier(n_neighbors = nNeighbors, metric = metric, p = p)
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        l = list()
        l.append(predictVal)
        l.append(regressor)
        return l
    
        # More elegant closed in function Logistic Regression model creator
    def LogisticRegressionModelCreator(self, xMatr, yVector, valueToPredict = None, randomState = 0):
        regressor = LogisticRegression(random_state = randomState) 
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        l = list()
        l.append(predictVal)
        l.append(regressor)
        return l
    
    
        # More elegant closed in function Decision Tree Regression model creator
    def RandomForestRegressionModelCreator(self, xMatr, yVector, valueToPredict = None, nEstimators = 10, randomState = 0):
        regressor = RandomForestRegressor(n_estimators = nEstimators, random_state = randomState) 
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        return predictVal
    
    
        # More elegant closed in function Decision Tree Regression model creator
    def DecisionTreeRegressionModelCreator(self, xMatr, yVector, valueToPredict = None, randomState = 0):
        regressor = DecisionTreeRegressor(random_state = randomState)
        regressor.fit(xMatr, yVector)
        if valueToPredict is None:
            predictVal = regressor.predict(xMatr)
        else:
            predictVal = regressor.predict(valueToPredict)
        return predictVal
    
        # More elegant closed in function SVR model creator
    def SVRModelCreator(self, xMatr, yVector, valueToPredict = None, typeOfKernel = 'rbf'):
        regressor = SVR(kernel = typeOfKernel)
        regressor.fit(X = xMatr, y = yVector)
        if valueToPredict is None:
            xSVR = regressor.predict(xMatr)
        else:
            xSVR = regressor.predict(valueToPredict)
        return xSVR

    # More elegant closed in function linearRegression model creator
    def linearRegressionModelCreator(self, xMatr, yVector, valueToPredict = None):
        regressor = LinearRegression()
        regressor.fit(X = xMatr, y = yVector)
        # Predicting X values in linearRegression
        if valueToPredict is None:
            xLinearPrediction = regressor.predict(xMatr)
        else:
            xLinearPrediction = regressor.predict([[valueToPredict]])
        return xLinearPrediction
    
    # More elegant closed in function polynomialRegression model creator
    def polynomialRegressionModelCreator(self, xMatr, yVector, degree = 2, valueToPredict = None):
        regressor = PolynomialFeatures(degree = degree)
        xPoly = regressor.fit_transform(xMatr)
        linRegressor = LinearRegression()
        linRegressor.fit(xPoly, yVector)
        if valueToPredict is None:
            xPolyPredictor = regressor.fit_transform(xMatr)
        else:
            xPolyPredictor = regressor.fit_transform([[valueToPredict]])
        xPolyPrediction = linRegressor.predict(xPolyPredictor)
        return xPolyPrediction
