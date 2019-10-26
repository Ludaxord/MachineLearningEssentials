# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class DataConverter:
    
    # Get separate correct predictions from real values
    def errorRateMatrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm
    
    def scallerTransform(self, value, xScaller = StandardScaler()):
        return xScaller.transform(np.array([[value]]))
    
    # Encoding categorical data
    def encodingCategoricalData(self, X, categoricalDataPlace, categorical_features):
        labelencoder_X = LabelEncoder()
        X[:, categoricalDataPlace] = labelencoder_X.fit_transform(X[:, categoricalDataPlace])
        onehotencoder = OneHotEncoder(categorical_features = [3])
        X = onehotencoder.fit_transform(X).toarray()
        return X

    
    def createGrid(self, xMatr, dType = 0.1, columns = 1):
        XGrid = np.arange(min(xMatr), max(xMatr), dType)
        XGrid = XGrid.reshape((len(XGrid), columns))
        return XGrid

    # Splitting the dataset into the Training set and Test set
    def splittingDataSetToTrainingSetAndTestSet(self, X, y, testSize = 0.2, randomState = 0):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        l = list()
        l.append(X_train)
        l.append(X_test)
        l.append(y_train)
        l.append(y_test)
        return l
        
    # Invert Scaling
    def invertScalling(self, sc_y, y_pred):
        return sc_y.inverse_transform(y_pred)
    
    ### !IMPORTANT! Only when model is based on euclidean distance
    # Feature Scaling
    def featureScalling(self, xScallers, yScallers):
        
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        
        xDict = {}
        yDict = {}
        
        xDict['x_scaller'] = sc_X
        yDict['y_scaller'] = sc_y
        
        for index, x in enumerate(xScallers):
            x_transformed = sc_X.fit_transform(x)
            xDict[f'x_{index}'] = x_transformed
        for index, y in enumerate(yScallers):
            y_transformed = sc_y.fit_transform(y)
            yDict[f'y_{index}'] = y_transformed
                
        return [xDict, yDict]
            