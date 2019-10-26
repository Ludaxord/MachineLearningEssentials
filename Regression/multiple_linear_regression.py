# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

Xx = dataset.iloc[:, :-1].values
yY = dataset.iloc[:, 4].values

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
# In normal environment numpy take care of dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fit multiple linear regression model to dataset

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# prediction of the test set results

y_pred = regressor.predict(X_test)

# Matrix of array

columns_size = len(X)
rows_size = len(X[0]) + 1

list_of_size = list(range(rows_size))

# building the optimal model using backward elimination

import statsmodels.formula.api as sm

X = np.append(arr=np.ones((columns_size, 1)).astype(int), values=X, axis=1)

# highest => P > |t| ===> 0.990

X_opt = X[:, list_of_size]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

# removing high values of P > |t| till is smaller from SL (e.g. SL => 0,05)

# highest => P > |t| ===> 0.940

if (2 in list_of_size):
    list_of_size.remove(2)

X_opt = X[:, list_of_size]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

# highest => P > |t| ===> 0.602

if (1 in list_of_size):
    list_of_size.remove(1)

X_opt = X[:, list_of_size]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

# highest => P > |t| ===> 0.060

if (4 in list_of_size):
    list_of_size.remove(4)

X_opt = X[:, list_of_size]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

# highest => P > |t| ===> 0.000

if (5 in list_of_size):
    list_of_size.remove(5)

X_opt = X[:, list_of_size]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

# Proper version with automatic backward elimination
import statsmodels.formula.api as sm


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


# call of function
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
