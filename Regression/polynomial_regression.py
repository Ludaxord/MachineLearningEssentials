# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #always Matrix
y = dataset.iloc[:, 2].values   #always Vector

# Fitting Linear regression model to the dataset

from sklearn.linear_model import LinearRegression

linearRegressor = LinearRegression()
linearRegressor.fit(X = X, y = y)

from sklearn.preprocessing import PolynomialFeatures

polynomialRegressor = PolynomialFeatures(degree = 2)
X_poly = polynomialRegressor.fit_transform(X)

linearRegressor_2 = LinearRegression()
linearRegressor_2.fit(X_poly, y)

#Polynomial Regression with higher degree == 3
polynomialRegressorHigherDegree = PolynomialFeatures(degree = 3)
X_poly_higher_degree = polynomialRegressorHigherDegree.fit_transform(X)

linearRegressor_3 = LinearRegression()
linearRegressor_3.fit(X_poly_higher_degree, y)

#Polynomial Regression with higher degree == 4
polynomialRegressorHigherDegree4 = PolynomialFeatures(degree = 4)
X_poly_higher_degree_4 = polynomialRegressorHigherDegree4.fit_transform(X)

linearRegressor_4 = LinearRegression()
linearRegressor_4.fit(X_poly_higher_degree_4, y)

# Predicting X values in linearRegression
X_pred = linearRegressor.predict(X)

# Predicting X values in polynomialRegression
X_poly_for_pred = polynomialRegressor.fit_transform(X)
X_poly_pred = linearRegressor_2.predict(X_poly_for_pred)

# Predicting X values in polynomialRegression with higher degree
X_poly_for_pred_higher_degree = polynomialRegressorHigherDegree.fit_transform(X)
X_poly_pred_higher_degree = linearRegressor_3.predict(X_poly_for_pred_higher_degree)

# Visualising Linear Regression results

plt.scatter(X, y, color = 'green')

plt.plot(X, X_pred, color = 'blue')

plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Visualising Polynomial Regression results

plt.scatter(X, y, color = 'green')

plt.plot(X, X_poly_pred_higher_degree, color = 'cyan')

plt.title('Truth or Bluff (Polynomial Regression with higer degree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Visualising Polynomial Regression results

plt.scatter(X, y, color = 'green')

plt.plot(X, X_poly_pred, color = 'red')

plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Visualising Polynomial Regression and Linear Regression results

plt.scatter(X, y, color = 'green')

plt.plot(X, X_poly_pred, color = 'red')
plt.plot(X, X_pred, color = 'blue')
plt.plot(X, X_poly_pred_higher_degree, color = 'cyan')

plt.title('Truth or Bluff (Polynomial Regression and Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

#Predict new results with Linear Regression

OnePred = linearRegressor.predict([[6.5]])
onePredPolyTransform = polynomialRegressorHigherDegree4.fit_transform([[6.5]])
OnePredPoly = linearRegressor_4.predict(onePredPolyTransform)
