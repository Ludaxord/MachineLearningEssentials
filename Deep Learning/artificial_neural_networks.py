# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import warnings
import os

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.simplefilter(action='ignore', category=FutureWarning)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

label_encoder_x_1 = LabelEncoder()
X[:, 1] = label_encoder_x_1.fit_transform(X[:, 1])
label_encoder_x_2 = LabelEncoder()
X[:, 2] = label_encoder_x_2.fit_transform(X[:, 2])

# Deprecated

# one_hot_encoder = OneHotEncoder(categorical_features=[1])
# Xx = one_hot_encoder.fit_transform(X).toarray()
# Xx = Xx[:, 1:]

# Actual

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
# first layer
classifier.add(Dense(units=int((11 + 1) / 2), kernel_initializer='uniform', activation='relu', input_dim=11))
# second layer
classifier.add(Dense(units=int((11 + 1) / 2), kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
