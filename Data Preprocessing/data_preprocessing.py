#Quick Code tutorial how to implement basic machine learning:
# Data Preprocessing

#lesson 1

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets

dataset = pd.read_csv('Data.csv')

print(f'Data.csv \n {dataset}')

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 3].values

print(f'x => \n {x}')

print(f'y => \n {y}')

#lesson 2

# Taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(f'missing_data x => \n {x}')

#lesson 3

#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_x = LabelEncoder()

x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

print(f'categorical_data x => \n {x}')

one_hot_encoder = OneHotEncoder(categorical_features = [0])

x = one_hot_encoder.fit_transform(x).toarray()

print(f'categorical_data x => \n {x}')

label_encoder_y = LabelEncoder()

y = label_encoder_y.fit_transform(y)

#lesson 4

# Splitting dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(f'splitting_data_sets x_train => \n {x_train}')

print(f'splitting_data_sets y_train => \n {y_train}')

print(f'splitting_data_sets x_test => \n {x_test}')

print(f'splitting_data_sets y_test => \n {y_test}')


#lesson 5

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print(f'scalled variables x_train => \n {x_train}')
print(f'scalled variables x_test => \n {x_test}')

