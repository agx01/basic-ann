import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------------------------#
#Data PreProcessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Data PreProcessing ends
#---------------------------------------------------------------#

#Evaluating, Improving and Tuning the ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense
import timeit
#Build ANN function
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

start_time = timeit.default_timer()
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
elapsed = timeit.default_timer() - start_time

mean = accuracies.mean()
variance = accuracies.std()
#Improving the ANN


#Tuning the ANN


#Evaluating, Improving and Tuning the ANN ends