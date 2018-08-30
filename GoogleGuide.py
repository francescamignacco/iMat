import pandas as pd
import numpy
import csv
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

# DataFrame, which you can imagine as a relational data table, with rows and named columns
# Series, which is a single column. A DataFrame contains one or more Series and a name for each Series.

predictors = pd.read_csv('iza.csv', sep = ";")
del predictors['name']
target = pd.read_csv('result.csv', sep=';')

n_cols = predictors.shape[1]

model = Sequential()
model.add(Dense(50, activation="relu", input_shape=(n_cols,)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(predictors, target)


