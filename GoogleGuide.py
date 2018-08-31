import pandas as pd
import numpy
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
#import matplotlib.pyplot as plt

#IMPORTING DATAS
predictors = pd.read_csv('izacsv.csv', sep = ";")
target = pd.read_csv('result_giusti.csv', sep = ';')

#PREPROCESSING
del predictors['largest_free_sphere']
del predictors['mode_dim']
del predictors['name']
del predictors['spg']
del predictors['SiOSi_mean']
del predictors['SiOSi_hmean']
del predictors['SiOSi_gmean']
del predictors['SiOSi_min']
del predictors['SiOSi_max']
del predictors['SiOSi_std']
del predictors['SiO_mean']
del predictors['SiO_hmean']
del predictors['SiO_gmean']
del predictors['SiO_min']
del predictors['SiO_max']
del predictors['SiO_std']
del predictors['AV']
del predictors['VolFrac']
del predictors['largest_included_sphere_free']
del predictors['max_dim']

index = predictors.columns

correlation = predictors.corr()
writer = pd.ExcelWriter('PearsonCorrelation.xlsx')
correlation.to_excel(writer, 'correlation')
writer.save()

predictors['ASA'].replace(0, numpy.nan, inplace=True)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(predictors)
predictors = imp.transform(predictors)

predictors['NASA'].replace(0, numpy.nan, inplace=True)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(predictors)
predictors = imp.transform(predictors)

predictors = pd.DataFrame(data=predictors, columns=index)

#predictors.replace(',','.')
#predictors['density'] = predictors['density'].astype('float') #per cambiare type
##predictors.var() per calcolare la varianza delle colonne se troppa alta standardizza
##np.log(predictors["col2"]) log standardization
##scaler = StandardScaler(); scaled_predictors = pd.DataFrame(scaler.fit_transform(predictors),columns=predictors.columns)
##usare lo scaling quando la varianza non è diversissima tra le colonne pero i dati hanno grandezze diverse
##feature engineering per categorical data: users["sub_enc"] = users["subscribed"].apply(lambda val: 1 if val == "y" else 0)
##feature selection (predictors.corr()) per calcolare il coefficiente di Pearson tra le colonne

#MODEL IMPLEMENTATION
n_cols = predictors.shape[1]

model = Sequential()
model.add(Dense(100, activation="relu", input_shape=(n_cols,)))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

Early_Stopping_Monitor = EarlyStopping(patience=3)
#model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
model.compile(optimizer="adam", loss="mean_squared_error")
#model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks=[Early_Stopping_Monitor])
model.fit(predictors, target, validation_split=0.3, epochs=2)
