import pandas as pd
import numpy
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
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
del predictors['min_dim']
del predictors['volume']

index = predictors.columns

correlation = predictors.corr()
writer1 = pd.ExcelWriter('PearsonCorrelation.xlsx')
correlation.to_excel(writer1, 'correlation')
writer1.save()

predictors_colonne=predictors.loc[:,['ASA','NASA']]
writer2 = pd.ExcelWriter('predictors_colonne.xlsx')
predictors_colonne.to_excel(writer2, 'predictors_colonne')
writer2.save()

predictors['ASA'].replace(0, numpy.nan, inplace=True)
#devo capire perchè non funziona il comando, perchè le colonne le seleziona, puoi vederlo in predictors_colonne di excel
#solo che poi non fa l'operazione di sostituire gli zeri con NaN
#predictors.loc[:,['ASA','NASA']].replace(0, numpy.nan, inplace=True)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(predictors)
predictors = imp.transform(predictors)

predictors = pd.DataFrame(data=predictors, columns=index)

predictors['NASA'].replace(0, numpy.nan, inplace=True)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(predictors)
predictors = imp.transform(predictors)

predictors = pd.DataFrame(data=predictors, columns=index)

predictors['SiOSi_var']=pd.DataFrame(numpy.log(predictors['SiOSi_var']))
predictors['SiO_var']=pd.DataFrame(numpy.log(predictors['SiO_var']))
predictors['ASA']=pd.DataFrame(numpy.log(predictors['ASA']))
predictors['NASA']=pd.DataFrame(numpy.log(predictors['NASA']))

#da verificare cosa sto facendo esattamente con questa roba
scaler=StandardScaler()
predictors_scaled=scaler.fit_transform(predictors)
predictors_scaled=pd.DataFrame(data=predictors_scaled, columns=index)
#se faccio questo la varianza diventa tutta 1
#esistono molti tipi di scaling, c'è il MaxMin o c'è lo scale (che fa una distribuzione normale)

writer3 = pd.ExcelWriter('Predictors_new.xlsx')
predictors.to_excel(writer3, 'predictors_excel')
writer3.save()

writer4 = pd.ExcelWriter('Predictors_scaled.xlsx')
predictors_scaled.to_excel(writer4, 'predictors_scaled')
writer4.save()

print(predictors.var())
print(predictors_scaled.var())

#print(predictors.max(axis=0,skipna=False,numeric_only=True))
#print(predictors.min(axis=0,skipna=False,numeric_only=True))
#print(predictors.max(axis=0,skipna=False,numeric_only=True)/predictors.min(axis=0,skipna=False,numeric_only=True))

stat=predictors.describe()
print(stat)
writer5 = pd.ExcelWriter('Predictors_stat.xlsx')
stat.to_excel(writer5, 'predictors_stat')
writer5.save()

correlation=predictors_scaled.corr()
writer6 = pd.ExcelWriter('Predictors_scaled_corr.xlsx')
correlation.to_excel(writer6, 'predictors_corr')
writer6.save()
#vediamo che alcune colonne hanno magari varianza quasi normale ma hanno dei valori completamente out of range
#cercare in letteratura come pulirli con metodi statistici automatici

#andrebbero processati anche gli output penso
indextarget=target.columns
target_scaled=scaler.fit_transform(target)
target_scaled=pd.DataFrame(data=target_scaled, columns=indextarget)

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
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1))

Early_Stopping_Monitor = EarlyStopping(patience=2)
#model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
model.compile(optimizer="adam", loss="mean_squared_error")
#model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks=[Early_Stopping_Monitor])
model.fit(predictors_scaled, target_scaled, validation_split=0.25, epochs=20)
#a=[0.3,0.2,0.23,0.24,0.27,0.25,0.29,0.26,0.21]
#for num in a:
#   model.fit(predictors, target, validation_split=num, epochs=1)

n_folds = 6
#data, labels, header_info = load_data()
skf = StratifiedKFold(n_splits=n_folds, shuffle=True,random_state=16457)
#(train,test)=skf.split(predictors,target)
#for i, (train, test) in enumerate(skf):
  #  print "Running Fold", i+1, "/" # n_folds
  #  model = None # Clearing the NN.
   # model = create_model()
   # train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test))


#random_state=15475
#rkf=RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
#for num in rkf.split(predictors,target):
    #model.fit(num, target, epochs=2)
   # a=1
#regression=svm.LinearSVR()
#score=cross_val_score(regression,predictors,target,cv=10)
#print(score)