import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Kidem_ve_Maas_VeriSeti.csv') ##Datasets


X = dataset.iloc[:,:-1].values ##Matris oluştur
y = dataset.iloc[:, 1].values   ##Vektör oluştur

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) ##Test


regressor = LinearRegression() 
regressor.fit(X_train, y_train) ##Train the model

plt.scatter(X_train, y_train, color = 'red') ##Renk seçimi
modelin_tahmin_ettigi_y = regressor.predict(X_train) ##Tahmin et
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'blue') ##Renk ve tahmin
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli') ##Başlıklar
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()
