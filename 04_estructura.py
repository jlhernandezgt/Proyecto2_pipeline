# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:57:41 2022

@author: luish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score  
import time



import funciones_proyecto as fn

#os.chdir('/Users/luish/Documents/Maestria/trimestre2/StatisticalLearning/proyecto2/Proyecto2')

data = pd.read_csv("churn.csv")

data.head()

col_numerics = fn.getContinuesCols(data)
col_categoricals = fn.select_categorical_cols(data)


data[col_numerics].isnull().mean()

numeric_data = data[col_numerics]
categorical_data = data[col_categoricals]

numeric_data[col_numerics].isnull().mean()

numeric_data.head()

for col in col_numerics:
    fn.plot_density_variable(numeric_data, col) 

fn.FillNaN_Corr_DF(numeric_data, 'points_in_wallet', 'age' )
numeric_data[col_numerics].isnull().mean()

dataset_final = fn.funcion_final(numeric_data, 'avg_transaction_value','age', 1.75)




  
fn.balanceo_datos(data, 'gender')    
data_balanceada = data  

#data_balanceada[data_balanceada['gender'] == 'Unknown'] = 'M'

fn.balanceo_datos(data_balanceada, 'gender')  



### funcion de reproceso de data  ---- por confirmar
#proceso de balanceo de data.
nMasculino = len(data_balanceada[data_balanceada['gender'] == "M"])
Masculino = data_balanceada[data_balanceada['gender'] == "M"]
Femenino = data_balanceada[data_balanceada['gender'] == "F"]
Femenino = Femenino.sample(nMasculino, random_state=1234, replace=True)
data_balanceada = Femenino.append(Masculino)
data_balanceada = data_balanceada.sample(frac=1, random_state=1234)
data_balanceada




fn.FillNaN_Corr_DF(data_balanceada, 'points_in_wallet', 'age' )
#X = data_balanceada[['avg_transaction_value', 'points_in_wallet']]
X = data_balanceada[['avg_transaction_value']]
y = data_balanceada['gender']

#Ingeniería de caracteristicas - Codificación del Target.
lableEncoder = LabelEncoder()
lableEncoder.fit(['M', 'F'])
y = lableEncoder.transform(y.values)


#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle=True, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle=True, random_state=1234)

modelo = SVC(C = 100, kernel = 'linear', random_state=123)
modelo.fit(X_train, y_train)

start = time.time()
y_preds_svm = fn.modelo_svm(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")

print("Accuracy: ", accuracy_score(y_test, y_preds_svm))
     
conf_matrix = pd.crosstab(y_test, y_preds_svm, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_svm(conf_matrix)




start = time.time()
y_preds_nb = fn.modelo_naive_bayes(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")


print("Accuracy: ", accuracy_score(y_test, y_preds_nb))

conf_matrix = pd.crosstab(y_test, y_preds_nb, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)    

fn.validacion_nb(conf_matrix)




start = time.time()
y_preds_tree = fn.modelo_arbol_decision(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")

print("Accuracy: ", accuracy_score(y_test, y_preds_tree))

conf_matrix = pd.crosstab(y_test, y_preds_tree, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_dt(conf_matrix)



start = time.time()
y_preds_knn = fn.modelo_knn(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")

print("Accuracy: ", accuracy_score(y_test, y_preds_knn))

conf_matrix = pd.crosstab(y_test, y_preds_knn, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_knn(conf_matrix)




fn.print_roc(y_test,y_preds_svm,y_preds_nb,y_preds_tree,y_preds_knn)


svm_prob, svm_prob_v, _ = roc_curve(y_test, y_preds_svm)
nb_prob, nb_prob_v, _ = roc_curve(y_test, y_preds_nb)
tree_prob, tree_prob_v, _ = roc_curve(y_test, y_preds_tree)
knn_prob, knn_prob_v, _ = roc_curve(y_test, y_preds_knn)

plt.plot(svm_prob, svm_prob_v, linestyle="--", label="SVM")
plt.plot(nb_prob, svm_prob_v, marker='.', label="NB")
plt.plot(tree_prob, svm_prob_v, marker='.', label="tree")
plt.plot(knn_prob, knn_prob_v, marker='.', label="KNN")

plt.title("ROC Plot")
plt.xlabel("False Posotive Rate")
plt.ylabel("True Posotive Rate")
plt.legend()
plt.show()



