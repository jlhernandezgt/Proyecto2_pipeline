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
    fn.graficar_data_densidad(numeric_data, col) 


dataset_final = fn.funcion_final(numeric_data, 'age', 'avg_transaction_value', 1.75)






