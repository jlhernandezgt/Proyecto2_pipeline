# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 17:55:35 2022

@author: luish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import funciones_proyecto as fn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score  
from sklearn.utils import resample
import scipy.stats as stats
import time
from sklearn.pipeline import Pipeline
import preprocessors as pp


df = pd.read_csv("churn.csv")
#df = df.dropna()
df = pd.DataFrame(df[0:2500])
df = df.sample(frac=1, random_state=1234)

categoric_vars, discrete_vars , continues_vars = fn.getColumnsDataTypes(df=df)




gender_level_map = df['gender_code'].value_counts().to_dict()
gender_level_map

df['gender_code'] = df['gender_code'].map(gender_level_map)
df.head()

#X = df[['avg_transaction_value','points_in_wallet']]
#y = df['gender']

X = df.drop(['gender_code', 'avg_transaction_value'], axis=1)
y = df['gender_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=2022)

def instanciatePipeline(df, y):
    categoric_vars, discrete_vars , continues_vars = fn.getColumnsDataTypes(df=df)    
    #categoric_vars.remove(y)
    bankChurner_Pipeline = Pipeline(steps=[
        ('categorical-encoder',
            pp.categoricalEncoderOperator(varNames=categoric_vars)),

        ('classifier',
            fn.modelo_qda(X_train, y_train, X_test))
    ])

    return bankChurner_Pipeline

dfSalida = instanciatePipeline(df, 'gender').fit_transform(X_train, y_train)
dfSalida
