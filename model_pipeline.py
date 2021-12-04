# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
#####################################

# https://www.kaggle.com/c/titanic/overview

# The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.



#####################################

# The Challenge
# The sinking of the Titanic is one of the most infamous shipwrecks in history.

# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).


# What Data Will I Use in This Competition?
# In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled `train.csv` and the other is titled `test.csv`.

# <<Train.csv>> will contain the details of a subset of the passengers on board (<<891>> to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

# The `test.csv` dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.

# Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

# Check out the “Data” tab to explore the datasets even further. Once you feel you’ve created a competitive model, submit it to Kaggle to see where your model stands on our leaderboard against other Kagglers.

#####################################

# The data has been split into two groups:

# training set (train.csv)
# test set (test.csv)
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

# We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.


# Data Dictionary
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower

# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

# sibsp: The dataset defines family relations in this way:
    # Sibling = brother, sister, stepbrother, stepsister
    # Spouse = husband, wife (mistresses and fiancés were ignored)

# parch: The dataset defines family relations in this way:
    # Parent = mother, father
    # Child = daughter, son, stepdaughter, stepson
    # Some children travelled only with a nanny, therefore parch=0 for them.


"""

#%% This file

"""
feature pre-processing and model fitting pipeline.

"""

"""
Pipeline with ColumnTransformer for pre-processing

We use sklearn.compose.ColumnTransformer to define pre-processing, which is to be passed to a pipeline.


# We also define custom estimators to do any column operations we want
# We create a custom estimator as a new estimator class; see 'Intro to Data Science' 1.6.
# We import from sklearn.base <<BaseEstimator>>, which provides a base class that enables to set and get parameters of the estimator, and <<TransformerMixin>>, which implements the combination of fit and transform, i.e. fit_transform

# A custom estimator consists of three methods:
# 1. __init__ : This is the constructor. Called when pipeline is initialized.
# 2. fit() : Called when we fit the pipeline.
# 3. transform() : Called when we use fit or transform on the pipeline.

# After that, the custom estimator is called in ColumnTransformer. The syntax is 
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('estimator1', Transformer_Example1(), <list of columns>), ...)
#     ], remainder='passthrough')
# Here, by default, remainder='drop' which drops all remaining columns. Here we can specify 'passthrough'. 
# The ColumnTransformer returns an array (each time the processed column is moved the the left)

# Then we pass the ColumnTransformer to a Pipeline


Summary: 

from sklearn.base import BaseEstimator, TransformerMixin
class Transformer_Example1(BaseEstimator, TransformerMixin) : 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # print('__init__ is called.\n')
        pass
    
    def fit(self, X, y=None) : 
        # print('fit is called.\n')
        return self
    
    def transform(self, X, y=None) : 
        # print('return is called.\n')
        X_ = X.copy() # create a copy to avoid changing the original dataset
        X_ =  2 * X_  # cook up some manipulation to column X2
        return X_

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('estimator1', Transformer_Example1(), <list of columns>), ...)
    ], remainder='passthrough')


"""

#%% Workflow

"""
# import training set
# drop columns: 'Name', 'Ticket', 'Cabin', 'PassengerId'
# train-validation split

# NaN: 
#     'Age': fillna with mean of the <<training set>>
#     'Embarked': fillna with most frequent occurrence (most likely 'S')
#     'Fare': fillna with median of the <<training set>>

# Categorical variables:
#     'Sex': target encode
#     'Pclass': target/frequency encode
#     'Embarked': one-hot encode

"""



#%% Preamble


import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns

# import os
# os.chdir(r'C:\Users\Cedric Yu\Desktop\Data Science\flask\titanic')

# for pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# for pre-processing
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import MinMaxScaler


from xgboost import XGBClassifier


#%% pre-processing with ColumnTransformer and Pipeline

# columns to drop
cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']

X_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



# pipeline_Embarked = Pipeline([
#     ('imputer_Embarked', SimpleImputer(strategy='most_frequent')),
#     ('OH_encode', OneHotEncoder(handle_unknown='ignore', sparse=False))
#     ])


# encoders = ColumnTransformer(
#     transformers=[
#         ('drop_cols', 'drop', cols_to_drop), 
#         ('imputer_Fare', SimpleImputer(strategy='median'), ['Fare']),
#         ('imputer_Age', SimpleImputer(strategy='mean'), ['Age']),
#         ('target_encode', ce.target_encoder.TargetEncoder(), ['Sex', 'Pclass']),
#         ('Embarked', pipeline_Embarked, ['Embarked'])
#         ],
#         remainder='passthrough')

# # scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
scale_pos_weight = 410./258.



# XGBC = XGBClassifier(n_estimators = 400, learning_rate = 0.01, max_depth = 3,  eval_metric="error", scale_pos_weight = scale_pos_weight, verbose = 0)


# """# define pipelines with the above ColumnTransformer"""
# pipeline_model = Pipeline([
#     ('encoders', encoders),
#     ('imputer', SimpleImputer(strategy='mean')),
#     # ('scaler', MinMaxScaler()),
#     ('XGBC', XGBC)
#     ])

"""


#%% load dataset train.csv

# import original training dataset
train_df_raw = pd.read_csv(r'datasets\train.csv', low_memory=False)
# train_df_raw.shape
# (891, 12)

train_df = train_df_raw.copy()

#%% drop columns: 'Name', 'Ticket', 'Cabin', 'PassengerId'

# train_df = train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)

X = train_df.drop(['Survived'], axis = 1)
y = train_df['Survived']

#%% train-validation split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)


# X_train_encoded0 = encoders.fit_transform(X_train, y_train)
# X_train_nontextp = pipeline_nontext.fit_transform(X_train_nontext, y_train)
# X_all = np.hstack([X_train_textp, X_train_nontextp])

pipeline_model.fit(X_train, y_train)

# pipeline_model.score(X_valid, y_valid)


# import joblib
from joblib import dump

# dump the pipeline model
dump(pipeline_Embarked, filename="pipeline_Embarked.joblib")
dump(encoders, filename="encoders.joblib")
dump(pipeline_model, filename="pipeline_model.joblib")

# from pickle import dump

# dump(pipeline_Embarked, open('pipeline_Embarked.pkl', 'wb'))
# dump(encoders, open('encoders.pkl', 'wb'))
# dump(pipeline_model, open('pipeline_model.pkl', 'wb'))



#%% load model

# load pipelines and model
from pickle import load
pipeline_Embarked = load(open('pipeline_Embarked.pkl', 'rb'))
encoders = load(open('encoders.pkl', 'rb'))
pipeline_model = load(open('pipeline_model.pkl', 'rb'))


# load pipelines and model
from joblib import load

pipeline_Embarked = load("pipeline_Embarked.joblib")
encoders = load("encoders.joblib")
pipeline_model = load("pipeline_model.joblib")




"""
