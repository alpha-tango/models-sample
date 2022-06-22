#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

import model010
import model005
import model006

from utils import get_features
from utils import get_target_column
from utils import read_dataset_training

class StackedModel:
    def __init__(self, current_round):
        self.base_models = list()
        self.combiner_model = RidgeCV()

        ##### Get pre-fit models ########
        print("initializing model006")  # Linear + Isotonic
        self.base_models.append(('model006', model006.get_model(current_round)))
        print("intitializing model005")  # Bagged RF
        self.base_models.append(('model005', model005.get_model(current_round)))
        print("initializing model010")  # Catboost-based
        self.base_models.append(('model010', model010.get_model(current_round)))

    def base_prediction_df(self, X):
        #### Make a dataframe of predictions from base models)
        data = dict()
        for name, model in self.base_models:
            data[name] = model.predict(X)
        return pd.DataFrame(data=data)

    def fit(self, X, y):
        #### Make and combine base model predictions #####
        print("predicting base models - training")
        A = self.base_prediction_df(X)

        #### Stack into RidgeCV Regression #####
        print("fitting ensemble - training")
        self.combiner_model.fit(A, y)

    def predict(self, X):
        #### Make and combine base model predictions #####
        A = self.base_prediction_df(X)
        p = self.combiner_model.predict(A)
        return np.interp(p, (p.min(), p.max()), (0, 1))


def get_model(current_round):
    training_data = read_dataset_training(current_round=current_round, include_validation_in_prod=False)
    target_column = get_target_column(training_data)

    model = StackedModel(current_round)
    model.fit(get_features(training_data), training_data[target_column])
    return model
