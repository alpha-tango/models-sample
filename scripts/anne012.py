#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

import model010
import anne007
import model006
import jef008

from utils import get_features
from utils import get_target_column
from utils import read_dataset_training

from utils import saved_model_path
from os import path
from joblib import dump, load


"""Ensemble like anne011 but with jef008"""

class StackedModel:
    def __init__(self, current_round):
        self.base_models = list()
        self.combiner_model = RidgeCV()
        self.current_round = current_round

        ##### Get pre-fit models ########
        self.base_models.append(('model006', model006.get_model))  #  Linear + Isotonic
        self.base_models.append(('anne007', anne007.get_model))  # Bagged RF
        self.base_models.append(('model010', model010.get_model))  # Catboost-based
        self.base_models.append(('jef008', jef008.get_model))  # Linear with Catboost trained on linear error

    def base_prediction_df(self, X):
        #### Make a dataframe of predictions from base models)
        data = dict()
        for name, get_model in self.base_models:
            model = get_model(self.current_round)
            data[name] = model.predict(X)
        return pd.DataFrame(data=data)

    def fit(self, X, y):
        #### try to use saved version
        model_name = path.basename(__file__).strip(".py")
        target_column = get_target_column(X)
        model_path = saved_model_path(X, target_column, model_name)
        try:
            self.combiner_model = load(model_path)
            print(f"Using saved model: {model_path}")
        except FileNotFoundError:
            print("Saved model not found. Training model.")

            #### Make and combine base model predictions #####
            print("predicting base models - training")
            A = self.base_prediction_df(get_features(X))

            #### Stack into RidgeCV Regression #####
            print("fitting ensemble - training")
            self.combiner_model.fit(A, y)
            dump(self.combiner_model, model_path)

    def predict(self, X):
        #### Make and combine base model predictions #####
        A = self.base_prediction_df(X)
        p = self.combiner_model.predict(A)
        return np.interp(p, (p.min(), p.max()), (0, 1))


def get_model(current_round):
    training_data = read_dataset_training(current_round=current_round, include_validation_in_prod=False)
    target_column = get_target_column(training_data)

    model = StackedModel(current_round)
    model.fit(training_data, training_data[target_column])
    return model
