#!/usr/bin/env python3

from catboost import CatBoostRegressor
import numpy as np

from utils import get_features, get_feature_columns
from utils import get_target_column
from utils import read_dataset_training

"""
Train per-era Catboost models and
average them together.
"""

class ScantlieMabADModel():
    def __init__(self):
        self.hyperparams = {
                                'iterations': 10,
                                'silent': True
        }
        self.models = list()

    def predict(self, X):
        predictions = list()
        for i, m in enumerate(self.models):
            predictions.append(m.predict(get_features(X)) * 100)

        predictions = np.array(predictions, dtype=np.uint8)
        return np.average(predictions, axis=0) / 100

    def fit(self, X, y):

        def to_int(era_str):
            """Map era column value to an integer"""
            era_str = era_str[3:]
            return int(era_str)

        X['era_int'] = X['era'].map(to_int)
        era_ints = np.array(X.groupby(by=X.era_int).count().index)

        era_ints = sorted(era_ints)
        era_ints = era_ints[5:]  # first 5 eras have fewer rows, so omitting

        feature_cols = get_feature_columns(X)
        target_col = get_target_column(X)

        for era in era_ints:
            print("training model {}".format(era))
            this_training = X.loc[X.era_int == era, feature_cols]
            this_target = X.loc[X.era_int == era, target_col]
            this_model = CatBoostRegressor(**self.hyperparams)
            this_model.fit(this_training, this_target)
            self.models.append(this_model)

        # del this_training
        # del this_target
        # del this_model


def get_model(current_round):
    training_data = read_dataset_training(current_round=current_round, include_validation_in_prod=False)
    target_column = get_target_column(training_data)

    # invoke
    model = ScantlieMabADModel()
    model.fit(training_data, training_data[target_column])
    return model
