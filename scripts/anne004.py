#!/usr/bin/env python3

from catboost import CatBoostRegressor
from joblib import dump, load
import random
import sys

from driver import main
from utils import get_features
from utils import get_target_column
from utils import read_dataset_training

"""
Catboost with column downsampling to reduce overfitting,
and noise added during training via feature sign reversals.
(Noisy version of alphaiota.)
Forum post here: https://forum.numer.ai/t/feature-reversing-input-noise/1416
"""

MODEL_FILE = "/data-external/modelaac.joblib"

class NoisyCatboostModel:
    def __init__(self):
        self.reverse_prop = 0.1
        self.model = CatBoostRegressor(
                        iterations=1,
                        depth=7,
                        learning_rate=0.19,  # match `alphaiota`
                        loss_function='RMSE',
                        colsample_bylevel=0.1,
                        silent=True
        )
        self.multiplier = []


    def reverse_features(self, X):
        """
        Reverse the signs of some of the features in the dataset
        to add noise and prevent overreliance on the sign
        of a certain feature.
        Instead of centering all the data and then reversing the signs,
        I just center the data to be reversed, then reversed that data,
        then recentered, which all simplifies to doing (1-x) on that
        particular data.
        """
        # make a list of columns
        columns = X.columns

        # undo previous inversion
        for i in self.multiplier:
            col_name = columns[i]
            X[col_name] = 1 - X.loc[:,col_name]

        # make a list of columns to invert
        num_columns = len(columns)
        num_invert_columns = round(num_columns * self.reverse_prop)
        self.multiplier = random.sample(range(num_columns), num_invert_columns)

        # multiply the columns of the dataset
        for i in self.multiplier:
            col_name = columns[i]
            X[col_name] = 1 - X.loc[:,col_name]
        return X

    def fit(self, X, y):
        """
        Fit a Catboost model on a small number of trees and then incrementally
        train with feature signs periodically reversed.
        """
        try:
            self.model = load(MODEL_FILE)

        except FileNotFoundError:
            print("Saved model not found. Proceeding to train.")
            self.model.fit(X, y)

            for i in range(499):
                print(f"SWITCHING SIGNS - ROUND {i}")
                self.model = self.model.fit(self.reverse_features(X), y, init_model=self.model)

            dump(self.model, MODEL_FILE)

    def predict(self, X):
        return self.model.predict(get_features(X))


################################
# Required driver function
################################

def get_model(current_round):
    training_data = read_dataset_training(current_round=current_round)
    target_column = get_target_column(training_data)
    model = NoisyCatboostModel()
    model.fit(get_features(training_data), training_data[target_column])
    return model


if __name__ == "__main__":
    sys.exit(main(get_model))
