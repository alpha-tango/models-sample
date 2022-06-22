#!/usr/bin/env python3

from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

from utils import get_features
from utils import get_target_column
from utils import read_dataset_training

"""
Bagged Random Forest models, using PCA-transformed data.
Using incrementalPCA for memory management.
"""


class ScantlieMabAEModel():
    def __init__(self):
        self.pca = IncrementalPCA(n_components=20, batch_size=502000, copy=False)
        self.pca_file = "/data-external/model005-pca_inc.joblib"
        self.regressor = RandomForestRegressor(n_estimators=5, max_depth=10, max_features=1)
        self.bag = BaggingRegressor(base_estimator=self.regressor, n_estimators=500, max_samples=1.0)
        self.bag_file = "/data-external/model005-bag.joblib"

    def fit(self, X, y):
        try:
            self.pca = load(self.pca_file)
            print("Loading pre-saved PCA")
        except:
            print("Unable to load pre-trained PCA.")
            self.pca.fit(X)
            dump(self.pca, self.pca_file)

        try:
            self.bag = load(self.bag_file)
            print("Loading pre-saved bag")
        except:
            print("Unable to load pre-trained bag.")
            self.bag.fit(self.pca.transform(X), y)
            dump(self.bag, self.bag_file)

    def predict(self, X):
        X = self.pca.transform(get_features(X))
        return self.bag.predict(X)


def get_model(current_round):
    training_data = read_dataset_training(current_round=current_round, include_validation_in_prod=False)
    target_column = get_target_column(training_data)

    # invoke
    model = ScantlieMabAEModel()
    model.fit(get_features(training_data), training_data[target_column])
    return model
