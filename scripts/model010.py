#!/usr/bin/env python3

from catboost import CatBoostRegressor

from utils import get_features
from utils import get_target_column
from utils import read_dataset_training

from utils import saved_model_path
from os import path
from joblib import dump, load

"""
Catboost, original version.
"""

class ScantlieMabModel(CatBoostRegressor):
    def __init__(self):
        self.hyperparams = {
                                'iterations': 500,
                                'depth': 7,
                                'loss_function': 'RMSE',
                                'learning_rate': 0.03,
                                'silent': True
        }
        super().__init__(**self.hyperparams)

    def predict(self, X):
        return super().predict(get_features(X))

    ## Inherits fit() from CatBoostRegressor


def get_model(current_round):
    training_data = read_dataset_training(current_round=current_round, include_validation_in_prod=False)
    target_column = get_target_column(training_data)
    model_name = path.basename(__file__).strip(".py")

    model_path = saved_model_path(training_data, target_column, model_name)

    try:
        model = load(model_path)
        print(f"Using saved model: {model_path}")

    except FileNotFoundError:
        print("Saved model not found. Training model.")

        # invoke
        model = ScantlieMabModel()
        model.fit(get_features(training_data), training_data[target_column])
        dump(model, model_path)
    return model
