import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from utils import get_feature_columns, get_target_column
from utils import read_dataset_training


class StableFeatureModel:
    def __init__(self, model, correlation_func, num_features):
        self.model = model
        self.correlation_func = correlation_func
        self.features = list()
        self.eras = list()
        self.num_features = num_features

    def build_corr_df(self, X, feature_cols):
        self.eras = X['era'].unique()
        out_cols = ['era'] + list(feature_cols)
        corr_df = pd.DataFrame(columns=out_cols)

        for era in self.eras:
            corrs = list()
            this_era = X[X['era'] == era]
            for col in feature_cols:
                pearson = self.correlation_func(this_era[col], this_era['target'])[0]
                corrs.append(pearson)
            row = [era] + list(corrs)
            row_df = pd.DataFrame([row], columns=out_cols)
            corr_df = corr_df.append(row_df)
        return corr_df

    def build_stats_df(self, corr_df, feature_cols):
        stats_df = pd.DataFrame(corr_df.std(), columns=['std'])
        stats_df['mean'] = corr_df.mean()
        stats_df['abs_sharpe'] = (stats_df['mean'] / stats_df['std']).abs()
        stats_df['pos_corr'] = corr_df[corr_df[feature_cols] > 0].count()

        def agreement(x):
            if x > len(self.eras) - x:
                return x/len(self.eras)
            else:
                return (len(self.eras) - x)/len(self.eras)

        stats_df['corr_agree'] = stats_df['pos_corr'].map(agreement)
        return stats_df

    def select_features(self, X):
        feature_cols = get_feature_columns(X)
        corr_df = self.build_corr_df(X, feature_cols)
        stats_df = self.build_stats_df(corr_df, feature_cols)

        # self.features = stats_df.sort_values(by='corr_agree', ascending=False)\
        #     .head(self.num_features).index
        # self.features = stats_df.sort_values(by='abs_sharpe', ascending=False)\
        #     .head(self.num_features).index
        self.features = stats_df.sort_values(by='std', ascending=True)\
            .head(self.num_features).index
        print(self.features)

    def dummy_select_features(self, X):
        self.features = get_feature_columns(X)

    def fit (self, X, y):
        self.select_features(X)
        self.model.fit(X[self.features], y)

    def predict(self, X):
        return self.model.predict(X[self.features])

def get_model(current_round):
    training_data = read_dataset_training(current_round=current_round)
    target_column = get_target_column(training_data)

    model = StableFeatureModel(
                model=LinearRegression(),
                correlation_func=pearsonr,
                num_features=150)

    model.fit(training_data, training_data[target_column])
    return model
