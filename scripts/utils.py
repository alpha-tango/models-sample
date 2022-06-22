# utils.py

import functools
import operator
import tempfile
import zipfile

import numerapi
import numpy
import pandas


def check_model(model, data):
    target = get_target(data)
    # data_scrubbed = data[[c for c in data.columns if not c.startswith("target")]]
    data_scrubbed = data[[c for c in data.columns if c.startswith("feature")]]

    prediction = model.predict(data_scrubbed)

    check_df = pandas.DataFrame(
        data={"era": data["era"], "prediction": prediction, "target": target},
        index=data.index,
    )

    check_correlations = compute_era_correlations(check_df)

    print(
        f"min/mean/max = {check_correlations.min():.4f}/{check_correlations.mean():.4f}/{check_correlations.max():.4f}"
    )
    print(f"     std dev = {check_correlations.std():.4f}")
    print(f"Sharpe ratio = {check_correlations.mean()/check_correlations.std():.2f}")
    print(f"   geo. mean = {geometric_mean(check_correlations):.4f}")
    print("")


def check_prod():
    # try to read model_id secret which will only be available in prod.
    try:
        model_id = read_secret("model_id")
    except FileNotFoundError:
        model_id = None

    return model_id is not None


def compute_era_correlations(df):
    def compute_era_correlation(df_era):
        # Derived from https://github.com/numerai/example-scripts/blob/master/example_model.py

        pct_ranks = get_prediction(df_era).rank(pct=True, method="first")
        targets = get_target(df_era)
        return numpy.corrcoef(targets, pct_ranks)[0, 1]

    return df.groupby("era").apply(compute_era_correlation)


def filter_training_data_types(training_data):
    if check_prod():
        expected_data_types = frozenset(["train", "validation"])
    else:
        expected_data_types = frozenset(["train"])

    data_types = set(training_data["data_type"].unique())
    if not data_types.issubset(expected_data_types):
        print("filtering training data")
        training_data = training_data[
            training_data["data_type"].isin(expected_data_types)
        ]

    assert set(training_data["data_type"].unique()).issubset(expected_data_types)

    return training_data


def geometric_mean(xs):
    return functools.reduce(operator.mul, [1.0 + x for x in xs]) ** (1.0 / len(xs))


def get_api():
    api_params = {}
    api_params["public_id"] = read_secret("public_id")
    api_params["secret_key"] = read_secret("secret_key")

    return numerapi.NumerAPI(**api_params)


def get_datasets_filename(current_round):
    return f"/data-shared/numerai_datasets-{current_round:d}.zip"


def get_feature_columns(df):
    return [c for c in df.columns if c.startswith("feature_")]


def get_features(df, include_era=False):
    feature_columns = get_feature_columns(df)

    if include_era:
        feature_columns[:0] = ["era"]

    return df[feature_columns]


def get_prediction(df):
    return df[get_prediction_column(df)]


def get_prediction_column(df):
    return "prediction"


def get_target(df):
    return df[get_target_column(df)]


def get_target_column(df):
    assert "target" in df.columns
    return "target"


def normalize_target(df):
    (target_column,) = [c for c in df.columns if c.startswith("target")]
    if target_column != "target":
        # expecting target_kazutsugi before round 238
        df = df.rename(columns={target_column: "target"})

    return df


def read_dataset(current_round, dataset_filename, data_type=None):
    datasets_filename = get_datasets_filename(current_round)

    read_csv_params = {}
    read_csv_params["index_col"] = "id"

    with zipfile.ZipFile(datasets_filename) as datasets_zipfile:
        with datasets_zipfile.open(dataset_filename) as dataset_fp:
            if data_type is None:
                return pandas.read_csv(dataset_fp, **read_csv_params)

            data_type = data_type.encode("utf8")

            def parse_line(line):
                return line.strip(b"\n").split(b",")

            header_line = dataset_fp.readline()
            header = parse_line(header_line)
            data_type_index = header.index(b"data_type")
            with tempfile.TemporaryFile() as temp_fp:
                temp_fp.write(header_line)

                for line in dataset_fp.readlines():
                    if parse_line(line)[data_type_index] == data_type:
                        temp_fp.write(line)

                temp_fp.flush()
                temp_fp.seek(0)
                return pandas.read_csv(temp_fp, **read_csv_params)


def read_dataset_chunked(current_round, dataset_filename, chunksize):
    datasets_filename = get_datasets_filename(current_round)

    with zipfile.ZipFile(datasets_filename) as datasets_zipfile:
        with datasets_zipfile.open(dataset_filename) as dataset_fp:
            for chunk in pandas.read_csv(
                dataset_fp, chunksize=chunksize, index_col="id"
            ):
                yield chunk


def read_dataset_training(current_round, include_validation_in_prod=True):
    # current_round <= 237: kazutsugi
    # current_round >= 238: nomi

    training_data = normalize_target(
        read_dataset(current_round, dataset_filename="numerai_training_data.csv")
    )
    validate_target(training_data)

    # normalize target column

    if check_prod() and include_validation_in_prod:
        validation_data = normalize_target(
            read_dataset(
                current_round,
                dataset_filename="numerai_tournament_data.csv",
                data_type="validation",
            )
        )
        if list(validation_data.columns) != list(training_data.columns):
            raise RuntimeError("column mismatch adding validation training data")
        validate_target(validation_data)

        training_data = training_data.append(validation_data)

    # sanity check values

    validate_target(training_data)

    # filter data types and return

    return filter_training_data_types(training_data)


def read_secret(secret_name, strip=True):
    secret_filename = f"/run/secrets/{secret_name:s}"
    with open(secret_filename) as secret_fp:
        output = secret_fp.read()

    if strip:
        output = output.strip()

    return output


def validate_target(df):
    """
    Validates values in the target column.

    Throws an exception for unexpected values.
    """

    target_values_expected = set([0.00, 0.25, 0.50, 0.75, 1.00])
    target_values_check = set(df["target"].unique())
    if not target_values_check.issubset(target_values_expected):
        print(
            "unexpected target values ",
            sorted(target_values_check - target_values_expected),
        )
        raise RuntimeError("unexpected target values")

    return df


def saved_model_path(X, target_col, model_name):
    n = int(X[target_col].sum())
    return f"/data-external/{model_name}-{n}.joblib"
