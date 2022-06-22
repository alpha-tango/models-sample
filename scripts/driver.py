#!/usr/bin/env python3

import argparse
import sys

import pandas

from model_mapping import MODEL_MAPPING
from utils import check_model
from utils import get_api
from utils import normalize_target
from utils import read_dataset
from utils import read_dataset_chunked
from utils import read_secret


def predict_checked(model, df):
    prediction = model.predict(df)
    assert 0.0 <= prediction.min() <= prediction.max() <= 1.0

    return pandas.DataFrame(data={"prediction": prediction}, index=df.index)


def main():
    # arguments

    parser = argparse.ArgumentParser(description="driver options")
    parser.add_argument("--no-stats-check", action='store_false', dest="check_stats", default=True)
    parser.add_argument("--no-write", action='store_false', dest="write_predictions", default=True)

    args = parser.parse_args()
    print(args.check_stats)

    # check config

    model_name = read_secret("model_name")
    if model_name not in MODEL_MAPPING:
        sys.stderr.write(f"unrecognized model name {model_name!r}\n")
        return 1

    model_module_name = MODEL_MAPPING[model_name]
    model_module = __import__(model_module_name)

    # api setup

    api = get_api()
    current_round = api.get_current_round()

    # get model

    print("#" * 60)

    model = model_module.get_model(current_round=current_round)

    # check performance stats on train and validation data

    print("#" * 60)

    def check_data_type(dataset_filename, data_type):
        print("CHECKING MODEL PERFORMANCE ON", data_type)

        check_data = normalize_target(
            read_dataset(
                current_round, dataset_filename=dataset_filename, data_type=data_type
            )
        )
        check_model(model, check_data)

    if args.check_stats:
        check_data_type("numerai_training_data.csv", "train")
        check_data_type("numerai_tournament_data.csv", "validation")
    else:
        print("Skipping performance checks.")

    # predictions to submit

    print("#" * 60)

    if args.write_predictions:
        predictions = []
        tournament_reader = read_dataset_chunked(
            current_round, "numerai_tournament_data.csv", chunksize=100000
        )
        for (i, tournament_chunk) in enumerate(tournament_reader):
            print("chunk", i)
            predictions.append(predict_checked(model, tournament_chunk))

        predictions = pandas.concat(predictions)

        ############################################################

        output_filename = f"/data/predictions-{model_name}-{current_round}.csv"
        print("writing", len(predictions), "predictions to", output_filename)

        predictions.to_csv(output_filename, index=True)
    else:
        print("Skipping write predictions.")

    return 0

############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
