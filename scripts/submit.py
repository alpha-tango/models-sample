#!/usr/bin/env python3

import shutil
import sys

import numerapi

import utils


def main():
    # api setup

    api = utils.get_api()
    current_round = api.get_current_round()
    model_id = utils.read_secret("model_id")
    model_name = utils.read_secret("model_name")

    predictions_filename = f"/data/predictions-{model_name:s}-{current_round:d}.csv"
    print(predictions_filename)

    submission_id = api.upload_predictions(predictions_filename, model_id=model_id)
    print(submission_id)

    submission_filename = (
        f"/data-submissions/submission-{model_name:s}-{current_round:d}.csv"
    )
    print(submission_filename)
    shutil.copyfile(predictions_filename, submission_filename)

    return 0


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
