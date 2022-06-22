#!/usr/bin/env python3

import argparse
import os.path
import sys

import requests

import utils


def main():
    # arguments

    parser = argparse.ArgumentParser(description="download current dataset")
    parser.add_argument("--current_round", type=int)
    parser.add_argument("--lazy", action="store_true")

    args = parser.parse_args()

    # api setup

    api = utils.get_api()
    current_round = (
        args.current_round
        if args.current_round is not None
        else api.get_current_round()
    )

    print(f"current round = {current_round:d}")

    # file location and lazy checks

    datasets_filename = utils.get_datasets_filename(current_round)
    if args.lazy and os.path.exists(datasets_filename):
        print("already downloaded")
        return 0

    # download data

    datasets_url = api.get_dataset_url()
    print(datasets_url)
    if f"/{current_round}/numerai_datasets.zip?" not in datasets_url:
        sys.stderr.write("please sanity check url^^\n")
        return 1

    datasets_response = requests.get(datasets_url, stream=True)

    with open(datasets_filename, "wb") as datasets_fp:
        for chunk in datasets_response.iter_content(chunk_size=1048576):
            datasets_fp.write(chunk)

    # done

    return 0


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
