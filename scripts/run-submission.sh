#!/bin/sh

set -e

cd `dirname $0`/..

scripts/run-prediction-prod.sh
scripts/submit.py
