#!/bin/sh

set -e

cd `dirname $0`/..

MODEL_NAME=`cat /run/secrets/model_name`

echo "$MODEL_NAME"
scripts/download.py --lazy
scripts/driver.py --no-write
