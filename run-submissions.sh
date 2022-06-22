#!/bin/sh

set -e

cd `dirname $0`

function submit {
    docker-compose run "$1-prod" scripts/run-submission.sh
}

docker-compose build

for model in $(cat admin/models.txt); do
    submit "$model"
done

