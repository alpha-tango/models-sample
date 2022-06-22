#!/bin/sh

set -e

cd `dirname $0`/..

admin/docker-compose.py > docker-compose.yml

for model in $(cat admin/models.txt); do
    echo "$model" > "secrets/$model-model_name.txt"
done
