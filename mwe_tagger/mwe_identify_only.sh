#!/bin/bash

# Predict with an existing MWE model.
# Usage: ./mwe_identify.sh model input

set -eu
set -o pipefail

input=$1  # word and POS tag on each line (tab-separated)

# predict MWEs with an existing model

python src/tags2mwe.py $input.pred.tags > $input.pred.mwe
