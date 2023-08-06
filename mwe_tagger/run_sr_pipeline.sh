#!/bin/bash

# Predict with an existing MWE model.
# Usage: ./mwe_identify.sh model input

set -eu
set -o pipefail

input=$1  # word and POS tag on each line (tab-separated)

# predict MWEs with an existing model

../data_processing/setup_spacy.sh

python process_hf_dataset.py -f $input

./preprocess.sh $input

./sst.sh $input

python src/tags2mwe.py $input.pred.tags > $input.pred.mwe

python make_csv.py -f $input.pred.mwe
