#!/bin/bash

# Predict with an existing MWE model.
# Usage: ./mwe_identify.sh model input

set -eu
set -o pipefail

input=$1  # word and POS tag on each line (tab-separated)
model=$2

# predict MWEs with an existing model
echo "setting up spacy"
bash setup_spacy.sh

echo "Processing hf dataset"
python process_hf_dataset.py -f $input

echo "Preprocessing"
./preprocess.sh $input

echo "sst"
./sst.sh $input

echo "tags2mwe"
python src/tags2mwe.py $input.pred.tags > $input.pred.mwe

echo "make_csv"
python make_csv.py -f $input.pred.mwe -m $model
