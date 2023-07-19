#!/bin/bash
set -eu

# Generate supersense lexicon from WordNet

mkdir lex

python src/sstFeatures.py
