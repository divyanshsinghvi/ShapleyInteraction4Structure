import sys
sys.path.append("../data_processing/")
import pandas as pd
from parse_dep import *
from datasets import load_dataset

import argparse


pipeline = spacy.load("en_core_web_sm")

def get_dataset():
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = wiki['train']['text'] + wiki['test']['text'] + wiki['validation']['text']
    #text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    # Remove empty lines
    return text

def parse_sentences_mwe(dataset):
    lines = []
    for d in dataset:
        doc = pipeline(d)
        if doc:
            for sent in doc.sents:
                for word in sent:
                    lines.append(f"{word}\t{word.pos_}\n")
                lines.append("\n")
    return lines

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", help="MWE tagged file")
    args = parser.parse_args()
    mwe_file = args.file
        
    data = get_dataset()
    parsed_data = parse_sentences_mwe(data)

    with open(mwe_file, "w") as dset:
        dset.writelines(parsed_data)


if __name__ == "__main__":
    main()
