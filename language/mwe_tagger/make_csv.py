import sys
sys.path.append("../data_processing/")
import pandas as pd
from parse_dep import *
import re
from ast import literal_eval
import tokenizations
from datasets import load_dataset

import argparse

def preprocess_tags(df):
    
    df['d'] = df.apply(lambda x: literal_eval(x['d']), axis=1)
    df = pd.concat([df, pd.json_normalize(df.d)],axis=1)
    df = df.drop(columns=[0])
    df['tokens'] = df['toks'].apply(lambda x: [y[0] for y in x])
    df['sentence'] = df['sentence'].apply(lambda x: literal_eval(x).decode('utf-8'))
    df['sent'] = df['sentence'].apply(lambda x: x.replace("_", " ").replace("~"," "))

    return df 

def list_to_index_dict(input_list):
    index_dict = {}
    
    for index, sublist in enumerate(input_list):
        for item in sublist:
            if item in index_dict:
                index_dict[item].append(index)
            else:
                index_dict[item] = [index]
    
    return index_dict

def map_mwes_together(x, mwe_type):
    """
    ~: for strong
    _: for weak
    """
    assert mwe_type in ["~", "_"], "MWE Type must be one of [~, _]"
    mapped_mwes = []
    for j, mwe in enumerate(x[mwe_type]):
        mapped_mwes.append([])
        for index in mwe:
            
            if index-1 in x['token_map_dict']:
                for val in x['token_map_dict'][index-1]:
                    mapped_mwes[j].append(val+1)
                
                
    return mapped_mwes
    
def strip_g(l: list):
    return [i.replace("Ġ", "") for i in l]

from tqdm.auto import tqdm


def main():
    
    tqdm.pandas()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", help="MWE tagged file")
    parser.add_argument("-m", "--model", help="bert|gpt")
    args = parser.parse_args()
    mwe_file = args.file

    
    model = args.model
    assert model in ['gpt2', 'bert']

    out_file = f"{mwe_file.split('.')[0]}_{model}.pkl"
    pipeline = get_spacy_pipeline(model)
    
    big_df = preprocess_tags(pd.read_csv(mwe_file, sep='\t', names=[0, 'sentence', 'd']))
    print("MWE file original shape, ", big_df.shape)
    big_df = big_df.reset_index(drop=True)


    tot = len(big_df)

    for i, k in enumerate([[0, tot//4], [tot//4, tot//2], [tot//2,tot*3//4], [tot*3//4 , tot]]):
        
        df = big_df.iloc[k[0]: k[1]]
        df["syntactic_distance_idx"] = df["sent"].progress_apply(lambda x: get_syntactic_distance(pipeline(x), index=True))

        df['tokens_to_map'] = df.progress_apply(lambda x: list(map(str, list(pipeline(x["sent"])))), axis=1)
        df['token_map'] = df.progress_apply(lambda x: tokenizations.get_alignments(x['tokens'],
                                                                      #x['tokens_to_map'])[1], axis=1)
                                                                      strip_g(x['tokens_to_map']))[1], axis=1)
        
        df['token_map_dict'] = df['token_map'].progress_apply(lambda x: list_to_index_dict(x))
        df["syntactic_distance_idx_mapped"] = df.progress_apply(lambda x: map_syntactic_distance(x), axis=1)
        df['weak_mwe'] = df.progress_apply(lambda x: map_mwes_together(x, "_"), axis=1)
        df['strong_mwe'] = df.progress_apply(lambda x: map_mwes_together(x, "~"), axis=1)

        df.to_pickle(f'{out_file}_{i}')


if __name__ == "__main__":
    main()
