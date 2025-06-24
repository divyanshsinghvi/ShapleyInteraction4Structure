
from transformers import BertForMaskedLM, BertTokenizerFast
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle
import random

class ExperimentRunner:
    def __init__(self, cuda, seq_len, model_name, method):
        assert model_name in ['gpt', 'bert']
        self.CUDA = cuda
        self.SEQ_LEN = seq_len
        self.MODEL_NAME = model_name
        self.METHOD = method
        self.SOFTMAX = False
        random.seed(42)
    
    def prepare_model(self):
        if self.MODEL_NAME == 'bert':
            if self.CUDA:
                self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
            else:
                self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

            self.tokenizer= BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)#, add_special_tokens=False)

        elif self.MODEL_NAME == 'gpt':
            if self.CUDA:
                self.model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
            else:
                self.model = GPT2LMHeadModel.from_pretrained('gpt2')

            self.tokenizer= GPT2TokenizerFast.from_pretrained('gpt2', use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            raise NotImplementedError(f"{self.MODEL_NAME} wrong model name passed")



    def load_data(self):
        if self.MODEL_NAME == 'bert':
            return pd.concat([pd.read_pickle(f'mwe_tagger/bert_bert.pkl_{i}').drop(columns= ['syntactic_distance_idx_mapped', 'syntactic_distance_idx', 'lemmas', 'd', 'toks', 'tags']) for i in range(4)])
        elif self.MODEL_NAME == 'gpt':
            return pd.concat([pd.read_pickle(f'mwe_tagger/gpt_gpt.pkl_{i}').drop(columns= ['syntactic_distance_idx_mapped', 'syntactic_distance_idx', 'lemmas', 'd', 'toks', 'tags']) for i in range(4)])


    def prepare_data(self):
        test = self.load_data()
        test['length'] = test['sentence'].str.split().str.len()
        test = test[~((test['weak_mwe'].str.len()==0) & (test['strong_mwe'].str.len()==0))]
        test = test.drop(columns = ['syntactic_distance_idx', 'lemmas', 'd', 'toks', 'tags'], errors='ignore')
        self.test = test


    def get_prediction_softmax(self, X, token_next):
        logits = self.model(X).logits
        if self.SOFTMAX:
            abc =  logits[0, token_next:, :].softmax(dim=-1)
        else:
            abc =  logits[0, token_next:, :]

        if not self.CUDA:
            return abc.detach().numpy()
        else:
            return abc
    
    def interaction_value_di(self, X, tokens):
        token1, token2 = tokens
        if self.MODEL_NAME == 'gpt':
            token_next = max(token1, token2) + 1
        elif self.MODEL_NAME == 'bert':
            token_next = 0

        AB = self.get_prediction_softmax(X, token_next)
        X_t1 = X.clone()
        X_t1[0, token1] = self.tokenizer.pad_token_id
        A = self.get_prediction_softmax(X_t1, token_next)
        
        X_t2 = X.clone()
        X_t2[0,token1] = self.tokenizer.pad_token_id
        B = self.get_prediction_softmax(X_t2, token_next)

        X_t12 = X.clone()
        X_t12[0,token2] = self.tokenizer.pad_token_id
        X_t12[0,token1] = self.tokenizer.pad_token_id
        phi = self.get_prediction_softmax(X_t12, token_next)

        # print(AB, A, B, phi)
        val = AB - A - B + phi
        

        if self.METHOD == 1:
            val = AB - A - B + phi
            val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
            res_list = [(1, val.detach(), token_next)]
            return res_list


    
    def mwe_distance_interaction(self, encoded_row, row, col, row_number):

        iv_mwe = []
        
        mwes = row[col]
        encoded_row = encoded_row.reshape(1, -1)
        for mwe in mwes:

            for i in range(len(mwe)):
                for j in range(len(mwe)):
                    if i > j:
                        if len([x for x in mwe if x >= self.SEQ_LEN]) > 0:
                            continue
                        iv = self.interaction_value_di(encoded_row, [mwe[i]-1, mwe[j]-1])
                        iv_mwe.append([iv, abs((mwe[i]-1-(mwe[j]-1))),mwe, row_number, i, j])
                        
        return iv_mwe
    
    def calculate_interaction(self, encoded_row, row_number):
        interactions = []
        encoded_len = len(encoded_row)
        for j in range( min(self.SEQ_LEN, encoded_len)):
            probability = 0.05
            if random.random() < probability: 
                for k in range(j+1, min(self.SEQ_LEN, encoded_len, j+9)):
                    if j+k >= encoded_len:
                        continue
                    if encoded_row[j] == self.tokenizer.unk_token_id or encoded_row[k] == self.tokenizer.unk_token_id:
                        continue

                    og = encoded_row.clone()
                    og = og.reshape(1, -1)
                    iv = self.interaction_value_di(og, [j, k])
                    interactions.append([iv, abs((j-k)), row_number, j, k])
                
        return interactions
    
    def run_mwe_interactions(self, suffix=''):
        weak_mwe_distance = []
        strong_mwe_distance = []


        for row_number, row in tqdm(self.test.iterrows(), total=self.test.shape[0]):
            # Tokenize each sentence with no padding and truncate if seq_len < tokenized sentence length
            encoded_row = self.tokenizer(row['sentence'], padding=False,  truncation=True, max_length=self.SEQ_LEN, return_tensors ='pt').input_ids[0]
            if self.CUDA:
                encoded_row = encoded_row.cuda()

            if len(row['weak_mwe']) != 0:
                weak_mwe_distance.append(self.mwe_distance_interaction(encoded_row, row, 'weak_mwe', row_number))
            
            if len(row['strong_mwe']) != 0:
                strong_mwe_distance.append(self.mwe_distance_interaction(encoded_row, row, 'strong_mwe', row_number))
            
            del encoded_row
        
        weak_mwe_distance = [y for x in weak_mwe_distance for y in x]
        strong_mwe_distance = [y for x in strong_mwe_distance for y in x]

        pd.DataFrame(weak_mwe_distance, columns = ['tensor', 'posdis', 'ignore', 'row_number', 'first_token', 'second_token']).to_pickle(f'weak_mwe_{self.MODEL_NAME}{suffix}.pkl', compression='gzip')
        pd.DataFrame(strong_mwe_distance, columns = ['tensor', 'posdis', 'ignore', 'row_number', 'first_token', 'second_token']).to_pickle(f'strong_mwe_{self.MODEL_NAME}{suffix}.pkl', compression='gzip')

    def run_avg_interactions(self, suffix=''):
        average_distance = []

        for row_number, row in tqdm(self.test.iterrows(), total=self.test.shape[0]):
            encoded_row =  self.tokenizer(row['sentence'], padding=False,  truncation=True, max_length=self.SEQ_LEN, return_tensors ='pt').input_ids[0]
            if self.CUDA:
                encoded_row = encoded_row.cuda()

            
            if len(row['weak_mwe']) != 0:
                average_distance.extend(self.calculate_interaction(encoded_row, row_number))
            if row_number % 1000 == 0:
                print(len(average_distance))
                pickle.dump(average_distance, open(f'avg_{self.MODEL_NAME}{suffix}.pkl','wb'))

    def run_experiment(self, mwe=True, avg=True, suffix=''):
        self.prepare_data()
        self.prepare_model()
        if mwe:
            self.run_mwe_interactions(suffix=suffix)
        if avg:
            self.run_avg_interactions(suffix=suffix)

if __name__  == '__main__':
    ExperimentRunner(cuda=True, seq_len=50, model_name = 'bert', method=1).run_experiment(suffix='') 
    ExperimentRunner(cuda=True, seq_len=50, model_name = 'gpt', method=1).run_experiment(suffix='')
