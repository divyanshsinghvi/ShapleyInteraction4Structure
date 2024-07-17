
from transformers import BertForMaskedLM, BertTokenizerFast
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle
import random
import gzip

class ExperimentRunner:
    def __init__(self, cuda, seq_len, model_name, method):
        assert model_name in ['gpt', 'bert']
        self.CUDA = cuda
        self.SEQ_LEN = seq_len
        self.MODEL_NAME = model_name
        self.METHOD = method
        self.SOFTMAX = False
        self.RIGHT_MAX = 4
        self.NUMBER_OF_PERM = 5
        self.NUMBER_OF_ROWS = 4
        random.seed(42)

    ## NEED TO MOVE IT TO SOME HELPER CLASS    
    def generate_perm(self, nperm):
        perm_stored=[]
        arg = np.arange(self.SEQ_LEN)
        i = 0
        while i < nperm: 
            perm_stored.append(np.random.permutation(arg))
            perm_stored.append(perm_stored[-1][::-1])
            i += 1
        return np.array(perm_stored)

    def generate_incremental_masks(self, permutations, l, r):
        n_rows, n_cols = permutations.shape
        
        masks = np.zeros((n_rows, n_cols), dtype=np.int8)
        
        stop_mask = (permutations == l) | (permutations == r)
        stop_indices = np.argmax(stop_mask, axis=1)
        col_indices = np.arange(n_cols)[np.newaxis, :]
        valid_mask = col_indices < stop_indices[:, np.newaxis]
        indices = permutations * valid_mask

        
        for inde, row in enumerate(indices):
            non_zero_indices = row[row != 0]
            masks[inde, non_zero_indices] = True
        return masks
    
    def get_smax(self, encoded_row, masks_tensor):
        expanded_input = encoded_row.unsqueeze(0).expand(masks_tensor.size(0), -1, -1)
        masks_tensor_expanded = masks_tensor.unsqueeze(1).expand(-1, encoded_row.shape[0], -1)
        masked_tensors = torch.where(masks_tensor_expanded == 1, expanded_input, torch.tensor(self.tokenizer.pad_token_id, device='cuda:0'))

        masked_tensors = masked_tensors.flatten(start_dim = 0, end_dim = 1)
        # masked_tensors = masked_tensors.reshape((masked_tensors.shape[0] * masked_tensors.shape[1], -1))

        
        masks_tensor_expanded = masks_tensor_expanded.flatten(start_dim = 0, end_dim = 1)
        # masks_tensor_expanded = masks_tensor_expanded.reshape((masks_tensor_expanded.shape[0] * masks_tensor_expanded.shape[1], -1))
        logits = self.model(masked_tensors, attention_mask=masks_tensor_expanded).logits    
        smax = logits.softmax(dim=-1)
        return smax, logits

    def prepare_model(self):
        if self.MODEL_NAME == 'bert':
            if self.CUDA:
                self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
            else:
                self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

            self.tokenizer= BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True, add_special_tokens=False)

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
            return pd.concat([pd.read_pickle(f'../mwe_tagger/bert_bert.pkl_{i}').drop(columns= ['syntactic_distance_idx_mapped', 'syntactic_distance_idx', 'lemmas', 'd', 'toks', 'tags']) for i in range(4)])
        elif self.MODEL_NAME == 'gpt':
            return pd.concat([pd.read_pickle(f'../mwe_tagger/gpt_gpt.pkl_{i}').drop(columns= ['syntactic_distance_idx_mapped', 'syntactic_distance_idx', 'lemmas', 'd', 'toks', 'tags']) for i in range(4)])


    def prepare_data(self):
        test = self.load_data()
        test['length'] = test['sentence'].str.split().str.len()
        test = test[~((test['weak_mwe'].str.len()==0) & (test['strong_mwe'].str.len()==0))]
        test = test.drop(columns = ['syntactic_distance_idx', 'lemmas', 'd', 'toks', 'tags'], errors='ignore')
        self.test = test


    def interaction_value_di(self, X, start_end_row):
        print(start_end_row)
        perms = self.generate_perm(self.NUMBER_OF_PERM)
        results = []


        for l in range(self.SEQ_LEN):
            for r in range(l+1, min(self.SEQ_LEN, l+self.RIGHT_MAX + 1)):
                masks = self.generate_incremental_masks(perms, l, r)
                masks_tensor = torch.tensor(masks, dtype=torch.long, device='cuda:0').detach()

                smax_ab, logits_ab = self.get_smax(X, masks_tensor.clone())

                masks_tensor_A = masks_tensor.clone()
                masks_tensor_A[:, l] = 1
                smax_a, logits_a = self.get_smax(X, masks_tensor_A)

                masks_tensor_B = masks_tensor.clone()
                masks_tensor_B[:, r] = 1
                smax_b, logits_b = self.get_smax(X, masks_tensor_B)

                masks_tensor_phi = masks_tensor.clone()
                masks_tensor_phi[:, r] = 1
                masks_tensor_phi[:, l] = 1
                smax_phi, logits_phi = self.get_smax(X, masks_tensor_phi)


                smax = smax_ab - smax_a - smax_b + smax_phi
                logits = logits_ab - logits_a - logits_b + logits_phi


                smax = smax.unflatten(0, (smax.shape[0]//X.shape[0], X.shape[0]))
                logits = logits.unflatten(0, (logits.shape[0]//X.shape[0], X.shape[0]))

                smax = torch.linalg.norm(smax, dim=-1)

                logits = torch.linalg.norm(logits, dim=-1)
                smax = torch.mean(smax, dim=0).cpu().detach()
                logits = torch.mean(logits, dim=0).cpu().detach()

                smax_ab = torch.linalg.norm(smax_ab, dim=-1).cpu().detach()
                smax_ab = torch.mean(smax_ab, dim=0).cpu().detach()

                results.append([l, r, smax, logits, smax_ab, start_end_row])
        return results

    
    def calculate_interaction(self, encoded_row, start_end_row):
        interactions = []
        iv = self.interaction_value_di(encoded_row, start_end_row)
        interactions.append(iv)
        return interactions
    

    def run_avg_interactions(self, suffix=''):
        average_distance = []

        i = 0
        for start_row in tqdm(range(0, self.test.shape[0], self.NUMBER_OF_ROWS), total=self.test.shape[0] // self.NUMBER_OF_ROWS + 1):
        # for row_number, row in tqdm(self.test.iterrows(), total=self.test.shape[0]):
            end_row = min(start_row +  self.NUMBER_OF_ROWS-1, self.test.shape[0])
            batch = self.test.iloc[start_row:end_row]['sentence'].to_list()
            encoded_row =  self.tokenizer(batch, padding=True,  truncation=True, max_length=self.SEQ_LEN, return_tensors ='pt').input_ids.detach()
            if self.CUDA:
                encoded_row = encoded_row.cuda()

            try:
                average_distance.extend(self.calculate_interaction(encoded_row, list(range(start_row, end_row+1))))
            except:
                pass
            if (end_row+1) > 2^i+100 == 0:
                i += 1
                print(len(average_distance))
                with gzip.open(f'avg_2_{self.MODEL_NAME}{suffix}.pkl','wb') as f:
                    pickle.dump(average_distance, f)
            if (end_row+1) == 10:
                print(len(average_distance))
                with gzip.open(f'avg_2_{self.MODEL_NAME}{suffix}.pkl','wb') as f:
                    pickle.dump(average_distance, f)


    def run_experiment(self, avg=True, suffix=''):
        self.prepare_data()
        self.prepare_model()
        if avg:
            self.run_avg_interactions(suffix=suffix)

if __name__  == '__main__':
    ExperimentRunner(cuda=True, seq_len=20, model_name = 'bert', method=105).run_experiment(suffix='100') 
    ExperimentRunner(cuda=True, seq_len=20, model_name = 'gpt', method=105).run_experiment(suffix='100')
    asdadsa
    #ExperimentRunner(cuda=True, seq_len=50, model_name = 'gpt', method=1).run_experiment(avg=False, suffix='1')
    #ExperimentRunner(cuda=True, seq_len=50, model_name = 'gpt', method=2).run_experiment(avg=False, suffix='2')
    #ExperimentRunner(cuda=True, seq_len=50, model_name = 'gpt', method=3).run_experiment(avg=False, suffix='3')

    ExperimentRunner(cuda=True, seq_len=50, model_name = 'bert', method=1).run_experiment(mwe=False, suffix='1') 
    ExperimentRunner(cuda=True, seq_len=50, model_name = 'gpt', method=1).run_experiment(mwe=False, suffix='1')
    ExperimentRunner(cuda=True, seq_len=50, model_name = 'bert', method=2).run_experiment(mwe=False, suffix='2') 
    ExperimentRunner(cuda=True, seq_len=50, model_name = 'bert', method=3).run_experiment(mwe=False, suffix='3')

    ExperimentRunner(cuda=True, seq_len=50, model_name = 'gpt', method=2).run_experiment(mwe=False, suffix='2')
    ExperimentRunner(cuda=True, seq_len=50, model_name = 'gpt', method=3).run_experiment(mwe=False, suffix='3')
