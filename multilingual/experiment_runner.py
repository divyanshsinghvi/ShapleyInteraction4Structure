
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast, XLMRobertaForMaskedLM
from datasets import load_dataset
import pandas as pd
import torch
from tqdm.auto import tqdm
import numpy as np
import pickle
import random
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class ExperimentRunner:
    def __init__(self, cuda, seq_len, model_name, method, lang, prob = 0.1):
        assert model_name in ['xlm-roberta-base', 'bert-base-multilingual-cased']
        self.CUDA = cuda
        self.SEQ_LEN = seq_len
        self.MODEL_NAME = model_name
        self.LANGUAGE = lang
        self.METHOD = method
        self.SOFTMAX = True
        self.PROB = prob
        random.seed(42)
    
    def prepare_model(self):
        if self.MODEL_NAME == 'xlm-roberta-base':
            if self.CUDA:
                self.model = XLMRobertaForMaskedLM.from_pretrained(self.MODEL_NAME).cuda()
            else:
                self.model = XLMRobertaForMaskedLM.from_pretrained(self.MODEL_NAME)

            self.tokenizer= XLMRobertaTokenizerFast.from_pretrained(self.MODEL_NAME, use_fast=True, add_special_tokens=False)
        else:
            raise NotImplementedError(f"{self.MODEL_NAME} wrong model name passed")



    def load_data(self):
        if self.LANGUAGE == 'turkish':
            dataset = load_dataset("turkish-nlp-suite/turkish-wikiNER")
            return dataset['test']['tokens'] + dataset['validation']['tokens'] + dataset['train']['tokens']
        elif self.LANGUAGE == 'german':
            dataset = load_dataset("germeval_14")
            return dataset['train']['tokens'] + dataset['validation']['tokens'] + dataset['train']['tokens']
        elif self.LANGUAGE == 'english':
            wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
            return wiki['train']['text'] + wiki['test']['text'] + wiki['validation']['text']
        # elif self.LANGUAGE == 'german':
        #     dataset = load_dataset("deepset/germanquad")
        #     return dataset['test']['context']

    def prepare_data(self):
        test = self.load_data()
        self.test = test


    def get_prediction_softmax(self, X, token_next, attention_mask):
        # logits = self.model(X).pooler_output 
        # print(logits.shape)
        g = self.model(X, labels=X.clone(), attention_mask=attention_mask)
        logits = g.logits
        
        if self.SOFTMAX:
            abc =  logits[0, token_next:, :].softmax(dim=-1)
        else:
            abc =  logits[0, token_next:, :]

        with torch.no_grad():
            celoss = torch.nn.CrossEntropyLoss(reduction ='none')
            loss1 = celoss(logits.view(-1, self.tokenizer.vocab_size), X.view(-1)).cpu().detach()
        
        if not self.CUDA:
            return abc.detach().numpy(), loss1.detach().numpy()#round(g.loss.item(), 2)
        else:
            return abc, loss1
    
    def interaction_value_di(self, X, tokens, attention_mask):
        token1, token2 = tokens

        if self.MODEL_NAME in ['xlm-roberta-base', 'bert-base-multilingual-cased']:
            token_next = 0
        else:
            raise Exception("Not Implemented")
        AB, loss_AB = self.get_prediction_softmax(X, token_next, attention_mask)
        X_t1 = X.clone()
        X_t1[0, token1] = self.tokenizer.pad_token_id
        A, loss_A = self.get_prediction_softmax(X_t1, token_next, attention_mask)
        
        X_t2 = X.clone()
        X_t2[0,token1] = self.tokenizer.pad_token_id
        B, loss_B = self.get_prediction_softmax(X_t2, token_next, attention_mask)

        X_t12 = X.clone()
        X_t12[0,token2] = self.tokenizer.pad_token_id
        X_t12[0,token1] = self.tokenizer.pad_token_id
        phi, loss_phi = self.get_prediction_softmax(X_t12, token_next, attention_mask)

        # print(AB, A, B, phi)
        val = AB - A - B + phi
        

        if self.METHOD == 105:
            val = AB - A - B + phi
            val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
            res_list = [(1, val.detach(), token_next, torch.linalg.norm(AB, dim=1).cpu().detach(), torch.linalg.norm(A, dim=1).cpu().detach(),torch.linalg.norm(B, dim=1).cpu().detach(),torch.linalg.norm(phi, dim=1).cpu().detach(), loss_AB, loss_A, loss_B, loss_phi)]
            # val = AB - A - B + phi
            # val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB - phi, dim=1)).cpu()
            # res_list.append((2, val.detach(), token_next))
            # val = AB - A - B + phi
            # val = torch.linalg.norm(val, dim=1).cpu()
            # res_list.append((3, val.detach(), token_next))
            # val = torch.divide(torch.linalg.norm(AB - A - B, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
            # res_list.append((4, val.detach(), token_next))
            return res_list

        assert False

        if self.METHOD == 1:
            val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
            return val.detach(), token_next
        elif self.METHOD == 2:
            val = torch.divide(torch.linalg.norm(val, dim=1), torch.linalg.norm(AB - phi, dim=1)).cpu()
            return val.detach(), token_next
        elif self.METHOD == 3:
            val = torch.linalg.norm(val, dim=1).cpu()
            return val.detach(), token_next
        elif self.method == 4:
            val = torch.divide(torch.linalg.norm(AB - A - B, dim=1), torch.linalg.norm(AB, dim=1)).cpu()
            return val.detach(), token_next

    

    
    def calculate_interaction(self, encoded_row, row_number, attention_mask):
        interactions = []
        encoded_len = len(encoded_row)
        for j in range( min(self.SEQ_LEN, encoded_len)):
            probability = self.PROB
            if random.random() <= probability: 
                for k in range(j+1, min(self.SEQ_LEN, encoded_len, j+9)):
                    if j+k >= encoded_len:
                        continue
                    if encoded_row[j] == self.tokenizer.unk_token_id or encoded_row[k] == self.tokenizer.unk_token_id:
                        continue

                    og = encoded_row.clone()
                    og = og.reshape(1, -1)
                    iv = self.interaction_value_di(og, [j, k], attention_mask)
                    interactions.append([iv, abs((j-k)), row_number, j, k])
                
        return interactions
    

    def run_avg_interactions(self, suffix=''):
        average_distance = []
        mwes = []

        for row_number, row in tqdm(enumerate(self.test), total=len(self.test)):
            
            if self.LANGUAGE == 'english': 
                encoded_row =  self.tokenizer(row, padding=False, truncation=True, max_length=self.SEQ_LEN, return_tensors ='pt')
            else:
                encoded_row =  self.tokenizer(row, padding=False, is_split_into_words=True, truncation=True, max_length=self.SEQ_LEN, return_tensors ='pt')
            
            if self.MODEL_NAME in ['bert-base-multilingual-cased']:
                idx = 0
                
                enc =[self.tokenizer.encode(x, padding=False,  truncation=True, max_length=self.SEQ_LEN) for x in row]
                
                # print(enc)
                desired_output = []
                
                for token in enc:
                    tokenoutput = []
                    for ids in token[1:-1]:
                      tokenoutput.append(idx)
                      idx +=1
                    desired_output.append(tokenoutput)
                g = {i : x for i, x in enumerate(desired_output)}
                mwe = [row_number, list(g.values())]
            else:
                g = {}
                for ix, el in enumerate(encoded_row.word_ids()):
                    if el is not None:
                        if el not in g:
                            g[el] = []
                        g[el].append(ix)
                mwe = [row_number, list(g.values())]

            attention_mask = encoded_row['attention_mask']
            encoded_row = encoded_row.input_ids[0]
            if self.CUDA:
                encoded_row = encoded_row.cuda()
                attention_mask = attention_mask.cuda()

            average_distance.extend(self.calculate_interaction(encoded_row, row_number, attention_mask))
            mwes.append(mwe)
            if row_number % 1000 == 0:
                print(len(average_distance))
                pickle.dump(average_distance, open(f'avg_{self.MODEL_NAME}{suffix}{self.LANGUAGE}{self.PROB}.pkl','wb'))
                pickle.dump(mwes, open(f'mwe_{self.MODEL_NAME}{suffix}{self.LANGUAGE}{self.PROB}.pkl','wb'))
                break
                
    def run_experiment(self, mwe=True, avg=True, suffix=''):
        self.prepare_data()
        self.prepare_model()
        self.run_avg_interactions(suffix=suffix)

if __name__  == '__main__':
    ExperimentRunner(cuda=True, seq_len=50, model_name = 'xlm-roberta-base', method=105, lang='english', prob=0.1).run_experiment(suffix='101') 
    # ExperimentRunner(cuda=True, seq_len=50, model_name = 'xlm-roberta-base', method=105, lang='german', prob=0.02).run_experiment(suffix='101') 
    # ExperimentRunner(cuda=True, seq_len=50, model_name = 'xlm-roberta-base', method=105, lang='turkish', prob=0.02).run_experiment(suffix='101') 
    
    # ExperimentRunner(cuda=True, seq_len=50, model_name = 'xlm-roberta-base', method=105, lang='turkish').run_experiment(suffix='100') 

