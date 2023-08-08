
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset

import numpy as np
import scipy
# data preprocessing functions



model_id = "gpt2"

def get_prediction_fn(model, pred_mode=1):
    # return lambda x : np.max(model(x).logits.detach().numpy())

    if pred_mode == 1:
        return lambda x : get_logodds(model(x).logits)
    elif pred_mode == 2:
        # Perplexity score
        return lambda x : np.exp(model(x, labels=x.clone()).loss.detach().numpy())


def get_model():
    # model = GPT2LMHeadModel.from_pretrained(model_id)#.to(device)
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()
    return model

def get_samples(seq_len, N, k):
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    testlist_for_tokenizer = []
    for t in test['text']:
        if '=' in t or t=='' or t.strip() == '\n':
            continue
        # Removing '\n'
        testlist_for_tokenizer.extend(t.split(".")[:-1])
        if len(testlist_for_tokenizer) >= N+k+20:
            break
    
    encoding = tokenizer(testlist_for_tokenizer, padding=True,  truncation=True, max_length=seq_len, return_tensors ='pt').input_ids
    # encoding = tokenizer([test["text"][num] for num in [4, 11, 12, 16]],padding=True,  truncation=True,max_length =seq_len, return_tensors ='pt').input_ids
    
    
    assert len(encoding)>= N+k
    
    print(encoding.shape)
    X = encoding[:N+k,:seq_len-1]
    y = encoding[:N+k,seq_len-1]
    return X, y, testlist_for_tokenizer[:N+k]

def get_logodds(logits):
    """ Calculates log odds from logits.

    This function passes the logits through softmax and then computes log odds for the output(target sentence) ids.
    Returns: Computes log odds for corresponding output ids.
    """
    # set output ids for which scores are to be extracted
    def calc_logodds(arr):
        probs = scipy.special.softmax(arr)
        return probs

    # pass logits through softmax, get the token corresponding score and convert back to log odds (as one vs all)
    logodds = np.apply_along_axis(calc_logodds, -1, logits.detach().numpy())
    logodds = np.linalg.norm(logodds, axis=-1)
    # logodds = np.max(logodds, axis=-1)
    logodds = np.linalg.norm(logodds, axis=-1)
    


    return logodds