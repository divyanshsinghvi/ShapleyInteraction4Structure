
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from datasets import load_dataset

import numpy as np
import scipy
# data preprocessing functions



model_id = "gpt2"

def get_prediction_fn(model, y = None):
    # return lambda x : np.max(model(x).logits.detach().numpy())
    if y is None:
        return lambda x : get_logodds(model(x).logits)
    else:
        return lambda x, y  : get_logodds(model(x).logits, y)

def get_model():
    return GPT2LMHeadModel.from_pretrained(model_id)#.to(device)

def get_samples(seq_len, N, k):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id, pad_token = '[PAD]')
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

def get_logodds(logits, y = None):
    """ Calculates log odds from logits.

    This function passes the logits through softmax and then computes log odds for the output(target sentence) ids.
    Returns: Computes log odds for corresponding output ids.
    """
    # set output ids for which scores are to be extracted
    def calc_logodds(arr):
        # probs = np.exp(arr) / np.exp(arr).sum(-1)
        probs = scipy.special.softmax(arr)
        logodds = scipy.special.logit(probs)
        return logodds

    # pass logits through softmax, get the token corresponding score and convert back to log odds (as one vs all)
    logodds = np.apply_along_axis(calc_logodds, -1, logits.detach().numpy())
    
    if y is None:
        logodds = np.max(logodds, axis=-1)
    else:
        logodds = logodds[:, np.array(range(logodds.shape[1])), np.repeat(y, logodds.shape[1])
                          .reshape(logodds.shape[0], logodds.shape[1], 1)]
    return logodds