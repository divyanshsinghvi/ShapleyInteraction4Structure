
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from datasets import load_dataset

import numpy as np
import scipy
# data preprocessing functions



model_id = "gpt2"

def get_prediction_fn(model):
    return lambda x : np.max(model(x).logits.detach().numpy())
    # predict_fn = lambda x : get_logodds(model(x).logits)

def get_model():
    return GPT2LMHeadModel.from_pretrained(model_id)#.to(device)

def get_samples(seq_len):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id, pad_token = '[PAD]')
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encoding = tokenizer([test["text"][num] for num in [4, 11, 12, 16]],padding=True,  truncation=True,max_length =seq_len, return_tensors ='pt').input_ids
    print(encoding.shape)
    X = encoding[:,:seq_len-1]
    return X

def get_logodds(logits):
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

    logodds = np.max(logodds, axis=-1)
    # logodds_for_output_ids = logodds[:, np.array(range(logodds.shape[1])), :]
    return logodds