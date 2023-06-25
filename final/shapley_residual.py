print("Importing dependencies..")

import scipy
import numpy as np
from scipy.linalg import lstsq
from scipy.linalg import norm 
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
from util import RegressionGame
from util_sparse import getShapleyResiduals, getShapleyProjection
import xgboost
from sklearn.model_selection import train_test_split
import sklearn
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import scipy
import shap
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# device = "cuda"
# model_id = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
# tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# from datasets import load_dataset

# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# encodings = tokenizer(test["text"][4], return_tensors="pt")
# explainer = shap.Explainer(model, tokenizer)
# shap_values = explainer([test["text"][4][:60]])




if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('__dcache__'):
    os.makedirs('__dcache__')

N = 2
print("Explanation count: %s" % N)
k = 2
print("SHAP sample count: %s" % k)
seq_len = 10
print("Seq Length: %s" % seq_len)



# Setup
np.random.seed(1)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# device = "cpu"
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id)#.to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id, pad_token = '[PAD]')

from datasets import load_dataset


# data preprocessing functions
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

encoding = tokenizer([test["text"][num] for num in [4, 11, 12, 16]],padding=True,  truncation=True,max_length =seq_len, return_tensors ='pt').input_ids

print(encoding.shape)
X = encoding[:,:seq_len-1]
# y = encoding[:,seq_len-1]



torch.no_grad()

def get_logodds(logits):
    """ Calculates log odds from logits.

    This function passes the logits through softmax and then computes log odds for the output(target sentence) ids.

    Parameters
    ----------
    logits: numpy.ndarray
        An array of logits generated from the model.

    Returns
    -------
    numpy.ndarray
        Computes log odds for corresponding output ids.
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

# predict_fn = lambda x : get_logodds(model(x).logits)
predict_fn = lambda x : np.max(model(x).logits.detach().numpy())

obj = RegressionGame(X = X[0:k], function = predict_fn, transform = torch.as_tensor)

X_samp = X[k:(N+k)]

shapley_values = np.empty((0, X.shape[1]))
partial_residuals = np.empty((0, X.shape[1]))
games = np.empty((0, 2 ** X.shape[1]))

print("SHape")
print(X.shape[1])

print("  ..ok!")
print("Generating explanations..")

for i in range(0, N):
    example_row = X_samp[i,:].reshape((1,X_samp.shape[1]))
    game = obj.getKernelSHAPGame(example_row)
    games = np.append(games, game.reshape((1,game.shape[0])), axis = 0)
    results, residualGame, origGame = getShapleyProjection(game)
    shapley_values = np.append(shapley_values,
                               np.array([np.flip(results[-1])]), axis=0)
    partial_residuals = np.append(partial_residuals,
                                  np.array([np.flip(norm(residualGame, axis = 0)/norm(origGame, axis = 0))]), axis = 0)
    print("%s/%s samples done." % (i+1, N))

print(" Explanations saved to data/llm_*.csv!")

pd.DataFrame(X_samp).to_csv('data/llm_input.csv')
pd.DataFrame(shapley_values).to_csv('data/llm_shapley_values.csv')
pd.DataFrame(partial_residuals).to_csv('data/llm_partial_residuals.csv')

