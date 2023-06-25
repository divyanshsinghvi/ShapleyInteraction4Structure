print("Importing dependencies..")

import scipy
import numpy as np
from scipy.linalg import lstsq
from scipy.linalg import norm 
import pandas as pd
import os
from util import RegressionGame
from util_sparse import getShapleyProjection
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import shap
import llm_helper
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

model = llm_helper.get_model()
X = llm_helper.get_samples(seq_len)

# TODO: Fix the prediction fn. Might have to incorporate
#  target variable to get the logit for the target variable instead of max
predict_fn = llm_helper.get_prediction_fn(model)



torch.no_grad()

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

