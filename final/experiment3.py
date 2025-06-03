import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoFeatureExtractor

import soundfile as sf
import torch
import numpy as np
import pandas as pd
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
# torch.cuda.set_device(1)
# print(torch.cuda.current_device())

OUTPUT_COMPLETE_MAPPED = 'common_voice_sample_phonemes.csv'
WAV_FILE = 'common_voice_sample.wav'

class Wav2VecExperimentRunner:
    def __init__(self, cuda, model_name):
        self.CUDA = cuda
        self.MODEL_NAME = model_name
        self.device = torch.device('cuda' if cuda else 'cpu')

    def prepare_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(self.MODEL_NAME).to(self.device)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.MODEL_NAME)
        self.extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_NAME)

    # def load_data(self):
    #     waveform, sample_rate = sf.read(WAV_FILE)
    #     return waveform, sample_rate

    # def prepare_data(self):
    #     dataset = load_dataset("mozilla-foundation/common_voice_12_0", "en", split="train[:10]")
    #     self.test = dataset

    def prepare_data(self):
        self.test=pd.read_csv(OUTPUT_COMPLETE_MAPPED)

        ########### NOTE THIS IS DONE ONLY FOR ONE WAV FILE ############
        self.test = self.test
        self.test = pd.DataFrame([{
            "phoneme": self.test["phoneme"].tolist(),
            "phoneme_start": self.test["start"].tolist(),
            "phoneme_end": self.test["end"].tolist(),
            "phoneme_duration": (self.test["end"] - self.test["start"]).tolist()
        }])

        # self.test = df.groupby(['wav_file_name', 'txt_file_name', 'textgrid_name']).agg({
        #     'phoneme': list,
        #     'phoneme_start': list,
        #     'phoneme_end': list,
        #     'phoneme_duration': list
        # }).reset_index()
        # self.test = self.test.sample(n=500, random_state=42)

    def pad_tensor_to_target(self,tensor, target_size):
        print(target_size, tensor.size)
        padding_size = target_size - tensor.size(1)
        return F.pad(tensor, (0, 0, 0, padding_size), 'constant', 0)

    def get_prediction_softmax(self, waveform, next_phoneme_start):
        print(waveform.size())
        logits = self.model(waveform).logits
        prediction=logits.softmax(dim=-1)
        # Return the numpy array if not using CUDA, else return the tensor
        if not self.CUDA:
            return prediction.detach().numpy()
        else:
            return prediction

    def generate_perm(self, nperm, n):
        perm_stored = []
        arg = np.arange(n)
        for _ in range(nperm):
            perm = np.random.permutation(arg)
            perm_stored.append(perm)
            perm_stored.append(perm[::-1])
        return np.array(perm_stored)
    

    #####  NEEDS TO BE VERIFIED 
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
    #####  NEEDS TO BE VERIFIED 

    def apply_phoneme_mask(self, waveform, phoneme_intervals, mask):
        waveform = waveform.copy()
        for i, keep in enumerate(mask):
            if not keep:
                start_sample = int(phoneme_intervals[i][0] * self.sample_rate)
                end_sample = int(phoneme_intervals[i][1] * self.sample_rate)
                waveform[start_sample:end_sample] = 0
        return waveform

    def interaction_value_shapley(self, waveform, phoneme_intervals, l, r, num_perm=4):
        print(waveform.size)
        print("-----------")
        print(waveform)
        perms = self.generate_perm(num_perm, len(phoneme_intervals))
        masks = self.generate_incremental_masks(perms, l, r)
        smax_values = []
        for mask in masks:
            base_mask = mask.copy()
            A_mask = base_mask.copy(); A_mask[l] = 0
            B_mask = base_mask.copy(); B_mask[r] = 0
            phi_mask = base_mask.copy(); phi_mask[l] = 0; phi_mask[r] = 0

            def get_softmax(mask):
                masked_waveform = self.apply_phoneme_mask(waveform, phoneme_intervals, mask)
                # input_values = self.tokenizer(masked_waveform, return_tensors="pt", sampling_rate=self.sample_rate, padding="max_length", max_length=399).input_values.to(self.device)
                input_values = self.tokenizer(masked_waveform, return_tensors="pt", sampling_rate=self.sample_rate, padding="max_length").input_values.to(self.device)
                return self.get_prediction_softmax(input_values, 0)

            AB = self.pad_tensor_to_target(get_softmax(base_mask), 399).squeeze(0)
            A = self.pad_tensor_to_target(get_softmax(A_mask), 399).squeeze(0)
            B = self.pad_tensor_to_target(get_softmax(B_mask), 399).squeeze(0)
            phi = self.pad_tensor_to_target(get_softmax(phi_mask), 399).squeeze(0)

            num = torch.linalg.norm(AB - A - B + phi, ord=2)
            den = torch.linalg.norm(AB, ord=2)
            smax_values.append((num / den).item())

        return np.mean(smax_values)

    def calculate_interaction(self, row, row_number):
        interactions = []
        phoneme_list = row['phoneme']
        start_times = row['phoneme_start']
        end_times = row['phoneme_end']
        phoneme_intervals = list(zip(start_times, end_times))
        waveform, self.sample_rate = sf.read(WAV_FILE)
        print(self.sample_rate)

        for i in range(1, len(phoneme_list) - 1):
            val = self.interaction_value_shapley(waveform, phoneme_intervals, i, i+1)
            interactions.append({
                "Row Number": WAV_FILE,
                "Phoneme From": phoneme_list[i],
                "Phoneme To": phoneme_list[i+1],
                "Shapley Value": val
            })
        return interactions

    def run_avg_interactions(self, suffix=''):
        results = []
        print(self.test)
        for index, row in tqdm(self.test.iterrows(), total=len(self.test)):
            print(f"Processing {index}: {WAV_FILE}")
            results.extend(self.calculate_interaction(row, index))
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'interactions_shapley_{suffix}.csv', index=False)

    def run_experiment(self, suffix=''):
        self.prepare_data()
        self.prepare_model()
        self.run_avg_interactions(suffix=suffix)


if __name__ == '__main__':
    Wav2VecExperimentRunner(cuda=False, model_name='facebook/wav2vec2-base-960h').run_experiment(suffix='100')

