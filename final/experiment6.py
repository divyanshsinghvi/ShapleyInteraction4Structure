import torch
import pandas as pd
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoFeatureExtractor
from tqdm import tqdm
import os
from time import perf_counter
import torchaudio
import librosa

hf_token = os.environ.get("HF_TOKEN")

class SpeechSTIIExperimentRunner:
    def __init__(self, model_name, right_max=1, num_perm=2, max_length=399):
        self.device = torch.device(0)
        print(self.device)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, token=hf_token).to(self.device)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name, token=hf_token)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name, token=hf_token)
        self.RIGHT_MAX = right_max
        self.NUMBER_OF_PERM = num_perm
        self.max_length = max_length

    def resample_waveform(self, waveform, original_sr, target_sr=16000):
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform
    

    def load_audio(self, audio_path):
        # waveform, sample_rate = sf.read(audio_path)
        # print(waveform.shape)
        # return waveform, sample_rate
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self.resample_waveform(waveform, sample_rate)  # Now at 16kHz
        # print(waveform.squeeze().numpy().shape)
        return waveform.squeeze().numpy(), 16000

    def load_phonemes(self, phoneme_csv_path):
        return pd.read_csv(phoneme_csv_path)

    def generate_perm(self, nperm, nphonemes):
        perm_stored = []
        arg = np.arange(nphonemes)
        i = 0
        while i < nperm:
            perm = np.random.permutation(arg)
            perm_stored.append(perm)
            perm_stored.append(perm[::-1])
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

    def mask_waveform(self, waveform, sample_rate, mask):
        # mask: 1 means mask (zero out), 0 means keep
        masked_waveform = waveform.copy()
        for idx, m in enumerate(mask):
            if m:
                start = int(idx)
                end = int((idx+1))
                masked_waveform[start:end] = 0
        return masked_waveform

    def get_prediction_softmax(self, input_values):
        with torch.no_grad():
            logits = self.model(input_values).logits
            prediction = logits.softmax(dim=-1)
        return prediction

    def pad_tensor_to_target(self, tensor, target_size):
        import torch.nn.functional as F
        padding_size = target_size - tensor.size(1)
        return F.pad(tensor, (0, 0, 0, padding_size), 'constant', 0)


    def interaction_value_stii(self, waveform, sample_rate, phonemes_df):
        # print(waveform.shape)

        # nframes = ((waveform.size//320) + (waveform.size % 320 > 0))
        ninput = waveform.size

        # nphonemes = len(phonemes_df)
        # print(phonemes_df)
        # phoneme_intervals = [(row['start'], row['end']) for _, row in phonemes_df.iterrows()]
        results = []

        perms = self.generate_perm(self.NUMBER_OF_PERM, ninput)

        for l in range(ninput):
            for r in range(l+1, min(ninput, l+self.RIGHT_MAX+1)):
                l_time = l / 16000
                r_time = r / 16000

                print(l_time, r_time)
                if phonemes_df[(l_time >= phonemes_df['start']) & (r_time <= phonemes_df['end'])].size > 0:
                    continue

                templ = phonemes_df[(l_time >= phonemes_df['start']) & (l_time <= phonemes_df['end'])]
                
                if templ.size  == 0:
                    continue
                else:
                    templ = templ.iloc[0]

                tempr = phonemes_df[(r_time >= phonemes_df['start']) & (r_time <= phonemes_df['end'])]
                
                if tempr.size  == 0:
                    continue
                else:
                    tempr = tempr.iloc[0]
                # Generate incremental masks for this (l, r) pair
                masks = self.generate_incremental_masks(perms, l, r)
                masks = np.array(masks)

                # For each permutation, create 4 waveform variants: AB, A, B, Phi
                smax_ab_list, smax_a_list, smax_b_list, smax_phi_list = [], [], [], []

                for mask in masks:
                    # t4 = perf_counter()
                    # AB: mask as per permutation up to l/r
                    waveform_ab = self.mask_waveform(waveform, sample_rate, mask)
                    input_values_ab = self.tokenizer(
                        waveform_ab, return_tensors="pt", padding='max_length', max_length=self.max_length, sampling_rate=16000,
                    ).input_values.to(self.device)
                    smax_ab = self.get_prediction_softmax(input_values_ab)[0]
                    smax_ab_list.append(smax_ab)

                    # A: mask + l
                    mask_a = mask.copy()
                    mask_a[l] = 1
                    waveform_a = self.mask_waveform(waveform, sample_rate, mask_a)
                    input_values_a = self.tokenizer(
                        waveform_a, return_tensors="pt", padding='max_length', max_length=self.max_length, sampling_rate=16000,
                    ).input_values.to(self.device)
                    smax_a = self.get_prediction_softmax(input_values_a)[0]
                    smax_a_list.append(smax_a)

                    # B: mask + r
                    mask_b = mask.copy()
                    mask_b[r] = 1
                    waveform_b = self.mask_waveform(waveform, sample_rate, mask_b)
                    input_values_b = self.tokenizer(
                        waveform_b, return_tensors="pt", padding='max_length', max_length=self.max_length, sampling_rate=16000,
                    ).input_values.to(self.device)
                    smax_b = self.get_prediction_softmax(input_values_b)[0]
                    smax_b_list.append(smax_b)

                    # Phi: mask + l + r
                    mask_phi = mask.copy()
                    mask_phi[l] = 1
                    mask_phi[r] = 1
                    waveform_phi = self.mask_waveform(waveform, sample_rate, mask_phi)
                    input_values_phi = self.tokenizer(
                        waveform_phi, return_tensors="pt", padding='max_length', max_length=self.max_length, sampling_rate=16000,
                    ).input_values.to(self.device)
                    smax_phi = self.get_prediction_softmax(input_values_phi)[0]
                    smax_phi_list.append(smax_phi)
                    # t5 = perf_counter()
                    # print(f"    AB waveform masked in {t5 - t4:.2f} seconds")

                # Stack and compute norms as in STII
                smax_ab = torch.stack(smax_ab_list)
                smax_a = torch.stack(smax_a_list)
                smax_b = torch.stack(smax_b_list)
                smax_phi = torch.stack(smax_phi_list)

                smax = smax_ab - smax_a - smax_b + smax_phi
                smax = torch.linalg.norm(smax, dim=-1)
                smax = torch.mean(smax, dim=0).cpu().detach().numpy()

                smax_ab_norm = torch.linalg.norm(smax_ab, dim=-1)
                smax_ab_norm = torch.mean(smax_ab_norm, dim=0).cpu().detach().numpy()



                phoneme_l = templ['phoneme']; start_l = templ['start']; end_l = templ['end']
                phoneme_r = tempr['phoneme']; start_r = tempr['start']; end_r = tempr['end']
                # print(smax.shape)
                # For each output frame, store the normalized interaction
                for idx in range(len(smax)):
                    results.append({
                        "Frame_L_index": l,
                        "Phoneme_L": phoneme_l,
                        "start_L": start_l,
                        "end_L": end_l,
                        "Frame_R_index": r, 
                        "Phoneme_R": phoneme_r,
                        "start_R": start_r,
                        "end_R": end_r,
                        "TotalFrames": ninput,
                        "STII_Interaction": smax[idx] / smax_ab_norm[idx] if smax_ab_norm[idx] != 0 else 0.0
                    })
        return results

    def run(self, audio_path, phoneme_csv_path, output_csv_path):
        waveform, sample_rate = self.load_audio(audio_path)
        phonemes_df = self.load_phonemes(phoneme_csv_path)

        print(f"Running STII Shapley for {len(phonemes_df)} phonemes...")
        results = self.interaction_value_stii(waveform, sample_rate, phonemes_df)
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv_path, index=False)
        print(f"Saved STII Shapley interactions to {output_csv_path}")

# Usage:

def run_sample(number):
    try:
        runner = SpeechSTIIExperimentRunner(model_name="facebook/wav2vec2-base-960h")
        print(f"🔁 Running sample {number}")
        audio_path = f"../speech_data/mfa_inp_new/data_{number}.wav"
        phoneme_csv_path = f"../speech_data/extracted_phonemes/phonemes_{number}.csv"
        output_csv_path = f"../speech_data/stii_outputs_fix1/stii_{number}.csv"
        
        # Only run if input files exist
        if not os.path.exists(audio_path) or not os.path.exists(phoneme_csv_path):
            print(f"⚠️ Skipping {number}: missing files.")
            return

        runner.run(audio_path, phoneme_csv_path, output_csv_path)
        print(f"✅ Finished sample {number}")
    except Exception as e:
        print(f"❌ Error in sample {number}: {e}")

# os.makedirs("../speech_data/stii_outputs_fix1/", exist_ok=True)
# from tqdm import tqdm
# from joblib import Parallel, delayed
# Parallel(n_jobs=11)(
#     delayed(run_sample)(num) for num in tqdm(range(1, 201))
# )

runner = SpeechSTIIExperimentRunner(model_name="facebook/wav2vec2-base-960h")

number = 19
runner.run(
    audio_path=f"../speech_data/mfa_inp_new/data_{number}.wav",
    phoneme_csv_path=f"../speech_data/extracted_phonemes/phonemes_{number}.csv",
    output_csv_path="../speech_data//common_voice_sample_stii_shapley_interactions.csv"
)
