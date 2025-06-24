import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoFeatureExtractor
from joblib import Parallel, delayed
from tqdm import tqdm

SPEECH_DATA_PATH = 'speech_data/'


class SpeechSTIIExperimentRunner:
    """
    This class runs a Shapley-based Temporal Interaction Index (STII) analysis 
    on speech audio using a Wav2Vec2 ASR model. It estimates how phoneme pairs 
    interact by masking parts of the waveform and comparing softmax predictions.
    """

    def __init__(self, model_name: str, right_max: int = 1, num_perm: int = 2, max_length: int = 399):
        """
        Initialize the experiment runner.
        Args:
            model_name: HuggingFace model name for Wav2Vec2.
            right_max: Maximum allowed frame distance between left and right.
            num_perm: Number of permutations for Shapley estimation.
            max_length: Max input length for tokenization.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, token=os.environ.get("HF_TOKEN")).to(self.device)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
        self.RIGHT_MAX = right_max
        self.NUMBER_OF_PERM = num_perm
        self.max_length = max_length

    def resample_waveform(self, waveform, original_sr, target_sr=16000):
        """Resample waveform to target sample rate if needed."""
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform

    def load_audio(self, audio_path):
        """Load audio and resample to 16kHz mono."""
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self.resample_waveform(waveform, sample_rate)
        return waveform.squeeze().numpy(), 16000

    def load_phonemes(self, phoneme_csv_path):
        """Load phoneme intervals from CSV."""
        return pd.read_csv(phoneme_csv_path)

    def generate_perm(self, nperm, nframes):
        """Generate random permutations for Shapley estimation."""
        perms = []
        indices = np.arange(nframes)
        for _ in range(nperm):
            perm = np.random.permutation(indices)
            perms.append(perm)
            perms.append(perm[::-1])
        return np.array(perms)

    def generate_incremental_masks(self, permutations, l, r):
        """
        Generate incremental masking patterns up to left/right indices for each permutation.
        """
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
        """
        Zero out masked frames in waveform based on binary mask.
        Each frame assumed to cover 20ms.
        """
        masked_waveform = waveform.copy()
        frame_len = int(0.02 * sample_rate)
        for idx, m in enumerate(mask):
            if m:
                start = idx * frame_len
                end = start + frame_len
                masked_waveform[start:end] = 0
        return masked_waveform

    def get_prediction_softmax(self, input_values):
        """Get model softmax probabilities for given input tensor."""
        with torch.no_grad():
            logits = self.model(input_values).logits
            return logits.softmax(dim=-1)

    def interaction_value_stii(self, waveform, sample_rate, phonemes_df):
        """
        Main STII computation for waveform + phonemes.
        Returns list of dictionaries with interaction scores.
        """
        results = []
        nframes = waveform.size // 320 + int(waveform.size % 320 > 0)
        permutations = self.generate_perm(self.NUMBER_OF_PERM, nframes)

        for l in range(nframes):
            for r in range(l + 1, min(nframes, l + self.RIGHT_MAX + 1)):
                # Only consider if l & r map to different phonemes
                if phonemes_df[(l * 0.02 >= phonemes_df['start']) & (r * 0.02 <= phonemes_df['end'])].size > 0:
                    continue

                phoneme_l_df = phonemes_df[(l * 0.02 >= phonemes_df['start']) & (l * 0.02 <= phonemes_df['end'])]
                phoneme_r_df = phonemes_df[(r * 0.02 >= phonemes_df['start']) & (r * 0.02 <= phonemes_df['end'])]
                if phoneme_l_df.empty or phoneme_r_df.empty:
                    continue

                phoneme_l = phoneme_l_df.iloc[0]
                phoneme_r = phoneme_r_df.iloc[0]

                # Generate masks
                masks = self.generate_incremental_masks(permutations, l, r)
                masks = np.array(masks)

                # For each mask, compute AB, A, B, Phi variants
                smax_ab, smax_a, smax_b, smax_phi = [], [], [], []

                for mask in masks:
                    for suffix, extra in zip(['ab', 'a', 'b', 'phi'],
                                             [[], [l], [r], [l, r]]):
                        m = mask.copy()
                        for e in extra:
                            m[e] = 1
                        wav_masked = self.mask_waveform(waveform, sample_rate, m)
                        input_values = self.tokenizer(
                            wav_masked, return_tensors="pt",
                            padding="max_length", max_length=self.max_length,
                            sampling_rate=16000
                        ).input_values.to(self.device)
                        smax = self.get_prediction_softmax(input_values)[0]
                        eval(f"smax_{suffix}").append(smax)

                # Stack and calculate normalized STII
                s_ab = torch.stack(smax_ab)
                s_a = torch.stack(smax_a)
                s_b = torch.stack(smax_b)
                s_phi = torch.stack(smax_phi)

                interaction = torch.linalg.norm(s_ab - s_a - s_b + s_phi, dim=-1).mean(0).cpu().detach().numpy()
                ab_norm = torch.linalg.norm(s_ab, dim=-1).mean(0).cpu().detach().numpy()

                for idx in range(len(interaction)):
                    results.append({
                        "Frame_L_index": l,
                        "Phoneme_L": phoneme_l['phoneme'],
                        "start_L": phoneme_l['start'],
                        "end_L": phoneme_l['end'],
                        "Frame_R_index": r,
                        "Phoneme_R": phoneme_r['phoneme'],
                        "start_R": phoneme_r['start'],
                        "end_R": phoneme_r['end'],
                        "TotalFrames": nframes,
                        "STII_Interaction": interaction[idx] / ab_norm[idx] if ab_norm[idx] != 0 else 0.0
                    })

        return results

    def run(self, audio_path, phoneme_csv_path, output_csv_path):
        """Full pipeline: load data, compute STII, save CSV."""
        waveform, sample_rate = self.load_audio(audio_path)
        phonemes_df = self.load_phonemes(phoneme_csv_path)
        print(f"Running STII for {len(phonemes_df)} phonemes.")
        results = self.interaction_value_stii(waveform, sample_rate, phonemes_df)
        pd.DataFrame(results).to_csv(output_csv_path, index=False)
        print(f"Saved results to {output_csv_path}")

def run_sample(number):
    """
    Utility to run SpeechSTIIExperimentRunner for a given sample number.
    """
    try:
        runner = SpeechSTIIExperimentRunner(model_name="facebook/wav2vec2-base-960h")
        print(f"Running sample {number}")
        audio_path = f"{SPEECH_DATA_PATH}/mfa_inp_new/data_{number}.wav"
        phoneme_csv_path = f"{SPEECH_DATA_PATH}/extracted_phonemes/phonemes_{number}.csv"
        output_csv_path = f"{SPEECH_DATA_PATH}/stii_outputs_fix/stii_{number}.csv"

        if not os.path.exists(audio_path) or not os.path.exists(phoneme_csv_path):
            print(f"Skipping {number}: missing files.")
            return

        runner.run(audio_path, phoneme_csv_path, output_csv_path)
        print(f"Finished sample {number}")
    except Exception as e:
        print(f"Error in sample {number}: {e}")


if __name__ == "__main__":
    # Ensure output folder exists
    os.makedirs(f"{SPEECH_DATA_PATH}/stii_outputs_fix/", exist_ok=True)

    # Run in parallel for multiple samples
    Parallel(n_jobs=11)(
        delayed(run_sample)(num) for num in tqdm(range(1, 201))
    )
