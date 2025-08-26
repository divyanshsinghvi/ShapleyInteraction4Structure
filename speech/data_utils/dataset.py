from datasets import load_dataset
import soundfile as sf
import os
import numpy as np

hf_token = os.environ.get("HF_TOKEN")

# Create output directory if it doesn't exist
output_dir = "mfa_inp_new"
os.makedirs(output_dir, exist_ok=True)

# Enable streaming to avoid downloading the full dataset
dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="train",
    token=hf_token,
    trust_remote_code=True,
    streaming=True
)

# Get the first 200 samples and save them
for i, sample in enumerate(dataset):
    if i >= 200:
        break
    audio_array = sample['audio']['array']
    sampling_rate = sample['audio']['sampling_rate']
    audio_array = np.array(audio_array, dtype=np.float32)
    if audio_array.size == 0:
        continue  # skip empty audio
    wav_path = os.path.join(output_dir, f"data_{i}.wav")
    txt_path = os.path.join(output_dir, f"data_{i}.txt")
    sf.write(wav_path, audio_array, sampling_rate)
    with open(txt_path, "w") as f:
        f.write(sample['sentence'])
    print(f"Saved {wav_path} and {txt_path}")
