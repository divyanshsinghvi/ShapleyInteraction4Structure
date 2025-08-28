from textgrid import TextGrid
import pandas as pd
import os

input_dir = "mfa_output_new"
output_dir = "extracted_phonemes"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".TextGrid"):
        tg_path = os.path.join(input_dir, fname)
        tg = TextGrid.fromFile(tg_path)

        # Find the 'phones' tier
        phones_tier = None
        for tier in tg.tiers:
            if tier.name.lower() in ["phones", "phone", "phoneme"]:
                phones_tier = tier
                break

        if phones_tier is None:
            print(f"No 'phones' tier found in {fname}. Skipping.")
            continue

        phonemes = []
        for interval in phones_tier.intervals:
            if interval.mark.strip():
                phonemes.append({
                    "phoneme": interval.mark.strip(),
                    "start": interval.minTime,
                    "end": interval.maxTime
                })

        df_phonemes = pd.DataFrame(phonemes)
        out_csv = os.path.join(output_dir, fname.replace(".TextGrid", ".csv"))
        df_phonemes.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
