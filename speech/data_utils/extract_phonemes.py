from textgrid import TextGrid

# Load the TextGrid file
tg = TextGrid.fromFile("mfa_output/common_voice_sample.TextGrid")

# Find the 'phones' tier (sometimes called 'phone' or 'phoneme')
phones_tier = None
for tier in tg.tiers:
    if tier.name.lower() in ["phones", "phone", "phoneme"]:
        phones_tier = tier
        break

if phones_tier is None:
    raise ValueError("No 'phones' tier found in the TextGrid.")

# Extract phonemes and their time intervals
phonemes = []
for interval in phones_tier.intervals:
    if interval.mark.strip():  # skip empty intervals
        phonemes.append({
            "phoneme": interval.mark,
            "start": interval.minTime,
            "end": interval.maxTime
        })

# Print the extracted phonemes
for p in phonemes:
    print(f"{p['phoneme']}: {p['start']:.2f} - {p['end']:.2f}")
