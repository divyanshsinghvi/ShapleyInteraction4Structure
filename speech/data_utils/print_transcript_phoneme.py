# Example: Print transcript and phonemes for one file
tg_path = os.path.join(input_dir, "data_0.TextGrid")
txt_path = os.path.join("mfa_inp_new", "data_0.txt")

# Print transcript
with open(txt_path, "r") as f:
    transcript = f.read()
print("Transcript:", transcript)

# Print phonemes from TextGrid
tg = TextGrid.fromFile(tg_path)
phones_tier = None
for tier in tg.tiers:
    if tier.name.lower() in ["phones", "phone", "phoneme"]:
        phones_tier = tier
        break

if phones_tier:
    print("Extracted phonemes:")
    for interval in phones_tier.intervals:
        if interval.mark.strip():
            print(interval.mark, f"{interval.minTime:.2f}-{interval.maxTime:.2f}")
else:
    print("No phones tier found.")
