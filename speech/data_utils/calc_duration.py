import os
from pydub import AudioSegment

def calculate_total_duration(directory):
    total_duration = 0.0
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            audio = AudioSegment.from_wav(filepath)
            total_duration += audio.duration_seconds
    return total_duration

total_duration = calculate_total_duration("mfa_inp_new")
print(f"Total duration of audio files in 'mfa_inp_new': {total_duration} seconds")
