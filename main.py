import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

audiofile = 'AudioFile.m4a'


y, sr = librosa.load(audiofile, sr=None)
print(f"Sample rate: {sr}")

n_fft = 2048
hop_length = 256


S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
print("Shape da STFT (freqs x tempo):", S.shape)

df = pd.DataFrame(S)
print(S)

plt.figure(figsize=(12, 4), dpi=300)

img = librosa.display.specshow(
    librosa.amplitude_to_db(S, ref=np.max),
    sr=sr,
    hop_length=hop_length,
    y_axis='linear',
    x_axis='time'
)

plt.savefig("spectrum.png", dpi=300)
df.to_csv('Spectrum.csv')

