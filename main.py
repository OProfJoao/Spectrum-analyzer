import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


audiofile = 'AudioFile.m4a'


y, sr = librosa.load(audiofile, sr=None)
print(f"Sample rate: {sr}")

frame_duration = 0.2
frame_samples = int(sr*frame_duration)

for i in range(0,len(y)-frame_samples,frame_samples):
    y_frame = y[i:i+frame_samples]
    freqs = np.fft.rfftfreq(len(y_frame),1/sr)
    spec = np.abs(np.fft.rfft(y_frame))
    spectrum_db = 20 * np.log10(spec + 1e-6)
        # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, spectrum_db)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Spectrum of frame starting at {i/sr:.2f}s')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
    

# n_fft = 2048
# hop_length = 256


# S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
# print("Shape da STFT (freqs x tempo):", S.shape)

# df = pd.DataFrame(S)
# print(S)

# plt.figure(figsize=(12, 4), dpi=300)

# img = librosa.display.specshow(
#     librosa.amplitude_to_db(S, ref=np.max),
#     sr=sr,
#     hop_length=hop_length,
#     y_axis='linear',
#     x_axis='time'
# )

# plt.savefig("spectrum.png", dpi=300)
# df.to_csv('Spectrum.csv')

