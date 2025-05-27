import pandas as pd
import numpy as np
import seaborn as sns
import librosa
import librosa.display
import matplotlib.pyplot as plt

audiofile = 'AudioFile.m4a'

y,sr = librosa.load(audiofile,
                    sr=None,
                    offset=1.0,
                    duration = 0.2)
S = np.abs(librosa.stft(y,
                        n_fft=1024,
                        hop_length = 256))


print (S.shape)

fig,ax = plt.subplots(dpi=300)
img = librosa.display.specshow(
    librosa.amplitude_to_db(
        S,
        ref = np.max,),
    y_axis='log',
    x_axis='time',
    ax=ax,)
plt.savefig("spectrum.png",dpi=300)