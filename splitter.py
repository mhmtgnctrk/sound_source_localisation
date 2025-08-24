import soundfile as sf
import numpy as np
import matplotlib as plt

data, fs = sf.read('-222_292_530.wav')  # data shape: (n_samples, 4)
data = data[8000:]

for ch in range(4):
    sf.write(f'mic{ch+1}.wav', data[:,ch], fs)