import soundfile as sf
import numpy as np
import matplotlib as plt

data, fs = sf.read(r'E:\git_projects\sound_source_localisation\sound_source_localisation\-212_112_300.wav')  # data shape: (n_samples, 4)
data = data[8000:]

for ch in range(4):
    sf.write(f'E:\git_projects\sound_source_localisation\sound_source_localisation\mic{ch+1}.wav', data[:,ch], fs)