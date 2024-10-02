import scipy.io.wavfile as wavfile
import numpy as np
from gcc_phat import *
from math import *

# Mikrofon kayıtlarını saklayacak liste
mic_data = []
fs = None

# 16 mikrofonun wav dosyalarını okuma
for i in range(1, 17):
    if i <=9:
        filename = f'D:\py_venvs\DLIVING_16k\DLIVING\ch0{i}.wav'
    else:
        filename = f'D:\py_venvs\DLIVING_16k\DLIVING\ch{i}.wav'
    fs, data = wavfile.read(filename)
    mic_data.append(data)

mic_data = np.array(mic_data)

mics, samples = mic_data.shape
print(f"Sampling rate: {fs}")
print(f"Data shape (microphones, samples): {mics, samples}")
alt_lim=round(samples*13/30)
ust_lim=round(samples*17/30)
trimmed_mic_data = []
for m in mic_data:
    trimmed_mic_data.append(m[alt_lim:ust_lim])
print(trimmed_mic_data)

trimmed_mic_data = np.array(trimmed_mic_data)


# Referans mikrofon
ref_mic = trimmed_mic_data[0]
for idx, m in enumerate(trimmed_mic_data):
    if np.array_equal(ref_mic, m):
        ref_index = idx
        continue

tdoas = gcc_phat_array(trimmed_mic_data,ref_mic,fs=fs, interp=32)

f = open("tdoa/tdoas.txt", "w")
    
for tdoa in tdoas:
    f.write((str(tdoa)+"\n"))
f.close()

print("TDOAS: ",tdoas)