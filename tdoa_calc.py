import scipy.io.wavfile as wavfile
import numpy as np
import gcc_phat_interp as gpi
import gcc_phat as gp
from math import *

# Mikrofon kayıtlarını saklayacak liste
mic_data = []
fs = None

# 4 mikrofonun wav dosyalarını okuma
for i in range(1, 5):
    pre_file=r"C:\Users\bn.y.mn\Desktop\sound_source\sound_source_localisation"
    if i <=9:
        filename = pre_file + f'\mic{i}.wav' 
  #  else:
  #      filename = f'C:\Users\bn.y.mn\Desktop\sound_source\sound_source_localisation\mic{i}.wav'
    fs, data = wavfile.read(filename)
    mic_data.append(data)

mic_data = np.array(mic_data)

mics, samples = mic_data.shape
print(f"Sampling rate: {fs}")
print(f"Data shape (microphones, samples): {mics, samples}")

# Referans mikrofon
ref_mic = mic_data[0]
for idx, m in enumerate(mic_data):
    if np.array_equal(ref_mic, m):
        ref_index = idx
        continue

tdoas = gpi.gcc_phat_array(mic_data,ref_mic,fs=fs)
f = open("tdoa/tdoas.txt", "w")
    
for tdoa in tdoas:
    f.write((str(tdoa)+"\n"))
f.close()

print("TDOAS: ",tdoas)