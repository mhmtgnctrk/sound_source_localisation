import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import soundfile as sf

# Load audio file
sig1, sample_rate = sf.read('D:\py_venvs\DLIVING_16k\DLIVING\ch01.wav')
sig2, sample_rate2 = sf.read('D:\py_venvs\DLIVING_16k\DLIVING\ch16.wav')

sig1 = sig1 / np.linalg.norm(sig1)
sig2 = sig2 / np.linalg.norm(sig2)

mpl.rc('lines', linewidth=0.5)

# Gerçek sinyalleri görselleştirme
plt.figure()
plt.plot(sig1, label="Microphone 1")
plt.plot(sig2, label="Microphone 2")
plt.legend()
plt.show()