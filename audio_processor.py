import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import soundfile as sf

# Load audio file
audio, sample_rate = sf.read('D:\py_venvs\DLIVING_16k\DLIVING\ch01.wav')
audio2, sample_rate2 = sf.read('D:\py_venvs\DLIVING_16k\DLIVING\ch02.wav')

# Create time axis
time = np.arange(0, len(audio)) / sample_rate
time2 = np.arange(0, len(audio2)) / sample_rate2

mpl.rc('lines', linewidth=0.5)
# Plot audio signal
#plt.plot(time, audio, linewidth=0.08)
#plt.plot(time2, audio2, linewidth2=0.08)
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
ax1.plot(time, audio)
ax2.plot(time2, audio2)

for ax in fig.get_axes():
    ax.label_outer()

plt.show()