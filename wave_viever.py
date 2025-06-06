import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

filenames = ['mic1.wav', 'mic2.wav', 'mic3.wav', 'mic4.wav']

# Create a figure with 4 subplots (stacked vertically)
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))

for idx, fname in enumerate(filenames):
    if not os.path.exists(fname):
        axes[idx].text(0.5, 0.5, f"Dosya bulunamadı: {fname}", 
                       horizontalalignment='center', 
                       verticalalignment='center',
                       transform=axes[idx].transAxes,
                       fontsize=12, color='red')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        continue
    
    # Read WAV file
    fs, data = wavfile.read(fname)
    
    # If multi-channel, use first channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Create time axis
    n_samples = data.shape[0]
    time_axis = np.arange(n_samples) / fs
    
    # Compute maximum absolute amplitude
    max_amp = np.max(np.abs(data))
    
    # Plot waveform
    axes[idx].plot(time_axis, data, linewidth=0.5)
    axes[idx].set_ylabel('Şiddet')
    axes[idx].set_title(f"{fname}  ('En Yüksek Şiddet: {max_amp})")

# Label the x-axis on the bottom subplot
axes[-1].set_xlabel('Zaman (saniye)')

plt.tight_layout()
plt.show()
