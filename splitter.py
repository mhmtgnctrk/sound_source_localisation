import soundfile as sf

data, fs = sf.read('kinect_raw.wav')  # data shape: (n_samples, 4)
for ch in range(4):
    sf.write(f'mic{ch+1}.wav', data[:,ch], fs)