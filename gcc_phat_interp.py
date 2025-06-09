import numpy as np
from numpy.fft import fft, ifft
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wavfile
from tqdm import tqdm

def __butter_bandpass(lowcut, highcut, fs, order=5):
    
    # Design a Butterworth bandpass filter.
    
    # Parameters:
    # lowcut (float): Lower frequency cutoff.
    # highcut (float): Upper frequency cutoff.
    # fs (float): Sampling frequency of the signal.
    # order (int): The order of the filter.
    
    # Returns:
    # b, a (tuple): Numerator (b) and denominator (a) polynomials of the IIR filter.
    
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def __butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    
    # Apply a Butterworth bandpass filter to a signal.
    
    # Parameters:
    # data (array): Input signal.
    # lowcut (float): Lower frequency cutoff.
    # highcut (float): Upper frequency cutoff.
    # fs (float): Sampling frequency of the signal.
    # order (int): The order of the filter.
    
    # Returns:
    # y (array): Filtered signal.
    
    b, a = __butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def __gcc_phat(sig, refsig, fs=1, interp=16, lowcut=None, highcut=None):
    
    # Perform GCC-PHAT with optional band-pass filtering.
    
    # Parameters:
    # sig (array): Signal from the first microphone.
    # refsig (array): Signal from the second microphone (reference).
    # fs (int): Sampling frequency of the signals.
    # interp (int): Interpolation factor to increase resolution.
    # lowcut (float): Lower frequency for band-pass filter.
    # highcut (float): Upper frequency for band-pass filter.
    
    # Returns:
    # tau (float): Estimated time difference of arrival.
    
    # bellek tüketim hatasının giderilmesi
    sig = sig.astype(np.float32)
    refsig = refsig.astype(np.float32)
    
    # Apply bandpass filtering if frequency cutoffs are provided
    if lowcut is not None and highcut is not None:
        sig = __butter_bandpass_filter(sig, lowcut, highcut, fs)
        refsig = __butter_bandpass_filter(refsig, lowcut, highcut, fs)

    # FFT of both signals
    n = sig.shape[0] + refsig.shape[0]
    SIG = fft(sig, n=n)
    REFSIG = fft(refsig, n=n)
    
    # Cross-power spectrum
    R = SIG * np.conj(REFSIG)
    
    # Apply PHAT weighting
    R /= np.abs(R) + 1e-15  # Avoid division by zero
    
    # Inverse FFT to get the cross-correlation
    cc = ifft(R)
    
    # Interpolation for higher resolution
    cc = np.concatenate([cc[-int(n / 2):], cc[:int(n / 2)]])
    
    # Resampling the cross-correlation with interpolation factor
    cc = np.interp(np.linspace(0, len(cc), len(cc) * interp), np.arange(len(cc)), np.abs(cc))
    
    # Find the index of the peak
    max_idx = np.argmax(cc)
    
    # Calculate time shift (TDOA)
    max_shift = len(cc) // 2
    tau = (max_idx - max_shift) / float(interp * fs)
    
    return tau

# Her mikrofon çifti arasındaki zaman farkını hesapla
def gcc_phat_array(mic_array, ref_mic, fs=1, interp=16, lowcut=None, highcut=None):
    time_delays=[]
    progress_bar1 = tqdm(total=len(mic_array)-1)
    for i in range(len(mic_array)):
        if np.array_equal(ref_mic,mic_array[i]):
            time_delays.append(0)
            continue
        tau = __gcc_phat(ref_mic, mic_array[i], fs=fs, interp=interp, lowcut=lowcut, highcut=highcut)
        time_delays.append(-1*tau)
        progress_bar1.update(1)
    return time_delays

# Zaman farklarını yazdır
def print_tdoa(time_delays):
    
    for mic1, mic2, tau in time_delays:
       print(f"\nTime delay between microphone {mic1} and microphone {mic2}: {tau:.6f} seconds")


def visualize_gcc_phat(act_mic_data):
    # Gerçek sinyalleri görselleştirme
    fig, axs = plt.subplots(len(act_mic_data))
    for idx,m in enumerate(act_mic_data):
        axs[idx]=plt.specgram(m, Fs=fs, vmin=-50, scale='dB')
    plt.colorbar()
    plt.grid()
    plt.tight_layout()
    plt.show()
