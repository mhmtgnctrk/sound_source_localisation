import numpy as np
from numpy.fft import fft, ifft
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wavfile
from tqdm import tqdm

def __butter_bandpass(lowcut, highcut, fs, order=5):
    
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

def __gcc_phat(sig, refsig, fs=1, lowcut=None, highcut=None):
    """
    GCC-PHAT without explicit interpolation step.

    Parameters:
        sig (array):      Birinci mikrofondan gelen sinyal.
        refsig (array):   Referans sinyal (ikinci mik).
        fs (float):       Örnekleme frekansı.
        lowcut (float):   (Opsiyonel) Alt frekans kesimi için band-pass filtresi.
        highcut (float):  (Opsiyonel) Üst frekans kesimi için band-pass filtresi.

    Returns:
        tau (float):      Tahmin edilen zaman farkı (TDOA) saniye cinsinden.
    """
    # Girişleri float32’ye çek
    sig = sig.astype(np.float32)
    refsig = refsig.astype(np.float32)

    # Opsiyonel band-pass
    if lowcut is not None and highcut is not None:
        sig = __butter_bandpass_filter(sig, lowcut, highcut, fs)
        refsig = __butter_bandpass_filter(refsig, lowcut, highcut, fs)
 
    # FFT boyutu: iki sinyal uzunluklarının toplamı
    n = sig.shape[0] + refsig.shape[0]

    # Spektral dönüşümler
    SIG    = fft(sig,    n=n)
    REFSIG = fft(refsig, n=n)

    # Cross-power spectrum ve PHAT ağırlığı
    R = SIG * np.conj(REFSIG)
    W = 1/np.abs(R)

    # Zamansal korelasyon
    cc = ifft(W)

    # “Zero‐lag” merkezi elde etmek için kaydır
    cc = np.concatenate([cc[-n//2:], cc[:n//2]])

    # En büyük tepe indeksi
    max_idx = np.argmax(np.abs(cc))

    # TDOA = (indeks kayması) / fs
    tau = (max_idx - n//2) / float(fs)

    return tau


# Her mikrofon çifti arasındaki zaman farkını hesapla
def gcc_phat_array(mic_array, ref_mic, fs=1, lowcut=None, highcut=None):
    time_delays=[]
    progress_bar1 = tqdm(total=len(mic_array)-1)
    for i in range(len(mic_array)):
        if np.array_equal(ref_mic,mic_array[i]):
            time_delays.append(0)
            continue
        tau = __gcc_phat(ref_mic, mic_array[i], fs=fs, lowcut=lowcut, highcut=highcut)
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
