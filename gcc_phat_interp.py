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


def __gcc_phat(sig, refsig, fs=1, interp=16, lowcut=None, highcut=None, band_mode="freq"):
    # bellek ve dtype
    sig = sig.astype(np.float32, copy=False)
    refsig = refsig.astype(np.float32, copy=False)
    eps = 1e-15

    # FFT
    n = sig.shape[0] + refsig.shape[0]
    SIG    = fft(sig, n=n)
    REFSIG = fft(refsig, n=n)
    R_raw  = SIG * np.conj(REFSIG)

    # ---- YENİ: Frekans domeninde bant maskeleme ----
    if (lowcut is not None) and (highcut is not None) and (band_mode == "freq"):
        freqs = np.fft.fftfreq(n, d=1.0/fs)            # [-fs/2..fs/2) düzeninde değil; ama maske için yeterli
        band_mask = ((np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)).astype(R_raw.real.dtype)
        R_raw = R_raw * band_mask                      # sadece  [lowcut, highcut]  bandı kalsın

    # if (lowcut is not None) and (highcut is not None) and (band_mode == "time"):
    #     sig    = __butter_bandpass_filter(sig,    lowcut, highcut, fs)
    #     refsig = __butter_bandpass_filter(refsig, lowcut, highcut, fs)
    #     SIG    = fft(sig, n=n);  REFSIG = fft(refsig, n=n);  R_raw = SIG * np.conj(REFSIG)

    # PHAT ağırlığı (güvenli)
    R = R_raw / (np.abs(R_raw) + eps)

    # IFFT -> korelasyon
    cc = ifft(R)
    cc = np.concatenate([cc[-n//2:], cc[:n//2]])      # zero-lag merkeze

    # Interpolasyon
    if interp and interp > 1:
        x_old = np.arange(len(cc))
        x_new = np.linspace(0, len(cc)-1, len(cc)*int(interp))
        cc = np.interp(x_new, x_old, np.abs(cc))
        scale = float(int(interp) * fs)
    else:
        cc = np.abs(cc)
        scale = float(fs)

    # Tepe
    max_idx   = int(np.argmax(cc))
    max_shift = len(cc) // 2
    tau = 2*(max_idx - max_shift) / scale
    return tau

def gcc_phat_array(mic_array, ref_mic, fs=1, interp=16, lowcut=None, highcut=None, band_mode="freq"):
    time_delays = []
    progress_bar1 = tqdm(total=len(mic_array)-1)
    for i in range(len(mic_array)):
        if np.array_equal(ref_mic, mic_array[i]):
            time_delays.append(0.0)
            continue
        tau = __gcc_phat(ref_mic, mic_array[i], fs=fs, interp=interp,
                         lowcut=lowcut, highcut=highcut, band_mode=band_mode)
        time_delays.append(-tau)
        progress_bar1.update(1)
    return time_delays

# # Her mikrofon çifti arasındaki zaman farkını hesapla
# def gcc_phat_array(mic_array, ref_mic, fs=1, interp=16, lowcut=None, highcut=None):
#     time_delays=[]
#     progress_bar1 = tqdm(total=len(mic_array)-1)
#     for i in range(len(mic_array)):
#         if np.array_equal(ref_mic,mic_array[i]):
#             time_delays.append(0)
#             continue
#         tau = __gcc_phat(ref_mic, mic_array[i], fs=fs, interp=interp, lowcut=lowcut, highcut=highcut)
#         time_delays.append(-1*tau)
#         progress_bar1.update(1)
#     return time_delays

# Zaman farklarını yazdır
def print_tdoa(time_delays):
    
    for mic1, mic2, tau in time_delays:
       print(f"\nTime delay between microphone {mic1} and microphone {mic2}: {tau:.6f} seconds")


def visualize_gcc_phat(act_mic_data, fs, nfft=512, noverlap=384, cmap="viridis",
                       vmin=-90, vmax=None):
    """
    Her mikrofon için spektrogram çizer; sağda tek bir colorbar kullanır.
    Colorbar artık grafiğin üstüne binmez.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    M = len(act_mic_data)

    # Sağda colorbar için ayrı sütun; layout otomatik
    fig = plt.figure(figsize=(12, 2.6*M), constrained_layout=True)
    gs = fig.add_gridspec(nrows=M, ncols=2, width_ratios=[20, 1])

    axs = []
    for i in range(M):
        ax = fig.add_subplot(gs[i, 0])
        axs.append(ax)

    # Colorbar ekseni (tek)
    cax = fig.add_subplot(gs[:, 1])

    # Her mikrofon için spektrogram
    im = None
    for idx, m in enumerate(act_mic_data):
        # scale='dB' ile colorbar dB olur; aspect='auto' taşmayı azaltır
        Pxx, freqs, bins, im = axs[idx].specgram(
            m, NFFT=nfft, Fs=fs, noverlap=noverlap,
            cmap=cmap, vmin=vmin, vmax=vmax, scale='dB'
        )
        axs[idx].set_title(f"Mikrofon {idx+1} – Spektrogram")
        axs[idx].set_ylabel("Frekans [Hz]")
        axs[idx].set_ylim(0, fs/2)
        axs[idx].set_aspect('auto')

    axs[-1].set_xlabel("Zaman [s]")

    # Tek colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Güç (dB)")

    plt.show()
