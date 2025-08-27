import scipy.io.wavfile as wavfile
import numpy as np
import gcc_phat_interp as gpi
import gcc_phat as gp
from math import *
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# Her adımı çizen yardımcı fonksiyon
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, rfft, irfft, rfftfreq, fftfreq

def plot_full_gcc_phat_pipeline_TR(sig, refsig, fs, lowcut=None, highcut=None, interp=None, baslik=""):
    """
    İstenen sırayla tüm adımları çizer ve tau'yu işaretler.
    Band geçişi 'freq-mode' ile frekans domeninde maskeyle yapılır (faz bozulmaz).
    """
    eps = 1e-12
    sig    = np.asarray(sig, dtype=np.float32)
    refsig = np.asarray(refsig, dtype=np.float32)

    # ---------------- (A) TEK MİKROFON PIPELINE (sadece 'sig' için) ----------------
    N   = len(sig)
    N_ref = len(refsig)
    t   = np.arange(N) / fs
    t_ref   = np.arange(N_ref) / fs
    SPEC_sig  = rfft(sig)
    SPEC_refsig = rfft(refsig)
    freqs_r   = rfftfreq(N, d=1.0/fs)
    freqs_r_ref = rfftfreq(N_ref, d=1.0/fs)
    mag_sig   = np.abs(SPEC_sig)
    mag_refsig = np.abs(SPEC_refsig)
    

    # ---------------- (B) ÇAPRAZ GÜÇ ve PHAT ----------------
    # GCC-PHAT için 'doğrusal' korelasyon elde etmek adına genellikle n = len(x)+len(y)
    n = len(sig) + len(refsig)
    freqs = fftfreq(n, d=1.0/fs)

    SIG    = fft(sig,    n=n)
    REFSIG = fft(refsig, n=n)

    # R_raw = S(f) * R*(f)  (çarpma -> korelasyon uzayına gidecek bilgi)
    R_raw = SIG * np.conj(REFSIG)

    # PHAT ağırlığı: genliği 1'e normalize eder, fazı korur.
    mag = np.abs(R_raw)
    # R_phat = np.zeros_like(R_raw, dtype=R_raw.dtype)
    # nz = mag > eps
    # R_phat[nz] = R_raw[nz] / mag[nz]      # |R_phat| ≈ 1
    R_phat = R_raw / mag

    #Fazlar
    phi_raw = np.angle(R_raw)
    phi = np.angle(R_phat)


    # ---------------- (C) IFFT -> KORELASYON (cc) ----------------
    # PHAT öncesi korelasyon (karşılaştırma için)
    cc_raw = ifft(R_raw)
    cc     = ifft(R_phat)                  # GCC-PHAT korelasyonu

    # Zero-lag merkezleme (merkez örnek = 0 gecikme)
    def zero_center(x):
        return np.concatenate([x[-n//2:], x[:n//2]])

    cc_raw_c = zero_center(cc_raw)
    cc_c     = zero_center(cc)

    cc_raw_centerd = np.fft.fftshift(cc_raw)
    cc_centerd = np.fft.fftshift(cc)

    # Büyüklükler
    cc_raw_mag = np.abs(cc_raw)
    cc_mag     = np.abs(cc)
    cc_raw_mag_c = np.abs(cc_raw_c)
    cc_mag_c     = np.abs(cc_c)

    # ---------------- (D) τ ----------------
    # PHAT
    L0 = len(cc_mag_c)
    center0 = L0 // 2
    lags0_s = (np.arange(L0) - center0) / fs           # saniye cinsinden gecikme ekseni
    lags0_ms = lags0_s * 1e3

    peak_idx = int(np.argmax(cc_mag_c))
    tau = (peak_idx - center0) / fs

    # Ham
    L0_raw = len(cc_raw_mag_c)
    center0_raw = L0_raw // 2
    lags0_s_raw = (np.arange(L0) - center0_raw) / fs           # saniye cinsinden gecikme ekseni
    lags0_ms_raw = lags0_s_raw * 1e3

    peak_idx_Raw = int(np.argmax(cc_raw_mag_c))
    tau_raw = (peak_idx_Raw - center0_raw) / fs

    # ---------------- (E) ÇİZİMLER ----------------
    # FIG-1: İstenen sıralama: Spektrum -> Zaman (ham) -> Zaman (band) -> FFT (band/ham)
    fig1, axs1 = plt.subplots(2,2, figsize=(14, 8))
    a1, a2, a3, a4 = axs1.ravel()

    # (1) Spektrum (ham)
    a1.plot(t, sig)
    a1.set_title(f"Mikrofon Spektrumu {baslik}")
    a1.set_xlabel("Frekans [Hz]"); a1.set_ylabel("Büyüklük")

    a2.plot(t_ref, refsig)
    a2.set_title("Mikrofon Spektrumu m_1")
    a2.set_xlabel("Frekans [Hz]"); a2.set_ylabel("Büyüklük")


    # (2) FFT büyüklüğü (band uygulanmışsa bantlı)

    a3.plot(freqs_r, mag_sig)
    a3.set_xlabel("Frekans [Hz]"); a3.set_ylabel("Büyüklük"); a3.set_xlim(0, fs/2)
    a3.set_title(f"FFT (Ham Büyüklük) {baslik}")

    a4.plot(freqs_r_ref, mag_refsig)
    a4.set_xlabel("Frekans [Hz]"); a4.set_ylabel("Büyüklük"); a4.set_xlim(0, fs/2)
    a4.set_title("FFT (Ham Büyüklük) m_1")
    

    fig1.tight_layout()

    # FIG-2: S·R*, PHAT, korelasyon
    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 9))
    b1, b2, b3, b4 = axs2.ravel()

    # (5) |R_raw|
    pos = freqs >= 0

    # (6) Korelasyon (IFFT): PHAT öncesi (merkezlenmemiş)
    b1.plot(lags0_ms_raw/1000,np.abs(cc_raw_centerd), label="Çapraz Korelasyon (PHAT öncesi)")
    b1.set_title("Korelasyon (PHAT öncesi) – merkezlenmiş")
    b1.set_xlabel("Gecikme (s)"); b1.set_ylabel("Büyüklük"); b1.legend()

    # (7) PHAT öncesi gecikme merkezli
    b2.plot(lags0_ms_raw, cc_raw_mag_c/np.max(cc_raw_mag_c+eps), label="|ÇK| (PHAT öncesi - sıfır-gecikme merkezli)")
    b2.axvline(0.0, linestyle=":", color="k", label="sıfır-gecikme")
    b2.axvline(tau_raw*1e3, linestyle="--", color="tab:red", label=f"tepe @ {tau_raw*1e3:.3f} ms")
    b2.set_xlim(-1.0, 1.0)
    b2.set_title(f"Korelasyon (PHAT öncesi, sıfır gecikme merkezli) – τ = {tau_raw:.6e} s")
    b2.set_xlabel("Gecikme [ms]"); b2.set_ylabel("Büyüklük (normalize)"); b2.legend()

    # (8) Korelasyon: PHAT sonrası (merkezlenmemiş ve merkezli)
    b3.plot(lags0_ms, np.abs(cc_centerd), label="Çapraz Korelasyon (PHAT sonrası)")
    b3.set_title("Korelasyon (PHAT) – merkezlenmiş")
    b3.set_xlabel("Gecikme (s)"); b3.set_ylabel("Büyüklük"); b3.legend()

    # (9) PHAT sornası gecikme
    b4.plot(lags0_ms, cc_mag_c/np.max(cc_mag_c+eps), label="|cc| (PHAT, norm)")
    b4.axvline(0.0, linestyle=":", color="k", label="sıfır-gecikme")
    b4.axvline(tau*1e3, linestyle="--", color="tab:red", label=f"tepe @ {tau*1e3:.3f} ms")
    b4.set_xlim(-1.0, 1.0)

    b4.set_title(f"Korelasyon (PHAT, zero-lag merkezli) – τ = {tau:.6e} s")
    b4.set_xlabel("Gecikme [ms]"); b4.set_ylabel("Büyüklük (normalize)"); b4.legend()

    fig2.tight_layout()
    
    plt.show()

    print(f"Tau (TDOA) = {tau:.9f} s  |  {tau*fs:.3f} örnek")
    return tau

# -------------------- KULLANIM --------------------
# plot_full_gcc_phat_pipeline_TR(sig=mic_data[1], refsig=mic_data[0], fs=fs,
#                                lowcut=4000, highcut=6000, interp=16,
#                                baslik="(Mik 2 vs Ref 1)")

# ===================== MEVCUT AKIŞ =====================

# Mikrofon kayıtlarını saklayacak liste
mic_data = []
fs = None

# 4 mikrofonun wav dosyalarını okuma
for i in range(1, 5):
    pre_file = r"E:\git_projects\sound_source_localisation\sound_source_localisation"
    if i <= 9:
        filename = pre_file + f'\mic{i}.wav'
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
    
# ---- YENİ: Her mic için adım adım görselleştirme ----
INTERP  = None      # interpolasyon faktörü (<=1 kapalı)
LOWCUT  = None    # örn: 300
HIGHCUT = None    # örn: 3400
tdoas=[0]
# (3) Her mikrofon için GCC-PHAT adımlarını çiz
for idx, m in enumerate(mic_data):
    if idx == 0:
        continue
    tdoas.append(plot_full_gcc_phat_pipeline_TR(m, ref_mic, fs, interp=INTERP, lowcut=LOWCUT, highcut=HIGHCUT,
                           baslik=f"(m_{idx+1})"))

f = open(r"E:\git_projects\sound_source_localisation\sound_source_localisation\tdoa\tdoas.txt", "w")
for tdoa in tdoas:
    f.write((str(tdoa)+"\n"))
f.close()
def print_tdoa_simple(tdoas, ref_index=0, fs=1):
    for i, tau in enumerate(tdoas):
        print(f"Mik {i+1} – Ref {ref_index+1}:  tau = {tau:+.6f} s  ({tau*fs:+.2f} örnek)")