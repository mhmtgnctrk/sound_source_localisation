import scipy.io.wavfile as wavfile
import numpy as np
import gcc_phat_interp as gpi
import gcc_phat as gp
from math import *

# ---- YENİ: Matplotlib ve sinyal işleme yardımcıları ----
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# (opsiyonel) band-pass için scipy.signal kullanımı
try:
    from scipy.signal import butter, lfilter
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _bandpass(x, lowcut, highcut, fs, order=5):
        if lowcut is None or highcut is None:
            return x
        b, a = _butter_bandpass(lowcut, highcut, fs, order)
        return lfilter(b, a, x)
except Exception:
    def _bandpass(x, lowcut, highcut, fs, order=5):
        return x  # scipy yoksa filtresiz

# ---- YENİ: Her adımı çizen yardımcı fonksiyon ----
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, rfft, irfft, rfftfreq, fftfreq

def plot_full_gcc_phat_pipeline_TR(sig, refsig, fs, lowcut=None, highcut=None, interp=16, baslik=""):
    """
    İstenen sırayla tüm adımları çizer ve tau'yu işaretler.
    Band geçişi 'freq-mode' ile frekans domeninde maskeyle yapılır (faz bozulmaz).
    """
    eps = 1e-12
    sig    = np.asarray(sig, dtype=np.float32)
    refsig = np.asarray(refsig, dtype=np.float32)

    # ---------------- (A) TEK MİKROFON PIPELINE (sadece 'sig' için) ----------------
    N   = len(sig)
    t   = np.arange(N) / fs
    SPEC_sig  = rfft(sig)
    freqs_r   = rfftfreq(N, d=1.0/fs)
    mag_sig   = np.abs(SPEC_sig)

    band_applied = (lowcut is not None) and (highcut is not None)
    if band_applied:
        mask_r = (freqs_r >= lowcut) & (freqs_r <= highcut)
        SPEC_sig_band = SPEC_sig * mask_r
        sig_band = irfft(SPEC_sig_band, n=N)      # sıfır-faz bant sınırlı sinyal
        mag_sig_band = np.abs(SPEC_sig_band)
    else:
        sig_band = None
        mag_sig_band = None

    # ---------------- (B) ÇAPRAZ GÜÇ ve PHAT ----------------
    # GCC-PHAT için 'doğrusal' korelasyon elde etmek adına genellikle n = len(x)+len(y)
    n = len(sig) + len(refsig)
    freqs = fftfreq(n, d=1.0/fs)

    SIG    = fft(sig,    n=n)
    REFSIG = fft(refsig, n=n)

    # R_raw = S(f) * R*(f)  (çarpma -> korelasyon uzayına gidecek bilgi)
    R_raw_full = SIG * np.conj(REFSIG)

    # Sadece istenen frekans bandı katkı yapsın (freq-mode)
    if band_applied:
        band_full = ((np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)).astype(R_raw_full.real.dtype)
        R_raw = R_raw_full * band_full
    else:
        R_raw = R_raw_full

    # PHAT ağırlığı: genliği 1'e normalize eder, fazı korur.
    mag = np.abs(R_raw)
    R_phat = np.zeros_like(R_raw, dtype=R_raw.dtype)
    nz = mag > eps
    R_phat[nz] = R_raw[nz] / mag[nz]      # |R_phat| ≈ 1

    # ---------------- (C) IFFT -> KORELASYON (cc) ----------------
    # PHAT öncesi korelasyon (karşılaştırma için)
    cc_raw = ifft(R_raw)
    cc     = ifft(R_phat)                  # GCC-PHAT korelasyonu

    # Zero-lag merkezleme (merkez örnek = 0 gecikme)
    def zero_center(x):
        return np.concatenate([x[-n//2:], x[:n//2]])

    cc_raw_c = zero_center(cc_raw)
    cc_c     = zero_center(cc)

    # Büyüklükler
    cc_raw_mag = np.abs(cc_raw)
    cc_mag     = np.abs(cc)
    cc_raw_mag_c = np.abs(cc_raw_c)
    cc_mag_c     = np.abs(cc_c)

    # ---------------- (D) INTERPOLASYON ve τ ----------------
    # Interpolasyon merkezlenmiş |cc| üzerinde yapılır
    L0 = len(cc_mag_c)
    center0 = L0 // 2
    lags0_s = (np.arange(L0) - center0) / fs           # saniye cinsinden gecikme ekseni
    lags0_ms = lags0_s * 1e3

    if interp and interp > 1:
        x_old = np.arange(L0)
        x_new = np.linspace(0, L0-1, L0*int(interp))
        cc_interp = np.interp(x_new, x_old, cc_mag_c)
        L1 = len(cc_interp)
        center1 = L1 // 2
        lags1_s  = (np.arange(L1) - center1) / (fs*int(interp))
        lags1_ms = lags1_s * 1e3
        peak_idx = int(np.argmax(cc_interp))
        tau = (peak_idx - center1) / (fs*int(interp))
    else:
        cc_interp = None
        lags1_ms = None
        center1 = None
        peak_idx = int(np.argmax(cc_mag_c))
        tau = (peak_idx - center0) / fs

    # ---------------- (E) ÇİZİMLER ----------------
    # FIG-1: İstenen sıralama: Spektrum -> Zaman (ham) -> Zaman (band) -> FFT (band/ham)
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 8))
    a1, a2, a3, a4 = axs1.ravel()

    # (1) Spektrum (ham)
    a1.plot(freqs_r, mag_sig)
    if band_applied:
        a1.axvspan(lowcut, highcut, color="tab:orange", alpha=0.15, label=f"Bant: {lowcut}-{highcut} Hz")
        a1.legend(loc="upper right")
    a1.set_title(f"Mikrofon Spektrumu {baslik}")
    a1.set_xlabel("Frekans [Hz]"); a1.set_ylabel("Büyüklük"); a1.set_xlim(0, fs/2)

    # (2) Zaman alanı (ham)
    a2.plot(t, sig)
    a2.set_title("Zaman Alanı (Ham)")
    a2.set_xlabel("Zaman [s]"); a2.set_ylabel("Genlik")

    # (3) Zaman alanı (band) – yoksa gizle
    if sig_band is not None:
        a3.plot(t, sig_band, color="tab:orange")
        a3.set_title(f"Zaman Alanı (Bant: {lowcut}-{highcut} Hz)")
        a3.set_xlabel("Zaman [s]"); a3.set_ylabel("Genlik")
    else:
        a3.axis("off")

    # (4) FFT büyüklüğü (band uygulanmışsa bantlı)
    if mag_sig_band is not None:
        a4.plot(freqs_r, mag_sig_band, color="tab:orange")
        a4.set_title("FFT (Bant Uygulanmış Büyüklük)")
    else:
        a4.plot(freqs_r, mag_sig)
        a4.set_title("FFT (Ham Büyüklük)")
    a4.set_xlabel("Frekans [Hz]"); a4.set_ylabel("Büyüklük"); a4.set_xlim(0, fs/2)

    fig1.tight_layout()

    # FIG-2: S·R*, PHAT, korelasyon
    fig2, axs2 = plt.subplots(2, 3, figsize=(16, 9))
    b1, b2, b3, b4, b5, b6 = axs2.ravel()

    # (5) |R_raw| (full vs band)
    pos = freqs >= 0
    Rfull_db = 20*np.log10(np.maximum(np.abs(R_raw_full[pos]), eps))
    Rfull_db -= Rfull_db.max()   # normalize: en büyük 0 dB
    Rband_db = 20*np.log10(np.maximum(np.abs(R_raw[pos]), eps))
    Rband_db -= Rband_db.max() if np.isfinite(Rband_db).any() else 0.0

    b1.plot(freqs[pos], Rfull_db, label="|R_raw| (ham, norm dB)")
    b1.plot(freqs[pos], Rband_db, label="|R_raw| (bant, norm dB)", linestyle="--")
    if band_applied:
        b1.axvspan(lowcut, highcut, color="tab:orange", alpha=0.12)
    b1.set_title("Çapraz Güç: |R_raw| (ham vs bant)")
    b1.set_xlabel("Frekans [Hz]"); b1.set_ylabel("dB"); b1.legend()

    # (6) |R_phat| (ayrı y-ekseni ile, ~1)
    b2a = b2
    ln1 = b2a.plot(freqs[pos], np.abs(R_raw[pos]), label="|R_raw| (bant)")
    b2a.set_xlabel("Frekans [Hz]"); b2a.set_ylabel("|R_raw|")
    b2a.set_title("PHAT Ağırlığı: |R_raw| & |R_phat|")

    b2b = b2a.twinx()
    ln2 = b2b.plot(freqs[pos], np.abs(R_phat[pos]), label="|R_phat| ≈ 1", linestyle="--", color="tab:orange")
    b2b.set_ylabel("|R_phat|"); b2b.set_ylim(0, 1.05); b2b.grid(False)
    lns  = ln1 + ln2
    labs = [l.get_label() for l in lns]
    b2a.legend(lns, labs, loc="upper right")

    # (7) Korelasyon: PHAT öncesi (merkezlenmemiş ve merkezli)
    b3.plot(np.abs(cc_raw), label="|cc_raw| (PHAT öncesi)")
    b3.set_title("Korelasyon (PHAT öncesi) – merkezlenmemiş")
    b3.set_xlabel("Örnek"); b3.set_ylabel("Büyüklük"); b3.legend()

    b4.plot(cc_raw_mag_c, label="|cc_raw| (zero-lag merkezli)")
    b4.axvline(len(cc_raw_mag_c)//2, linestyle=":", color="k", label="sıfır-gecikme")
    b4.set_title("Korelasyon (PHAT öncesi) – zero-lag merkezli")
    b4.set_xlabel("Örnek"); b4.set_ylabel("Büyüklük"); b4.legend()

    # (8) Korelasyon: PHAT sonrası (merkezlenmemiş ve merkezli)
    b5.plot(np.abs(cc), label="|cc| (PHAT)")
    b5.set_title("Korelasyon (PHAT) – merkezlenmemiş")
    b5.set_xlabel("Örnek"); b5.set_ylabel("Büyüklük"); b5.legend()

    b6.plot(lags0_ms, cc_mag_c/np.max(cc_mag_c+eps), label="|cc| (PHAT, norm)")
    b6.axvline(0.0, linestyle=":", color="k", label="sıfır-gecikme")
    if interp and interp > 1:
        # yakınlaştırılmış eğriyi ek çizgi olarak göster
        b6.plot(lags1_ms, cc_interp/np.max(cc_interp+eps), linestyle="--", label=f"interp x{interp}")
        tau_ms = tau*1e3
        b6.axvline(tau_ms, linestyle="--", color="tab:red", label=f"tepe @ {tau_ms:.3f} ms")
        # mantıklı bir pencere (±1 ms) – gerekirse ayarla
        b6.set_xlim(-1.0, 1.0)
    else:
        b6.axvline(tau*1e3, linestyle="--", color="tab:red", label=f"tepe @ {tau*1e3:.3f} ms")
        b6.set_xlim(-1.0, 1.0)

    b6.set_title(f"Korelasyon (PHAT, zero-lag merkezli) – τ = {tau:.6e} s")
    b6.set_xlabel("Gecikme [ms]"); b6.set_ylabel("Büyüklük (normalize)"); b6.legend()

    fig2.tight_layout()
    
    plt.show()

    print(f"Tau (TDOA) = {tau:.9f} s  |  {tau*fs:.3f} örnek")

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
    pre_file = r"D:\github\ssl_new\sound_source_localisation"
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
INTERP  = 16      # interpolasyon faktörü (<=1 kapalı)
LOWCUT  = None    # örn: 300
HIGHCUT = None    # örn: 3400

# TDOA hesaplama (mevcut yapıyı BOZMADAN)
tdoas = gpi.gcc_phat_array(mic_data, ref_mic, fs=fs, interp=INTERP,
                           lowcut=LOWCUT, highcut=HIGHCUT)
f = open("tdoa/tdoas.txt", "w")
for tdoa in tdoas:
    f.write((str(tdoa)+"\n"))
f.close()
def print_tdoa_simple(tdoas, ref_index=0, fs=1):
    for i, tau in enumerate(tdoas):
        print(f"Mik {i+1} – Ref {ref_index+1}:  tau = {tau:+.6f} s  ({tau*fs:+.2f} örnek)")




# (1) TDOA’lar (senin mevcut fonksiyonunla)
tdoas = gpi.gcc_phat_array(mic_data, ref_mic, fs=fs, interp=16)
print_tdoa_simple(tdoas, ref_index=0, fs=fs)

# (2) Spektrogramlar
gpi.visualize_gcc_phat(mic_data, fs)

# (3) Her mikrofon için GCC-PHAT adımlarını çiz
for idx, m in enumerate(mic_data):
    if idx == 0:
        continue
    plot_full_gcc_phat_pipeline_TR(m, ref_mic, fs, interp=INTERP, lowcut=LOWCUT, highcut=HIGHCUT,
                           baslik=f"(Mik {idx+1} vs Ref 1)")

