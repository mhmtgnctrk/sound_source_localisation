import numpy as np
from numpy import linalg as LA
import sympy as sp
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from math import *
from tqdm import tqdm

#act_mic_data = np.array([mic_data[0], mic_data[3], mic_data[12], mic_data[6], mic_data[15]])

# Mikrofon arası en düşük mesafe (mm cinsinden)
# mic_span = 40.0

# Ses hızı (mm/s)
c = 343000

# Ref mikrofon
ref_index=0

# Mikrofon pozisyonları (mm)
d1 = sqrt(50**2-25**2)
mpos = np.array([[0, 0, 0],     [0, 0, 50],     [0, 0, 100],    [0, 0, 150], 
                [d1, 0, 25],    [d1, 0, 75],    [d1, 0, 125],   [d1, 0, 175], 
                [2*d1, 0, 0],   [2*d1, 0, 50],  [2*d1, 0, 100], [2*d1, 0, 150], 
                [3*d1, 0, 25],  [3*d1, 0, 75],  [3*d1, 0, 125], [3*d1, 0, 175]])

# Sanal ses kaynağının konumu (mm cinsinden)
rng = np.random.default_rng()
vir_source = np.array([rng.random()*400-200, rng.random()*200.0+300, rng.random()*30.0-15])

# Mikrofon konumları (mm cinsinden)
# m1 = np.array([mic_span / 2, 0.0, -mic_span * 3**0.5 / 2])
# m2 = np.array([-mic_span / 2, 0.0, -mic_span * 3**0.5 / 2])
# m3 = np.array([-mic_span, 0.0, 0.0])
# m4 = np.array([-mic_span / 2, 0.0, mic_span * 3**0.5 / 2])
# m5 = np.array([mic_span / 2, 0.0, mic_span * 3**0.5 / 2])
# m6 = np.array([mic_span, 0.0, 0.0])
# m7 = np.array([0.0, 0.0, 0.0])

act_mpos= np.array([mpos[0], mpos[3], mpos[12], mpos[6], mpos[15]])

# Mikrofonları bir diziye ekleyelim
# mic_array6 = np.array([m1, m2, m3, m4, m5, m6, m7])

# Mikrofonlar arası sesin kat ettiği mesafeleri hesaplayalım
# dist_travs = np.array([LA.norm(vir_source - m) for m in mic_array6])


# Mesafe farkları (Referans mikrofona göre)
# range_difs = dist_travs - dist_travs[ref_index]

# Zaman farkları (mesafe farklarını ses hızına böleriz)
# tdoas = range_difs / c


# # GHerçek ses konumunu yazdıralım
# print("\nGerçek ses konumu: ", vir_source)

# # Hesaplanan TDOA'ları yazdıralım
# print("\nHesaplanan TDOA'lar:", tdoas)

# Ağırlık matrisi (mikrofonlardan gelen veriye güvenimiz)
weights = np.array([1,      0.1,    0.1,    1, 
                    0.1,    1,      0.1,    0.1, 
                    0.5,    0.1,    0.1,    0.5,
                    1,      0.1,    0.1,    1])

### HESAPLANAN TDOA'YI INPUT OLARAK KULLANIP KONUM TAHMINİ

# A ve B matrislerini sembolik olarak oluşturma fonksiyonu
def compute_A_and_B_sym(guess, sym_mic_array, calc_ran_difs):
    A = []
    B = []

    # Mikrofon sayısı (n)
    # num_mics = len(sym_mic_array)

    for i, mic in enumerate(sym_mic_array):
        if i == ref_index:
            continue  # Referans mikrofonu atla, çünkü ona göre farklar hesaplanıyor

        # Referans mikrofon ile tahmin arasındaki mesafe
        d_ref = sp.sqrt((guess - sym_ref_mic).dot(guess - sym_ref_mic))

        # Diğer mikrofonlarla tahmin arasındaki mesafe
        d_i = sp.sqrt((guess - mic).dot(guess - mic))

        # A matrisi: her mikrofonun tahmine göre doğrusal oranları
        A_row = [(g - m) / d_i * weights[i] for g, m in zip(guess, mic)]
        A.append(A_row)

        # B vektörü: mesafe farklarının ifadesi
        B_val = (calc_ran_difs[i] - (d_i - d_ref)) * weights[i]
        B.append(B_val)

    # Liste olarak toplanan A ve B'yi matris formuna dönüştürelim
    A = sp.Matrix(A)  # (n-1) x 3 boyutunda olmalı
    B = sp.Matrix(B)  # (n-1) x 1 boyutunda olmalı
    return A, B

f = open("tdoa/tdoas.txt", "r")
tdoas=[]
for i in f.readlines():
    tdoas.append(float(i))
tdoas=np.array(tdoas)

# Başlangıç tahmini konum (ilk varsayım)
initial_guess = np.array([10,10,10])

# Sembolik değişkenleri tanımlayalım
x, y, z = sp.symbols('x y z')
guess = sp.Matrix([x, y, z])

# Mikrofon pozisyonlarının sembolik matrise dönüştürülmesi
sym_mic_array = [sp.Matrix(mic) for mic in mpos]

# Referans mikrofon sembolik
sym_ref_mic = sym_mic_array[ref_index] 

# Hesaplanan mesafe farkları (mm)
calc_ran_difs = tdoas * c

# İterasyon sayısı ve hata toleransı belirlenir
max_iterations = 100
tolerance = 10

# Başlangıç tahmini
current_guess = initial_guess
print(f"\nBaşlangıç Tahmini: {current_guess}")

#giris=time.time()

progress_bar2 = tqdm(total=max_iterations)

plt.ion()
# 3D Grafik Oluşturma
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Grafik detayları
ax.set_xlabel('X Ekseni (mm)')
ax.set_ylabel('Y Ekseni (mm)')
ax.set_zlabel('Z Ekseni (mm)')
ax.set_title('Mikrofonlar, Gerçek ve Tahmin Edilen Ses Kaynağı Konumları')
ax.legend()
plt.grid(True)
ax.set_xlim(-300,300)
ax.set_ylim(0,600)
ax.set_zlim(-300,300)

# Mikrofonların konumlarını çiz
mic_positions = np.array(mpos)
ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', label='Mikrofonlar', s=100)

# İterasyon döngüsü
for iteration in range(max_iterations):
    # Sembolik A ve B matrislerini hesapla
    A_sym, B_sym = compute_A_and_B_sym(guess, sym_mic_array, calc_ran_difs)

    # Sayısal A ve B matrislerini mevcut tahmine göre hesapla
    A_numeric = np.array(A_sym.subs({x: current_guess[0], y: current_guess[1], z: current_guess[2]})).astype(np.float64)
    B_numeric = np.array(B_sym.subs({x: current_guess[0], y: current_guess[1], z: current_guess[2]})).astype(np.float64)

    # En küçük kareler yöntemi ile doğrusal sistemi çöz
    delta_x = np.linalg.lstsq(A_numeric, B_numeric, rcond=None)[0].flatten()

    # Yeni tahmini konumu hesapla
    new_guess = current_guess + delta_x

    # Güncellemeleri yazdır
    print(f"\nİterasyon {iteration + 1}:")
    print(f"Delta X: {delta_x}")
    print(f"Yeni Tahmin: {new_guess}")

    # Hata kontrolü: Güncellemenin büyüklüğü küçükse dur
    if np.linalg.norm(delta_x) < tolerance:
        print("\nTolerans sınırına ulaşıldı, iterasyon sonlandırıldı.")
        break
    # Tahmin edilen ses kaynağını çiz
    ax.scatter(current_guess[0], current_guess[1], current_guess[2], c='r', marker='^', label=f'Tahmin Edilen Konum: {np.around(current_guess,2)}', s=150)

    fig.canvas.draw()
    
    fig.canvas.flush_events()
    #time.sleep(0.1)
    # Güncel tahmini konum, bir sonraki iterasyon için kullanılır
    current_guess = new_guess
    progress_bar2.update(1)
#cikis=time.time()

print(f"\nSon Tahmin: {current_guess}")
#print(f"\nIterasyon süresi (sn): {cikis-giris}")





plt.ioff()
# Gerçek ses kaynağını çiz
#ax.scatter(vir_source[0], vir_source[1], vir_source[2], c='g', marker='*', label=f'Gerçek Ses Kaynağı: {np.around(vir_source,2)}', s=200)
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Grafik detayları
ax2.set_xlabel('X Ekseni (mm)')
ax2.set_ylabel('Y Ekseni (mm)')
ax2.set_zlabel('Z Ekseni (mm)')
ax2.set_title('Mikrofonlar, Gerçek ve Tahmin Edilen Ses Kaynağı Konumları')
ax2.legend()
plt.grid(True)
ax2.set_xlim(-300,300)
ax2.set_ylim(0,600)
ax2.set_zlim(-300,300)
ax2.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', label='Mikrofonlar', s=100)
ax2.scatter(current_guess[0], current_guess[1], current_guess[2], c='r', marker='^', label=f'Tahmin Edilen Konum: {np.around(current_guess,2)}', s=150)
plt.show()

"""
# Sembolik matrisleri göstermek için (isteğe bağlı)
print("\nA Matrisi (Sembolik):")
sp.pprint(A_sym)

print("\nB Vektörü (Sembolik):")
sp.pprint(B_sym)

print("\nA Matrisi (Sayısal):")
print(A_numeric)

print("\nB Vektörü (Sayısal):")
print(B_numeric)
"""