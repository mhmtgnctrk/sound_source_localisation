import numpy as np
from numpy import linalg as LA
import sympy as sp
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math import *
from tqdm import tqdm

# Ses hızı (mm/s)
c = 343200

# Ref mikrofon
ref_index=0
a=225.0 # mm cinsinden kare kenar uzunluğu

mpos = np.array([[-a/2, a/2, 9],     [a/2, a/2, 9],     [a/2, -a/2, 0],    [-a/2, -a/2, 0]])
#mpos = np.array([[113, 0, 0],     [-36, 0, 0],     [-72, 0, 0],    [-113, 0, 0]])
# Sanal ses kaynağının konumu (mm cinsinden)
rng = np.random.default_rng()

### HESAPLANAN TDOA'YI INPUT OLARAK KULLANIP KONUM TAHMINİ

# A ve B matrislerini sembolik olarak oluşturma fonksiyonu
def compute_A_and_B_sym(guess, sym_mic_array, calc_ran_difs):
    A = []
    B = []

    for i, mic in enumerate(sym_mic_array):
        if i == ref_index:
            continue  # Referans mikrofonu atla, çünkü ona göre farklar hesaplanıyor

        # Referans mikrofon ile tahmin arasındaki mesafe
        d_ref = sp.sqrt((guess - sym_mic_array[ref_index]).dot(guess - sym_mic_array[ref_index]))

        # Diğer mikrofonlarla tahmin arasındaki mesafe
        d_i = sp.sqrt((guess - mic).dot(guess - mic))

        # A matrisi: her mikrofonun tahmine göre doğrusal oranları
        A_row = [(g - m) / d_i  for g, m in zip(guess, mic)]
        A.append(A_row)

        # B vektörü: mesafe farklarının ifadesi
        B_val = (calc_ran_difs[i] - (d_i - d_ref))
        B.append(B_val)

    # Liste olarak toplanan A ve B'yi matris formuna dönüştürelim
    A = sp.Matrix(A)  # (n-1) x 3 boyutunda olmalı
    B = sp.Matrix(B)  # (n-1) x 1 boyutunda olmalı
    return A, B

f = open(r"E:\git_projects\sound_source_localisation\sound_source_localisation\tdoa\tdoas.txt", "r")
tdoas=[]
for i in f.readlines():
    tdoas.append(float(i))  # ms to seconds
tdoas=np.array(tdoas) 

# Başlangıç tahmini konum (ilk varsayım)
#initial_guess = np.array([0,0,50])
init_array=[]
for i in range(10,1000,1):
    init_array.append(np.array([0,0,i]))

#print(init_array)
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
max_iterations = 20
tolerance = 21.45

def angle_between_vectors_np(u, v):
    u = np.array(u)
    v = np.array(v)
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg
out_array=[]
mesafe_h=[]
aci_hatalari=[]
for i in init_array:
    initial_guess = i
    current_guess=initial_guess
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

        # Hata kontrolü: Güncellemenin büyüklüğü küçükse dur
        if np.linalg.norm(delta_x) < tolerance:
            #print("\nTolerans sınırına ulaşıldı, iterasyon sonlandırıldı.")
            break

        # Güncel tahmini konum, bir sonraki iterasyon için kullanılır
        current_guess = new_guess
        #progress_bar2.update(1)
    out_array.append(current_guess)
    print(f"\nİlk Tahmin: {initial_guess}")
    print(f"\nSon Tahmin: {current_guess}")
    gercek_konum=[-222,292,530]
    vector_u = [1, 0, 0]
    vector_guess = current_guess 
    vector_real = gercek_konum
    angle_rad_guess, angle_deg_guess = angle_between_vectors_np(vector_guess, vector_u)
    angle_rad_real, angle_deg_real = angle_between_vectors_np(vector_real, vector_u)
    tahmin_mesafe=np.linalg.norm(current_guess)
    gercek_mesafe=np.linalg.norm(gercek_konum)

    print(f"Açı hatası (in degrees): {angle_deg_real - angle_deg_guess}")

    mesafe_hatasi = gercek_mesafe-tahmin_mesafe
    aci_hatasi=angle_deg_real - angle_deg_guess
    aci_hatalari.append(aci_hatasi)
    print(f"Mesafe Hatası: {mesafe_hatasi}")
    mesafe_h.append(mesafe_hatasi)
    if len(mesafe_h) >=2 and np.abs(mesafe_hatasi)<= 10.0: break
    
out_array = np.array(out_array)
init_array = np.array(init_array)
mesafe_h=np.array(mesafe_h)

# x, y, z bileşenlerini ayır
x = out_array[:, 0]
y = out_array[:, 1]
z = out_array[:, 2]
n = len(out_array)
t = init_array[:n, 2]          # sadece üretilen nokta kadarını al

#local_min=min(z)
#idx = np.where(z == local_min)[0]
sonuc=[x[-1],y[-1],z[-1]]
print("Sonuç:",sonuc)

# Grafik çiz
fig1, axs1 = plt.subplots(3,1, figsize=(14, 8))
a1, a2, a3= axs1.ravel()

a1.plot(t, x, label='X', color='r')
a1.plot(t, y, label='Y', color='g')
a1.plot(t, z, label='Z', color='b')
a1.set_xlabel('İterasyon')
a1.set_ylabel('Değer (mm)')
a1.set_title('X, Y, Z Ekseni Koordinatları')
a1.legend()
a1.grid(True)

a2.plot(t, mesafe_h, label='Mesafe Hatası', color='r')
a2.set_xlabel('İterasyon')
a2.set_ylabel('Mesafe (mm)')
a2.set_title('Mesafe Hatası (mm)')
a2.legend()
a2.grid(True)

a3.plot(t, aci_hatalari, label='Açı Hatası', color='r')
a3.set_xlabel('İterasyon')
a3.set_ylabel('Açı (°)')
a3.set_title('Açısal Hata (° derece)')
a3.legend()
a3.grid(True)

plt.show()

