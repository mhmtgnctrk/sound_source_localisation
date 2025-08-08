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

mpos = np.array([[-225/2, 225/2, 9],     [225/2, 225/2, 9],     [225/2, -225/2, 0],    [-225/2, -225/2, 0]])
#mpos = np.array([[113, 0, 0],     [-36, 0, 0],     [-72, 0, 0],    [-113, 0, 0]])
# Sanal ses kaynağının konumu (mm cinsinden)
rng = np.random.default_rng()


#act_mpos= np.array([mpos[0], mpos[3], mpos[12], mpos[6], mpos[15]])
act_mpos= np.array([mpos[0], mpos[1], mpos[2], mpos[3]])

# Ağırlık matrisi (mikrofonlardan gelen veriye güvenimiz)
weights = np.array([1, 1, 1, 1])

### HESAPLANAN TDOA'YI INPUT OLARAK KULLANIP KONUM TAHMINİ

# A ve B matrislerini sembolik olarak oluşturma fonksiyonu
def compute_A_and_B_sym(guess, sym_mic_array, calc_ran_difs):
    A = []
    B = []

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
initial_guess = np.array([0,0,50])
init_array=[]
for i in range(10,500,10):
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

# Başlangıç tahmini
current_guess = initial_guess
#print(f"\nBaşlangıç Tahmini: {current_guess}")

# progress_bar2 = tqdm(total=max_iterations)

# plt.ion()
# # 3D Grafik Oluşturma
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Grafik detayları
# ax.set_xlabel('X Ekseni (mm)')
# ax.set_ylabel('Y Ekseni (mm)')
# ax.set_zlabel('Z Ekseni (mm)')
# ax.set_title('Mikrofonlar, Gerçek ve Tahmin Edilen Ses Kaynağı Konumları')

# plt.grid(True)
# ax.set_xlim(-500,500)
# ax.set_ylim(0,600)
# ax.set_zlim(-300,300)
# ax.legend()
# # Mikrofonların konumlarını çiz
# mic_positions = np.array(mpos)
# ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', label='Mikrofonlar', s=100)
def angle_between_vectors_np(u, v):
    u = np.array(u)
    v = np.array(v)
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg
out_array=[]
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

        # Güncellemeleri yazdır
        #print(f"\nİterasyon {iteration + 1}:")
        
        #print(f"Delta X: {delta_x}")
        #print(f"Yeni Tahmin: {new_guess}")

        # Hata kontrolü: Güncellemenin büyüklüğü küçükse dur
        if np.linalg.norm(delta_x) < tolerance:
            #print("\nTolerans sınırına ulaşıldı, iterasyon sonlandırıldı.")
            break
        # # Tahmin edilen ses kaynağını çiz
        # ax.scatter(current_guess[0], current_guess[1], current_guess[2], c='r', marker='^', label=f'Tahmin Edilen Konum: {np.around(current_guess,2)}', s=150)

        # fig.canvas.draw()
        
        # fig.canvas.flush_events()

        # Güncel tahmini konum, bir sonraki iterasyon için kullanılır
        current_guess = new_guess
        #progress_bar2.update(1)
    out_array.append(current_guess)
    print(f"\nİlk Tahmin: {initial_guess}")
    print(f"\nSon Tahmin: {current_guess}")
    gercek_konum=[225/2,225/2,250]
    vector_u = [1, 0, 0]
    vector_guess = current_guess 
    vector_real = gercek_konum
    angle_rad_guess, angle_deg_guess = angle_between_vectors_np(vector_guess, vector_u)
    angle_rad_real, angle_deg_real = angle_between_vectors_np(vector_real, vector_u)
    tahmin_mesafe=np.linalg.norm(current_guess)
    gercek_mesafe=np.linalg.norm(gercek_konum)
    #print(f"Tahmin açısı (in radians): {angle_rad_guess}")
    #print(f"Tahmin açısı (in degrees): {angle_deg_guess}")
    #print(f"Angle between vectors of real (in radians): {angle_rad_real}")
    #print(f"Gerçek açı (in degrees): {angle_deg_real}")
    #print(f"Angle error (in radians): {angle_rad_real - angle_rad_guess}")
    print(f"Açı hatası (in degrees): {angle_deg_real - angle_deg_guess}")
    #print(f"Tahmin mesafe: {tahmin_mesafe}")
    #print(f"Gercek mesafe: {gercek_mesafe}")
    print(f"Mesafe Hatası: {gercek_mesafe-tahmin_mesafe}")
out_array = np.array(out_array)
init_array = np.array(init_array)

# x, y, z bileşenlerini ayır
x = out_array[:, 0]
y = out_array[:, 1]
z = out_array[:, 2]
t = init_array[:, 2]

local_min=min(z)
idx = np.where(z == local_min)[0]
sonuc=[x[idx],y[idx],z[idx]]
print("Sonuç:",sonuc)

# Grafik çiz
plt.plot(t, x, label='X', color='r')
plt.plot(t, y, label='Y', color='g')
plt.plot(t, z, label='Z', color='b')

plt.xlabel('Zaman / İndeks')
plt.ylabel('Değer')
plt.title('X, Y, Z Ekseni Değerleri')
plt.legend()
plt.grid(True)
plt.show()

# plt.ioff()
# # Gerçek ses kaynağını çiz
# fig2 = plt.figure(figsize=(10, 8))
# ax2 = fig2.add_subplot(111, projection='3d')

# # Grafik detayları
# ax2.set_xlabel('X Ekseni (mm)')
# ax2.set_ylabel('Y Ekseni (mm)')
# ax2.set_zlabel('Z Ekseni (mm)')
# ax2.set_title('Mikrofonlar, Gerçek ve Tahmin Edilen Ses Kaynağı Konumları')

# plt.grid(True)
# ax2.set_xlim(-500,500)
# ax2.set_ylim(0,600)
# ax2.set_zlim(-300,300)
# ax2.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', label='Mikrofonlar', s=100)
# ax2.scatter(current_guess[0], current_guess[1], current_guess[2], c='r', marker='^', label=f'Tahmin Edilen Konum: {np.around(current_guess,2)}', s=150)
# ax2.scatter(gercek_konum[0], gercek_konum[1], gercek_konum[2], c='r', marker='x', label=f'Gerçek Konum: {gercek_konum}', s=150)
# ax2.plot([0,current_guess[0],gercek_konum[0]],[0,current_guess[1],gercek_konum[1]],[0,current_guess[2],gercek_konum[2]])
# ax2.legend(loc="upper left")
# plt.show()

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