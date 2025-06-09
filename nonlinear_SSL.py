from sympy import symbols, Eq, sqrt, nsolve, solve
import numpy as np

# Bilinmeyenler
x_s, y_s, z_s = symbols('x_s y_s z_s')

# Mikrofon pozisyonları (mm)
mpos = np.array([[113,0,0],[-36,0,0],[-76,0,0],[-113,0,0]])
ref_idx = 0
ref = mpos[ref_idx]

# Örnek time_differences ve c
c = 343000
tdoas = np.loadtxt("tdoa/tdoas.txt")  # [t0, t1, t2, t3]
ranges = tdoas * c

# # Denklemleri oluştur
# eqs = []
# for i in range(len(ranges)):
#     if i == ref_idx: continue
#     xi, yi, zi = mpos[i]
#     di_ref = sqrt((x_s-ref[0])**2 + (y_s-ref[1])**2 + (z_s-ref[2])**2)
#     di_i   = sqrt((x_s-xi)**2     + (y_s-yi)**2     + (z_s-zi)**2)
#     # d_i - d_ref = ranges[i]
#     eqs.append(Eq(di_i - di_ref, ranges[i]))

# Başlangıç tahmini (ör. dizi merkezine yakın)
guess = (0, 500, 0)
# z_s = 0 olarak sabit
eqs_2d = []
for i in range(len(ranges)):
    if i==ref_idx: continue
    xi, yi, _ = mpos[i]
    d_ref = sqrt((x_s-ref[0])**2 + (y_s-ref[1])**2)
    d_i   = sqrt((x_s-xi)**2     + (y_s-yi)**2)
    eqs_2d.append(Eq(d_i - d_ref + ranges[i],0))
sol_xy = nsolve(eqs_2d, (x_s, y_s), (100,100))
print(sol_xy)
# Çözümü sayısal olarak bul
# sol = nsolve(eqs, (x_s, y_s, z_s), guess, maxsteps=50, tol=1e-6)
# print("Çözüm:", sol)