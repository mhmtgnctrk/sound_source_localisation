import numpy as np

def closed_form_tdoa(mics, delta_r):
    """
    mics: array shape (M, D)  D=2 veya 3
    delta_r: array shape (M-1,)  -> [Δr21, Δr31, ..., ΔrM1]

    """
    mics = np.asarray(mics, float)
    M, D = mics.shape
    assert delta_r.shape[0] == M-1

    m1 = mics[0]
    Mi = mics[1:]                  # (M-1, D)
    dr = delta_r.reshape(-1, 1)    # (M-1, 1)


    # A matrisi
    A_geom = Mi - m1               # (M-1, D)
    A = np.hstack([A_geom, -dr])   # (M-1, 4)

    # b vektörü
    mi2 = np.sum(Mi**2, axis=1)    # ||m_i||^2
    m12 = np.sum(m1**2)            # ||m_1||^2
    
    b = 0.5 * (mi2 - m12 - (delta_r**2))

    b = b.reshape(-1, 1)

    # En küçük kareler çözümü
    x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    x_hat = x_hat.ravel()

    sx, sy, sz, r1 = x_hat
    s_hat = np.array([sx, sy, sz])

    return s_hat, float(r1)

a=225.0 # mm cinsinden kare kenar uzunluğu

mics = np.array([[-a/2, a/2, 9],     [a/2, a/2, 9],     [a/2, -a/2, 0],    [-a/2, -a/2, 0]])

# Ses hızı (mm/s)
c = 343200

f = open(r"E:\git_projects\sound_source_localisation\sound_source_localisation\tdoa\tdoas.txt", "r")
tdoas=[]

for i in f.readlines():
    tdoas.append(float(i))  
tdoas=np.array(tdoas) 

delta_r = c * np.array(tdoas[1:])

s_hat, r1_hat = closed_form_tdoa(mics, delta_r)
print("Konum tahmini (x,y,z):", s_hat, "  r1:", r1_hat)