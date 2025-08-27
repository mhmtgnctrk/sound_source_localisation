import numpy as np
import sympy as sp
from closed_form import closed_form_tdoa as cft

# ---------- 1) Sembolik kurulum (mics ve Δr sabit olarak gömülecek) ----------
def build_sympy_fJ(mics, delta_r):
    # mics: (M,3), delta_r: (M-1,) = [r2-r1, r3-r1, ...]
    x, y, z = sp.symbols('x y z', real=True)
    sx = [x, y, z]

    mics = np.asarray(mics, float)
    M = mics.shape[0]
    assert delta_r.shape[0] == M-1

    # referans m1
    x1, y1, z1 = map(sp.Float, mics[0])
    r1 = sp.sqrt( (x - x1)**2 + (y - y1)**2 + (z - z1)**2 )

    f_list = []
    for i in range(1, M):
        xi, yi, zi = map(sp.Float, mics[i])
        ri = sp.sqrt( (x - xi)**2 + (y - yi)**2 + (z - zi)**2 )
        dr = sp.Float(delta_r[i-1])
        fi = (ri - r1) - dr
        f_list.append(fi)

    f_vec = sp.Matrix(f_list)                 # (M-1)×1
    J = f_vec.jacobian(sp.Matrix(sx))         # (M-1)×3

    f_func = sp.lambdify((x, y, z), f_vec, 'numpy')
    J_func = sp.lambdify((x, y, z), J,     'numpy')
    return f_func, J_func

# ---------- 2) Maliyet ve tek GN adımı ----------
def cost_from_f(f): return 0.5 * float(f @ f)

def F_and_J_at(s, f_func, J_func):
    f = np.array(f_func(s[0], s[1], s[2]), dtype=float).reshape(-1)
    J = np.array(J_func(s[0], s[1], s[2]), dtype=float)
    F = 0.5 * float(f @ f)
    return F, f, J

def lm_solve_3d_sympy(f_func, J_func, s0,
                      max_iter=50,
                      lam0=None,
                      D_mode="marquardt",   # "I" veya "marquardt"
                      eps_H=1e-12):
    """
    Standart LM: (J^T J + λ D) d = - J^T f
    D: "I" (Levenberg) veya diag(J^T J) (Marquardt)
    λ güncellemesi: ρ (gain ratio) ile adaptif
    """
    s = np.array(s0, float).copy()
    F, f, J = F_and_J_at(s, f_func, J_func)
    JTJ = J.T @ J
    # başlangıç λ
    if lam0 is None:
        lam = 1e-3 * (np.max(np.diag(JTJ)) + 1.0)
    else:
        lam = float(lam0)

    for _ in range(max_iter):
        JTJ = J.T @ J
        g   = J.T @ f
        if D_mode.lower().startswith("m"):  # "marquardt"
            D = np.diag(np.diag(JTJ)) + eps_H*np.eye(3)
        else:                               # "I"
            D = np.eye(3)

        # LM sistemi
        H_lm = JTJ + lam * D
        try:
            d = -np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            # son çare: küçük ek düzenleme
            d = -np.linalg.lstsq(H_lm + eps_H*np.eye(3), g, rcond=None)[0]

        # model azalışı (predicted decrease)
        # m(0) - m(d) = 0.5*||f||^2 - [0.5*||f+Jd||^2 + 0.5*lam*d^T D d]
        f_lin = f + J @ d
        pred  = 0.5*(f @ f) - (0.5*(f_lin @ f_lin) + 0.5*lam*float(d.T @ (D @ d)))
        if pred <= 0:
            # sayısal tuhaflıkta pred negatifse küçük artışla lam'ı büyütüp tekrar dene
            lam *= 10.0
            continue

        # adımı dene (actual decrease)
        s_try = s + d
        F_try, f_try, J_try = F_and_J_at(s_try, f_func, J_func)
        ared = F - F_try
        rho  = ared / (pred + 1e-18)

        if rho > 0:  # kabul
            s, F, f, J = s_try, F_try, f_try, J_try
            # λ azalt (Nielsen önerisi – yumuşak ve stabil)
            lam = lam * max(1/3, 1 - (2*rho - 1)**3)
            # yakınsaklık testleri
            if np.linalg.norm(d) < 1e-6*(np.linalg.norm(s)+1e-6):
                break
            if np.linalg.norm(J.T @ f, ord=np.inf) < 1e-6:
                break
        else:        # ret
            lam *= 10.0  # daha temkinli ol
            # aynı noktadan, daha büyük λ ile yeniden d dene
            continue

    return s, F
# ----- verilerin -----

# === LM core + history + plots =========================
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def residual_and_jacobian(s, mics, delta_r, eps=1e-12):
    """
    s: (3,) [x,y,z] (mm)
    mics: (M,3) (mm)
    delta_r: (M-1,) = [r2-r1, r3-r1, ...] (mm)
    return: f (M-1,), J (M-1,3)
    """
    s = s.reshape(3)
    m1 = mics[0]
    Mi = mics[1:]

    diff1 = s - m1
    r1 = sqrt(np.dot(diff1, diff1) + eps)

    diffi = s - Mi
    ri = np.sqrt(np.sum(diffi**2, axis=1) + eps)

    # residuals: (ri - r1) - Δr
    f = (ri - r1) - delta_r

    # Jacobian rows: (s-mi)/ri - (s-m1)/r1
    Ji = diffi / ri[:, None]
    J1 = (diff1 / r1)
    J = Ji - J1
    return f, J

def lm_with_history(mics, delta_r, s0, max_iter=50, lam0=None,
                    D_mode="I", tol_step=1e-6, tol_grad=1e-6, eps_H=1e-12, z_min=10):
    """
    Levenberg–Marquardt:
       (J^T J + λ D) d = - J^T f
    D_mode: "I" (Levenberg) veya "marquardt" (diag(J^T J)).
    Döndürür: s_hat, history (list of dicts)
    """
    s = np.array(s0, dtype=float).copy()
    f, J = residual_and_jacobian(s, mics, delta_r)
    F = 0.5*float(f @ f)
    JTJ = J.T @ J
    g = J.T @ f

    lam = float(lam0) if lam0 is not None else 1e-3 * (np.max(np.diag(JTJ)) + 1.0)

    history = []
    def log(iter_idx, step_vec=None):
        history.append({
            "iter": iter_idx,
            "F": F,
            "lambda": lam,
            "step_norm": (np.linalg.norm(step_vec) if step_vec is not None else np.nan),
            "grad_inf": float(np.linalg.norm(g, ord=np.inf)),
            "x": s[0], "y": s[1], "z": s[2],
        })

    k = 0
    log(0)
    while k < max_iter:
        if np.linalg.norm(g, ord=np.inf) < tol_grad:
            break

        # D seçimi
        if D_mode.lower().startswith("m"):  # "marquardt"
            D = np.diag(np.diag(JTJ)) + eps_H*np.eye(3)
        else:                               # "I" (Levenberg)
            D = np.eye(3)

        # LM sistemi
        H_lm = JTJ + lam * D
        try:
            d = -np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            d = -np.linalg.lstsq(H_lm + eps_H*np.eye(3), g, rcond=None)[0]

        # Tahmini azalış (predicted decrease)
        d_used = d.copy()
        s_try = s + d_used
        if s_try[2] < z_min:
            s_try[2] = z_min
            d_used = s_try - s
        f_lin = f + J @ d_used
        pred = 0.5*(f @ f) - (0.5*(f_lin @ f_lin) + 0.5*lam*float(d_used.T @ (D @ d_used)))
        if pred <= 0:           # sayısal tuhaflık → λ büyüt, aynı iterasyonu tekrar dene
            lam *= 10.0
            continue

        # Adayı dene
        f_try, J_try = residual_and_jacobian(s_try, mics, delta_r)
        F_try = 0.5*float(f_try @ f_try)
        ared = F - F_try
        rho = ared / (pred + 1e-18)

        if rho > 0:  # adımı kabul
            s, f, J = s_try, f_try, J_try
            F = F_try
            JTJ = J.T @ J
            g = J.T @ f
            # Nielsen güncellemesi (stabil)
            lam = lam * max(1/3, 1 - (2*rho - 1)**3)

            k += 1
            log(k, d)

            # küçük adım kriteri
            if float(np.linalg.norm(d)) < tol_step*(np.linalg.norm(s)+tol_step):
                break
        else:        # adımı reddet → λ büyüt ve aynı iterasyonda tekrar dene
            lam *= 10.0
            continue

    return s, history

def plot_convergence(history, title_suffix="(LM)"):
    it   = [h["iter"] for h in history]
    Fv   = [h["F"] for h in history]
    lamv = [h["lambda"] for h in history]
    step = [h["step_norm"] for h in history]
    xv   = [h["x"] for h in history]
    yv   = [h["y"] for h in history]
    zv   = [h["z"] for h in history]

    # Aynı figür ve eksenler üzerinde çizelim
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 8))
    a1, a2, a3, a4 = axs2.ravel()

    # 1) F(s)
    a1.plot(it, Fv, marker="o")
    a1.set_title(f"F(s) vs Iteration {title_suffix}")
    a1.set_xlabel("Iteration")
    a1.set_ylabel(r"F = 0.5 ||f||²")
    a1.grid(True)

    # 2) lambda
    a2.plot(it, lamv, marker="o")
    a2.set_title(f"λ vs Iteration {title_suffix}")
    a2.set_xlabel("Iteration")
    a2.set_ylabel("lambda")
    a2.grid(True)

    # 3) step norm (step vektör ise norma çeviriyoruz)
    step_norm = np.linalg.norm(step, axis=1) if hasattr(step, "ndim") and step.ndim > 1 else step
    a3.plot(it, step_norm, marker="o")
    a3.set_title(f"||Δ|| vs Iteration {title_suffix}")
    a3.set_xlabel("Iteration")
    a3.set_ylabel(r"||delta||")
    a3.grid(True)

    # 4) parametreler
    a4.plot(it, xv, marker="o", label="x")
    a4.plot(it, yv, marker="o", label="y")
    a4.plot(it, zv, marker="o", label="z")
    a4.set_title(f"Parameters vs Iteration {title_suffix}")
    a4.set_xlabel("Iteration")
    a4.set_ylabel("mm")
    a4.legend()
    a4.grid(True)

    fig2.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- mics ve delta_r senin kodundan gelsin ---
    a = 225.0
    mics = np.array([
        [-a/2,  a/2,  9.0],
        [ a/2,  a/2,  9.0],
        [ a/2, -a/2,  0.0],
        [-a/2, -a/2,  0.0],
    ], float)

    # Ses hızı (mm/s)
    c = 343200

    f = open(r"E:\git_projects\sound_source_localisation\sound_source_localisation\tdoa\tdoas.txt", "r")
    tdoas=[]

    for i in f.readlines():
        tdoas.append(float(i))  
    tdoas=np.array(tdoas) 

    delta_r = c * np.array(tdoas[1:])

    # build_sympy_fJ(mics, delta_r) zaten sende var:
    f_func, J_func = build_sympy_fJ(mics, delta_r)


    s0,_=cft(mics=mics,delta_r=delta_r)
    #s0,_ = cft(mics, delta_r)

    # Levenberg (D=I) öneririm:
    s_hat, hist = lm_with_history(mics, delta_r, s0,
                                max_iter=50,
                                D_mode="I",      # "marquardt" da deneyebilirsin
                                lam0=None)

    print("LM (D=I) s_hat [mm]:", np.round(s_hat, 6))

    # Doğrulama: pred Δr
    r_all = np.linalg.norm(mics - s_hat, axis=1)
    dr_pred = r_all[1:] - r_all[0]
    print("input Δr [mm]:", np.round(delta_r, 6))
    print("pred  Δr [mm]:", np.round(dr_pred, 6))
    print("diff  Δr [mm]:", np.round(dr_pred - delta_r, 9))

    # Yakınsama grafikleri
    plot_convergence(hist, title_suffix="(D=I)")

    gercek_konum=[-212,112,300]


    tahmin_mesafe=np.linalg.norm(s_hat)
    gercek_mesafe=np.linalg.norm(gercek_konum)

    mesafe_hatasi = gercek_mesafe-tahmin_mesafe
    print(f"Mesafe Hatası: {mesafe_hatasi}")


