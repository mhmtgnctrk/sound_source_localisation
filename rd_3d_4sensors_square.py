
import numpy as np
from typing import Optional, Tuple

def build_G_b(sensors_xy: np.ndarray, r_i1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert sensors_xy.shape == (4, 2), "sensors_xy must be (4,2) for planar sensors"
    assert r_i1.shape == (3,), "r_i1 must be [r21, r31, r41]"
    x1, y1 = sensors_xy[0]
    G = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for row, i in enumerate([1, 2, 3]):
        xi, yi = sensors_xy[i]
        dx, dy = xi - x1, yi - y1
        ri1 = r_i1[row]
        G[row, :] = [dx, dy, ri1]
        b[row] = 0.5 * ((xi**2 + yi**2) - (x1**2 + y1**2) - ri1**2)
    return G, b

def solve_xy_R1s(G: np.ndarray, b: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    if W is None:
        try:
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(G) @ b
    GTWG = G.T @ W @ G
    GTWb = G.T @ W @ b
    try:
        return np.linalg.solve(GTWG, GTWb)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(GTWG) @ GTWb

def recover_z_positive(x: float, y: float, R1s: float, sensor1_xyz: np.ndarray) -> float:
    x1, y1, z1 = sensor1_xyz
    val = R1s**2 - (x - x1)**2 - (y - y1)**2
    if val <= 0:
        return 0.0
    return max(0.0, np.sqrt(val) + z1)

def rd_from_positions(xs: np.ndarray, sensors_xyz: np.ndarray) -> np.ndarray:
    R = np.linalg.norm(sensors_xyz - xs[None, :], axis=1)
    return np.array([R[1] - R[0], R[2] - R[0], R[3] - R[0]], dtype=float)

def residuals_and_jacobian(x: np.ndarray, sensors_xyz: np.ndarray, r_i1: np.ndarray):
    x1 = sensors_xyz[0]
    f = np.zeros(3, dtype=float)
    J = np.zeros((3, 3), dtype=float)
    R1 = np.linalg.norm(x - x1)
    if R1 < 1e-9: R1 = 1e-9
    g1 = (x - x1) / R1
    for row, i in enumerate([1, 2, 3]):
        xi = sensors_xyz[i]
        Ri = np.linalg.norm(x - xi)
        if Ri < 1e-9: Ri = 1e-9
        gi = (x - xi) / Ri
        f[row] = Ri - R1 - r_i1[row]
        J[row, :] = gi - g1
    return f, J

def lm_refine(x0: np.ndarray, sensors_xyz: np.ndarray, r_i1: np.ndarray,
              weights: Optional[np.ndarray] = None, max_iter: int = 25, lam0: float = 1e-2) -> np.ndarray:
    x = x0.astype(float).copy()
    lam = lam0
    W = None if weights is None else np.diag(weights)
    for _ in range(max_iter):
        f, J = residuals_and_jacobian(x, sensors_xyz, r_i1)
        JTJ = J.T @ J if W is None else J.T @ W @ J
        JTf = J.T @ f if W is None else J.T @ W @ f
        H = JTJ + lam * np.eye(3)
        try:
            step = -np.linalg.solve(H, JTf)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(H) @ JTf
        x_new = x + step
        x_new[2] = max(0.0, x_new[2])  # z >= 0
        f_new, _ = residuals_and_jacobian(x_new, sensors_xyz, r_i1)
        cost = f@f if W is None else f.T@W@f
        cost_new = f_new@f_new if W is None else f_new.T@W@f_new
        if cost_new < cost:
            x = x_new
            lam *= 0.5
            if np.linalg.norm(step) < 1e-6: break
        else:
            lam *= 2.0
    return x

if __name__ == '__main__':
    # Example with square frame (225 mm), sensors on z=0 plane.
    a = 225.0
    sensors_xyz = np.array([
        [ -a/2, +a/2, 0.0],
        [ +a/2, +a/2, 0.0],
        [ +a/2, -a/2, 0.0],
        [ -a/2, -a/2, 0.0],
    ], dtype=float)
    x_true = np.array([-212.0, 112.0, 300.0], dtype=float)
    r_meas = rd_from_positions(x_true, sensors_xyz)  # noiseless demo
    G, b = build_G_b(sensors_xyz[:, :2], r_meas)
    u = solve_xy_R1s(G, b)
    x, y, R1s = u
    z = recover_z_positive(x, y, R1s, sensors_xyz[0])
    x0 = np.array([x, y, z])
    x_ref = lm_refine(x0, sensors_xyz, r_meas)
    print('Estimated (initial):', x0)
    print('Estimated (refined):', x_ref)
    print('True:', x_true)
