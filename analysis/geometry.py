import numpy as np

def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-9
    return (dx * ddy - dy * ddx) / denom


def compute_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    mag = np.hypot(dx, dy) + 1e-9
    nx = -dy / mag
    ny = dx / mag
    return nx, ny


def compute_distance(ref_x, ref_y, drv_x, drv_y):
    d = np.hypot(
        drv_x[:, None] - ref_x[None, :],
        drv_y[:, None] - ref_y[None, :]
    )
    idx = np.argmin(d, axis=1)
    return np.min(d, axis=1), idx
