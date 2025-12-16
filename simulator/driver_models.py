import numpy as np
from analysis.geometry import curvature

def compute_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    mag = np.hypot(dx, dy) + 1e-9
    nx = -dy / mag
    ny = dx / mag
    return nx, ny


def generate_driver_from_centerline(
    x,
    y,
    profile="normal",
    noise_scale=0.3,
    bias=0.0,
    seed=0
):
    """
    Generate a physically plausible driver trajectory
    as a perturbation of the centerline.
    """

    np.random.seed(seed)
    n = len(x)

    nx, ny = compute_normals(x, y)
    kappa = np.abs(curvature(x, y))

    if profile == "normal":
        offset = bias + np.random.normal(0, noise_scale, n)

    elif profile == "aggressive":
        offset = bias + np.random.normal(0, noise_scale * 2.0, n)
        offset += 0.6 * np.tanh(5 * kappa)

    elif profile == "smooth":
        slow_drift = np.sin(np.linspace(0, 2 * np.pi, n))
        offset = bias + 0.2 * slow_drift

    else:
        raise ValueError(f"Unknown driver profile: {profile}")

    drv_x = x + offset * nx
    drv_y = y + offset * ny

    return drv_x, drv_y
