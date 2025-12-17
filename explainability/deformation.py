# explainability/deformation.py
import numpy as np
from analysis.geometry import curvature, arc_length, compute_normals

def deform_reference_track(ref_x, ref_y, weakness_vec, max_offset=3.0):
    s = arc_length(ref_x, ref_y)
    s_norm = s / (s[-1] + 1e-9)

    kappa = np.abs(curvature(ref_x, ref_y))
    nx, ny = compute_normals(ref_x, ref_y)

    labels = ["apex", "slalom", "cornering"]
    focus = labels[np.argmax(weakness_vec)]

    delta = np.zeros_like(s)

    if focus == "apex":
        mask = kappa > np.percentile(kappa, 75)
        strength = weakness_vec[0]
    elif focus == "slalom":
        mask = kappa < np.percentile(kappa, 30)
        strength = weakness_vec[1]
    else:
        mask = np.ones_like(kappa, dtype=bool)
        strength = weakness_vec[2]

    delta[mask] = max_offset * strength * np.sin(2 * np.pi * 4 * s_norm[mask])
    delta = np.convolve(delta, np.ones(15) / 15, mode="same")

    return ref_x + delta * nx, ref_y + delta * ny, delta, focus
