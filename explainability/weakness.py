 # explainability/weakness.py
import numpy as np
from analysis.geometry import curvature
from analysis.distance import compute_distance

def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
    kappa = np.abs(curvature(ref_x, ref_y))

    apex_mask = kappa > np.percentile(kappa, 75)

    w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
    w_slalom = np.std(np.diff(dist))
    w_corner = np.mean(dist)

    w = np.array([w_apex, w_slalom, w_corner])
    return w / (np.linalg.norm(w) + 1e-9)
