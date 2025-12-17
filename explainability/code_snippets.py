# explainability/code_snippets.py

DEFORM_CODE = """
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

    new_x = ref_x + delta * nx
    new_y = ref_y + delta * ny

    return new_x, new_y, delta, focus
"""

ANALYZE_CODE = """
def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
    kappa = np.abs(curvature(ref_x, ref_y))

    apex_mask = kappa > np.percentile(kappa, 75)

    w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
    w_slalom = np.std(np.diff(dist))
    w_corner = np.mean(dist)

    w = np.array([w_apex, w_slalom, w_corner])
    return w / (np.linalg.norm(w) + 1e-9)
"""
