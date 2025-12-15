import numpy as np
from .geometry import compute_distance

def analyze_driver_weakness(ref_x, ref_y, drv_x, drv_y, segments):
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)

    segment_errors = {}

    for seg in segments:
        mask = (idx >= seg["start"]) & (idx < seg["end"])
        if not np.any(mask):
            continue

        err = dist[mask]
        label = seg["label"]

        if label not in segment_errors:
            segment_errors[label] = []

        segment_errors[label].append({
            "mean_error": np.mean(err),
            "std_error": np.std(err)
        })

    scores = {}
    for label, values in segment_errors.items():
        mean_err = np.mean([v["mean_error"] for v in values])
        std_err = np.mean([v["std_error"] for v in values])
        scores[label] = mean_err + 0.5 * std_err

    primary_weakness = max(scores, key=scores.get)

    return scores, primary_weakness
