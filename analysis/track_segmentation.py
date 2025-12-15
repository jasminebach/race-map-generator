import numpy as np
from .geometry import curvature

def segment_track(x, y, window=20, kappa_straight=0.002, kappa_slalom_std=0.01):
    """
    Segments the track based on curvature.
    """
    kappa = curvature(x, y)
    segments = []

    i = 0
    while i < len(x) - window:
        k_win = kappa[i:i+window]
        mean_k = np.mean(k_win)
        std_k = np.std(k_win)

        if abs(mean_k) < kappa_straight:
            label = "straight"
        elif std_k > kappa_slalom_std:
            label = "slalom"
        elif mean_k > 0:
            label = "left_turn"
        else:
            label = "right_turn"

        segments.append({
            "start": i,
            "end": i + window,
            "label": label
        })

        i += window

    return segments


# import numpy as np
# from .geometry import curvature

# def segment_track(x, y, window=20,
#                   kappa_straight=0.002,
#                   kappa_slalom_std=0.01):
#     """
#     Returns:
#         segments: list of dicts with
#         - start_idx
#         - end_idx
#         - label
#     """

#     kappa = curvature(x, y)
#     segments = []

#     i = 0
#     while i < len(x) - window:
#         k_win = kappa[i:i+window]
#         mean_k = np.mean(k_win)
#         std_k = np.std(k_win)

#         if abs(mean_k) < kappa_straight:
#             label = "straight"
#         elif std_k > kappa_slalom_std:
#             label = "slalom"
#         elif mean_k > 0:
#             label = "left_turn"
#         else:
#             label = "right_turn"

#         segments.append({
#             "start": i,
#             "end": i + window,
#             "label": label
#         })

#         i += window

#     return segments
