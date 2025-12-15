import os
import argparse
import numpy as np
import pandas as pd

# ======================================================
# CLI arguments
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch-dir",
    required=True,
    help="Path to SIMULINK-DATA/batch_n directory"
)
args = parser.parse_args()

BATCH_DIR = os.path.abspath(args.batch_dir)

CENTERLINE_CSV = os.path.join(
    BATCH_DIR, "Simulation_Data_Export - Centerline.csv"
)
SIMULATION_CSV = os.path.join(
    BATCH_DIR, "Simulation_Data_Export - Simulation.csv"
)
OUTPUT_CSV = os.path.join(
    BATCH_DIR, "derived_tracking_dataset.csv"
)

# ======================================================
# Sanity checks
# ======================================================
if not os.path.exists(CENTERLINE_CSV):
    raise FileNotFoundError(CENTERLINE_CSV)

if not os.path.exists(SIMULATION_CSV):
    raise FileNotFoundError(SIMULATION_CSV)

# ======================================================
# Load data
# ======================================================
ref = pd.read_csv(CENTERLINE_CSV)
drv = pd.read_csv(SIMULATION_CSV)

# MATLAB / Simulink coordinate convention
#   Simulink: X forward, Y left
#   MATLAB:   plot(Y, X)
ref_x = -ref["yRef"].to_numpy()
ref_y =  ref["xRef"].to_numpy()

drv_x = -drv["<Y>"].to_numpy()
drv_y =  drv["<X>"].to_numpy()

# ======================================================
# Projection helper
# ======================================================
def project_point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    denom = np.dot(ab, ab)

    if denom == 0:
        return a, np.linalg.norm(p - a), 0.0

    t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return proj, dist, t

# ======================================================
# Precompute longitudinal s along centerline
# ======================================================
centerline = np.column_stack([ref_x, ref_y])
segment_lengths = np.linalg.norm(
    np.diff(centerline, axis=0), axis=1
)

s_ref = np.concatenate([[0.0], np.cumsum(segment_lengths)])

# ======================================================
# Generate derived dataset
# ======================================================
rows = []

for i in range(len(drv_x)):
    p = np.array([drv_x[i], drv_y[i]])

    best_dist = np.inf
    best_proj = None
    best_s = None

    for j in range(len(centerline) - 1):
        a = centerline[j]
        b = centerline[j + 1]

        proj, dist, t = project_point_to_segment(p, a, b)

        if dist < best_dist:
            best_dist = dist
            best_proj = proj
            best_s = s_ref[j] + t * segment_lengths[j]

    # Vehicle heading (finite difference)
    if i > 0:
        heading = np.arctan2(
            drv_y[i] - drv_y[i - 1],
            drv_x[i] - drv_x[i - 1]
        )
    else:
        heading = 0.0

    rows.append({
        "idx": i,
        "x_vehicle": drv_x[i],
        "y_vehicle": drv_y[i],
        "x_ref": best_proj[0],
        "y_ref": best_proj[1],
        "lateral_error": best_dist,
        "longitudinal_s": best_s,
        "heading_vehicle": heading,
    })

# ======================================================
# Save
# ======================================================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Derived dataset saved to: {OUTPUT_CSV}")
