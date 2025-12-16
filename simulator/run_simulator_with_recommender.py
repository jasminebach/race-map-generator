import argparse
import time
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

from simulator.track_loader import load_track_csv
from simulator.driver_models import generate_driver_from_centerline
from analysis.geometry import curvature

# ======================================================
# Argument parsing
# ======================================================
parser = argparse.ArgumentParser(description="Synthetic Telemetry Simulator")

parser.add_argument("--track", required=True, help="Centerline CSV filename")
parser.add_argument("--driver-id", default="normal",
                    choices=["normal", "aggressive", "smooth"])
parser.add_argument("--target-laps", type=int, default=1)

parser.add_argument("--data-dir", required=True,
                    help="Data directory (for realtime.json)")
parser.add_argument("--track-dir", required=True,
                    help="Directory containing track CSVs")
parser.add_argument("--log-dir", required=True,
                    help="Directory to store persistent logs")

parser.add_argument("--progress-file", required=True,
                    help="Progress JSON file")

args = parser.parse_args()

# ======================================================
# Paths & directories
# ======================================================
REALTIME_FILE = os.path.join(args.data_dir, "realtime.json")
STOP_FILE = os.path.join(args.data_dir, "stop_signal.txt")

os.makedirs(args.log_dir, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(args.log_dir, f"run_{run_id}")
os.makedirs(run_dir, exist_ok=True)

TELEMETRY_LOG_PATH = os.path.join(run_dir, "telemetry.jsonl")
TRAJECTORY_CSV_PATH = os.path.join(run_dir, "trajectory.csv")
SUMMARY_PATH = os.path.join(run_dir, "summary.json")

# ======================================================
# Load track
# ======================================================
track_path = os.path.join(args.track_dir, args.track)
track_pts = load_track_csv(track_path)

ref_x = np.array([p[0] for p in track_pts])
ref_y = np.array([p[1] for p in track_pts])

n_points = len(ref_x)
kappa = np.abs(curvature(ref_x, ref_y))

# ======================================================
# Generate synthetic driver trajectory
# ======================================================
drv_x, drv_y = generate_driver_from_centerline(
    ref_x,
    ref_y,
    profile=args.driver_id,
    noise_scale=0.4,
    seed=42
)

# ======================================================
# Open telemetry log
# ======================================================
telemetry_log = open(TELEMETRY_LOG_PATH, "w")

# ======================================================
# Simulation loop
# ======================================================
lap = 0
idx = 0
step = 0
t0 = time.time()

print(f"[SIM] Started run {run_id}")
print(f"[SIM] Driver profile: {args.driver_id}")
print(f"[SIM] Target laps: {args.target_laps}")

while lap < args.target_laps:

    # Stop signal (from Streamlit)
    if os.path.exists(STOP_FILE):
        print("[SIM] Stop signal received")
        break

    # Loop around track
    if idx >= n_points:
        idx = 0
        lap += 1
        continue

    # --------------------------------------------------
    # Telemetry synthesis (physically plausible)
    # --------------------------------------------------
    curvature_factor = min(kappa[idx] * 15.0, 1.0)

    speed = 90 - 55 * curvature_factor
    speed += np.random.normal(0, 1.2)

    yaw = np.sign(kappa[idx]) * np.sqrt(kappa[idx]) * 35
    yaw += np.random.normal(0, 1.5)

    brake = min(1.0, kappa[idx] * 8.0)
    throttle = max(0.0, 1.0 - brake)

    packet = {
        "timestamp": time.time() - t0,
        "step": step,
        "lap": lap,

        "gps": {
            "x": float(drv_x[idx]),
            "y": float(drv_y[idx])
        },

        "true": {
            "speed_kmh": float(speed),
            "yaw_deg": float(yaw),
            "coolant_temp": float(85 + np.random.normal(0, 0.4)),
            "throttle": float(throttle)
        },

        "sensors": {
            "brake_pressure": float(brake * 40.0)
        }
    }

    # --------------------------------------------------
    # Write LIVE telemetry (overwrite)
    # --------------------------------------------------
    with open(REALTIME_FILE, "w") as f:
        json.dump(packet, f, indent=2)

    # --------------------------------------------------
    # Append persistent telemetry
    # --------------------------------------------------
    telemetry_log.write(json.dumps(packet) + "\n")
    telemetry_log.flush()

    # --------------------------------------------------
    # Progress file
    # --------------------------------------------------
    with open(args.progress_file, "w") as f:
        json.dump({
            "lap": lap,
            "index": idx,
            "step": step
        }, f)

    idx += 1
    step += 1
    time.sleep(0.05)

# ======================================================
# Save final trajectory
# ======================================================
df_traj = pd.DataFrame({
    "x": drv_x,
    "y": drv_y
})
df_traj.to_csv(TRAJECTORY_CSV_PATH, index=False)

# ======================================================
# Save run summary
# ======================================================
summary = {
    "run_id": run_id,
    "driver_profile": args.driver_id,
    "target_laps": args.target_laps,
    "completed_laps": lap,
    "num_steps": step,
    "track_file": args.track,
    "start_time": run_id
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)

telemetry_log.close()

print(f"[SIM] Run finished → {run_dir}")


# import argparse
# import time
# import json
# import os
# import numpy as np

# from simulator.track_loader import load_track_csv
# from simulator.driver_models import generate_driver_from_centerline
# from analysis.geometry import curvature

# # ------------------------------------------------------
# # Arguments
# # ------------------------------------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--track", required=True)
# parser.add_argument("--driver-id", default="normal")
# parser.add_argument("--target-laps", type=int, default=1)
# parser.add_argument("--progress-file", required=True)
# parser.add_argument("--data-dir", required=True)
# parser.add_argument("--track-dir", required=True)
# parser.add_argument("--log-dir", required=True)

# args = parser.parse_args()

# # ------------------------------------------------------
# # Paths
# # ------------------------------------------------------
# REALTIME_FILE = os.path.join(args.data_dir, "realtime.json")
# STOP_FILE = os.path.join(args.data_dir, "stop_signal.txt")

# # ------------------------------------------------------
# # Load track
# # ------------------------------------------------------
# track_path = os.path.join(args.track_dir, args.track)
# track = load_track_csv(track_path)

# ref_x = np.array([p[0] for p in track])
# ref_y = np.array([p[1] for p in track])

# kappa = np.abs(curvature(ref_x, ref_y))

# # ------------------------------------------------------
# # Generate driver
# # ------------------------------------------------------
# drv_x, drv_y = generate_driver_from_centerline(
#     ref_x,
#     ref_y,
#     profile=args.driver_id,
#     noise_scale=0.4,
#     seed=42
# )

# # ------------------------------------------------------
# # Simulation loop
# # ------------------------------------------------------
# lap = 0
# idx = 0
# t0 = time.time()

# while lap < args.target_laps:

#     if os.path.exists(STOP_FILE):
#         break

#     # Loop around track
#     if idx >= len(ref_x):
#         idx = 0
#         lap += 1

#     # --- Telemetry synthesis ---
#     speed = 85 - 50 * min(kappa[idx] * 15, 1.0)
#     speed += np.random.normal(0, 1.0)

#     yaw = np.sign(kappa[idx]) * np.sqrt(kappa[idx]) * 30
#     yaw += np.random.normal(0, 1.0)

#     brake = max(0, min(1, kappa[idx] * 8))
#     throttle = max(0, 1 - brake)

#     packet = {
#         "timestamp": time.time() - t0,
#         "lap": lap,
#         "gps": {
#             "x": float(drv_x[idx]),
#             "y": float(drv_y[idx])
#         },
#         "true": {
#             "speed_kmh": float(speed),
#             "yaw_deg": float(yaw),
#             "coolant_temp": float(85 + np.random.normal(0, 0.4)),
#             "throttle": float(throttle)
#         },
#         "sensors": {
#             "brake_pressure": float(brake * 40)
#         }
#     }

#     # Write realtime packet
#     with open(REALTIME_FILE, "w") as f:
#         json.dump(packet, f, indent=2)

#     # Progress
#     with open(args.progress_file, "w") as f:
#         json.dump({
#             "lap": lap,
#             "index": idx
#         }, f)

#     idx += 1
#     time.sleep(0.05)

# print("Simulation finished.")
