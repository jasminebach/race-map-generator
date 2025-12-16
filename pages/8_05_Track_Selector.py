import sys
import os
import time
import csv
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulator.track_loader import load_track_csv
from simulator.driver_models import generate_driver_from_centerline
from analysis.geometry import curvature

# ------------------------------------------------------
# Storage paths
# ------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# ------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------
st.set_page_config(layout="wide")
st.title("🏎 Synthetic Driver Simulation & Telemetry Viewer")

# ------------------------------------------------------
# Track selection
# ------------------------------------------------------
TRACK_DIR = os.path.join(ROOT, "SIMULINK-DATA")
tracks = sorted([f for f in os.listdir(TRACK_DIR) if "Centerline" in f])

track_name = st.selectbox("Select Track Centerline", tracks)
track_path = os.path.join(TRACK_DIR, track_name)

ref = load_track_csv(track_path)
ref_x = np.array([p[0] for p in ref])
ref_y = np.array([p[1] for p in ref])

# ------------------------------------------------------
# Driver settings
# ------------------------------------------------------
st.subheader("Driver Profile")

driver_profile = st.selectbox(
    "Driver Type",
    ["normal", "aggressive", "smooth"]
)

noise_scale = st.slider("Lateral Noise Scale (m)", 0.1, 1.5, 0.4)
seed = st.number_input("Random Seed", 0, 9999, 42)

# ------------------------------------------------------
# Generate synthetic driver
# ------------------------------------------------------
drv_x, drv_y = generate_driver_from_centerline(
    ref_x,
    ref_y,
    profile=driver_profile,
    noise_scale=noise_scale,
    seed=seed
)

kappa = np.abs(curvature(ref_x, ref_y))

# ------------------------------------------------------
# Telemetry synthesis
# ------------------------------------------------------
def synthesize_packet(i):
    speed = 85 - 50 * min(kappa[i] * 15, 1.0)
    speed += np.random.normal(0, 1.5)

    yaw = np.sign(kappa[i]) * np.sqrt(abs(kappa[i])) * 30
    yaw += np.random.normal(0, 1.5)

    brake = max(0, min(1, kappa[i] * 8))
    throttle = max(0, 1 - brake)

    return {
        "speed": speed,
        "yaw": yaw,
        "brake": brake * 40,
        "throttle": throttle
    }

# ------------------------------------------------------
# UI placeholders
# ------------------------------------------------------
metrics_placeholder = st.empty()
track_col, graph_col = st.columns([2, 1])
track_placeholder = track_col.empty()
graph_placeholder = graph_col.empty()

# ------------------------------------------------------
# Telemetry history
# ------------------------------------------------------
if "telemetry_history" not in st.session_state:
    st.session_state.telemetry_history = {
        "t": [],
        "speed": [],
        "brake": [],
        "yaw": [],
    }

# ------------------------------------------------------
# Draw helpers
# ------------------------------------------------------
def draw_numeric(p):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Speed", f"{p['speed']:.1f} km/h")
    c2.metric("Throttle", f"{p['throttle']:.2f}")
    c3.metric("Brake", f"{p['brake']:.1f} bar")
    c4.metric("Yaw", f"{p['yaw']:.2f}°")


def draw_track(i):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ref_x, ref_y, "--", color="gray", label="Centerline")
    ax.plot(drv_x[:i], drv_y[:i], color="blue", label="Driver Path")
    ax.scatter(drv_x[i], drv_y[i], color="red", s=80)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title("Driver Position on Track")
    return fig


def draw_graphs():
    t = st.session_state.telemetry_history["t"]

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    axes[0].plot(t, st.session_state.telemetry_history["speed"])
    axes[0].set_ylabel("Speed")

    axes[1].plot(t, st.session_state.telemetry_history["brake"], color="green")
    axes[1].set_ylabel("Brake")

    axes[2].plot(t, st.session_state.telemetry_history["yaw"], color="purple")
    axes[2].set_ylabel("Yaw")
    axes[2].set_xlabel("Step")

    for ax in axes:
        ax.grid(True)

    return fig

# ------------------------------------------------------
# Simulation control
# ------------------------------------------------------
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

colA, colB = st.columns(2)

# ---------- START ----------
if colA.button("🚀 Start Simulation"):
    st.session_state.sim_running = True
    st.session_state.telemetry_history = {k: [] for k in st.session_state.telemetry_history}

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "driver_trajectory.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["<X>", "<Y>"])

    st.session_state.csv_file = csv_file
    st.session_state.csv_writer = csv_writer
    st.session_state.run_dir = run_dir

    st.success(f"Simulation started — saving to {csv_path}")

# ---------- STOP ----------
if colB.button("🛑 Stop Simulation"):
    st.session_state.sim_running = False

    if "csv_file" in st.session_state:
        st.session_state.csv_file.close()
        del st.session_state.csv_file
        del st.session_state.csv_writer

    st.warning("Simulation stopped — CSV saved")

# ------------------------------------------------------
# Main loop
# ------------------------------------------------------
if st.session_state.sim_running:
    for i in range(len(ref_x)):
        if not st.session_state.sim_running:
            break

        p = synthesize_packet(i)

        st.session_state.telemetry_history["t"].append(i)
        st.session_state.telemetry_history["speed"].append(p["speed"])
        st.session_state.telemetry_history["brake"].append(p["brake"])
        st.session_state.telemetry_history["yaw"].append(p["yaw"])

        # --- SAVE CSV ROW (<X>,<Y>) ---
        st.session_state.csv_writer.writerow([
            float(drv_x[i]),
            float(drv_y[i])
        ])

        with metrics_placeholder.container():
            draw_numeric(p)

        track_placeholder.pyplot(draw_track(i))
        graph_placeholder.pyplot(draw_graphs())

        time.sleep(0.05)

else:
    st.info("Press **Start Simulation** to begin synthetic driving.")


# import sys
# import os
# import time
# import json

# ROOT = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT)

# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# from simulator.track_loader import load_track_csv
# from simulator.driver_models import generate_driver_from_centerline
# from analysis.geometry import curvature

# # ------------------------------------------------------
# # Streamlit setup
# # ------------------------------------------------------
# st.set_page_config(layout="wide")
# st.title("🏎 Synthetic Driver Simulation & Telemetry Viewer")

# # ------------------------------------------------------
# # Track selection
# # ------------------------------------------------------
# TRACK_DIR = os.path.join(ROOT, "SIMULINK-DATA")

# tracks = sorted([f for f in os.listdir(TRACK_DIR) if "Centerline" in f])
# track_name = st.selectbox("Select Track Centerline", tracks)
# track_path = os.path.join(TRACK_DIR, track_name)

# ref = load_track_csv(track_path)
# ref_x = np.array([p[0] for p in ref])
# ref_y = np.array([p[1] for p in ref])

# # ------------------------------------------------------
# # Driver settings
# # ------------------------------------------------------
# st.subheader("Driver Profile")

# driver_profile = st.selectbox(
#     "Driver Type",
#     ["normal", "aggressive", "smooth"]
# )

# noise_scale = st.slider("Lateral Noise Scale (m)", 0.1, 1.5, 0.4)
# seed = st.number_input("Random Seed", 0, 9999, 42)

# # ------------------------------------------------------
# # Generate synthetic driver
# # ------------------------------------------------------
# drv_x, drv_y = generate_driver_from_centerline(
#     ref_x,
#     ref_y,
#     profile=driver_profile,
#     noise_scale=noise_scale,
#     seed=seed
# )

# kappa = np.abs(curvature(ref_x, ref_y))

# # ------------------------------------------------------
# # Telemetry synthesis
# # ------------------------------------------------------
# def synthesize_packet(i):
#     speed = 85 - 50 * min(kappa[i] * 15, 1.0)
#     speed += np.random.normal(0, 1.5)

#     yaw = np.sign(kappa[i]) * np.sqrt(abs(kappa[i])) * 30
#     yaw += np.random.normal(0, 1.5)

#     brake = max(0, min(1, kappa[i] * 8))
#     throttle = max(0, 1 - brake)

#     return {
#         "true": {
#             "speed_kmh": speed,
#             "yaw_deg": yaw,
#             "coolant_temp": 85 + np.random.normal(0, 0.4),
#             "throttle": throttle
#         },
#         "sensors": {
#             "brake_pressure": brake * 40
#         },
#         "gps": {
#             "x": drv_x[i],
#             "y": drv_y[i]
#         }
#     }

# # ------------------------------------------------------
# # UI placeholders
# # ------------------------------------------------------
# metrics_placeholder = st.empty()
# track_col, graph_col = st.columns([2, 1])
# track_placeholder = track_col.empty()
# graph_placeholder = graph_col.empty()

# # ------------------------------------------------------
# # Telemetry history
# # ------------------------------------------------------
# telemetry_history = {
#     "t": [],
#     "speed": [],
#     "brake": [],
#     "yaw": [],
# }

# # ------------------------------------------------------
# # Draw helpers
# # ------------------------------------------------------
# def draw_numeric(packet):
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Speed", f"{packet['true']['speed_kmh']:.1f} km/h")
#     c2.metric("Throttle", f"{packet['true']['throttle']:.2f}")
#     c3.metric("Brake", f"{packet['sensors']['brake_pressure']:.1f} bar")
#     c4.metric("Yaw", f"{packet['true']['yaw_deg']:.2f}°")


# def draw_track(i):
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.plot(ref_x, ref_y, "--", color="gray", label="Centerline")
#     ax.plot(drv_x[:i], drv_y[:i], color="blue", label="Driver Path")
#     ax.scatter(drv_x[i], drv_y[i], color="red", s=80)
#     ax.set_aspect("equal")
#     ax.grid(True)
#     ax.legend()
#     ax.set_title("Driver Position on Track")
#     return fig


# def draw_graphs():
#     t = telemetry_history["t"]

#     fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

#     axes[0].plot(t, telemetry_history["speed"])
#     axes[0].set_ylabel("Speed (km/h)")

#     axes[1].plot(t, telemetry_history["brake"], color="green")
#     axes[1].set_ylabel("Brake (bar)")

#     axes[2].plot(t, telemetry_history["yaw"], color="purple")
#     axes[2].set_ylabel("Yaw (deg)")
#     axes[2].set_xlabel("Step")

#     for ax in axes:
#         ax.grid(True)

#     return fig

# # ------------------------------------------------------
# # Simulation control
# # ------------------------------------------------------
# if "sim_running" not in st.session_state:
#     st.session_state.sim_running = False

# colA, colB = st.columns(2)

# if colA.button("🚀 Start Simulation"):
#     st.session_state.sim_running = True
#     telemetry_history = {k: [] for k in telemetry_history}
#     st.success("Simulation Started")

# if colB.button("🛑 Stop Simulation"):
#     st.session_state.sim_running = False
#     st.warning("Simulation Stopped")

# # ------------------------------------------------------
# # Main loop
# # ------------------------------------------------------
# if st.session_state.sim_running:
#     for i in range(len(ref_x)):
#         if not st.session_state.sim_running:
#             break

#         packet = synthesize_packet(i)

#         telemetry_history["t"].append(i)
#         telemetry_history["speed"].append(packet["true"]["speed_kmh"])
#         telemetry_history["brake"].append(packet["sensors"]["brake_pressure"])
#         telemetry_history["yaw"].append(packet["true"]["yaw_deg"])

#         with metrics_placeholder.container():
#             draw_numeric(packet)

#         track_placeholder.pyplot(draw_track(i))
#         graph_placeholder.pyplot(draw_graphs())

#         time.sleep(0.05)

# else:
#     st.info("Press **Start Simulation** to begin synthetic driving.")
