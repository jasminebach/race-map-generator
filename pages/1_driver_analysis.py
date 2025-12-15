import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================
# Page setup
# =================================================
st.set_page_config(page_title="Driver Analysis", layout="wide")
st.title("📊 Driver vs Centerline — Analysis View")

# =================================================
# Load data
# =================================================
ref = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Centerline.csv")
drv = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Simulation.csv")

# =================================================
# MATLAB-consistent coordinate transform
# =================================================
ref_x = -ref["yRef"].to_numpy()
ref_y =  ref["xRef"].to_numpy()
drv_x = -drv["<Y>"].to_numpy()
drv_y =  drv["<X>"].to_numpy()

centerline = np.column_stack([ref_x, ref_y])
driver = np.column_stack([drv_x, drv_y])

# =================================================
# Geometry: orthogonal projection
# =================================================
def project_point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return proj, np.linalg.norm(p - proj)

proj = np.zeros_like(driver)
error = np.zeros(len(driver))

for i, p in enumerate(driver):
    best_d = np.inf
    for j in range(len(centerline) - 1):
        q, d = project_point_to_segment(p, centerline[j], centerline[j + 1])
        if d < best_d:
            best_d = d
            proj[i] = q
            error[i] = d

# =================================================
# Frenet distance s
# =================================================
s = np.zeros(len(centerline))
s[1:] = np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))

# =================================================
# Plots
# =================================================
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ref_x, ref_y, "--", label="Centerline")
    sc = ax.scatter(drv_x, drv_y, c=error, cmap="inferno", s=8)
    plt.colorbar(sc, ax=ax, label="Lateral Error [m]")
    ax.set_aspect("equal")
    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.set_title("Trajectory Error Map")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(error)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Lateral Error [m]")
    ax2.set_title("Lateral Error vs Time")
    ax2.grid(True)
    st.pyplot(fig2)

# =================================================
# Metrics
# =================================================
c1, c2, c3 = st.columns(3)
c1.metric("Mean Error [m]", f"{error.mean():.3f}")
c2.metric("Max Error [m]", f"{error.max():.3f}")
c3.metric("95th Percentile [m]", f"{np.percentile(error, 95):.3f}")
