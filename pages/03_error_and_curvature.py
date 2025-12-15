import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Error & Curvature", layout="wide")
st.title("📊 Error & Curvature Diagnostics")

CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

ref = pd.read_csv(CL_PATH)
drv = pd.read_csv(DRV_PATH)

ref_x, ref_y = ref["xRef"].to_numpy(), ref["yRef"].to_numpy()
drv_x, drv_y = drv["<X>"].to_numpy(), drv["<Y>"].to_numpy()

# --- curvature ---
dx = np.gradient(ref_x)
dy = np.gradient(ref_y)
ddx = np.gradient(dx)
ddy = np.gradient(dy)
curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-9)

# --- distance ---
dist = np.min(
    np.hypot(
        drv_x[:, None] - ref_x[None, :],
        drv_y[:, None] - ref_y[None, :]
    ),
    axis=1
)
dist_smooth = gaussian_filter1d(dist, sigma=3)

# --- plots ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

axs[0].plot(curvature)
axs[0].set_title("Centerline Curvature")
axs[0].set_ylabel("κ [1/m]")
axs[0].grid(True)

axs[1].plot(dist_smooth, label="Tracking Error")
axs[1].set_title("Driver Tracking Error (Smoothed)")
axs[1].set_ylabel("Error [m]")
axs[1].set_xlabel("Sample Index")
axs[1].grid(True)

st.pyplot(fig)

st.subheader("Statistics")
st.json({
    "mean_error": float(dist.mean()),
    "max_error": float(dist.max()),
    "curvature_95pct": float(np.percentile(np.abs(curvature), 95))
})
