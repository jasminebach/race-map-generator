import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# ======================================================
# CLI arguments
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--batch-dir", required=True, help="Path to batch folder")
args = parser.parse_args()

BATCH_DIR = os.path.abspath(args.batch_dir)

CENTERLINE_CSV = os.path.join(
    BATCH_DIR, "Simulation_Data_Export - Centerline.csv"
)
SIMULATION_CSV = os.path.join(
    BATCH_DIR, "Simulation_Data_Export - Simulation.csv"
)
VIDEO_PATH = os.path.join(BATCH_DIR, "driver_simulation.mp4")

# ======================================================
# Load data
# ======================================================
ref = pd.read_csv(CENTERLINE_CSV)
drv = pd.read_csv(SIMULATION_CSV)

# MATLAB-consistent transform
ref_x = -ref["yRef"].to_numpy()
ref_y =  ref["xRef"].to_numpy()
drv_x = -drv["<Y>"].to_numpy()
drv_y =  drv["<X>"].to_numpy()

# ======================================================
# Road geometry
# ======================================================
dx = np.gradient(ref_x)
dy = np.gradient(ref_y)
norm = np.sqrt(dx**2 + dy**2)

nx = -dy / norm
ny =  dx / norm

road_width = 6.0
left_x  = ref_x + road_width / 2 * nx
left_y  = ref_y + road_width / 2 * ny
right_x = ref_x - road_width / 2 * nx
right_y = ref_y - road_width / 2 * ny

# ======================================================
# Figure
# ======================================================
fig, ax = plt.subplots(figsize=(8, 8))

ax.fill(
    np.concatenate([left_x, right_x[::-1]]),
    np.concatenate([left_y, right_y[::-1]]),
    color="gray",
    alpha=0.4
)
ax.plot(ref_x, ref_y, "--", color="black")
ax.set_aspect("equal")
ax.set_xlim(ref_x.min() - 10, ref_x.max() + 10)
ax.set_ylim(ref_y.min() - 10, ref_y.max() + 10)

trail, = ax.plot([], [], color="orange", lw=2)
car = Rectangle((0, 0), 4.5, 2.0, color="blue")
ax.add_patch(car)

# ======================================================
# Animation
# ======================================================
def update(i):
    if i < 2:
        return trail, car

    trail.set_data(drv_x[:i], drv_y[:i])

    heading = np.arctan2(
        drv_y[i] - drv_y[i - 1],
        drv_x[i] - drv_x[i - 1]
    )

    car.set_xy((drv_x[i] - 2.25, drv_y[i] - 1.0))
    car.angle = np.degrees(heading)

    return trail, car

ani = FuncAnimation(fig, update, frames=len(drv_x), interval=33)
ani.save(VIDEO_PATH, dpi=120)

plt.close(fig)
print(f"✅ Video saved to {VIDEO_PATH}")


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.animation import FuncAnimation

# # ==================================================
# # Project root (robust)
# # ==================================================
# ROOT = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR = os.path.join(ROOT, "SIMULINK-DATA")
# VIDEO_PATH = os.path.join(ROOT, "driver_simulation.mp4")

# CENTERLINE_CSV = os.path.join(DATA_DIR, "Simulation_Data_Export - Centerline.csv")
# SIMULATION_CSV = os.path.join(DATA_DIR, "Simulation_Data_Export - Simulation.csv")

# # ==================================================
# # Load data
# # ==================================================
# ref = pd.read_csv(CENTERLINE_CSV)
# drv = pd.read_csv(SIMULATION_CSV)

# # MATLAB-consistent transform
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()
# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # ==================================================
# # Road geometry
# # ==================================================
# dx = np.gradient(ref_x)
# dy = np.gradient(ref_y)
# norm = np.sqrt(dx**2 + dy**2)

# nx = -dy / norm
# ny =  dx / norm

# road_width = 6.0
# left_x = ref_x + road_width / 2 * nx
# left_y = ref_y + road_width / 2 * ny
# right_x = ref_x - road_width / 2 * nx
# right_y = ref_y - road_width / 2 * ny

# # ==================================================
# # Figure setup
# # ==================================================
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.fill(
#     np.concatenate([left_x, right_x[::-1]]),
#     np.concatenate([left_y, right_y[::-1]]),
#     color="gray",
#     alpha=0.4
# )

# ax.plot(ref_x, ref_y, "--", color="black")
# ax.set_aspect("equal")
# ax.set_xlim(ref_x.min() - 10, ref_x.max() + 10)
# ax.set_ylim(ref_y.min() - 10, ref_y.max() + 10)
# ax.set_xlabel("Y [m]")
# ax.set_ylabel("X [m]")
# ax.set_title("Driver Simulation")

# trail_line, = ax.plot([], [], color="orange", lw=2)
# car = Rectangle((0, 0), 4.5, 2.0, color="blue")
# ax.add_patch(car)

# # ==================================================
# # Animation update
# # ==================================================
# def update(i):
#     if i < 2:
#         return trail_line, car

#     trail_line.set_data(drv_x[:i], drv_y[:i])

#     heading = np.arctan2(drv_y[i] - drv_y[i-1], drv_x[i] - drv_x[i-1])
#     car.set_xy((drv_x[i] - 2.25, drv_y[i] - 1.0))
#     car.angle = np.degrees(heading)

#     return trail_line, car

# ani = FuncAnimation(fig, update, frames=len(drv_x), interval=33)

# ani.save(VIDEO_PATH, dpi=120)
# plt.close(fig)

# print(f"✅ Video generated: {VIDEO_PATH}")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.animation import FuncAnimation

# # -------------------------------------------------
# # Load data
# # -------------------------------------------------
# ref = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Centerline.csv")
# drv = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Simulation.csv")

# # MATLAB-consistent transform
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()
# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # -------------------------------------------------
# # Build road
# # -------------------------------------------------
# dx = np.gradient(ref_x)
# dy = np.gradient(ref_y)
# norm = np.sqrt(dx**2 + dy**2)
# nx = -dy / norm
# ny =  dx / norm

# road_width = 6.0
# left_x = ref_x + road_width/2 * nx
# left_y = ref_y + road_width/2 * ny
# right_x = ref_x - road_width/2 * nx
# right_y = ref_y - road_width/2 * ny

# # -------------------------------------------------
# # Figure setup
# # -------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.fill(
#     np.concatenate([left_x, right_x[::-1]]),
#     np.concatenate([left_y, right_y[::-1]]),
#     color="gray",
#     alpha=0.4
# )

# ax.plot(ref_x, ref_y, "--", color="black")

# ax.set_aspect("equal")
# ax.set_xlim(ref_x.min() - 10, ref_x.max() + 10)
# ax.set_ylim(ref_y.min() - 10, ref_y.max() + 10)
# ax.set_xlabel("Y [m]")
# ax.set_ylabel("X [m]")
# ax.set_title("Driver Simulation")

# trail_line, = ax.plot([], [], color="orange", lw=2)

# car = Rectangle((0, 0), 4.5, 2.0, color="blue")
# ax.add_patch(car)

# # -------------------------------------------------
# # Animation update
# # -------------------------------------------------
# def update(i):
#     if i < 2:
#         return trail_line, car

#     trail_line.set_data(drv_x[:i], drv_y[:i])

#     heading = np.arctan2(drv_y[i] - drv_y[i-1], drv_x[i] - drv_x[i-1])
#     car.set_xy((drv_x[i] - 2.25, drv_y[i] - 1.0))
#     car.angle = np.degrees(heading)

#     return trail_line, car

# # -------------------------------------------------
# # Create animation
# # -------------------------------------------------
# ani = FuncAnimation(
#     fig,
#     update,
#     frames=len(drv_x),
#     interval=33  # ~30 FPS
# )

# ani.save("driver_simulation.mp4", dpi=120)
# plt.close(fig)

# print("Video saved as driver_simulation.mp4")
