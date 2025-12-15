import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =================================================
# Page config
# =================================================
st.set_page_config(page_title="Driver vs Centerline", layout="wide")
st.title("🚗 Driver Trajectory vs Centerline (True Orthogonal Projection)")

# =================================================
# Paths
# =================================================
CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# =================================================
# Load data
# =================================================
ref = pd.read_csv(CL_PATH)
drv = pd.read_csv(DRV_PATH)

# =================================================
# Coordinate transformation (MATLAB-consistent)
#
# Simulink vehicle frame:
#   X → forward
#   Y → left
#
# MATLAB visualization:
#   plot(Y, X)
#
# Python needs:
#   - axis swap
#   - Y sign flip
# =================================================
ref_x = -ref["yRef"].to_numpy()
ref_y =  ref["xRef"].to_numpy()

drv_x = -drv["<Y>"].to_numpy()
drv_y =  drv["<X>"].to_numpy()

centerline = np.column_stack([ref_x, ref_y])
driver_pts = np.column_stack([drv_x, drv_y])

# =================================================
# Geometry helpers
# =================================================
def project_point_to_segment(p, a, b):
    """
    Orthogonal projection of point p onto segment ab.
    Returns projected point and distance.
    """
    ab = b - a
    ap = p - a
    denom = np.dot(ab, ab)

    # Degenerate segment safeguard
    if denom == 0:
        return a, np.linalg.norm(p - a)

    t = np.dot(ap, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return proj, dist

# =================================================
# True orthogonal projection onto polyline
# =================================================
proj_x = np.zeros(len(driver_pts))
proj_y = np.zeros(len(driver_pts))
lateral_error = np.zeros(len(driver_pts))

for i, p in enumerate(driver_pts):
    min_dist = np.inf
    best_proj = None

    for j in range(len(centerline) - 1):
        a = centerline[j]
        b = centerline[j + 1]

        proj, d = project_point_to_segment(p, a, b)
        if d < min_dist:
            min_dist = d
            best_proj = proj

    proj_x[i], proj_y[i] = best_proj
    lateral_error[i] = min_dist

# =================================================
# Plot
# =================================================
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(ref_x, ref_y, "--", linewidth=2, label="Centerline")
ax.plot(drv_x, drv_y, ".", alpha=0.6, label="Driver Trajectory")

# Error vectors (subsampled for clarity)
step = max(len(drv_x) // 150, 1)
for i in range(0, len(drv_x), step):
    ax.plot(
        [drv_x[i], proj_x[i]],
        [drv_y[i], proj_y[i]],
        color="red",
        alpha=0.15,
        linewidth=0.8
    )

ax.set_xlabel("Y Distance [m]")
ax.set_ylabel("X Distance [m]")
ax.set_title("Driver vs Reference Track (Orthogonal Projection)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# =================================================
# Metrics
# =================================================
c1, c2 = st.columns(2)
c1.metric("Mean Lateral Error [m]", f"{lateral_error.mean():.3f}")
c2.metric("Max Lateral Error [m]", f"{lateral_error.max():.3f}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.spatial import cKDTree

# # -------------------------------------------------
# # Page config
# # -------------------------------------------------
# st.set_page_config(page_title="Driver vs Centerline", layout="wide")
# st.title("🚗 Driver Trajectory vs Centerline (Frenet-based, Correct)")

# # -------------------------------------------------
# # Paths
# # -------------------------------------------------
# CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
# DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# # -------------------------------------------------
# # Load data
# # -------------------------------------------------
# ref = pd.read_csv(CL_PATH)
# drv = pd.read_csv(DRV_PATH)

# # -------------------------------------------------
# # Coordinate transform (MATLAB-consistent)
# # Simulink:
# #   X forward, Y left
# # MATLAB plot:
# #   plot(Y, X)
# # -------------------------------------------------
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()

# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # -------------------------------------------------
# # Build KD-tree for centerline projection
# # -------------------------------------------------
# centerline_pts = np.column_stack([ref_x, ref_y])
# driver_pts = np.column_stack([drv_x, drv_y])

# tree = cKDTree(centerline_pts)
# dist, idx = tree.query(driver_pts)

# proj_x = ref_x[idx]
# proj_y = ref_y[idx]

# # -------------------------------------------------
# # Plot
# # -------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(ref_x, ref_y, "--", linewidth=2, label="Centerline")
# ax.plot(drv_x, drv_y, ".", alpha=0.6, label="Driver Trajectory")

# # Optional: error vectors (thin)
# for i in range(0, len(drv_x), max(len(drv_x)//150, 1)):
#     ax.plot(
#         [drv_x[i], proj_x[i]],
#         [drv_y[i], proj_y[i]],
#         "r-",
#         alpha=0.15,
#         linewidth=0.8
#     )

# ax.set_xlabel("Y Distance [m]")
# ax.set_ylabel("X Distance [m]")
# ax.set_title("Driver vs Reference Track (Orthogonal Projection)")
# ax.set_aspect("equal", adjustable="box")
# ax.grid(True)
# ax.legend()

# st.pyplot(fig)

# # -------------------------------------------------
# # Metrics (physically meaningful)
# # -------------------------------------------------
# c1, c2 = st.columns(2)
# c1.metric("Mean Lateral Error [m]", f"{dist.mean():.3f}")
# c2.metric("Max Lateral Error [m]", f"{dist.max():.3f}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # -------------------------------------------------
# # Page config
# # -------------------------------------------------
# st.set_page_config(page_title="Driver vs Centerline", layout="wide")
# st.title("🚗 Driver Trajectory vs Centerline (Physically Correct)")

# # -------------------------------------------------
# # Paths
# # -------------------------------------------------
# CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
# DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# # -------------------------------------------------
# # Load data
# # -------------------------------------------------
# ref = pd.read_csv(CL_PATH)
# drv = pd.read_csv(DRV_PATH)

# # -------------------------------------------------
# # Coordinate transformation
# #
# # Simulink vehicle frame:
# #   X = forward
# #   Y = left
# #
# # MATLAB visualization:
# #   plot(Y, X)
# #
# # Python requires:
# #   - axis swap
# #   - Y sign flip
# # -------------------------------------------------

# # Centerline
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()

# # Driver trajectory
# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # -------------------------------------------------
# # Origin alignment (visual consistency with MATLAB)
# # -------------------------------------------------
# dx0 = ref_x[0] - drv_x[0]
# dy0 = ref_y[0] - drv_y[0]

# drv_x += dx0
# drv_y += dy0

# # -------------------------------------------------
# # Plot
# # -------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(ref_x, ref_y, "--", linewidth=2, label="Centerline")
# ax.plot(drv_x, drv_y, ".", alpha=0.6, label="Driver Trajectory")

# ax.set_xlabel("Y Distance [m]")
# ax.set_ylabel("X Distance [m]")
# ax.set_title("Driver vs Reference Track")
# ax.set_aspect("equal", adjustable="box")
# ax.grid(True)
# ax.legend()

# st.pyplot(fig)

# # -------------------------------------------------
# # Tracking error (lateral distance to centerline)
# # -------------------------------------------------
# dx = drv_x[:, None] - ref_x[None, :]
# dy = drv_y[:, None] - ref_y[None, :]
# dist = np.sqrt(dx**2 + dy**2).min(axis=1)

# # -------------------------------------------------
# # Metrics
# # -------------------------------------------------
# c1, c2 = st.columns(2)
# c1.metric("Mean Tracking Error [m]", f"{dist.mean():.3f}")
# c2.metric("Max Tracking Error [m]", f"{dist.max():.3f}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# st.set_page_config(page_title="Driver vs Centerline", layout="wide")
# st.title("🚗 Driver Trajectory vs Centerline (MATLAB-consistent)")

# CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
# DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# ref = pd.read_csv(CL_PATH)
# drv = pd.read_csv(DRV_PATH)

# # --- MATLAB plotting convention ---
# ref_plot_x = -ref["yRef"].to_numpy()
# ref_plot_y =  ref["xRef"].to_numpy()

# drv_plot_x = -drv["<Y>"].to_numpy()
# drv_plot_y =  drv["<X>"].to_numpy()

# # --- ALIGN ORIGINS (critical missing step) ---
# dx0 = ref_plot_x[0] - drv_plot_x[0]
# dy0 = ref_plot_y[0] - drv_plot_y[0]

# drv_plot_x += dx0
# drv_plot_y += dy0

# # --- Plot ---
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(ref_plot_x, ref_plot_y, "--", linewidth=2, label="Centerline")
# ax.plot(drv_plot_x, drv_plot_y, ".", alpha=0.6, label="Driver")

# ax.set_xlabel("Y Distance [m]")
# ax.set_ylabel("X Distance [m]")
# ax.set_title("Driver vs Reference Track")
# ax.set_aspect("equal", adjustable="box")
# ax.grid(True)
# ax.legend()

# st.pyplot(fig)

# # --- Tracking error ---
# dx = drv_plot_x[:, None] - ref_plot_x[None, :]
# dy = drv_plot_y[:, None] - ref_plot_y[None, :]
# dist = np.sqrt(dx**2 + dy**2).min(axis=1)

# col1, col2 = st.columns(2)
# col1.metric("Mean Tracking Error [m]", f"{dist.mean():.3f}")
# col2.metric("Max Tracking Error [m]", f"{dist.max():.3f}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # -------------------------------------------------
# # Page config
# # -------------------------------------------------
# st.set_page_config(page_title="Driver vs Centerline", layout="wide")
# st.title("🚗 Driver Trajectory vs Centerline (MATLAB-consistent)")

# # -------------------------------------------------
# # Paths
# # -------------------------------------------------
# CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
# DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# # -------------------------------------------------
# # Load data
# # -------------------------------------------------
# ref = pd.read_csv(CL_PATH)
# drv = pd.read_csv(DRV_PATH)

# # -------------------------------------------------
# # MATLAB plotting convention + vehicle frame fix
# #
# # Simulink vehicle frame:
# #   X = forward
# #   Y = left
# #
# # MATLAB visualization:
# #   plot(Y, X)
# #   (implicitly flips Y for screen coordinates)
# #
# # Python needs this explicitly:
# #   - swap axes
# #   - flip Y sign
# # -------------------------------------------------

# # Centerline (CSV columns: yRef, xRef)
# ref_plot_x = -ref["yRef"].to_numpy()   # 🔥 sign flip
# ref_plot_y =  ref["xRef"].to_numpy()

# # Driver trajectory (CSV columns: <X>, <Y>)
# drv_plot_x = -drv["<Y>"].to_numpy()    # 🔥 sign flip
# drv_plot_y =  drv["<X>"].to_numpy()

# # -------------------------------------------------
# # Plot
# # -------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(
#     ref_plot_x,
#     ref_plot_y,
#     "--",
#     linewidth=2,
#     label="Centerline"
# )

# ax.plot(
#     drv_plot_x,
#     drv_plot_y,
#     ".",
#     alpha=0.6,
#     label="Driver"
# )

# ax.set_xlabel("Y Distance [m]")
# ax.set_ylabel("X Distance [m]")
# ax.set_title("Driver vs Reference Track")
# ax.set_aspect("equal", adjustable="box")
# ax.grid(True)
# ax.legend()

# st.pyplot(fig)

# # -------------------------------------------------
# # Tracking error (nearest distance to centerline)
# # IMPORTANT: must be computed in SAME frame as plot
# # -------------------------------------------------
# dx = drv_plot_x[:, None] - ref_plot_x[None, :]
# dy = drv_plot_y[:, None] - ref_plot_y[None, :]
# dist = np.sqrt(dx**2 + dy**2).min(axis=1)

# # -------------------------------------------------
# # Metrics
# # -------------------------------------------------
# col1, col2 = st.columns(2)
# col1.metric("Mean Tracking Error [m]", f"{dist.mean():.3f}")
# col2.metric("Max Tracking Error [m]", f"{dist.max():.3f}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # -------------------------------------------------
# # Page config
# # -------------------------------------------------
# st.set_page_config(page_title="Driver vs Centerline", layout="wide")
# st.title("🚗 Driver Trajectory vs Centerline (MATLAB-consistent)")

# # -------------------------------------------------
# # Paths
# # -------------------------------------------------
# CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
# DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# # -------------------------------------------------
# # Load data
# # -------------------------------------------------
# ref = pd.read_csv(CL_PATH)
# drv = pd.read_csv(DRV_PATH)

# # -------------------------------------------------
# # MATLAB plotting convention + vehicle frame fix
# #
# # Simulink vehicle frame:
# #   X = forward
# #   Y = left
# #
# # MATLAB visualization:
# #   plot(Y, X)
# #   (implicitly flips Y for screen coordinates)
# #
# # Python needs this explicitly:
# #   - swap axes
# #   - flip Y sign
# # -------------------------------------------------

# # Centerline (CSV columns: yRef, xRef)
# ref_plot_x = -ref["yRef"].to_numpy()   # 🔥 sign flip
# ref_plot_y =  ref["xRef"].to_numpy()

# # Driver trajectory (CSV columns: <X>, <Y>)
# drv_plot_x = -drv["<Y>"].to_numpy()    # 🔥 sign flip
# drv_plot_y =  drv["<X>"].to_numpy()

# # -------------------------------------------------
# # Plot
# # -------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(
#     ref_plot_x,
#     ref_plot_y,
#     "--",
#     linewidth=2,
#     label="Centerline"
# )

# ax.plot(
#     drv_plot_x,
#     drv_plot_y,
#     ".",
#     alpha=0.6,
#     label="Driver"
# )

# ax.set_xlabel("Y Distance [m]")
# ax.set_ylabel("X Distance [m]")
# ax.set_title("Driver vs Reference Track")
# ax.set_aspect("equal", adjustable="box")
# ax.grid(True)
# ax.legend()

# st.pyplot(fig)

# # -------------------------------------------------
# # Tracking error (nearest distance to centerline)
# # IMPORTANT: must be computed in SAME frame as plot
# # -------------------------------------------------
# dx = drv_plot_x[:, None] - ref_plot_x[None, :]
# dy = drv_plot_y[:, None] - ref_plot_y[None, :]
# dist = np.sqrt(dx**2 + dy**2).min(axis=1)

# # -------------------------------------------------
# # Metrics
# # -------------------------------------------------
# col1, col2 = st.columns(2)
# col1.metric("Mean Tracking Error [m]", f"{dist.mean():.3f}")
# col2.metric("Max Tracking Error [m]", f"{dist.max():.3f}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # -------------------------------------------------
# # Page config
# # -------------------------------------------------
# st.set_page_config(page_title="Driver vs Centerline", layout="wide")
# st.title("🚗 Driver Trajectory vs Centerline (MATLAB-consistent)")

# # -------------------------------------------------
# # Paths
# # -------------------------------------------------
# CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
# DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# # -------------------------------------------------
# # Load data
# # -------------------------------------------------
# ref = pd.read_csv(CL_PATH)
# drv = pd.read_csv(DRV_PATH)

# # -------------------------------------------------
# # IMPORTANT: MATLAB PLOTTING CONVENTION
# # MATLAB uses: plot(Y, X)
# # -------------------------------------------------

# # Centerline (CSV: yRef, xRef)
# ref_plot_x = ref["yRef"].to_numpy()   # horizontal axis
# ref_plot_y = ref["xRef"].to_numpy()   # vertical axis

# # Driver trajectory (CSV: <X>, <Y>)
# drv_plot_x = drv["<Y>"].to_numpy()    # horizontal axis
# drv_plot_y = drv["<X>"].to_numpy()    # vertical axis

# # -------------------------------------------------
# # Plot
# # -------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(
#     ref_plot_x,
#     ref_plot_y,
#     "--",
#     linewidth=2,
#     label="Centerline"
# )

# ax.plot(
#     drv_plot_x,
#     drv_plot_y,
#     ".",
#     alpha=0.6,
#     label="Driver"
# )

# ax.set_xlabel("Y Distance [m]")
# ax.set_ylabel("X Distance [m]")
# ax.set_title("Driver vs Reference Track")
# ax.set_aspect("equal", adjustable="box")
# ax.grid(True)
# ax.legend()

# st.pyplot(fig)

# # -------------------------------------------------
# # Tracking error (distance to nearest centerline point)
# # -------------------------------------------------
# # NOTE: error must be computed in SAME coordinate frame
# dx = drv_plot_x[:, None] - ref_plot_x[None, :]
# dy = drv_plot_y[:, None] - ref_plot_y[None, :]
# dist = np.sqrt(dx**2 + dy**2).min(axis=1)

# # -------------------------------------------------
# # Metrics
# # -------------------------------------------------
# col1, col2 = st.columns(2)
# col1.metric("Mean Tracking Error [m]", f"{dist.mean():.3f}")
# col2.metric("Max Tracking Error [m]", f"{dist.max():.3f}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# st.set_page_config(page_title="Driver vs Centerline", layout="wide")
# st.title("🚗 Driver Trajectory vs Centerline")

# CL_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"
# DRV_PATH = "SIMULINK-DATA/Simulation_Data_Export - Simulation.csv"

# ref = pd.read_csv(CL_PATH)
# drv = pd.read_csv(DRV_PATH)

# ref_x, ref_y = ref["xRef"].to_numpy(), ref["yRef"].to_numpy()
# drv_x, drv_y = drv["<X>"].to_numpy(), drv["<Y>"].to_numpy()

# fig, ax = plt.subplots(figsize=(8, 8))
# ax.plot(ref_x, ref_y, "--", linewidth=2, label="Centerline")
# ax.plot(drv_x, drv_y, ".", alpha=0.6, label="Driver")
# ax.set_aspect("equal")
# ax.set_title("Driver vs Reference Track")
# ax.legend()
# ax.grid(True)

# st.pyplot(fig)

# # Quick metrics
# dist = np.min(
#     np.hypot(
#         drv_x[:, None] - ref_x[None, :],
#         drv_y[:, None] - ref_y[None, :]
#     ),
#     axis=1
# )

# st.metric("Mean Tracking Error [m]", f"{dist.mean():.3f}")
# st.metric("Max Tracking Error [m]", f"{dist.max():.3f}")
