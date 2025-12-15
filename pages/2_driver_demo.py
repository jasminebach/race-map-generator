import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# =================================================
# Page setup
# =================================================
st.set_page_config(page_title="Driver Simulation (Plotly)", layout="wide")
st.title("🚗 Driving Simulation — Live Scene (Non-Blinking)")

# =================================================
# Load data
# =================================================
ref = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Centerline.csv")
drv = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Simulation.csv")

# =================================================
# MATLAB-consistent transform
# =================================================
ref_x = -ref["yRef"].to_numpy()
ref_y =  ref["xRef"].to_numpy()

drv_x = -drv["<Y>"].to_numpy()
drv_y =  drv["<X>"].to_numpy()

N = len(drv_x)

# =================================================
# Build road polygon (STATIC, ALWAYS PRESENT)
# =================================================
dx = np.gradient(ref_x)
dy = np.gradient(ref_y)
norm = np.sqrt(dx**2 + dy**2)

nx = -dy / norm
ny =  dx / norm

road_width = 6.0

left_x = ref_x + road_width / 2 * nx
left_y = ref_y + road_width / 2 * ny
right_x = ref_x - road_width / 2 * nx
right_y = ref_y - road_width / 2 * ny

road_x = np.concatenate([left_x, right_x[::-1]])
road_y = np.concatenate([left_y, right_y[::-1]])

# =================================================
# Session state
# =================================================
if "frame" not in st.session_state:
    st.session_state.frame = 1  # 🔴 NEVER start at 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# =================================================
# Controls
# =================================================
st.sidebar.header("🎬 Controls")

fps = st.sidebar.slider("FPS", 1, 60, 30)
trail_len = st.sidebar.slider("Trail Length", 20, 400, 120)
ego_view = st.sidebar.checkbox("Ego-centric Camera", True)

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Play"):
    st.session_state.playing = True
if c2.button("⏸ Pause"):
    st.session_state.playing = False
if c3.button("⏹ Reset"):
    st.session_state.playing = False
    st.session_state.frame = 1

st.session_state.frame = st.sidebar.slider(
    "Frame",
    1,
    N - 2,
    st.session_state.frame
)

# =================================================
# Frame data
# =================================================
i = st.session_state.frame
start = max(0, i - trail_len)

# Vehicle heading
heading = np.arctan2(
    drv_y[i] - drv_y[i - 1],
    drv_x[i] - drv_x[i - 1]
)

# Vehicle geometry
L, W = 4.5, 2.0
corners = np.array([
    [ L/2,  W/2],
    [ L/2, -W/2],
    [-L/2, -W/2],
    [-L/2,  W/2],
    [ L/2,  W/2]
])

R = np.array([
    [np.cos(heading), -np.sin(heading)],
    [np.sin(heading),  np.cos(heading)]
])

car_xy = corners @ R.T
car_x = car_xy[:, 0] + drv_x[i]
car_y = car_xy[:, 1] + drv_y[i]

# =================================================
# Plotly Scene (NO BLINKING)
# =================================================
fig = go.Figure()

# Road
fig.add_trace(go.Scatter(
    x=road_x,
    y=road_y,
    fill="toself",
    fillcolor="rgba(130,130,130,0.4)",
    line=dict(color="rgba(130,130,130,0.4)"),
    name="Road"
))

# Centerline
fig.add_trace(go.Scatter(
    x=ref_x,
    y=ref_y,
    mode="lines",
    line=dict(color="black", dash="dash"),
    name="Centerline"
))

# Trail
fig.add_trace(go.Scatter(
    x=drv_x[start:i],
    y=drv_y[start:i],
    mode="lines",
    line=dict(color="orange", width=3),
    name="Trajectory"
))

# Ego vehicle
fig.add_trace(go.Scatter(
    x=car_x,
    y=car_y,
    fill="toself",
    mode="lines",
    fillcolor="rgba(0,90,255,0.95)",
    line=dict(color="blue"),
    name="Ego Vehicle"
))

# =================================================
# Layout — FIXED WORLD SCALE
# =================================================
fig.update_layout(
    height=750,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(
        title="Y [m]",
        scaleanchor="y",
        range=[ref_x.min() - 10, ref_x.max() + 10]
    ),
    yaxis=dict(
        title="X [m]",
        range=[ref_y.min() - 10, ref_y.max() + 10]
    ),
    showlegend=False
)

# Ego-centric camera
if ego_view:
    fig.update_xaxes(range=[drv_x[i] - 20, drv_x[i] + 20])
    fig.update_yaxes(range=[drv_y[i] - 20, drv_y[i] + 20])

st.plotly_chart(fig, use_container_width=True)

# =================================================
# Playback advance
# =================================================
if st.session_state.playing:
    time.sleep(1 / fps)
    if st.session_state.frame < N - 2:
        st.session_state.frame += 1
        st.rerun()
    else:
        st.session_state.playing = False


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import time

# # =================================================
# # Page setup
# # =================================================
# st.set_page_config(page_title="Driver Demo Live", layout="wide")
# st.title("🎥 Driver Trajectory — Live Playback (No Blink)")

# # =================================================
# # Load data
# # =================================================
# ref = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Centerline.csv")
# drv = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Simulation.csv")

# # =================================================
# # Coordinate transform (MATLAB-consistent)
# # =================================================
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()
# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # =================================================
# # Road surface (static)
# # =================================================
# dx = np.gradient(ref_x)
# dy = np.gradient(ref_y)
# norm = np.sqrt(dx**2 + dy**2)
# nx = -dy / norm
# ny = dx / norm

# track_width = 6.0
# left_x = ref_x + track_width / 2 * nx
# left_y = ref_y + track_width / 2 * ny
# right_x = ref_x - track_width / 2 * nx
# right_y = ref_y - track_width / 2 * ny

# # =================================================
# # Session state
# # =================================================
# if "frame" not in st.session_state:
#     st.session_state.frame = 0
# if "playing" not in st.session_state:
#     st.session_state.playing = False

# # =================================================
# # Sidebar controls
# # =================================================
# st.sidebar.header("🎬 Playback")

# fps = st.sidebar.slider("FPS", 1, 60, 20)
# trail_len = st.sidebar.slider("Trail Length", 20, 500, 150)
# ego_view = st.sidebar.checkbox("Ego View", True)

# colA, colB, colC = st.sidebar.columns(3)

# if colA.button("▶ Play"):
#     st.session_state.playing = True
# if colB.button("⏸ Pause"):
#     st.session_state.playing = False
# if colC.button("⏹ Reset"):
#     st.session_state.playing = False
#     st.session_state.frame = 0

# st.session_state.frame = st.sidebar.slider(
#     "Frame",
#     0,
#     len(drv_x) - 2,
#     st.session_state.frame
# )

# # =================================================
# # Persistent figure (KEY FIX)
# # =================================================
# if "fig" not in st.session_state:
#     fig, ax = plt.subplots(figsize=(8, 8))
#     st.session_state.fig = fig
#     st.session_state.ax = ax

# ax = st.session_state.ax
# fig = st.session_state.fig

# # =================================================
# # Clear only plot contents, NOT figure
# # =================================================
# ax.clear()

# # Road
# ax.fill(
#     np.concatenate([left_x, right_x[::-1]]),
#     np.concatenate([left_y, right_y[::-1]]),
#     color="lightgray",
#     alpha=0.3
# )

# # Centerline
# ax.plot(ref_x, ref_y, "--", color="black", linewidth=1.5)

# # Driver trail
# i = st.session_state.frame
# start = max(0, i - trail_len)
# ax.plot(drv_x[start:i], drv_y[start:i], color="orange", linewidth=2)

# # Ego vehicle
# heading = np.arctan2(drv_y[i + 1] - drv_y[i], drv_x[i + 1] - drv_x[i])
# car = Rectangle(
#     (drv_x[i] - 2.25, drv_y[i] - 1.0),
#     4.5,
#     2.0,
#     angle=np.degrees(heading),
#     color="blue",
#     alpha=0.9
# )
# ax.add_patch(car)

# # View
# ax.set_aspect("equal")
# ax.grid(True)
# ax.set_xlabel("Y [m]")
# ax.set_ylabel("X [m]")
# ax.set_title("Live Driver Playback")

# if ego_view:
#     ax.set_xlim(drv_x[i] - 20, drv_x[i] + 20)
#     ax.set_ylim(drv_y[i] - 20, drv_y[i] + 20)

# # =================================================
# # Draw without blinking
# # =================================================
# plot_placeholder = st.empty()
# plot_placeholder.pyplot(fig, clear_figure=False, use_container_width=True)

# # =================================================
# # Advance frame (non-blocking)
# # =================================================
# if st.session_state.playing:
#     time.sleep(1 / fps)

#     if st.session_state.frame < len(drv_x) - 2:
#         st.session_state.frame += 1
#         st.experimental_rerun()
#     else:
#         st.session_state.playing = False


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import time

# # =================================================
# # Page setup
# # =================================================
# st.set_page_config(page_title="Driver Demo Live", layout="wide")
# st.title("🎥 Driver Trajectory — Live Playback")

# # =================================================
# # Load data
# # =================================================
# ref = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Centerline.csv")
# drv = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Simulation.csv")

# # =================================================
# # MATLAB-consistent coordinate transform
# # =================================================
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()

# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # =================================================
# # Road surface (static)
# # =================================================
# dx = np.gradient(ref_x)
# dy = np.gradient(ref_y)
# norm = np.sqrt(dx**2 + dy**2)
# nx = -dy / norm
# ny = dx / norm

# track_width = 6.0
# left_x = ref_x + track_width / 2 * nx
# left_y = ref_y + track_width / 2 * ny
# right_x = ref_x - track_width / 2 * nx
# right_y = ref_y - track_width / 2 * ny

# # =================================================
# # Session state initialization
# # =================================================
# if "playing" not in st.session_state:
#     st.session_state.playing = False

# if "frame" not in st.session_state:
#     st.session_state.frame = 0

# # =================================================
# # Sidebar controls
# # =================================================
# st.sidebar.header("🎬 Playback Controls")

# fps = st.sidebar.slider("Playback Speed (FPS)", 1, 60, 20)
# trail_len = st.sidebar.slider("Trail Length", 10, 500, 150)
# ego_view = st.sidebar.checkbox("Ego-Centric View", True)

# colA, colB, colC = st.sidebar.columns(3)

# if colA.button("▶ Play"):
#     st.session_state.playing = True

# if colB.button("⏸ Pause"):
#     st.session_state.playing = False

# if colC.button("⏹ Reset"):
#     st.session_state.playing = False
#     st.session_state.frame = 0

# # Manual scrub (always works)
# st.session_state.frame = st.sidebar.slider(
#     "Frame",
#     0,
#     len(drv_x) - 2,
#     st.session_state.frame
# )

# # =================================================
# # Plot placeholder
# # =================================================
# plot_placeholder = st.empty()

# # =================================================
# # Render one frame
# # =================================================
# def render_frame(i):
#     fig, ax = plt.subplots(figsize=(8, 8))

#     # Road
#     ax.fill(
#         np.concatenate([left_x, right_x[::-1]]),
#         np.concatenate([left_y, right_y[::-1]]),
#         color="lightgray",
#         alpha=0.3,
#         label="Road"
#     )

#     # Centerline
#     ax.plot(ref_x, ref_y, "--", color="black", linewidth=1.5, label="Centerline")

#     # Driver trail
#     start = max(0, i - trail_len)
#     ax.plot(
#         drv_x[start:i],
#         drv_y[start:i],
#         color="orange",
#         linewidth=2,
#         label="Driver Path"
#     )

#     # Ego vehicle
#     heading = np.arctan2(
#         drv_y[i + 1] - drv_y[i],
#         drv_x[i + 1] - drv_x[i]
#     )

#     car = Rectangle(
#         (drv_x[i] - 2.25, drv_y[i] - 1.0),
#         4.5,
#         2.0,
#         angle=np.degrees(heading),
#         color="blue",
#         alpha=0.9
#     )
#     ax.add_patch(car)

#     # View control
#     ax.set_aspect("equal")
#     ax.grid(True)
#     ax.set_xlabel("Y [m]")
#     ax.set_ylabel("X [m]")
#     ax.set_title("Live Driver Playback")

#     if ego_view:
#         ax.set_xlim(drv_x[i] - 20, drv_x[i] + 20)
#         ax.set_ylim(drv_y[i] - 20, drv_y[i] + 20)

#     ax.legend(loc="upper right")

#     plot_placeholder.pyplot(fig)
#     plt.close(fig)

# # =================================================
# # Draw current frame
# # =================================================
# render_frame(st.session_state.frame)

# # =================================================
# # Advance frame if playing (NON-BLOCKING)
# # =================================================
# if st.session_state.playing:
#     time.sleep(1 / fps)

#     if st.session_state.frame < len(drv_x) - 2:
#         st.session_state.frame += 1
#         st.experimental_rerun()
#     else:
#         st.session_state.playing = False


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import time

# # =================================================
# # Page setup
# # =================================================
# st.set_page_config(page_title="Driver Demo Live", layout="wide")
# st.title("🎥 Driver Trajectory — Live Playback")

# # =================================================
# # Load data
# # =================================================
# ref = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Centerline.csv")
# drv = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Simulation.csv")

# # =================================================
# # MATLAB-consistent coordinate transform
# # =================================================
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()

# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # =================================================
# # Road surface (static)
# # =================================================
# dx = np.gradient(ref_x)
# dy = np.gradient(ref_y)
# norm = np.sqrt(dx**2 + dy**2)
# nx = -dy / norm
# ny = dx / norm

# track_width = 6.0
# left_x = ref_x + track_width / 2 * nx
# left_y = ref_y + track_width / 2 * ny
# right_x = ref_x - track_width / 2 * nx
# right_y = ref_y - track_width / 2 * ny

# # =================================================
# # Playback controls
# # =================================================
# st.sidebar.header("🎬 Playback Controls")

# fps = st.sidebar.slider("Playback Speed (FPS)", 1, 60, 20)
# trail_len = st.sidebar.slider("Trail Length", 10, 500, 150)
# ego_view = st.sidebar.checkbox("Ego-Centric View", True)

# colA, colB = st.sidebar.columns(2)

# if "playing" not in st.session_state:
#     st.session_state.playing = False
# if "frame" not in st.session_state:
#     st.session_state.frame = 0

# if colA.button("▶ Play"):
#     st.session_state.playing = True

# if colB.button("⏸ Pause"):
#     st.session_state.playing = False

# # Manual scrub
# st.session_state.frame = st.sidebar.slider(
#     "Frame",
#     0,
#     len(drv_x) - 2,
#     st.session_state.frame
# )

# # =================================================
# # Plot placeholder
# # =================================================
# plot_placeholder = st.empty()

# # =================================================
# # Frame rendering function
# # =================================================
# def render_frame(i):
#     fig, ax = plt.subplots(figsize=(8, 8))

#     # Road
#     ax.fill(
#         np.concatenate([left_x, right_x[::-1]]),
#         np.concatenate([left_y, right_y[::-1]]),
#         color="lightgray",
#         alpha=0.3,
#         label="Road"
#     )

#     # Centerline
#     ax.plot(ref_x, ref_y, "--", color="black", linewidth=1.5, label="Centerline")

#     # Driver trail
#     start = max(0, i - trail_len)
#     ax.plot(
#         drv_x[start:i],
#         drv_y[start:i],
#         color="orange",
#         linewidth=2,
#         label="Driver Path"
#     )

#     # Ego vehicle
#     heading = np.arctan2(
#         drv_y[i + 1] - drv_y[i],
#         drv_x[i + 1] - drv_x[i]
#     )

#     car = Rectangle(
#         (drv_x[i] - 2.25, drv_y[i] - 1.0),
#         4.5,
#         2.0,
#         angle=np.degrees(heading),
#         color="blue",
#         alpha=0.9
#     )
#     ax.add_patch(car)

#     # View control
#     ax.set_aspect("equal")
#     ax.grid(True)
#     ax.set_xlabel("Y [m]")
#     ax.set_ylabel("X [m]")
#     ax.set_title("Live Driver Playback")

#     if ego_view:
#         ax.set_xlim(drv_x[i] - 20, drv_x[i] + 20)
#         ax.set_ylim(drv_y[i] - 20, drv_y[i] + 20)

#     ax.legend(loc="upper right")

#     plot_placeholder.pyplot(fig)
#     plt.close(fig)

# # =================================================
# # Live playback loop
# # =================================================
# if st.session_state.playing:
#     render_frame(st.session_state.frame)

#     time.sleep(1 / fps)

#     if st.session_state.frame < len(drv_x) - 2:
#         st.session_state.frame += 1
#     else:
#         st.session_state.playing = False

# else:
#     render_frame(st.session_state.frame)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# # =================================================
# # Page setup
# # =================================================
# st.set_page_config(page_title="Driver Demo", layout="wide")
# st.title("🚗 Driver Trajectory — Demo View")

# # =================================================
# # Load data
# # =================================================
# ref = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Centerline.csv")
# drv = pd.read_csv("SIMULINK-DATA/Simulation_Data_Export - Simulation.csv")

# # =================================================
# # Coordinate transform
# # =================================================
# ref_x = -ref["yRef"].to_numpy()
# ref_y =  ref["xRef"].to_numpy()
# drv_x = -drv["<Y>"].to_numpy()
# drv_y =  drv["<X>"].to_numpy()

# # =================================================
# # Slider
# # =================================================
# idx = st.slider("Simulation Step", 0, len(drv_x) - 2, 0)

# # =================================================
# # Road surface
# # =================================================
# dx = np.gradient(ref_x)
# dy = np.gradient(ref_y)
# norm = np.sqrt(dx**2 + dy**2)
# nx = -dy / norm
# ny = dx / norm

# width = 6.0
# left_x = ref_x + width / 2 * nx
# left_y = ref_y + width / 2 * ny
# right_x = ref_x - width / 2 * nx
# right_y = ref_y - width / 2 * ny

# # =================================================
# # Plot
# # =================================================
# fig, ax = plt.subplots(figsize=(8, 8))

# ax.fill(
#     np.concatenate([left_x, right_x[::-1]]),
#     np.concatenate([left_y, right_y[::-1]]),
#     color="gray",
#     alpha=0.3,
#     label="Road"
# )

# ax.plot(ref_x, ref_y, "--", label="Centerline")
# ax.plot(drv_x[:idx], drv_y[:idx], ".", color="orange", label="Driver Path")

# # Ego vehicle
# heading = np.arctan2(drv_y[idx+1] - drv_y[idx], drv_x[idx+1] - drv_x[idx])
# rect = Rectangle(
#     (drv_x[idx] - 2.25, drv_y[idx] - 1.0),
#     4.5, 2.0,
#     angle=np.degrees(heading),
#     color="blue",
#     alpha=0.9
# )
# ax.add_patch(rect)

# ax.set_aspect("equal")
# ax.set_xlabel("Y [m]")
# ax.set_ylabel("X [m]")
# ax.set_title("Driving Scenario (Ego View)")
# ax.grid(True)
# ax.legend()

# # Ego-centric zoom
# zoom = st.checkbox("Ego-centric View", value=True)
# if zoom:
#     ax.set_xlim(drv_x[idx] - 20, drv_x[idx] + 20)
#     ax.set_ylim(drv_y[idx] - 20, drv_y[idx] + 20)

# st.pyplot(fig)
