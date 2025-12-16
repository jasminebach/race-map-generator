import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Streamlit setup
# =====================================================

st.set_page_config(page_title="Generative Practice Track", layout="wide")
st.title("🏎️ Generative Practice Track (Weakness-Aware, Time-Series Driven)")

# =====================================================
# Geometry utilities
# =====================================================

def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-9
    return (dx * ddy - dy * ddx) / denom


def arc_length(x, y):
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.insert(np.cumsum(ds), 0, 0.0)


def compute_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    mag = np.hypot(dx, dy) + 1e-9
    return -dy / mag, dx / mag


def compute_distance(ref_x, ref_y, drv_x, drv_y):
    d = np.hypot(
        drv_x[:, None] - ref_x[None, :],
        drv_y[:, None] - ref_y[None, :]
    )
    idx = np.argmin(d, axis=1)
    return np.min(d, axis=1), idx


# =====================================================
# Weakness analysis (vector form)
# =====================================================

def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
    kappa = np.abs(curvature(ref_x, ref_y))

    apex_mask = kappa > np.percentile(kappa, 75)

    w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
    w_slalom = np.std(np.diff(dist))
    w_corner = np.mean(dist)

    w = np.array([w_apex, w_slalom, w_corner])
    return w / (np.linalg.norm(w) + 1e-9)


# =====================================================
# Time-series decision logic (SLIDE-LEVEL)
# =====================================================

def decision_from_timeseries(kappa_ts, error_ts):
    high_k = np.percentile(kappa_ts, 75)
    low_k = np.percentile(kappa_ts, 30)

    apex_error = np.mean(error_ts[kappa_ts > high_k])
    straight_error = np.mean(error_ts[kappa_ts < low_k])
    oscillation = np.std(np.diff(error_ts))
    mean_error = np.mean(error_ts)

    if apex_error > oscillation and apex_error > straight_error:
        decision = "APEX"
    elif oscillation > apex_error:
        decision = "SLALOM"
    else:
        decision = "CORNERING"

    metrics = {
        "apex_error": apex_error,
        "slalom_instability": oscillation,
        "mean_error": mean_error
    }

    return decision, metrics

def detect_problem_segments(s, kappa_ts, error_ts):
    threshold = np.percentile(error_ts, 80)
    bad_idx = error_ts > threshold

    return {
        "segments": list(zip(s[bad_idx][::10], error_ts[bad_idx][::10])),
        "coverage": np.mean(bad_idx)
    }


# =====================================================
# Track deformation
# =====================================================

def deform_reference_track(ref_x, ref_y, weakness_vec, max_offset=3.0):
    s = arc_length(ref_x, ref_y)
    s_norm = s / (s[-1] + 1e-9)

    kappa = np.abs(curvature(ref_x, ref_y))
    nx, ny = compute_normals(ref_x, ref_y)

    labels = ["apex", "slalom", "cornering"]
    focus = labels[np.argmax(weakness_vec)]

    delta = np.zeros_like(s)

    if focus == "apex":
        mask = kappa > np.percentile(kappa, 75)
        strength = weakness_vec[0]
    elif focus == "slalom":
        mask = kappa < np.percentile(kappa, 30)
        strength = weakness_vec[1]
    else:
        mask = np.ones_like(kappa, dtype=bool)
        strength = weakness_vec[2]

    delta[mask] = max_offset * strength * np.sin(2 * np.pi * 4 * s_norm[mask])
    delta = np.convolve(delta, np.ones(15) / 15, mode="same")

    new_x = ref_x + delta * nx
    new_y = ref_y + delta * ny

    return new_x, new_y, delta, focus


# =====================================================
# Visualization helpers
# =====================================================

def plot_curvature(x, y):
    s = arc_length(x, y)
    k = curvature(x, y)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(s, k)
    ax.set_title("Curvature κ(s)")
    ax.set_xlabel("Arc length (m)")
    ax.set_ylabel("κ")
    st.pyplot(fig)


def plot_time_series(s, kappa, dist, delta):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(s, kappa)
    ax[0].set_ylabel("κ")
    ax[0].set_title("Curvature")

    ax[1].plot(s, dist)
    ax[1].set_ylabel("Error (m)")
    ax[1].set_title("Lateral Tracking Error")

    ax[2].plot(s, delta)
    ax[2].set_ylabel("Δ (m)")
    ax[2].set_title("Track Deformation")
    ax[2].set_xlabel("Arc Length (m)")

    st.pyplot(fig)


def plot_weakness_vector(w):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Apex", "Slalom", "Cornering"], w,
           color=["red", "blue", "green"])
    ax.set_ylim(0, 1)
    ax.set_title("Normalized Weakness Vector")
    st.pyplot(fig)


def build_time_series(ref_x, ref_y, drv_x, drv_y, sample_rate=10.0):
    """
    Builds true time-series signals from driver data
    """
    # time axis
    t = np.arange(len(drv_x)) / sample_rate

    # spatial error
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)

    # curvature experienced at driver locations
    kappa_ref = np.abs(curvature(ref_x, ref_y))
    kappa_t = kappa_ref[idx]

    return t, kappa_t, dist

def plot_true_time_series(t, kappa_t, error_t):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(t, kappa_t)
    ax[0].set_title("Curvature vs Time")
    ax[0].set_ylabel("|κ|")

    ax[1].plot(t, error_t)
    ax[1].set_title("Lateral Error vs Time")
    ax[1].set_ylabel("Error (m)")
    ax[1].set_xlabel("Time (s)")

    st.pyplot(fig)


def plot_error_frequency(error_t, sample_rate=10.0):
    fft = np.fft.rfft(error_t - np.mean(error_t))
    freq = np.fft.rfftfreq(len(error_t), d=1/sample_rate)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freq, np.abs(fft))
    ax.set_title("Lateral Error Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, 2.0)

    st.pyplot(fig)


def plot_arc_diagnostics(s, kappa, dist, delta):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(s, kappa)
    ax[0].set_ylabel("κ(s)")
    ax[0].set_title("Arc-Length Diagnostics")

    ax[1].plot(s, dist)
    ax[1].set_ylabel("Error (m)")

    ax[2].plot(s, delta)
    ax[2].set_ylabel("Δ(s)")
    ax[2].set_xlabel("Arc length (m)")

    st.pyplot(fig)

# =====================================================
# UI
# =====================================================

with st.sidebar:
    st.header("📂 Input Data")
    cl_file = st.file_uploader("Reference Centerline CSV", type="csv")
    drv_file = st.file_uploader("Driver Run CSV", type="csv")
    run = st.button("Analyze & Generate AI Track")



# =====================================================
# Run pipeline
# =====================================================

if run and cl_file and drv_file:
    ref = pd.read_csv(cl_file)
    drv = pd.read_csv(drv_file)

    ref_x, ref_y = ref.iloc[:, 0].values, ref.iloc[:, 1].values
    drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

    # ---- TIME-SERIES ALIGNMENT ----
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
    s_ref = arc_length(ref_x, ref_y)
    s_ts = s_ref[idx]                     # spatial
    kappa_ts = np.abs(curvature(ref_x, ref_y))[idx]
    t = np.arange(len(dist))              # TRUE time index

    # ---- DECISION ----
    w = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
    decision, metrics = decision_from_timeseries(kappa_ts, dist)

    # ---- TRACK GENERATION ----
    gx, gy, delta, focus = deform_reference_track(ref_x, ref_y, w)

    # =====================================================
    # OUTPUT
    # =====================================================

    st.success(f"🎯 **Training Decision: {decision}**")
    st.json(metrics)

    col1, col2 = st.columns(2)

    with col1:
        plot_weakness_vector(w)
        plot_time_series(t, kappa_ts, dist, delta)

    with col2:
        plot_arc_diagnostics(s_ts, kappa_ts, dist, delta)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ref_x, ref_y, "--", label="Reference")
    ax.plot(gx, gy, label="Generated")
    ax.axis("equal")
    ax.legend()
    st.pyplot(fig)


# if run and cl_file and drv_file:
#     ref = pd.read_csv(cl_file)
#     drv = pd.read_csv(drv_file)

#     ref_x, ref_y = ref.iloc[:, 0].values, ref.iloc[:, 1].values
#     drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

#     # --- time-series alignment ---
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     s_ref = arc_length(ref_x, ref_y)
#     s_ts = s_ref[idx]
#     kappa_ts = np.abs(curvature(ref_x, ref_y))[idx]

#     # --- weakness & decision ---
#     w = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
#     decision, metrics = decision_from_timeseries(kappa_ts, dist)

#     # --- track deformation ---
#     gx, gy, delta, focus = deform_reference_track(ref_x, ref_y, w)

#     # =====================================================
#     # UI OUTPUT
#     # =====================================================

#     st.success(f"🎯 **Training Decision: {decision}**")
#     st.write(metrics)

#     col1, col2 = st.columns(2)

#     with col1:
#         plot_weakness_vector(w)
#         plot_curvature(ref_x, ref_y)

#     with col2:
#         plot_time_series(s_ts, kappa_ts, dist, delta)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.plot(ref_x, ref_y, "--", label="Reference")
#     ax.plot(gx, gy, label="Generated")
#     ax.axis("equal")
#     ax.legend()
#     st.pyplot(fig)




# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # =====================================================
# # Streamlit setup
# # =====================================================

# st.set_page_config(page_title="Generative Practice Track", layout="wide")
# st.title("🏎️ Generative Practice Track (Weakness-Aware)")

# # =====================================================
# # Geometry utilities
# # =====================================================

# def curvature(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#     denom = (dx**2 + dy**2)**1.5 + 1e-9
#     return (dx * ddy - dy * ddx) / denom


# def arc_length(x, y):
#     ds = np.hypot(np.diff(x), np.diff(y))
#     return np.insert(np.cumsum(ds), 0, 0.0)


# def compute_normals(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     mag = np.hypot(dx, dy) + 1e-9
#     return -dy / mag, dx / mag


# def compute_distance(ref_x, ref_y, drv_x, drv_y):
#     d = np.hypot(
#         drv_x[:, None] - ref_x[None, :],
#         drv_y[:, None] - ref_y[None, :]
#     )
#     idx = np.argmin(d, axis=1)
#     return np.min(d, axis=1), idx


# # =====================================================
# # Weakness analysis (vector form)
# # =====================================================

# def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     kappa = np.abs(curvature(ref_x, ref_y))

#     apex_mask = kappa > np.percentile(kappa, 75)

#     w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
#     w_slalom = np.std(np.diff(dist))
#     w_corner = np.mean(dist)

#     w = np.array([w_apex, w_slalom, w_corner])
#     return w / (np.linalg.norm(w) + 1e-9)


# # =====================================================
# # Time-series decision logic (SLIDE LEVEL)
# # =====================================================

# def decision_from_timeseries(kappa_ts, error_ts):
#     high_k = np.percentile(kappa_ts, 75)
#     low_k = np.percentile(kappa_ts, 30)

#     apex_error = np.mean(error_ts[kappa_ts > high_k])
#     straight_error = np.mean(error_ts[kappa_ts < low_k])
#     oscillation = np.std(np.diff(error_ts))

#     if apex_error > oscillation and apex_error > straight_error:
#         return "APEX", apex_error, oscillation

#     if oscillation > apex_error:
#         return "SLALOM", apex_error, oscillation

#     return "CORNERING", apex_error, oscillation


# # =====================================================
# # Track deformation
# # =====================================================

# def deform_reference_track(ref_x, ref_y, weakness_vec, max_offset=3.0):
#     s = arc_length(ref_x, ref_y)
#     s_norm = s / (s[-1] + 1e-9)

#     kappa = np.abs(curvature(ref_x, ref_y))
#     nx, ny = compute_normals(ref_x, ref_y)

#     labels = ["apex", "slalom", "cornering"]
#     focus = labels[np.argmax(weakness_vec)]

#     delta = np.zeros_like(s)

#     if focus == "apex":
#         mask = kappa > np.percentile(kappa, 75)
#         strength = weakness_vec[0]
#     elif focus == "slalom":
#         mask = kappa < np.percentile(kappa, 30)
#         strength = weakness_vec[1]
#     else:
#         mask = np.ones_like(kappa, dtype=bool)
#         strength = weakness_vec[2]

#     delta[mask] = max_offset * strength * np.sin(2 * np.pi * 4 * s_norm[mask])
#     delta = np.convolve(delta, np.ones(15) / 15, mode="same")

#     new_x = ref_x + delta * nx
#     new_y = ref_y + delta * ny

#     return new_x, new_y, delta, focus


# # =====================================================
# # Visualization helpers
# # =====================================================

# def plot_curvature(x, y):
#     s = arc_length(x, y)
#     k = curvature(x, y)
#     fig, ax = plt.subplots(figsize=(10, 3))
#     ax.plot(s, k)
#     ax.set_title("Curvature κ(s)")
#     ax.set_xlabel("Arc length (m)")
#     ax.set_ylabel("κ")
#     st.pyplot(fig)


# def plot_time_series(s, kappa, dist, delta):
#     fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#     ax[0].plot(s, kappa)
#     ax[0].set_title("Curvature vs Arc Length")

#     ax[1].plot(s, dist)
#     ax[1].set_title("Lateral Error vs Arc Length")

#     ax[2].plot(s, delta)
#     ax[2].set_title("Track Deformation vs Arc Length")
#     ax[2].set_xlabel("Arc Length (m)")

#     st.pyplot(fig)


# def plot_weakness_vector(w):
#     fig, ax = plt.subplots(figsize=(5, 4))
#     ax.bar(["Apex", "Slalom", "Cornering"], w,
#            color=["red", "blue", "green"])
#     ax.set_ylim(0, 1)
#     ax.set_title("Normalized Weakness Vector")
#     st.pyplot(fig)


# # =====================================================
# # UI
# # =====================================================

# with st.sidebar:
#     cl_file = st.file_uploader("Reference Centerline CSV", type="csv")
#     drv_file = st.file_uploader("Driver Run CSV", type="csv")
#     run = st.button("Analyze & Generate AI Track")


# # =====================================================
# # Run
# # =====================================================

# if run and cl_file and drv_file:
#     ref = pd.read_csv(cl_file)
#     drv = pd.read_csv(drv_file)

#     ref_x, ref_y = ref.iloc[:, 0].values, ref.iloc[:, 1].values
#     drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     s = arc_length(ref_x, ref_y)[idx]
#     kappa_ts = np.abs(curvature(ref_x, ref_y))[idx]

#     w = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
#     gx, gy, delta, focus = deform_reference_track(ref_x, ref_y, w)

#     decision, apex_err, osc = decision_from_timeseries(kappa_ts, dist)

#     st.success(f"🎯 Training Decision: **{decision}**")
#     st.write({
#         "Apex Error": apex_err,
#         "Slalom Oscillation": osc,
#         "Mean Error": np.mean(dist)
#     })

#     plot_weakness_vector(w)
#     plot_curvature(ref_x, ref_y)
#     plot_time_series(s, kappa_ts, dist, delta)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.plot(ref_x, ref_y, "--", label="Reference")
#     ax.plot(gx, gy, label="Generated")
#     ax.axis("equal")
#     ax.legend()
#     st.pyplot(fig)

# def weakness_vector_from_metrics(metrics):
#     w = np.array([
#         metrics["apex_error"],
#         metrics["slalom_instability"],
#         metrics["mean_error"]
#     ])
#     return w / (np.linalg.norm(w) + 1e-9)



# NameError: name 'plot_curvature' is not defined
# Traceback:
# File "/Users/darisdzakwanhoesien/.pyenv/versions/3.10.11/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 542, in _run_script
#     exec(code, module.__dict__)
# File "/Users/darisdzakwanhoesien/Documents/race-map-generator/pages/track_generator_FINAL_new.py", line 230, in <module>
#     plot_curvature(d["ref_x"], d["ref_y"])

# I like my code above, fix that and also I need some decision making on top of it based on the slides on why is it like that

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # =====================================================
# # Streamlit setup
# # =====================================================

# st.set_page_config(page_title="Generative Practice Track", layout="wide")
# st.title("🏎️ Generative Practice Track (Weakness-Aware)")

# # =====================================================
# # Geometry utilities
# # =====================================================

# def curvature(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#     denom = (dx**2 + dy**2)**1.5 + 1e-9
#     return (dx * ddy - dy * ddx) / denom


# def arc_length(x, y):
#     ds = np.hypot(np.diff(x), np.diff(y))
#     return np.insert(np.cumsum(ds), 0, 0.0)


# def compute_normals(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     mag = np.hypot(dx, dy) + 1e-9
#     return -dy / mag, dx / mag


# def compute_distance(ref_x, ref_y, drv_x, drv_y):
#     d = np.hypot(drv_x[:, None] - ref_x[None, :],
#                  drv_y[:, None] - ref_y[None, :])
#     idx = np.argmin(d, axis=1)
#     return np.min(d, axis=1), idx


# # =====================================================
# # Weakness analysis
# # =====================================================

# def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     kappa = np.abs(curvature(ref_x, ref_y))

#     apex_mask = kappa > np.percentile(kappa, 75)

#     w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
#     w_slalom = np.std(np.diff(dist))
#     w_corner = np.mean(dist)

#     w = np.array([w_apex, w_slalom, w_corner])
#     return w / (np.linalg.norm(w) + 1e-9)


# # =====================================================
# # Track deformation
# # =====================================================

# def deform_reference_track(ref_x, ref_y, weakness_vec, max_offset=3.0):
#     s = arc_length(ref_x, ref_y)
#     s_norm = s / (s[-1] + 1e-9)

#     kappa = np.abs(curvature(ref_x, ref_y))
#     nx, ny = compute_normals(ref_x, ref_y)

#     labels = ["apex", "slalom", "cornering"]
#     focus = labels[np.argmax(weakness_vec)]

#     delta = np.zeros_like(s)

#     if focus == "apex":
#         mask = kappa > np.percentile(kappa, 75)
#         strength = weakness_vec[0]
#     elif focus == "slalom":
#         mask = kappa < np.percentile(kappa, 30)
#         strength = weakness_vec[1]
#     else:
#         mask = np.ones_like(kappa, dtype=bool)
#         strength = weakness_vec[2]

#     delta[mask] = (
#         max_offset
#         * strength
#         * np.sin(2 * np.pi * 4 * s_norm[mask])
#     )

#     delta = np.convolve(delta, np.ones(15) / 15, mode="same")

#     new_x = ref_x + delta * nx
#     new_y = ref_y + delta * ny

#     return new_x, new_y, delta, focus


# # =====================================================
# # Decision logic (SLIDE-LEVEL)
# # =====================================================

# def decision_from_timeseries(kappa_ts, error_ts):
#     """
#     Translate time-series diagnostics into training decision
#     """

#     high_k = np.percentile(kappa_ts, 75)
#     low_k = np.percentile(kappa_ts, 30)

#     apex_error = np.mean(error_ts[kappa_ts > high_k])
#     straight_error = np.mean(error_ts[kappa_ts < low_k])
#     oscillation = np.std(np.diff(error_ts))

#     if apex_error > straight_error and apex_error > oscillation:
#         return "APEX", {
#             "apex_error": apex_error,
#             "slalom_instability": oscillation,
#             "mean_error": np.mean(error_ts)
#         }

#     if oscillation > apex_error:
#         return "SLALOM", {
#             "apex_error": apex_error,
#             "slalom_instability": oscillation,
#             "mean_error": np.mean(error_ts)
#         }

#     return "CORNERING", {
#         "apex_error": apex_error,
#         "slalom_instability": oscillation,
#         "mean_error": np.mean(error_ts)
#     }


# # =====================================================
# # Visualization helpers
# # =====================================================

# def plot_time_series(s, kappa, dist, delta):
#     fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#     ax[0].plot(s, kappa)
#     ax[0].set_title("Curvature κ(s)")
#     ax[0].set_ylabel("κ")

#     ax[1].plot(s, dist)
#     ax[1].set_title("Lateral Error")
#     ax[1].set_ylabel("Error (m)")

#     ax[2].plot(s, delta)
#     ax[2].set_title("Track Deformation Δ(s)")
#     ax[2].set_ylabel("Offset (m)")
#     ax[2].set_xlabel("Arc Length (m)")

#     st.pyplot(fig)


# def plot_weakness_vector(w):
#     fig, ax = plt.subplots(figsize=(5, 4))
#     ax.bar(["Apex", "Slalom", "Cornering"], w,
#            color=["red", "blue", "green"])
#     ax.set_ylim(0, 1)
#     ax.set_title("Normalized Weakness Vector")
#     st.pyplot(fig)


# # =====================================================
# # UI
# # =====================================================

# with st.sidebar:
#     cl_file = st.file_uploader("Reference Centerline CSV", type="csv")
#     drv_file = st.file_uploader("Driver Run CSV", type="csv")
#     run = st.button("Analyze & Generate AI Track")


# # =====================================================
# # Run analysis
# # =====================================================

# if run and cl_file is not None and drv_file is not None:
#     ref = pd.read_csv(cl_file)
#     drv = pd.read_csv(drv_file)

#     ref_x, ref_y = ref.iloc[:, 1].values, ref.iloc[:, 0].values
#     drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     s = arc_length(ref_x, ref_y)[idx]
#     kappa_ts = np.abs(curvature(ref_x, ref_y))[idx]

#     w = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
#     gx, gy, delta, focus = deform_reference_track(ref_x, ref_y, w)

#     decision, metrics = decision_from_timeseries(kappa_ts, dist)

#     st.success(f"🎯 **Training Decision: {decision}**")
#     st.write(metrics)

#     plot_weakness_vector(w)
#     plot_time_series(s, kappa_ts, dist, delta)

#     fig, ax = plt.subplots(figsize=(8, 7))
#     ax.plot(ref_x, ref_y, "--", label="Reference")
#     ax.plot(gx, gy, "g", label="Generated")
#     ax.axis("equal")
#     ax.legend()
#     st.pyplot(fig)




# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # =====================================================
# # Streamlit setup
# # =====================================================

# st.set_page_config(page_title="Generative Practice Track", layout="wide")
# st.title("🏎️ Generative Practice Track (Weakness-Aware)")

# # =====================================================
# # Geometry utilities
# # =====================================================

# def curvature(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#     denom = (dx**2 + dy**2)**1.5 + 1e-9
#     return (dx * ddy - dy * ddx) / denom


# def arc_length(x, y):
#     ds = np.hypot(np.diff(x), np.diff(y))
#     return np.insert(np.cumsum(ds), 0, 0.0)


# def compute_normals(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     mag = np.hypot(dx, dy) + 1e-9
#     return -dy / mag, dx / mag


# def compute_distance(ref_x, ref_y, drv_x, drv_y):
#     d = np.hypot(drv_x[:, None] - ref_x[None, :],
#                  drv_y[:, None] - ref_y[None, :])
#     idx = np.argmin(d, axis=1)
#     return np.min(d, axis=1), idx


# # =====================================================
# # Weakness analysis
# # =====================================================

# def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     kappa = np.abs(curvature(ref_x, ref_y))

#     apex_mask = kappa > np.percentile(kappa, 75)

#     w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
#     w_slalom = np.std(np.diff(dist))
#     w_corner = np.mean(dist)

#     w = np.array([w_apex, w_slalom, w_corner])
#     return w / (np.linalg.norm(w) + 1e-9)


# # =====================================================
# # Track deformation
# # =====================================================

# def deform_reference_track(ref_x, ref_y, weakness_vec, max_offset=3.0):
#     s = arc_length(ref_x, ref_y)
#     s_norm = s / (s[-1] + 1e-9)

#     kappa = np.abs(curvature(ref_x, ref_y))
#     nx, ny = compute_normals(ref_x, ref_y)

#     labels = ["apex", "slalom", "cornering"]
#     focus = labels[np.argmax(weakness_vec)]

#     delta = np.zeros_like(s)

#     if focus == "apex":
#         mask = kappa > np.percentile(kappa, 75)
#         strength = weakness_vec[0]
#     elif focus == "slalom":
#         mask = kappa < np.percentile(kappa, 30)
#         strength = weakness_vec[1]
#     else:
#         mask = np.ones_like(kappa, dtype=bool)
#         strength = weakness_vec[2]

#     delta[mask] = (
#         max_offset
#         * strength
#         * np.sin(2 * np.pi * 4 * s_norm[mask])
#     )

#     delta = np.convolve(delta, np.ones(15) / 15, mode="same")

#     new_x = ref_x + delta * nx
#     new_y = ref_y + delta * ny

#     return new_x, new_y, delta, focus


# # =====================================================
# # Decision logic (SLIDE-LEVEL)
# # =====================================================

# def decision_from_timeseries(kappa_ts, error_ts):
#     """
#     Translate time-series diagnostics into training decision
#     """

#     high_k = np.percentile(kappa_ts, 75)
#     low_k = np.percentile(kappa_ts, 30)

#     apex_error = np.mean(error_ts[kappa_ts > high_k])
#     straight_error = np.mean(error_ts[kappa_ts < low_k])
#     oscillation = np.std(np.diff(error_ts))

#     if apex_error > straight_error and apex_error > oscillation:
#         return "APEX", {
#             "apex_error": apex_error,
#             "slalom_instability": oscillation,
#             "mean_error": np.mean(error_ts)
#         }

#     if oscillation > apex_error:
#         return "SLALOM", {
#             "apex_error": apex_error,
#             "slalom_instability": oscillation,
#             "mean_error": np.mean(error_ts)
#         }

#     return "CORNERING", {
#         "apex_error": apex_error,
#         "slalom_instability": oscillation,
#         "mean_error": np.mean(error_ts)
#     }


# # =====================================================
# # Visualization helpers
# # =====================================================

# def plot_time_series(s, kappa, dist, delta):
#     fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#     ax[0].plot(s, kappa)
#     ax[0].set_title("Curvature κ(s)")
#     ax[0].set_ylabel("κ")

#     ax[1].plot(s, dist)
#     ax[1].set_title("Lateral Error")
#     ax[1].set_ylabel("Error (m)")

#     ax[2].plot(s, delta)
#     ax[2].set_title("Track Deformation Δ(s)")
#     ax[2].set_ylabel("Offset (m)")
#     ax[2].set_xlabel("Arc Length (m)")

#     st.pyplot(fig)


# def plot_weakness_vector(w):
#     fig, ax = plt.subplots(figsize=(5, 4))
#     ax.bar(["Apex", "Slalom", "Cornering"], w,
#            color=["red", "blue", "green"])
#     ax.set_ylim(0, 1)
#     ax.set_title("Normalized Weakness Vector")
#     st.pyplot(fig)


# # =====================================================
# # UI
# # =====================================================

# with st.sidebar:
#     cl_file = st.file_uploader("Reference Centerline CSV", type="csv")
#     drv_file = st.file_uploader("Driver Run CSV", type="csv")
#     run = st.button("Analyze & Generate AI Track")


# # =====================================================
# # Run analysis
# # =====================================================

# if run and cl_file is not None and drv_file is not None:
#     ref = pd.read_csv(cl_file)
#     drv = pd.read_csv(drv_file)

#     ref_x, ref_y = ref.iloc[:, 1].values, ref.iloc[:, 0].values
#     drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     s = arc_length(ref_x, ref_y)[idx]
#     kappa_ts = np.abs(curvature(ref_x, ref_y))[idx]

#     w = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
#     gx, gy, delta, focus = deform_reference_track(ref_x, ref_y, w)

#     decision, metrics = decision_from_timeseries(kappa_ts, dist)

#     st.success(f"🎯 **Training Decision: {decision}**")
#     st.write(metrics)

#     plot_weakness_vector(w)
#     plot_time_series(s, kappa_ts, dist, delta)

#     fig, ax = plt.subplots(figsize=(8, 7))
#     ax.plot(ref_x, ref_y, "--", label="Reference")
#     ax.plot(gx, gy, "g", label="Generated")
#     ax.axis("equal")
#     ax.legend()
#     st.pyplot(fig)


# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # =====================================================
# # Streamlit setup
# # =====================================================

# st.set_page_config(page_title="Generative Practice Track", layout="wide")
# st.title("🏎️ Generative Practice Track (Weakness-Aware)")

# # =====================================================
# # Geometry utilities
# # =====================================================

# def curvature(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#     denom = (dx**2 + dy**2)**1.5 + 1e-9
#     return (dx * ddy - dy * ddx) / denom


# def arc_length(x, y):
#     ds = np.hypot(np.diff(x), np.diff(y))
#     return np.insert(np.cumsum(ds), 0, 0.0)


# def compute_normals(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     mag = np.hypot(dx, dy) + 1e-9
#     return -dy / mag, dx / mag


# def compute_distance(ref_x, ref_y, drv_x, drv_y):
#     d = np.hypot(drv_x[:, None] - ref_x[None, :],
#                  drv_y[:, None] - ref_y[None, :])
#     idx = np.argmin(d, axis=1)
#     return np.min(d, axis=1), idx


# # =====================================================
# # Weakness analysis
# # =====================================================

# def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     kappa = np.abs(curvature(ref_x, ref_y))

#     apex_mask = kappa > np.percentile(kappa, 75)

#     w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
#     w_slalom = np.std(np.diff(dist))
#     w_corner = np.mean(dist)

#     w = np.array([w_apex, w_slalom, w_corner])
#     return w / (np.linalg.norm(w) + 1e-9)


# # =====================================================
# # Track deformation
# # =====================================================

# def deform_reference_track(ref_x, ref_y, weakness_vec, max_offset=3.0):
#     s = arc_length(ref_x, ref_y)
#     s_norm = s / (s[-1] + 1e-9)

#     kappa = np.abs(curvature(ref_x, ref_y))
#     nx, ny = compute_normals(ref_x, ref_y)

#     labels = ["apex", "slalom", "cornering"]
#     focus = labels[np.argmax(weakness_vec)]

#     delta = np.zeros_like(s)

#     if focus == "apex":
#         mask = kappa > np.percentile(kappa, 75)
#         strength = weakness_vec[0]
#     elif focus == "slalom":
#         mask = kappa < np.percentile(kappa, 30)
#         strength = weakness_vec[1]
#     else:
#         mask = np.ones_like(kappa, dtype=bool)
#         strength = weakness_vec[2]

#     delta[mask] = (
#         max_offset
#         * strength
#         * np.sin(2 * np.pi * 4 * s_norm[mask])
#     )

#     delta = np.convolve(delta, np.ones(15) / 15, mode="same")

#     new_x = ref_x + delta * nx
#     new_y = ref_y + delta * ny

#     return new_x, new_y, delta, focus


# # =====================================================
# # Visualization helpers
# # =====================================================

# def plot_curvature(x, y):
#     s = arc_length(x, y)
#     k = curvature(x, y)

#     fig, ax = plt.subplots(figsize=(10, 3))
#     ax.plot(s, k)
#     ax.set_title("Curvature κ(s)")
#     ax.set_xlabel("Arc length (m)")
#     ax.set_ylabel("κ")
#     st.pyplot(fig)


# def plot_error_vs_curvature(ref_x, ref_y, drv_x, drv_y):
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     kappa = np.abs(curvature(ref_x, ref_y))[idx]

#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.scatter(kappa, dist, alpha=0.6)
#     ax.set_xlabel("|κ|")
#     ax.set_ylabel("Lateral error")
#     ax.set_title("Driver Error vs Curvature")
#     st.pyplot(fig)


# def plot_weakness_vector(w):
#     fig, ax = plt.subplots(figsize=(5, 4))
#     ax.bar(["Apex", "Slalom", "Cornering"], w,
#            color=["red", "blue", "green"])
#     ax.set_ylim(0, 1)
#     ax.set_title("Normalized Weakness Vector")
#     st.pyplot(fig)


# def plot_curvature_comparison(ref_x, ref_y, new_x, new_y):
#     s = arc_length(ref_x, ref_y)
#     fig, ax = plt.subplots(figsize=(10, 3))
#     ax.plot(s, np.abs(curvature(ref_x, ref_y)), label="Original")
#     ax.plot(s, np.abs(curvature(new_x, new_y)), "--", label="Deformed")
#     ax.set_title("Curvature Before vs After Deformation")
#     ax.legend()
#     st.pyplot(fig)


# # =====================================================
# # UI (defined ONCE)
# # =====================================================

# with st.sidebar:
#     st.header("📂 Input Data")

#     cl_file = st.file_uploader(
#         "Reference Centerline CSV",
#         type="csv",
#         key="centerline_uploader"
#     )

#     drv_file = st.file_uploader(
#         "Driver Run CSV",
#         type="csv",
#         key="driver_uploader"
#     )

#     run = st.button("Analyze & Generate AI Track")


# # =====================================================
# # Run analysis
# # =====================================================

# if run:
#     if cl_file is None or drv_file is None:
#         st.error("Please upload both files.")
#     else:
#         ref = pd.read_csv(cl_file)
#         drv = pd.read_csv(drv_file)

#         ref_x, ref_y = ref.iloc[:, 0].values, ref.iloc[:, 1].values
#         drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

#         w = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
#         gx, gy, delta, focus = deform_reference_track(ref_x, ref_y, w)

#         st.session_state.data = {
#             "ref_x": ref_x,
#             "ref_y": ref_y,
#             "drv_x": drv_x,
#             "drv_y": drv_y,
#             "gx": gx,
#             "gy": gy,
#             "delta": delta,
#             "w": w,
#             "focus": focus,
#         }


# # =====================================================
# # Render results
# # =====================================================

# if "data" in st.session_state:
#     d = st.session_state.data

#     st.success(f"🎯 Practice Track Focus: **{d['focus'].upper()}**")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.header("Track Geometry")
#         plot_curvature(d["ref_x"], d["ref_y"])

#         st.header("Driver Weakness")
#         plot_weakness_vector(d["w"])

#     with col2:
#         st.header("Driver Error Evidence")
#         plot_error_vs_curvature(
#             d["ref_x"], d["ref_y"],
#             d["drv_x"], d["drv_y"]
#         )

#         st.header("Effect of Track Deformation")
#         plot_curvature_comparison(
#             d["ref_x"], d["ref_y"],
#             d["gx"], d["gy"]
#         )

#     st.header("Generated Practice Track")

#     fig, ax = plt.subplots(figsize=(8, 7))
#     ax.plot(d["ref_x"], d["ref_y"], "--", label="Reference")
#     ax.plot(d["gx"], d["gy"], "g", label="Generated")
#     ax.axis("equal")
#     ax.legend()
#     st.pyplot(fig)
