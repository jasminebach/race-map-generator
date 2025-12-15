import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

st.set_page_config(page_title="Track Weakness Analyzer", layout="wide")
st.title("Practice Track Generator")

# ------------------ Compute curvature ------------------
def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-9
    return (dx * ddy - dy * ddx) / denom

# ------------------ Distance to centerline ------------------
def compute_distance(ref_x, ref_y, drv_x, drv_y):
    d = np.hypot(
        drv_x[:, None] - ref_x[None, :],
        drv_y[:, None] - ref_y[None, :]
    )
    nearest = np.min(d, axis=1)
    nearest_idx = np.argmin(d, axis=1)
    return nearest, nearest_idx

# ------------------ Weakness analysis ------------------
def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
    kappa = curvature(ref_x, ref_y)

    # High-curvature zones (apex)
    k_thresh = np.percentile(np.abs(kappa), 80)
    highK = np.where(np.abs(kappa) > k_thresh)[0]

    apex_mask = np.isin(idx, highK)
    apex_error = np.mean(dist[apex_mask]) if np.any(apex_mask) else 0.0

    # Slalom oscillation (smoothed)
    dist_smooth = gaussian_filter1d(dist, sigma=3)
    slalom_score = np.std(np.diff(dist_smooth))

    mean_error = np.mean(dist)
    max_error = np.max(dist)

    if apex_error > 1.0:
        primary = "apex"
    elif slalom_score > 0.5:
        primary = "slalom"
    elif mean_error > 0.8:
        primary = "cornering"
    else:
        primary = None

    return {
        "mean_error": float(mean_error),
        "max_error": float(max_error),
        "apex_error": float(apex_error),
        "slalom_score": float(slalom_score),
        "primary": primary
    }

# ------------------ Generate new centerline ------------------
def generate_centerline(seed=None, focus=None, lap_target=1000):
    np.random.seed(seed or int(time.time()))

    if focus == "apex":
        probs = np.array([0.15, 0.15, 0.35, 0.30, 0.05])
    elif focus == "slalom":
        probs = np.array([0.10, 0.10, 0.10, 0.10, 0.60])
    elif focus == "cornering":
        probs = np.array([0.20, 0.40, 0.20, 0.20, 0.00])
    else:
        probs = np.array([0.25, 0.25, 0.20, 0.15, 0.15])

    probs /= probs.sum()
    segs = ["straight", "sweeper", "constant", "hairpin", "short_straight"]

    X, Y = [], []
    x, y = 0.0, 0.0
    heading = 0.0
    total = 0.0

    while total < lap_target * 0.95:
        stype = np.random.choice(segs, p=probs)

        if stype in ["straight", "short_straight"]:
            L = np.random.uniform(10, 80)
            xs = np.linspace(0, L, 20)
            ys = np.zeros_like(xs)
            dtheta = 0.0
        else:
            R = np.random.uniform(10, 40)
            ang = np.deg2rad(np.random.uniform(40, 140))
            sign = np.random.choice([-1, 1])
            t = np.linspace(0, sign * ang, 40)
            xs = R * np.sin(t)
            ys = sign * R * (1 - np.cos(t))
            dtheta = t[-1]

        c, s = np.cos(heading), np.sin(heading)
        xr = c * xs - s * ys + x
        yr = s * xs + c * ys + y

        X.append(xr)
        Y.append(yr)

        total += np.sum(np.hypot(np.diff(xr), np.diff(yr)))
        x, y = xr[-1], yr[-1]
        heading += dtheta

    return np.hstack(X), np.hstack(Y)

# ------------------ Streamlit UI ------------------
cl_file = st.sidebar.file_uploader("Centerline CSV", type="csv")
drv_file = st.sidebar.file_uploader("Driver Run CSV", type="csv")
lap_target = st.sidebar.number_input("Lap length (m)", 400, 5000, 1000)

if st.sidebar.button("Analyze & Generate"):
    if cl_file is None or drv_file is None:
        st.error("Please upload BOTH files.")
        st.stop()

    ref = pd.read_csv(cl_file)
    drv = pd.read_csv(drv_file)

    ref_x = ref["xRef"].to_numpy()
    ref_y = ref["yRef"].to_numpy()
    drv_x = drv["<X>"].to_numpy()
    drv_y = drv["<Y>"].to_numpy()

    results = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
    st.subheader("Detected Metrics")
    st.json(results)

    focus = results["primary"]
    if focus:
        st.warning(f"Detected primary weakness: **{focus}**")
    else:
        st.success("No dominant weakness → balanced track generated.")

    gx, gy = generate_centerline(focus=focus, lap_target=lap_target)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(ref_x, ref_y, "--", label="Reference")
    ax.plot(drv_x, drv_y, ".", alpha=0.6, label="Driver")
    ax.plot(gx, gy, "-", label="Generated Track")
    ax.axis("equal")
    ax.legend()
    st.pyplot(fig)

    st.download_button(
        "Download New Practice Track",
        pd.DataFrame({"x": gx, "y": gy}).to_csv(index=False),
        file_name="generated_track.csv"
    )


# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev
# from scipy.ndimage import gaussian_filter1d
# import time

# st.set_page_config(page_title="Track Weakness Analyzer", layout="wide")
# st.title("Practice Track Generator")

# # ------------------ Compute curvature ------------------
# def curvature(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#     denom = (dx**2 + dy**2)**1.5 + 1e-12
#     return (dx*ddy - dy*ddx) / denom

# # ------------------ Distance to centerline ------------------
# def compute_distance(ref_x, ref_y, drv_x, drv_y):
#     # compute nearest point distance
#     d = np.hypot(drv_x[:,None] - ref_x[None,:], drv_y[:,None] - ref_y[None,:])
#     nearest = np.min(d, axis=1)
#     nearest_idx = np.argmin(d, axis=1)
#     return nearest, nearest_idx

# # ------------------ Weakness analysis ------------------
# def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     kappa = curvature(ref_x, ref_y)

#     # 1. Apex → large error on high curvature
#     highK = np.where(abs(kappa) > np.percentile(abs(kappa), 80))[0]
#     apex_error = np.mean(dist[np.isin(idx, highK)])

#     # 2. Slalom weakness → oscillating error pattern
#     diffs = np.abs(np.diff(dist))
#     slalom_score = np.std(diffs)

#     # 3. General tracking error
#     mean_error = np.mean(dist)
#     max_error = np.max(dist)

#     # Decide primary weakness
#     if apex_error > 1.0:
#         primary = "apex"
#     elif slalom_score > 0.5:
#         primary = "slalom"
#     elif mean_error > 0.8:
#         primary = "cornering"
#     else:
#         primary = None

#     return {
#         "mean_error": mean_error,
#         "max_error": max_error,
#         "apex_error": apex_error,
#         "slalom_score": slalom_score,
#         "primary": primary
#     }

# # ------------------ Generate new centerline (biased) ------------------
# def generate_centerline(seed=None, focus=None, lap_target=1000):
#     np.random.seed(seed or int(time.time()))

#     # Bias probabilities
#     if focus == "apex":
#         probs = np.array([0.15,0.15,0.35,0.30,0.05])
#     elif focus == "slalom":
#         probs = np.array([0.10,0.10,0.10,0.10,0.60])
#     elif focus == "cornering":
#         probs = np.array([0.20,0.40,0.20,0.20,0.00])
#     else:
#         probs = np.array([0.25,0.25,0.20,0.15,0.15])

#     probs = probs / probs.sum()

#     segs = ["straight","sweeper","constant","hairpin","short_straight"]
#     X = []
#     Y = []
#     x = y = 0
#     heading = 0
#     total = 0

#     while total < lap_target * 0.95:
#         stype = np.random.choice(segs, p=probs)

#         if stype == "straight":
#             L = np.random.uniform(40,80)
#             xs = np.linspace(0,L,20)
#             ys = np.zeros_like(xs)

#         elif stype == "short_straight":
#             L = np.random.uniform(10,25)
#             xs = np.linspace(0,L,15)
#             ys = np.zeros_like(xs)

#         else:
#             R = np.random.uniform(10,40)
#             ang = np.deg2rad(np.random.uniform(40,140))
#             sign = 1 if np.random.rand() < 0.5 else -1
#             t = np.linspace(0,sign*ang,40)
#             xs = R*np.sin(t)
#             ys = R*(1-np.cos(t))*sign

#         c,s = np.cos(heading), np.sin(heading)
#         xr = c*xs - s*ys + x
#         yr = s*xs + c*ys + y

#         X.append(xr)
#         Y.append(yr)

#         total += np.sum(np.hypot(np.diff(xr), np.diff(yr)))
#         x, y = xr[-1], yr[-1]
#         heading = np.arctan2(yr[-1]-yr[0], xr[-1]-xr[0])

#     X = np.hstack(X)
#     Y = np.hstack(Y)
#     return X, Y

# # ------------------ Streamlit UI ------------------
# cl_file = st.sidebar.file_uploader("Centerline CSV", type="csv")
# drv_file = st.sidebar.file_uploader("Driver Run CSV", type="csv")
# lap_target = st.sidebar.number_input("Lap length (m)", 400, 5000, 1000)

# if st.sidebar.button("Analyze & Generate"):
#     if cl_file is None or drv_file is None:
#         st.error("Please upload BOTH files.")
#     else:
#         ref = pd.read_csv(cl_file)
#         ref_x = ref["xRef"].values
#         ref_y = ref["yRef"].values

#         drv = pd.read_csv(drv_file)
#         drv_x = drv["<X>"].values
#         drv_y = drv["<Y>"].values

#         # analyze
#         results = analyze_weakness(ref_x, ref_y, drv_x, drv_y)
#         st.subheader("Detected Metrics")
#         st.json(results)

#         focus = results["primary"]
#         if focus is None:
#             st.success("Driver has no major weakness → generating balanced track.")
#         else:
#             st.warning(f"Detected primary weakness: **{focus}**")

#         # generate new map
#         gx, gy = generate_centerline(focus=focus, lap_target=lap_target)

#         # visualize
#         fig, ax = plt.subplots(figsize=(8,7))
#         ax.plot(ref_x, ref_y, "--", label="Reference")
#         ax.plot(drv_x, drv_y, ".", alpha=0.6, label="Driver")
#         ax.plot(gx, gy, "-", label="Generated Track")
#         ax.axis("equal")
#         ax.legend()
#         st.pyplot(fig)

#         st.download_button(
#             "Download New Practice Track",
#             pd.DataFrame({"x":gx,"y":gy}).to_csv(index=False),
#             file_name="generated_track.csv"
#         )
