import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Track Weakness Analyzer", layout="wide")
st.title("Practice Track Generator (Weakness-Based)")

# Geometry helpers
def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-9
    return (dx*ddy - dy*ddx) / denom


def compute_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    mag = np.hypot(dx, dy) + 1e-9
    nx = -dy / mag
    ny = dx / mag
    return nx, ny


def generate_track_boundaries(x, y, width=6.0):
    nx, ny = compute_normals(x, y)
    lx = x + nx * width / 2
    ly = y + ny * width / 2
    rx = x - nx * width / 2
    ry = y - ny * width / 2
    return lx, ly, rx, ry


def compute_distance(ref_x, ref_y, drv_x, drv_y):
    d = np.hypot(drv_x[:,None]-ref_x[None,:],
                 drv_y[:,None]-ref_y[None,:])
    idx = np.argmin(d, axis=1)
    return np.min(d, axis=1), idx


# Previous Driver Simulation
def bbox(x, y):
    return x.min(), x.max(), y.min(), y.max()


def normalize(x, y):
    xmin, xmax, ymin, ymax = bbox(x, y)
    xn = (x - xmin) / (xmax - xmin + 1e-9)
    yn = (y - ymin) / (ymax - ymin + 1e-9)
    return xn, yn


def map_to_target(xn, yn, target_x, target_y):
    txmin, txmax, tymin, tymax = bbox(target_x, target_y)

    x_new = xn * (txmax - txmin) + txmin
    y_new = yn * (tymax - tymin) + tymin

    return x_new, y_new



# Weakness analysis
def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
    kappa = np.abs(curvature(ref_x, ref_y))

    highK = np.where(kappa > np.percentile(kappa, 75))[0]
    apex_error = np.mean(dist[np.isin(idx, highK)])
    slalom_score = np.std(np.diff(dist))
    mean_error = np.mean(dist)

    scores = {
        "apex": apex_error / (mean_error + 1e-3),
        "slalom": slalom_score / (mean_error + 1e-3),
        "cornering": mean_error
    }

    primary = max(scores, key=scores.get)

    return scores, primary

# Track generation
def generate_segment(stype):
    if stype == "hairpin":
        R = np.random.uniform(6, 10)
        ang = np.deg2rad(np.random.uniform(150, 190))
    elif stype == "constant":
        R = np.random.uniform(18, 30)
        ang = np.deg2rad(np.random.uniform(70, 110))
    elif stype == "sweeper":
        R = np.random.uniform(40, 70)
        ang = np.deg2rad(np.random.uniform(40, 80))
    elif stype == "slalom":
        xs = np.linspace(0, 60, 80)
        ys = 6 * np.sin(xs / 6)
        return xs, ys, 0
    else:
        L = np.random.uniform(30, 80)
        xs = np.linspace(0, L, 40)
        ys = np.zeros_like(xs)
        return xs, ys, 0

    t = np.linspace(0, ang, 80)
    xs = R * np.sin(t)
    ys = R * (1 - np.cos(t))
    return xs, ys, ang


def generate_centerline(focus, lap_target=800):
    np.random.seed(int(time.time()))
    X, Y = [], []
    x = y = 0
    heading = 0
    total = 0

    if focus == "apex":
        pool = ["hairpin", "constant", "straight"]
    elif focus == "slalom":
        pool = ["slalom", "straight"]
    else:
        pool = ["constant", "sweeper", "straight"]

    while total < lap_target:
        stype = np.random.choice(pool)
        xs, ys, dtheta = generate_segment(stype)

        c, s = np.cos(heading), np.sin(heading)
        xr = c*xs - s*ys + x
        yr = s*xs + c*ys + y

        X.append(xr)
        Y.append(yr)

        total += np.sum(np.hypot(np.diff(xr), np.diff(yr)))
        x, y = xr[-1], yr[-1]
        heading += dtheta

    return np.hstack(X), np.hstack(Y)

# UI
cl_file = st.sidebar.file_uploader("Centerline CSV", type="csv")
drv_file = st.sidebar.file_uploader("Driver Run CSV", type="csv")
lap_target = st.sidebar.slider("Lap Length (m)", 50, 1000, 500)
track_width = st.sidebar.slider("Track Width (m)", 4.0, 10.0, 6.0)

if st.sidebar.button("Analyze & Generate Track"):
    if cl_file is None or drv_file is None:
        st.error("Please upload BOTH centerline and driver files.")
    else:
        ref = pd.read_csv(cl_file)
        drv = pd.read_csv(drv_file)

        ref_x, ref_y = ref.iloc[:,0].values, ref.iloc[:,1].values
        drv_x, drv_y = drv.iloc[:,0].values, drv.iloc[:,1].values

        scores, focus = analyze_weakness(ref_x, ref_y, drv_x, drv_y)

        st.subheader("Detected Weakness Scores")
        st.json(scores)
        st.success(f"Primary weakness detected: **{focus.upper()}**")

        gx, gy = generate_centerline(focus, lap_target)
        lx, ly, rx, ry = generate_track_boundaries(gx, gy, track_width)

        fig, ax = plt.subplots(figsize=(8,7))

        # map reference + driver into generated track space 
        ref_xn, ref_yn = normalize(ref_x, ref_y)
        drv_xn, drv_yn = normalize(drv_x, drv_y)

        ref_xp, ref_yp = map_to_target(ref_xn, ref_yn, gx, gy)
        drv_xp, drv_yp = map_to_target(drv_xn, drv_yn, gx, gy)

        # plotting 
        fig, ax = plt.subplots(figsize=(8,7))

        ax.fill(
            np.r_[lx, rx[::-1]],
            np.r_[ly, ry[::-1]],
            color="lightgray",
            alpha=0.7,
            label="Track Surface"
        )

        ax.plot(lx, ly, "k")
        ax.plot(rx, ry, "k")
        ax.plot(gx, gy, "g--", label="Generated Centerline")

        ax.plot(ref_xp, ref_yp, "--", label="Reference (mapped)")
        ax.plot(drv_xp, drv_yp, ".", alpha=0.4, label="Driver (mapped)")

        ax.axis("equal")
        ax.legend()
        st.pyplot(fig)




