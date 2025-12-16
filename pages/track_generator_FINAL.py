import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Practice Track Generator", layout="wide")
st.title("Generative Practice Track")

def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-9
    return (dx * ddy - dy * ddx) / denom


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
    d = np.hypot(drv_x[:, None] - ref_x[None, :],
                 drv_y[:, None] - ref_y[None, :])
    idx = np.argmin(d, axis=1)
    return np.min(d, axis=1), idx


# Weakness analysis
def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
    dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
    kappa = np.abs(curvature(ref_x, ref_y))

    highK = np.where(kappa > np.percentile(kappa, 75))[0]
    apex_error = np.mean(dist[np.isin(idx, highK)])
    slalom_score = np.std(np.diff(dist))
    mean_error = np.mean(dist)

    vec = np.array([apex_error, slalom_score, mean_error])
    return vec / (np.linalg.norm(vec) + 1e-9)


# Track segment primitives
def generate_segment(stype, mean_radius=20):
    if stype == "hairpin":
        R = np.random.uniform(0.6, 0.8) * mean_radius
        ang = np.deg2rad(np.random.uniform(140, 190))
    elif stype == "sweeper":
        R = np.random.uniform(1.5, 2.5) * mean_radius
        ang = np.deg2rad(np.random.uniform(40, 80))
    elif stype == "constant":
        R = mean_radius
        ang = np.deg2rad(np.random.uniform(70, 110))
    elif stype == "slalom":
        xs = np.linspace(0, 50, 80)
        ys = 5 * np.sin(xs / 6)
        return xs, ys, 0
    else:  # straight
        L = np.random.uniform(30, 80)
        xs = np.linspace(0, L, 40)
        ys = np.zeros_like(xs)
        return xs, ys, 0

    t = np.linspace(0, ang, 80)
    xs = R * np.sin(t)
    ys = R * (1 - np.cos(t))
    return xs, ys, ang


# Parameter-based track generator
def generate_centerline_from_params(params, lap_target):
    X, Y = [], []
    x = y = heading = total = 0

    while total < lap_target:
        r = np.random.rand()

        if r < params["slalom"]:
            stype = "slalom"
        elif r < params["slalom"] + params["hairpin"]:
            stype = "hairpin"
        elif r < params["slalom"] + params["hairpin"] + params["sweeper"]:
            stype = "sweeper"
        elif r < 1 - params["straight"]:
            stype = "constant"
        else:
            stype = "straight"

        xs, ys, dtheta = generate_segment(stype, params["mean_radius"])

        c, s = np.cos(heading), np.sin(heading)
        xr = c * xs - s * ys + x
        yr = s * xs + c * ys + y

        X.append(xr)
        Y.append(yr)

        total += np.sum(np.hypot(np.diff(xr), np.diff(yr)))
        x, y = xr[-1], yr[-1]
        heading += dtheta

    return np.hstack(X), np.hstack(Y)


# Track metrics

def extract_track_metrics(x, y):
    k = np.abs(curvature(x, y))
    apex = np.percentile(k, 85)
    slalom = np.std(np.diff(np.sign(k)))
    corner = np.mean(k)
    vec = np.array([apex, slalom, corner])
    return vec / (np.linalg.norm(vec) + 1e-9)


def fitness(track_x, track_y, weakness_vec):
    track_vec = extract_track_metrics(track_x, track_y)
    return np.dot(track_vec, weakness_vec)


# optimizer
def arc_length(x, y):
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.insert(np.cumsum(ds), 0, 0.0)
    return s


def find_apex_indices(kappa, percentile=80):
    thresh = np.percentile(np.abs(kappa), percentile)
    return np.where(np.abs(kappa) > thresh)[0]



def integrate_centerline_from_curvature(x, y, kappa_new):
    s = arc_length(x, y)
    ds = np.gradient(s)

    # initial heading
    dx = np.gradient(x)
    dy = np.gradient(y)
    theta0 = np.arctan2(dy[0], dx[0])

    theta = np.zeros_like(s)
    theta[0] = theta0

    for i in range(1, len(s)):
        theta[i] = theta[i-1] + kappa_new[i] * ds[i]

    xn = np.zeros_like(x)
    yn = np.zeros_like(y)
    xn[0], yn[0] = x[0], y[0]

    for i in range(1, len(s)):
        xn[i] = xn[i-1] + ds[i] * np.cos(theta[i])
        yn[i] = yn[i-1] + ds[i] * np.sin(theta[i])

    return xn, yn



def deform_reference_track(
    ref_x,
    ref_y,
    weakness_vec,
    max_offset=3.0
):
    """
    Deforms the reference centerline based on driver weakness.
    Preserves topology and flow.
    """

    # --- parameterize by arc length ---
    s = arc_length(ref_x, ref_y)
    s_norm = s / (s[-1] + 1e-9)

    # --- curvature & normals ---
    kappa = np.abs(curvature(ref_x, ref_y))
    nx, ny = compute_normals(ref_x, ref_y)

    # --- identify focus ---
    labels = ["apex", "slalom", "cornering"]
    focus = labels[np.argmax(weakness_vec)]

    # --- build deformation field Δ(s) ---
    delta = np.zeros_like(s_norm)

    if focus == "apex":
        # Emphasize high-curvature regions
        mask = kappa > np.percentile(kappa, 75)
        strength = weakness_vec[0]

    elif focus == "slalom":
        # Emphasize low-curvature regions (straights)
        mask = kappa < np.percentile(kappa, 30)
        strength = weakness_vec[1]

    else:  # cornering
        # Global smooth difficulty increase
        mask = np.ones_like(kappa, dtype=bool)
        strength = weakness_vec[2]

    # --- smooth sinusoidal offset ---
    freq = 4.0
    delta[mask] = (
        max_offset
        * strength
        * np.sin(2 * np.pi * freq * s_norm[mask])
    )

    # --- smooth the deformation ---
    delta = np.convolve(delta, np.ones(15) / 15, mode="same")

    # --- apply deformation ---
    new_x = ref_x + delta * nx
    new_y = ref_y + delta * ny

    return new_x, new_y, focus


def random_params():
    return {
        "hairpin": np.random.uniform(0.1, 0.4),
        "slalom": np.random.uniform(0.1, 0.4),
        "sweeper": np.random.uniform(0.1, 0.4),
        "straight": np.random.uniform(0.1, 0.3),
        "mean_radius": np.random.uniform(12, 35),
    }


def optimize_track(weakness_vec, lap_target, generations=40, pop_size=30):
    population = [random_params() for _ in range(pop_size)]

    for _ in range(generations):
        scored = []
        for p in population:
            x, y = generate_centerline_from_params(p, lap_target)
            s = fitness(x, y, weakness_vec)
            scored.append((s, p))

        scored.sort(reverse=True, key=lambda x: x[0])
        elites = [p for _, p in scored[:pop_size // 4]]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            parent = elites[np.random.randint(len(elites))]
            child = parent.copy()
            for k in child:
                child[k] += np.random.normal(0, 0.05)
                if k != "mean_radius":
                    child[k] = np.clip(child[k], 0.05, 0.9)
            new_pop.append(child)

        population = new_pop

    return scored[0][1]


# UI

cl_file = st.sidebar.file_uploader("Reference Centerline CSV", type="csv")
drv_file = st.sidebar.file_uploader("Driver Run CSV", type="csv")
lap_target = st.sidebar.slider("Target Lap Length (m)", 100, 1000, 600)
# track_width = st.sidebar.slider("Track Width (m)", 4.0, 10.0, 6.0)

if st.sidebar.button("Analyze & Generate AI Track"):
    if cl_file is None or drv_file is None:
        st.error("Upload both centerline and driver data.")
    else:
        ref = pd.read_csv(cl_file)
        drv = pd.read_csv(drv_file)

        ref_x, ref_y = ref.iloc[:, 0].values, ref.iloc[:, 1].values
        drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

        weakness_vec = analyze_weakness(ref_x, ref_y, drv_x, drv_y)

        st.subheader("Detected Driver Weakness Vector")
        st.write({
            "Apex": weakness_vec[0],
            "Slalom": weakness_vec[1],
            "Cornering": weakness_vec[2],
        })

        best_params = optimize_track(weakness_vec, lap_target)
        # gx, gy = generate_centerline_from_params(best_params, lap_target)
        gx, gy, focus = deform_reference_track(ref_x, ref_y, weakness_vec)
        # lx, ly, rx, ry = generate_track_boundaries(gx, gy, track_width)
        lx, ly, rx, ry = generate_track_boundaries(gx, gy)


        st.success(f"Practice Track Focus: **{focus.upper()}**")

        fig, ax = plt.subplots(figsize=(8, 7))
        # ax.fill(np.r_[lx, rx[::-1]], np.r_[ly, ry[::-1]],
        #         color="lightgray", alpha=0.7)
        ax.plot(lx, ly, "k")
        ax.plot(rx, ry, "k")
        ax.plot(gx, gy, "g--", label="Generated Centerline")
        ax.plot(ref_x, ref_y, "--", label="Reference (mapped)")

        ax.axis("equal")
        ax.legend()
        st.pyplot(fig)
