# train_track_generator.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import time, os

st.set_page_config(page_title="Practice Track Generator", layout="wide")
st.title("🏁 Practice Track Generator — Weakness Analyzer → Track")

# ------------------ Small geometry & generator utilities (based on your simulator) ------------------
def compute_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    n = np.hypot(dx, dy) + 1e-12
    nx = -dy / n
    ny = dx / n
    return nx, ny

def curvature(x,y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-12
    return (dx*ddy - dy*ddx) / denom

def speed_limit_from_kappa(kappa, max_corner_g=1.6):
    a_lat = max_corner_g * 9.81
    with np.errstate(divide='ignore'):
        vmax = np.sqrt(np.clip(a_lat / (np.abs(kappa) + 1e-12), 0.5, 50.0))
    vmax = gaussian_filter1d(vmax, sigma=8)
    return vmax

def place_cones(xc, yc, spacing, half_w):
    nx, ny = compute_normals(xc, yc)
    d = np.hypot(np.diff(xc), np.diff(yc))
    cum = np.hstack(([0.0], np.cumsum(d)))
    last = 0.0
    cones = []
    side = 1
    for i in range(len(xc)):
        if cum[i] - last >= spacing:
            cones.append((float(xc[i] + side*half_w*nx[i]), float(yc[i] + side*half_w*ny[i]), int(side)))
            side *= -1
            last = cum[i]
    return np.array(cones)

def generate_centerline(seed=None, lap_target=1000.0, bias=None):
    """
    bias: None or dict like {"focus":"slalom"} or {"focus":"apex"} to bias segment types.
    This re-uses your original random segment composition but tweaks probabilities for training.
    """
    if seed is None:
        seed = int(time.time() * 1000) % 2**31
    np.random.seed(seed)

    base_seg_types = ['straight','sweeper','constant','hairpin','short_straight']
    # default probabilities
    probs = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    if bias is not None:
        focus = bias.get("focus","")
        if focus == "slalom":
            # more short straights + small-radius turns
            probs = np.array([0.10, 0.15, 0.15, 0.15, 0.45])
        elif focus == "apex":
            # add hairpins and constant radius corners
            probs = np.array([0.15, 0.15, 0.30, 0.30, 0.10])
        elif focus == "braking":
            # straights and hairpins (deceleration practice)
            probs = np.array([0.35, 0.10, 0.15, 0.30, 0.10])
        elif focus == "smooth_steer":
            probs = np.array([0.15, 0.50, 0.20, 0.05, 0.10])
        elif focus == "throttle":
            probs = np.array([0.45, 0.20, 0.10, 0.05, 0.20])

    probs = probs / probs.sum()

    X_all = []
    Y_all = []
    heading = 0.0
    total_len = 0.0

    while total_len < lap_target * 0.98:
        stype = np.random.choice(base_seg_types, p=probs)
        if stype == 'straight':
            L = np.random.uniform(30, min(77, lap_target/6))
            n = max(8, int(L/1.0))
            xs = np.linspace(0, L, n)
            ys = np.zeros_like(xs)

        elif stype == 'short_straight':
            L = np.random.uniform(8, 25)
            n = max(6, int(L/1.0))
            xs = np.linspace(0, L, n)
            ys = np.zeros_like(xs)

        elif stype == 'sweeper':
            R = np.random.uniform(20, 60)
            ang = np.deg2rad(np.random.uniform(25,120))
            sign = 1 if np.random.rand() < 0.5 else -1
            n = max(12, int(R*abs(ang)/1.0))
            thetas = np.linspace(0, sign*ang, n)
            xs = R * np.sin(thetas)
            ys = R * (1 - np.cos(thetas)) * sign

        elif stype == 'constant':
            R = np.random.uniform(12,30)
            ang = np.deg2rad(np.random.uniform(30,100))
            sign = 1 if np.random.rand() < 0.5 else -1
            n = max(12, int(R*abs(ang)/1.0))
            thetas = np.linspace(0, sign*ang, n)
            xs = R * np.sin(thetas)
            ys = R * (1 - np.cos(thetas)) * sign

        else:  # hairpin
            R = np.random.uniform(4.5, 10.0)
            ang = np.deg2rad(np.random.uniform(100,220))
            sign = 1 if np.random.rand() < 0.5 else -1
            n = max(16, int(R*abs(ang)/0.5))
            thetas = np.linspace(0, sign*ang, n)
            xs = R * np.sin(thetas)
            ys = R * (1 - np.cos(thetas)) * sign

        c, s = np.cos(heading), np.sin(heading)
        rx = c * xs - s * ys
        ry = s * xs + c * ys

        if len(X_all) == 0:
            ox, oy = 0.0, 0.0
        else:
            ox, oy = X_all[-1][-1], Y_all[-1][-1]

        rx = rx + ox
        ry = ry + oy
        X_all.append(rx)
        Y_all.append(ry)

        dx = rx[-1] - rx[0]
        dy = ry[-1] - ry[0]
        heading = np.arctan2(dy, dx)
        total_len += np.sum(np.hypot(np.diff(rx), np.diff(ry)))

    X = np.hstack(X_all)
    Y = np.hstack(Y_all)

    try:
        tck, u = splprep([X, Y], s=0.0, per=True)
        u_fine = np.linspace(0, 1, 2000)
        Xs, Ys = splev(u_fine, tck)
    except Exception:
        Xs, Ys = X, Y

    Xs = np.array(Xs) - np.min(Xs) + 10.0
    Ys = np.array(Ys) - np.min(Ys) + 10.0
    return Xs, Ys

# ------------------ Simplified simulation metrics (reuse of your compute_metrics idea) ------------------
def compute_metrics(df_sim, xc, yc, cones, vmax, dt=0.02):
    # nearest mapping
    dists = np.hypot(df_sim['x'].values[:,None] - xc[None,:], df_sim['y'].values[:,None] - yc[None,:])
    nearest_idx = np.argmin(dists, axis=1)

    # Brake onset: use drop in vmax as proxy
    dv = np.diff(vmax)
    brakes_idx = np.where(dv < -0.5)[0]
    sim_brake_on = df_sim.index[df_sim['brake'] > 0.2].tolist()
    boe = None
    if len(brakes_idx)>0 and len(sim_brake_on)>0:
        t_ideal = brakes_idx[0] * dt
        t_driver = float(df_sim.loc[sim_brake_on[0], 'time'])
        boe = abs(t_driver - t_ideal)
    else:
        boe = np.nan

    # Steering behavior
    steer = df_sim['steer'].values
    steer_rate = np.abs(np.diff(steer) / np.diff(df_sim['time'].values + 1e-12))
    steer_osc = float(np.mean(steer_rate)) if steer_rate.size>0 else np.nan
    steer_jerk = float(np.mean(np.abs(np.diff(steer_rate)))) if steer_rate.size>1 else np.nan

    # Apex miss: measure distance to high-curvature points
    k = curvature(xc, yc)
    pk = np.percentile(np.abs(k), 80)
    apex_indices = np.where(np.abs(k) >= pk)[0]
    apex_dists = []
    for ai in apex_indices[::max(1, len(apex_indices)//6)]:
        ax, ay = xc[ai], yc[ai]
        d = np.min(np.hypot(df_sim['x']-ax, df_sim['y']-ay))
        apex_dists.append(d)
    apex_miss = float(np.mean(apex_dists)) if len(apex_dists)>0 else np.nan

    # Slalom RMS compared to cones
    slalom_rms = np.nan
    if cones.size>0:
        cones_xy = cones[:,0:2].astype(float)
        cinds = np.linspace(0, cones.shape[0]-1, min(30, cones.shape[0])).astype(int)
        dlist = []
        for ci in cinds:
            cx, cy = cones_xy[ci]
            dlist.append(np.min(np.hypot(df_sim['x']-cx, df_sim['y']-cy)))
        slalom_rms = float(np.sqrt(np.mean(np.array(dlist)**2)))

    # Throttle aggression
    thr = df_sim['throttle'].values
    if len(thr) > 2:
        thr_rate = np.diff(thr) / np.diff(df_sim['time'].values + 1e-12)
        tai = float(np.mean(np.maximum(0.0, thr_rate)))
    else:
        tai = np.nan

    return {
        'Brake Onset Error (s)': boe,
        'Steering rate (rad/s, mean)': steer_osc,
        'Steering jerk (rad/s^2, mean)': steer_jerk,
        'Apex miss (m)': apex_miss,
        'Slalom RMS (m)': slalom_rms,
        'Throttle Aggression (m/s^2)': tai
    }

# ------------------ Analyzer: decide which weakness(es) are present ------------------
def analyze_weaknesses(metrics, thresholds=None):
    # thresholds are tunable; defaults chosen empirically
    if thresholds is None:
        thresholds = {
            'apex_miss': 0.9,         # >0.9 m is poor
            'slalom_rms': 1.2,       # >1.2 m is poor
            'brake_onset': 0.7,      # >0.7 s late is poor
            'steer_rate': 0.6,       # >0.6 rad/s mean
            'throttle_aggr': 1.6,    # >1.6
        }

    weak = {}
    weak['apex'] = metrics.get('Apex miss (m)', np.nan) > thresholds['apex_miss']
    weak['slalom'] = metrics.get('Slalom RMS (m)', np.nan) > thresholds['slalom_rms']
    boe = metrics.get('Brake Onset Error (s)', np.nan)
    weak['braking'] = (boe is not None and not np.isnan(boe) and boe > thresholds['brake_onset'])
    weak['steering_smoothness'] = metrics.get('Steering rate (rad/s, mean)', np.nan) > thresholds['steer_rate']
    weak['throttle'] = metrics.get('Throttle Aggression (m/s^2)', np.nan) > thresholds['throttle_aggr']

    # Rank by severity (normalized difference to threshold where possible)
    scores = {}
    for k in weak.keys():
        if k == 'apex':
            val = metrics.get('Apex miss (m)', 0.0)
            scores[k] = (val - thresholds['apex_miss']) / (thresholds['apex_miss'] + 1e-12)
        elif k == 'slalom':
            val = metrics.get('Slalom RMS (m)', 0.0)
            scores[k] = (val - thresholds['slalom_rms']) / (thresholds['slalom_rms'] + 1e-12)
        elif k == 'braking':
            val = metrics.get('Brake Onset Error (s)', 0.0)
            scores[k] = (val - thresholds['brake_onset']) / (thresholds['brake_onset'] + 1e-12)
        elif k == 'steering_smoothness':
            val = metrics.get('Steering rate (rad/s, mean)', 0.0)
            scores[k] = (val - thresholds['steer_rate']) / (thresholds['steer_rate'] + 1e-12)
        elif k == 'throttle':
            val = metrics.get('Throttle Aggression (m/s^2)', 0.0)
            scores[k] = (val - thresholds['throttle_aggr']) / (thresholds['throttle_aggr'] + 1e-12)

    # Primary weakness = max positive score
    primary = max(scores, key=lambda k: scores[k])
    if scores[primary] <= 0:
        primary = None

    return weak, scores, primary

# ------------------ Map weakness -> generator bias ------------------
def weakness_to_focus(primary):
    # mapping rules (simple): you can extend to more structured segment lists
    if primary == 'apex':
        return 'apex'
    if primary == 'slalom':
        return 'slalom'
    if primary == 'braking':
        return 'braking'
    if primary == 'steering_smoothness':
        return 'smooth_steer'
    if primary == 'throttle':
        return 'throttle'
    return None

# ------------------ Streamlit UI and orchestration ------------------
st.sidebar.header("Input / Options")
st.sidebar.markdown("Upload a centerline CSV and a simulated run CSV, or run the simulator first then upload.")
centerline_file = st.sidebar.file_uploader("Centerline CSV (optional)", type="csv")
driver_file = st.sidebar.file_uploader("Driver run CSV", type="csv")
lap_length = st.sidebar.number_input("Lap target length (m) for new track", min_value=400, max_value=5000, value=1000)
cone_spacing = st.sidebar.slider("Cone spacing (m)", 6.0, 20.0, 12.0)
track_width = st.sidebar.slider("Track width (m)", 3.0, 10.0, 4.5)
analyze_btn = st.sidebar.button("Analyze & Generate Practice Track")

st.markdown("## How it works")
st.write("""
1. Loads centerline + driver run.  
2. Recomputes metrics (apex miss, slalom RMS, brake-onset error, steering smoothness, throttle aggression).  
3. Detects dominant weakness and maps it to a **generation bias** (e.g., slalom bias → more short straights and cone alternations).  
4. Generates a new centerline biased to practice that weakness and shows recommended exercises and downloads.
""")

if analyze_btn:
    # minimal validation
    if driver_file is None:
        st.error("Please upload a driver run CSV (simulated run). Optionally upload a centerline CSV; if not provided a default centerline will be generated.")
    else:
        # load or make base centerline
        if centerline_file is None:
            st.info("No centerline uploaded — making a default base centerline to compare against.")
            base_xc, base_yc = generate_centerline(seed=42, lap_target=lap_length)
        else:
            cl = pd.read_csv(centerline_file)
            base_xc = cl['x'].values
            base_yc = cl['y'].values

        df_sim = pd.read_csv(driver_file)
        # safety: ensure time column exists
        if 'time' not in df_sim.columns:
            df_sim['time'] = np.linspace(0, len(df_sim)*0.02, len(df_sim))

        # compute cones and vmax for mapping metrics
        cones = place_cones(base_xc, base_yc, cone_spacing, track_width/2.0)
        kappa = curvature(base_xc, base_yc)
        vmax = speed_limit_from_kappa(kappa)

        metrics = compute_metrics(df_sim, base_xc, base_yc, cones, vmax)
        st.subheader("Computed Metrics")
        for k,v in metrics.items():
            st.write(f"**{k}:** {v}")

        weak, scores, primary = analyze_weaknesses(metrics)
        st.subheader("Detected Weaknesses")
        st.write(weak)
        st.write("Severity scores (normalized):")
        st.write({k: float(f"{v:.3f}") for k,v in scores.items()})

        if primary is None:
            st.success("No dominant weakness detected — driver appears within thresholds. Generated track will be balanced.")
            focus = None
        else:
            st.warning(f"Primary weakness detected: **{primary}** (this will bias the generated track toward training this skill).")
            focus = weakness_to_focus(primary)

        # generate new track biased by focus
        new_xc, new_yc = generate_centerline(seed=None, lap_target=lap_length, bias={"focus": focus} if focus else None)
        new_cones = place_cones(new_xc, new_yc, cone_spacing, track_width/2.0)

        # Visualize
        col1, col2 = st.columns([3,1])
        with col1:
            fig, ax = plt.subplots(figsize=(10,8))
            ax.plot(base_xc, base_yc, '--', linewidth=2.0, label='Reference centerline')
            ax.plot(new_xc, new_yc, '-', linewidth=2.0, label='Generated practice centerline')
            ax.scatter(new_cones[:,0], new_cones[:,1], s=30, label='Cones', alpha=0.8)
            ax.set_aspect('equal')
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Practice plan (text)")
            if primary is None:
                st.write("Balanced session: mix of sweepers, straights, and a few hairpins.")
            elif primary == 'apex':
                st.write("- Focus: Apex accuracy drills\n- Practice exercises: multiple hairpins, double-apex corners, radius-constant corners\n- Reps: 6–10 hairpin entries")
            elif primary == 'slalom':
                st.write("- Focus: Cone-to-cone slaloms\n- Practice exercises: short-straight slalom sections with alternating cones; tight spacing 6–9 m\n- Reps: 6 slalom runs")
            elif primary == 'braking':
                st.write("- Focus: Braking timing and modulation\n- Practice exercises: long straights into decreasing-radius hairpins; 3–4 braking markers\n- Reps: 5–8 strong decel entries")
            elif primary == 'steering_smoothness':
                st.write("- Focus: Smooth steering on sweepers\n- Practice exercises: long radius sweepers at varying speeds, aim for low steering rate\n- Reps: 4–6 sweepers")
            elif primary == 'throttle':
                st.write("- Focus: Throttle modulation on exit\n- Practice exercises: short straights after medium corners; progressive throttle ramps\n- Reps: 6–10 exits")
            else:
                st.write("Generic practice session.")

            st.write("")
            st.subheader("Downloads")
            st.download_button("Download generated centerline CSV",
                               pd.DataFrame({'x':new_xc,'y':new_yc}).to_csv(index=False),
                               file_name='practice_centerline.csv')
            st.download_button("Download generated cones CSV",
                               pd.DataFrame(new_cones, columns=['x','y','side']).to_csv(index=False),
                               file_name='practice_cones.csv')

            st.markdown("---")
            st.write("Tip: feed the generated practice track into your simulator and re-run the driver. Iterate until the weakness metric improves!")

else:
    st.write("Configure inputs in the sidebar. Upload a simulated run to analyze, then click **Analyze & Generate Practice Track**.")
