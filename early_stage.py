# streamlit_stage1_2_simulator.py
# FSAE 1-lap generator + dynamic bicycle simulator + metrics + CSV save/load
# - If user uploads a CSV (time,x,y,v,steer,throttle,brake,yaw) the app will use it as the simulated run
# - Otherwise the app simulates a run and writes a CSV to /mnt/data and offers it for download

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import time, os

st.set_page_config(page_title="FSAE Lap Simulator + Metrics", layout="wide")
st.title("🏁 FSAE: Track + Dynamic Driver Simulator + Metrics")

# ------------------ Sidebar ------------------
st.sidebar.header('Track / Simulation')
lap_target_m = st.sidebar.number_input('Lap length (m)', min_value=600, max_value=2000, value=1000)
cone_spacing = st.sidebar.slider('Cone spacing (m)', 9.0, 15.0, 12.0)
track_width = st.sidebar.slider('Track width (m)', 4.5, 10.0, 4.5)
randomize = st.sidebar.checkbox('Randomize each generation', value=True)

st.sidebar.markdown('---')
st.sidebar.header('Simulator')
dt = st.sidebar.number_input('Simulator dt (s)', value=0.02, step=0.01)
wheelbase = st.sidebar.number_input('Wheelbase (m)', value=1.0, step=0.01)
mass = st.sidebar.number_input('Mass (kg)', value=300.0, step=10.0)
max_steer_deg = st.sidebar.slider('Max steer (deg)', 10, 45, 30)
max_acc = st.sidebar.number_input('Max acc (m/s^2)', value=4.0, step=0.5)
max_brake = st.sidebar.number_input('Max decel (m/s^2)', value=8.0, step=0.5)

st.sidebar.markdown('---')
st.sidebar.header('Driver behavior')
skill_level = st.sidebar.selectbox('Driver skill', ['Beginner','Intermediate','Advanced'], index=1)
if skill_level == 'Beginner':
    steer_noise = 2.5  # deg
    throttle_noise = 0.6
    follow_aggressiveness = 0.7
elif skill_level == 'Intermediate':
    steer_noise = 1.2
    throttle_noise = 0.3
    follow_aggressiveness = 0.9
else:
    steer_noise = 0.4
    throttle_noise = 0.12
    follow_aggressiveness = 1.05

st.sidebar.markdown('---')
st.sidebar.header('Use previous run?')
uploaded = st.sidebar.file_uploader('Upload simulated CSV to use as driver run (optional)', type=['csv'])

simulate_btn = st.sidebar.button('Generate track & simulate')

# ------------------ Utilities ------------------

def generate_centerline(seed=None, lap_target=1000.0):
    # Simple segment-based generator that produces ~lap_target meters of track
    if seed is None:
        seed = int(time.time() * 1000) % 2**31
    np.random.seed(seed)

    # segment types and probabilities (enforced constraints)
    seg_types = ['straight','sweeper','constant','hairpin','short_straight']
    probs = [0.25, 0.25, 0.2, 0.15, 0.15]

    X_all = []
    Y_all = []
    heading = 0.0
    total_len = 0.0

    while total_len < lap_target * 0.98:
        stype = np.random.choice(seg_types, p=probs)
        if stype == 'straight':
            L = np.random.uniform(30, min(77, lap_target/6))
            n = max(8, int(L/1.0))
            xs = np.linspace(0, L, n)
            ys = np.zeros_like(xs)
        elif stype == 'short_straight':
            L = np.random.uniform(15, 35)
            n = max(6, int(L/1.0))
            xs = np.linspace(0, L, n)
            ys = np.zeros_like(xs)
        elif stype == 'sweeper':
            R = np.random.uniform(15, 30)
            ang = np.deg2rad(np.random.uniform(25,90))
            sign = 1 if np.random.rand() < 0.5 else -1
            n = max(12, int(R*abs(ang)/1.0))
            thetas = np.linspace(0, sign*ang, n)
            xs = R * np.sin(thetas)
            ys = R * (1 - np.cos(thetas)) * sign
        elif stype == 'constant':
            R = np.random.uniform(15,27)
            ang = np.deg2rad(np.random.uniform(30,120))
            sign = 1 if np.random.rand() < 0.5 else -1
            n = max(12, int(R*abs(ang)/1.0))
            thetas = np.linspace(0, sign*ang, n)
            xs = R * np.sin(thetas)
            ys = R * (1 - np.cos(thetas)) * sign
        else: # hairpin
            R = np.random.uniform(4.5, 8.0)
            ang = np.deg2rad(np.random.uniform(110,230))
            sign = 1 if np.random.rand() < 0.5 else -1
            n = max(16, int(R*abs(ang)/0.5))
            thetas = np.linspace(0, sign*ang, n)
            xs = R * np.sin(thetas)
            ys = R * (1 - np.cos(thetas)) * sign

        # rotate and append
        # rotate by current heading
        c, s = np.cos(heading), np.sin(heading)
        rx = c * xs - s * ys
        ry = s * xs + c * ys
        # translate to current tip
        if len(X_all) == 0:
            ox, oy = 0.0, 0.0
        else:
            ox, oy = X_all[-1][-1], Y_all[-1][-1]
        rx = rx + ox
        ry = ry + oy
        X_all.append(rx)
        Y_all.append(ry)

        # update heading and total length (approx last segment)
        dx = rx[-1] - rx[0]
        dy = ry[-1] - ry[0]
        heading = np.arctan2(dy, dx)
        total_len += np.sum(np.hypot(np.diff(rx), np.diff(ry)))

    # assemble and smooth with periodic spline
    X = np.hstack(X_all)
    Y = np.hstack(Y_all)
    try:
        tck, u = splprep([X, Y], s=0.0, per=True)
        u_fine = np.linspace(0, 1, 2000)
        Xs, Ys = splev(u_fine, tck)
    except Exception:
        Xs, Ys = X, Y

    # shift to positive coords
    Xs = np.array(Xs) - np.min(Xs) + 10.0
    Ys = np.array(Ys) - np.min(Ys) + 10.0
    return Xs, Ys


def compute_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    n = np.hypot(dx, dy) + 1e-12
    nx = -dy / n
    ny = dx / n
    return nx, ny


def place_cones(xc, yc, spacing, half_w):
    nx, ny = compute_normals(xc, yc)
    d = np.hypot(np.diff(xc), np.diff(yc))
    cum = np.hstack(([0.0], np.cumsum(d)))
    last = 0.0
    cones = []
    side = 1
    for i in range(len(xc)):
        if cum[i] - last >= spacing:
            cones.append((xc[i] + side*half_w*nx[i], yc[i] + side*half_w*ny[i], side))
            side *= -1
            last = cum[i]
    return np.array(cones)

# ------------------ Ideal speed profile (curvature) ------------------

def curvature(x,y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-12
    k = (dx*ddy - dy*ddx) / denom
    return k


def speed_limit_from_kappa(kappa, max_corner_g=1.6):
    a_lat = max_corner_g * 9.81
    with np.errstate(divide='ignore'):
        vmax = np.sqrt(np.clip(a_lat / (np.abs(kappa) + 1e-12), 0.5, 50.0))
    vmax = gaussian_filter1d(vmax, sigma=8)
    return vmax

# ------------------ Dynamic bicycle model (simple) ------------------

def simulate_bicycle(xc, yc, vmax, dt=0.02, wheelbase=1.0, mass=300.0,
                     max_steer=np.deg2rad(30), max_acc=4.0, max_brake=8.0,
                     steer_noise_deg=1.0, throttle_noise=0.2, aggressiveness=1.0):
    # initial state placed at start of centerline, heading toward next point
    x, y = xc[0], yc[0]
    heading = np.arctan2(yc[2]-yc[0], xc[2]-xc[0])
    v = 2.0
    delta = 0.0

    Nx = []
    Ny = []
    V = []
    STEER = []
    STEER_CMD = []
    THR = []
    BRA = []
    YAW = []
    T = []

    # controller gains
    Kp_v = 1.2 * aggressiveness
    Kp_lat = 1.0

    nx, ny = compute_normals(xc, yc)
    path_len = np.sum(np.hypot(np.diff(xc), np.diff(yc)))

    # simulation loop until one lap
    traveled = 0.0
    target_idx = 0
    sim_time = 0.0
    max_steps = int(200000)

    for step in range(max_steps):
        # find closest centerline index
        dists = np.hypot(xc - x, yc - y)
        idx = int(np.argmin(dists))

        # desired speed
        v_des = vmax[idx]
        # throttle/brake simple proportional
        acc_cmd = np.clip(Kp_v * (v_des - v), -max_brake, max_acc)
        # add throttle noise
        acc_cmd += np.random.normal(0, throttle_noise)

        # steering controller: aim at lookahead point
        lookahead = np.clip(1.5 + 0.8 * v, 2.0, 12.0)
        # find point ahead by arc length
        cumd = np.hstack(([0.0], np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))))
        targ_idx = np.searchsorted(cumd, cumd[idx] + lookahead) % len(xc)
        tx, ty = xc[targ_idx], yc[targ_idx]
        angle_to_target = np.arctan2(ty - y, tx - x)
        alpha = angle_to_target - heading
        alpha = (alpha + np.pi) % (2*np.pi) - np.pi
        # geometric pure pursuit-ish
        steer_cmd = np.arctan2(2.0 * wheelbase * np.sin(alpha), max(0.1, lookahead))
        # add skill noise
        steer_cmd += np.deg2rad(np.random.normal(0, steer_noise_deg))
        # rate-limit and saturate
        max_delta_change = max_steer * dt * 10.0
        delta = np.clip(steer_cmd, -max_steer, max_steer)

        # vehicle dynamics integration (kinematic bicycle with simple longitudinal accel)
        yaw_rate = v * np.tan(delta) / wheelbase
        heading = (heading + yaw_rate * dt)
        x += v * np.cos(heading) * dt
        y += v * np.sin(heading) * dt
        v += acc_cmd * dt
        v = max(0.0, v)

        # log
        Nx.append(x); Ny.append(y); V.append(v); STEER.append(delta); STEER_CMD.append(steer_cmd);
        THR.append(max(0.0, acc_cmd)); BRA.append(max(0.0, -acc_cmd)); YAW.append(heading);
        T.append(sim_time)

        sim_time += dt
        if step > 100 and np.hypot(x - xc[0], y - yc[0]) < 3.0 and sim_time > 10.0:
            # finished one lap
            break

    df = pd.DataFrame({'time': T, 'x': Nx, 'y': Ny, 'v': V, 'steer': STEER, 'steer_cmd': STEER_CMD,
                       'throttle': THR, 'brake': BRA, 'yaw': YAW})
    return df

# ------------------ Metric computations ------------------

def compute_metrics(df_sim, xc, yc, cones, vmax):
    # map sim points to nearest centerline index
    dists = np.hypot(df_sim['x'].values[:,None] - xc[None,:], df_sim['y'].values[:,None] - yc[None,:])
    nearest_idx = np.argmin(dists, axis=1)

    # Brake Onset Error: find ideal braking times where vmax drops significantly
    dv = np.diff(vmax)
    brakes_idx = np.where(dv < -0.5)[0]
    # times in sim when driver applies brake>threshold
    sim_brake_on = df_sim.index[df_sim['brake'] > 0.2].tolist()
    boe = 0.0
    if len(brakes_idx)>0 and len(sim_brake_on)>0:
        t_ideal = brakes_idx[0] * 0.02
        t_driver = df_sim.loc[sim_brake_on[0], 'time']
        boe = abs(t_driver - t_ideal)

    # Steering rate / oscillation
    steer = df_sim['steer'].values
    steer_rate = np.abs(np.diff(steer) / np.diff(df_sim['time'].values))
    steer_osc = np.mean(steer_rate)
    steer_jerk = np.mean(np.abs(np.diff(steer_rate)))

    # Apex miss distance: find local curvature maxima as apexes, compute min distance
    k = curvature(xc, yc)
    pk = np.percentile(np.abs(k), 80)
    apex_indices = np.where(np.abs(k) >= pk)[0]
    apex_dists = []
    for ai in apex_indices[::max(1, len(apex_indices)//6)]:
        ax, ay = xc[ai], yc[ai]
        d = np.min(np.hypot(df_sim['x']-ax, df_sim['y']-ay))
        apex_dists.append(d)
    apex_miss = float(np.mean(apex_dists)) if len(apex_dists)>0 else 0.0

    # Slalom deviation: for cone-based slalom positions, compute lateral error to centerline
    slalom_rms = 0.0
    if cones.size>0:
        cones_xy = cones[:,0:2]
        # find sim points near cone chain and compute lateral distance to line between cones
        # approximate: RMS distance from sim path to cone positions (sampled)
        cinds = np.linspace(0, cones.shape[0]-1, min(30, cones.shape[0])).astype(int)
        dlist = []
        for ci in cinds:
            cx, cy = cones_xy[ci]
            dlist.append(np.min(np.hypot(df_sim['x']-cx, df_sim['y']-cy)))
        slalom_rms = float(np.sqrt(np.mean(np.array(dlist)**2)))

    # Throttle Aggression Index: mean positive throttle derivative
    thr = df_sim['throttle'].values
    if len(thr) > 2:
        thr_rate = np.diff(thr) / np.diff(df_sim['time'].values)
        tai = float(np.mean(np.maximum(0.0, thr_rate)))
    else:
        tai = 0.0

    return {
        'Brake Onset Error (s)': boe,
        'Steering rate (rad/s, mean)': float(steer_osc),
        'Steering jerk (rad/s^2, mean)': float(steer_jerk),
        'Apex miss (m)': apex_miss,
        'Slalom RMS (m)': slalom_rms,
        'Throttle Aggression (m/s^2)': tai
    }

# ------------------ Main flow ------------------
if simulate_btn:
    # seed management
    seed = None if randomize else 42
    Xc, Yc = generate_centerline(seed=seed, lap_target=lap_target_m)
    cones = place_cones(Xc, Yc, cone_spacing, track_width/2.0)
    kappa = curvature(Xc, Yc)
    vmax = speed_limit_from_kappa(kappa)

    # if user uploaded CSV, use it; otherwise simulate
    if uploaded is not None:
        df_sim = pd.read_csv(uploaded)
        st.success('Using uploaded simulated run for metrics and visualization.')
    else:
        # adjust driver noise and aggressiveness by skill
        if skill_level == 'Beginner':
            aggr = 0.8
            s_noise = 3.0
            t_noise = 0.6
        elif skill_level == 'Intermediate':
            aggr = 1.0
            s_noise = 1.2
            t_noise = 0.3
        else:
            aggr = 1.15
            s_noise = 0.5
            t_noise = 0.12
        df_sim = simulate_bicycle(Xc, Yc, vmax, dt=dt, wheelbase=wheelbase, mass=mass,
                                  max_steer=np.deg2rad(max_steer_deg), max_acc=max_acc, max_brake=max_brake,
                                  steer_noise_deg=s_noise, throttle_noise=t_noise, aggressiveness=aggr)
        # save CSV to /mnt/data with timestamp so next run you can upload it
        out_folder = "runs"
        os.makedirs(out_folder, exist_ok=True)
        fname = os.path.join(out_folder, f'simulated_run_{int(time.time())}.csv')
        df_sim.to_csv(fname, index=False)
        st.info(f'Simulated run saved to: `{fname}` — download below or upload next time to reuse.')

    # compute metrics
    metrics = compute_metrics(df_sim, Xc, Yc, cones, vmax)

    # Visualization
    col1, col2 = st.columns([3,1])
    with col1:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(Xc, Yc, '--', linewidth=2.5, color='black', label='Ideal centerline')
        ax.plot(df_sim['x'], df_sim['y'], '-', linewidth=2.0, color='red', label='Simulated run')
        if cones.size>0:
            ax.scatter(cones[:,0], cones[:,1], c='orange', s=40, label='Cones')
        # boundaries
        nx, ny = compute_normals(Xc, Yc)
        ax.plot(Xc + nx*(track_width/2.0), Yc + ny*(track_width/2.0), color='gray', alpha=0.5)
        ax.plot(Xc - nx*(track_width/2.0), Yc - ny*(track_width/2.0), color='gray', alpha=0.5)
        ax.set_aspect('equal')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader('Metrics')
        for k,v in metrics.items():
            st.metric(k, f"{v:.3f}")

    # time-series plots
    st.subheader('Time series telemetry')
    fig2, axs = plt.subplots(4,1, figsize=(10,8), sharex=True)
    axs[0].plot(df_sim['time'], df_sim['v']*3.6)
    axs[0].set_ylabel('Speed (km/h)')
    axs[1].plot(df_sim['time'], df_sim['throttle'])
    axs[1].set_ylabel('Throttle (m/s^2 approx)')
    axs[2].plot(df_sim['time'], df_sim['brake'])
    axs[2].set_ylabel('Brake (m/s^2)')
    axs[3].plot(df_sim['time'], df_sim['steer'])
    axs[3].set_ylabel('Steer (rad)')
    axs[3].set_xlabel('Time (s)')
    st.pyplot(fig2)

    # downloads
    st.download_button('⬇️ Download centerline CSV', pd.DataFrame({'x':Xc,'y':Yc}).to_csv(index=False), file_name='centerline.csv')
    st.download_button('⬇️ Download cones CSV', pd.DataFrame(cones, columns=['x','y','side']).to_csv(index=False), file_name='cones.csv')
    st.download_button('⬇️ Download simulated run CSV', df_sim.to_csv(index=False), file_name='simulated_run.csv')

else:
    st.write('Configure options in the sidebar and press **Generate track & simulate**. If you upload a CSV (past run) it will be used instead of simulating a new run.')
