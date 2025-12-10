import streamlit as st
import numpy as np
import pandas as pd

#  ------------------ Utility: curvature (same method you used in simulator)  ------------------ 
def compute_curvature(x, y):
    dx  = np.gradient(x)
    dy  = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-12
    return (dx*ddy - dy*ddx) / denom


#  ------------------  Utility: distance between two sets of points ------------------ 
def point_dist(px, py, qx, qy):
    return np.sqrt((px - qx)**2 + (py - qy)**2)


#  ------------------  MAIN ANALYZER ------------------ 
def analyze_driver_weakness(centerline_csv, driver_csv,
                            kappa_threshold=0.02,
                            slalom_min_flips=6):

    # Load CSVs
    cl = pd.read_csv(centerline_csv)
    dr = pd.read_csv(driver_csv)

    xc = cl['x'].values
    yc = cl['y'].values
    x  = dr['x'].values
    y  = dr['y'].values

    # Curvature along reference centerline
    kappa = compute_curvature(xc, yc)

    # Identify turn types  
    left_idxs  = np.where(kappa > kappa_threshold)[0]
    right_idxs = np.where(kappa < -kappa_threshold)[0]

    #  ------------------ LEFT TURN SCORE ------------------ 
    left_errors = [
        np.min(point_dist(x, y, xc[i], yc[i])) for i in left_idxs
    ]
    left_score = np.mean(left_errors) if len(left_errors) > 0 else 0

    #  ------------------ RIGHT TURN SCORE  ------------------ 
    right_errors = [
        np.min(point_dist(x, y, xc[i], yc[i])) for i in right_idxs
    ]
    right_score = np.mean(right_errors) if len(right_errors) > 0 else 0

    #  ------------------ SLALOM SCORE (curvature sign flips)  ------------------ 
    signs = np.sign(kappa)
    flip_idxs = [
        i for i in range(1, len(signs)) if signs[i] * signs[i-1] < 0
    ]

    slalom_errors = []
    if len(flip_idxs) >= slalom_min_flips:
        for i in flip_idxs:
            slalom_errors.append(np.min(point_dist(x, y, xc[i], yc[i])))

    slalom_score = np.mean(slalom_errors) if len(slalom_errors) > 0 else 0

    #  ------------------ Determine dominant weakness ------------------ 
    error_dict = {
        "left_turn":  left_score,
        "right_turn": right_score,
        "slalom":     slalom_score
    }

    priority = max(error_dict, key=error_dict.get)

    return error_dict, priority



#  ------------------  SIMPLE TEXT TRACK GENERATOR  ------------------ 
def generate_text_practice_track(priority, length=12):

    track = []

    if priority == "left_turn":
        for _ in range(length):
            track.append("LEFT")
            if np.random.rand() < 0.20:
                track.append("STRAIGHT")

    elif priority == "right_turn":
        for _ in range(length):
            track.append("RIGHT")
            if np.random.rand() < 0.20:
                track.append("STRAIGHT")

    else:  # SLALOM
        for _ in range(length // 2):
            track.append("SLALOM LEFT→RIGHT")
            track.append("SLALOM RIGHT→LEFT")

    track.append("FINISH")
    return track



#  ------------------  PAGE UI ------------------ 
def main():
    st.title("🏁 Practice Trainer")
    st.write("Analyze your driving weaknesses and generate a simple text-only practice track.")

    st.subheader("Upload Data")

    centerline_file = st.file_uploader("Centerline CSV", type="csv")
    driver_file     = st.file_uploader("Driver Run CSV", type="csv")

    if centerline_file and driver_file:
        st.success("Files loaded!")

        if st.button("Analyze Weakness"):
            errors, priority = analyze_driver_weakness(centerline_file, driver_file)

            st.subheader("Weakness Scores")
            st.write(f"**Left Turns:**  {errors['left_turn']:.4f}")
            st.write(f"**Right Turns:** {errors['right_turn']:.4f}")
            st.write(f"**Slalom:**      {errors['slalom']:.4f}")

            st.subheader("Priority Weakness")
            if priority == "left_turn":
                st.write("**Primary Weakness:** Left Turns")
            elif priority == "right_turn":
                st.write("**Primary Weakness:** Right Turns")
            else:
                st.write("**Primary Weakness:** Slaloms")

            st.subheader("Generated Practice Track (Text-Only)")
            text_track = generate_text_practice_track(priority)

            st.code(" — ".join(text_track))


if __name__ == "__main__":
    main()
