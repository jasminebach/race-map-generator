import streamlit as st
import matplotlib.pyplot as plt
from utils.io import list_simulink_pairs, load_pair
from analysis.track_segmentation import segment_track
from analysis.driver_performance import analyze_driver_weakness
from generation.practice_track_generator import generate_practice_track

st.set_page_config(page_title="Practice Track Generator", layout="wide")
st.title("🏁 Custom Practice Track Generator")

pairs = list_simulink_pairs()
dataset = st.selectbox("Select Dataset", pairs.keys())
lap_target = st.slider("Practice Lap Length (m)", 200, 1000, 600)

ref, drv = load_pair(pairs[dataset])
ref_x, ref_y = ref.iloc[:, 0].values, ref.iloc[:, 1].values
drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values


segments = segment_track(ref_x, ref_y)
scores, weakness = analyze_driver_weakness(
    ref_x, ref_y, drv_x, drv_y, segments
)

st.header("🎯 Driver Weakness Detection")

x, y = ref.iloc[:, 0].values, ref.iloc[:, 1].values

segments = segment_track(x, y)

fig, ax = plt.subplots(figsize=(9, 7))

color_map = {
    "straight": "gray",
    "left_turn": "blue",
    "right_turn": "red",
    "slalom": "purple"
}

for seg in segments:
    ax.plot(
        x[seg["start"]:seg["end"]],
        y[seg["start"]:seg["end"]],
        color=color_map[seg["label"]],
        linewidth=2
    )

ax.set_title("Track Sections")
ax.axis("equal")
st.pyplot(fig)

st.markdown("### Segment Summary")
st.write(
    {k: sum(1 for s in segments if s["label"] == k)
     for k in color_map}
)



st.header("🎯 Driver Weakness Detection")


st.subheader("Weakness Scores by Section")
st.json(scores)

st.success(f"Primary Weakness: **{weakness.upper()}**")

# Visualization
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(ref_x, ref_y, "k--", label="Centerline")
ax.plot(drv_x, drv_y, ".", alpha=0.4, label="Driver")
ax.axis("equal")
ax.legend()
st.pyplot(fig)


st.header("🏁 Custom Practice Track Generator")




st.info(f"Detected Weakness → **{weakness.upper()}**")

gx, gy = generate_practice_track(weakness, lap_target)

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(gx, gy, "g", linewidth=2, label="Generated Practice Track")
ax.axis("equal")
ax.legend()
st.pyplot(fig)
