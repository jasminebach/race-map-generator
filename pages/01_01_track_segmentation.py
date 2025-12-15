import streamlit as st
import matplotlib.pyplot as plt
from utils.io import list_simulink_pairs, load_pair
from analysis.track_segmentation import segment_track

st.set_page_config(page_title="Track Segmentation", layout="wide")
st.title("🧭 Track Section Classification")

pairs = list_simulink_pairs()
dataset = st.selectbox("Select Dataset", pairs.keys())

ref, _ = load_pair(pairs[dataset])
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
