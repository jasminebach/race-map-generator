import streamlit as st
import matplotlib.pyplot as plt
from utils.io import list_simulink_pairs, load_pair
from analysis.track_segmentation import segment_track
from analysis.driver_performance import analyze_driver_weakness

st.set_page_config(page_title="Driver Weakness Analysis", layout="wide")
st.title("🎯 Driver Weakness Detection")

pairs = list_simulink_pairs()
dataset = st.selectbox("Select Dataset", pairs.keys())

ref, drv = load_pair(pairs[dataset])

ref_x, ref_y = ref.iloc[:, 0].values, ref.iloc[:, 1].values
drv_x, drv_y = drv.iloc[:, 0].values, drv.iloc[:, 1].values

segments = segment_track(ref_x, ref_y)
scores, weakness = analyze_driver_weakness(
    ref_x, ref_y, drv_x, drv_y, segments
)

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
