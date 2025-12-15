import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Centerline Viewer", layout="wide")
st.title("🏁 Centerline Geometry Viewer")

DATA_PATH = "SIMULINK-DATA/Simulation_Data_Export - Centerline.csv"

df = pd.read_csv(DATA_PATH)

st.subheader("Raw Data Preview")
st.dataframe(df.head(20), use_container_width=True)

x = df["xRef"].to_numpy()
y = df["yRef"].to_numpy()

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x, y, "-", linewidth=2)
ax.set_aspect("equal")
ax.set_title("Track Centerline")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)

st.pyplot(fig)

st.metric("Total Track Points", len(df))
