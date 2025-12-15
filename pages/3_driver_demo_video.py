import streamlit as st
import subprocess
import sys
import os

# ======================================================
# Project root
# ======================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMULINK_ROOT = os.path.join(ROOT, "SIMULINK-DATA")

GEN_SCRIPT = os.path.join(ROOT, "generate_driver_video.py")

# ======================================================
# Page setup
# ======================================================
st.set_page_config(layout="wide")
st.title("🎬 Driver Simulation — Batch Video Viewer")

# ======================================================
# Discover batches dynamically
# ======================================================
batches = sorted([
    d for d in os.listdir(SIMULINK_ROOT)
    if os.path.isdir(os.path.join(SIMULINK_ROOT, d))
])

if not batches:
    st.error("No batches found in SIMULINK-DATA/")
    st.stop()

batch = st.selectbox("Select Batch", batches)
BATCH_DIR = os.path.join(SIMULINK_ROOT, batch)
VIDEO_PATH = os.path.join(BATCH_DIR, "driver_simulation.mp4")

# ======================================================
# Controls
# ======================================================
colA, colB = st.columns(2)

with colA:
    generate = st.button("🎬 Generate / Regenerate Video")

with colB:
    st.write("")

# ======================================================
# Generate video
# ======================================================
if generate:
    with st.spinner(f"Generating video for {batch}…"):
        try:
            subprocess.run(
                [
                    sys.executable,
                    GEN_SCRIPT,
                    "--batch-dir",
                    BATCH_DIR,
                ],
                cwd=ROOT,
                check=True,
            )
            st.success("Video generated successfully!")
        except subprocess.CalledProcessError as e:
            st.error("Video generation failed")
            st.exception(e)

# ======================================================
# Display video
# ======================================================
st.markdown("---")

if os.path.exists(VIDEO_PATH):
    st.subheader(f"▶ Playback — {batch}")
    st.video(VIDEO_PATH)
else:
    st.info("No video yet. Click **Generate / Regenerate Video**.")


# import streamlit as st
# import subprocess
# import sys
# import os
# import time

# # ==================================================
# # Project root (IMPORTANT)
# # ==================================================
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# VIDEO_PATH = os.path.join(ROOT, "driver_simulation.mp4")
# SCRIPT_PATH = os.path.join(ROOT, "generate_driver_video.py")

# # ==================================================
# # Page setup
# # ==================================================
# st.set_page_config(layout="wide")
# st.title("🚗 Driver Simulation (Matplotlib Video)")

# # ==================================================
# # Controls
# # ==================================================
# colA, colB = st.columns(2)

# with colA:
#     generate = st.button("🎬 Generate / Regenerate Video")

# with colB:
#     autoplay = st.checkbox("Autoplay", True)

# # ==================================================
# # Run generator
# # ==================================================
# if generate:
#     with st.spinner("Generating simulation video…"):
#         try:
#             subprocess.run(
#                 [sys.executable, SCRIPT_PATH],
#                 cwd=ROOT,            # 🔥 critical
#                 check=True
#             )
#             st.success("Video generated successfully!")
#         except subprocess.CalledProcessError as e:
#             st.error("Video generation failed")
#             st.exception(e)

# # ==================================================
# # Display video
# # ==================================================
# st.markdown("---")

# if os.path.exists(VIDEO_PATH):
#     st.subheader("▶ Simulation Playback")
#     st.video(VIDEO_PATH)
# else:
#     st.info("No video found. Click **Generate / Regenerate Video**.")


# import streamlit as st
# import subprocess
# import os
# import time

# # -------------------------------------------------
# # Page setup
# # -------------------------------------------------
# st.set_page_config(layout="wide")
# st.title("🚗 Driver Simulation (Matplotlib Video)")

# VIDEO_PATH = "driver_simulation.mp4"
# SCRIPT_PATH = "generate_driver_video.py"

# # -------------------------------------------------
# # Helper: check video exists
# # -------------------------------------------------
# def video_exists():
#     return os.path.exists(VIDEO_PATH)

# # -------------------------------------------------
# # Controls
# # -------------------------------------------------
# colA, colB = st.columns(2)

# with colA:
#     generate = st.button("🎬 Generate / Regenerate Video")

# with colB:
#     auto_play = st.checkbox("Autoplay after generation", True)

# # -------------------------------------------------
# # Trigger video generation
# # -------------------------------------------------
# if generate:
#     with st.spinner("Generating simulation video… this may take a moment"):
#         try:
#             subprocess.run(
#                 ["python", SCRIPT_PATH],
#                 check=True
#             )
#             st.success("Video generation completed!")
#         except subprocess.CalledProcessError as e:
#             st.error("Video generation failed")
#             st.exception(e)

# # -------------------------------------------------
# # Display video if available
# # -------------------------------------------------
# st.markdown("---")

# if video_exists():
#     st.subheader("▶ Simulation Playback")

#     if auto_play:
#         st.video(VIDEO_PATH, start_time=0)
#     else:
#         st.video(VIDEO_PATH)

# else:
#     st.info("No video found. Click **Generate / Regenerate Video** to create one.")
