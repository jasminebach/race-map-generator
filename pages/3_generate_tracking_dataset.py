import streamlit as st
import subprocess
import sys
import os

# ======================================================
# Project root
# ======================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMULINK_ROOT = os.path.join(ROOT, "SIMULINK-DATA")

GEN_SCRIPT = os.path.join(ROOT, "generate_tracking_dataset.py")

# ======================================================
# Page setup
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 Generate Derived Tracking Dataset")

# ======================================================
# Discover batches
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
OUTPUT_CSV = os.path.join(BATCH_DIR, "derived_tracking_dataset.csv")

# ======================================================
# Controls
# ======================================================
if st.button("⚙️ Generate Derived Dataset"):
    with st.spinner(f"Processing {batch}…"):
        try:
            subprocess.run(
                [
                    sys.executable,
                    GEN_SCRIPT,
                    "--batch-dir",
                    BATCH_DIR,
                ],
                cwd=ROOT,
                check=True
            )
            st.success("Derived dataset generated successfully!")
        except subprocess.CalledProcessError as e:
            st.error("Dataset generation failed")
            st.exception(e)

# ======================================================
# Preview result
# ======================================================
st.markdown("---")

if os.path.exists(OUTPUT_CSV):
    st.subheader("📄 Preview: derived_tracking_dataset.csv")
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    st.dataframe(df.head(20))
else:
    st.info("No derived dataset yet.")
