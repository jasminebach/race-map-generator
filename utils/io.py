import os
import pandas as pd

DATA_DIR = "SIMULINK-DATA"

def list_simulink_pairs():
    files = os.listdir(DATA_DIR)
    pairs = {}

    for f in files:
        if "Centerline" in f and f.endswith(".csv"):
            key = f.replace(" - Centerline.csv", "")
            pairs[key] = {
                "centerline": os.path.join(DATA_DIR, f)
            }

    for f in files:
        if "Simulation" in f and f.endswith(".csv"):
            key = f.replace(" - Simulation.csv", "")
            if key in pairs:
                pairs[key]["simulation"] = os.path.join(DATA_DIR, f)

    return {k: v for k, v in pairs.items() if len(v) == 2}


def load_pair(pair):
    ref = pd.read_csv(pair["centerline"])
    drv = pd.read_csv(pair["simulation"])
    return ref, drv
