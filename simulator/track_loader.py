import pandas as pd

def load_track_csv(path):
    """
    Load centerline CSV from SIMULINK export.

    Expected formats:
    - time, yRef, xRef
    - x, y
    """

    df = pd.read_csv(path)

    cols = [c.lower() for c in df.columns]

    if "xref" in cols and "yref" in cols:
        x = df[df.columns[cols.index("xref")]].values
        y = df[df.columns[cols.index("yref")]].values

    elif "x" in cols and "y" in cols:
        x = df[df.columns[cols.index("x")]].values
        y = df[df.columns[cols.index("y")]].values

    else:
        raise ValueError(f"Unrecognized track CSV format: {path}")

    return list(zip(x, y))
