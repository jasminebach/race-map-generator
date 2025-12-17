import streamlit as st
from pyvis.network import Network
import tempfile
import os

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Weakness Extraction – Horizontal Diagram")

st.markdown("""
This diagram represents the **horizontal data flow**
for weakness extraction based on geometry and distance.

• Left → Right processing  
• Nodes are **draggable**  
• Hover nodes for meaning  
""")

# =====================================================
# Build horizontal PyVis network
# =====================================================
def build_horizontal_network():
    net = Network(
        height="650px",
        width="100%",
        directed=True
    )

    # Disable physics initially for clean layout
    net.toggle_physics(False)

    # -------------------------------------------------
    # Color scheme
    # -------------------------------------------------
    COLORS = {
        "input": "#97C2FC",
        "process": "#D3D3D3",
        "decision": "#FFB347",
        "output": "#77DD77",
    }

    # -------------------------------------------------
    # Nodes with fixed x positions (horizontal layout)
    # -------------------------------------------------
    nodes = [
        # Inputs
        ("ref_xy", "ref_x, ref_y", "input", 0, 200),
        ("dxy", "dx, dy", "input", 0, 350),

        # Geometry
        ("kappa", "curvature\n(kappa)", "process", 300, 200),
        ("distance", "compute\ndistance", "process", 300, 350),

        # Intermediate
        ("apex_mask", "apex mask", "decision", 600, 200),
        ("dist_idx", "dist, idx", "process", 600, 350),

        # Weakness branches
        ("w_apex", "w_apex", "output", 900, 120),
        ("w_slalom", "w_slalom", "output", 900, 260),
        ("w_corner", "w_corner", "output", 900, 400),

        # Final output
        ("weakness_vec", "weakness_vec", "output", 1200, 260),
    ]

    for node_id, label, typ, x, y in nodes:
        net.add_node(
            node_id,
            label=label,
            title=label,
            x=x,
            y=y,
            fixed=True,
            color=COLORS[typ],
            shape="box" if typ != "decision" else "diamond"
        )

    # -------------------------------------------------
    # Edges (exactly matching your sketch)
    # -------------------------------------------------
    edges = [
        ("ref_xy", "kappa"),
        ("dxy", "kappa"),

        ("ref_xy", "distance"),
        ("dxy", "distance"),

        ("kappa", "apex_mask"),
        ("distance", "dist_idx"),

        ("apex_mask", "w_apex"),
        ("apex_mask", "w_slalom"),
        ("apex_mask", "w_corner"),

        ("dist_idx", "w_slalom"),

        ("w_apex", "weakness_vec"),
        ("w_slalom", "weakness_vec"),
        ("w_corner", "weakness_vec"),
    ]

    for src, dst in edges:
        net.add_edge(src, dst, arrows="to")

    return net

# =====================================================
# Render network
# =====================================================
net = build_horizontal_network()

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    net.save_graph(tmp.name)
    html_path = tmp.name

st.components.v1.html(
    open(html_path, "r", encoding="utf-8").read(),
    height=700,
    scrolling=True
)

os.unlink(html_path)

# =====================================================
# Explanation panel
# =====================================================
st.markdown("---")
st.markdown("""
### 🧩 Interpretation

- **Left**: raw geometric signals  
- **Middle**: curvature & distance analysis  
- **Right**: weakness scores per driving skill  
- **Final**: aggregated `weakness_vec`

This diagram mirrors your handwritten logic exactly.
""")
