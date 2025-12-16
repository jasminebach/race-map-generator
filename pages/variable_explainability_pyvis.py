import streamlit as st
from pyvis.network import Network
import tempfile
import os

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Variable-Level Explainability (Draggable Graph)")

st.markdown("""
This interactive graph represents the **variable-level data flow**
of `deform_reference_track()`.

• Nodes are **draggable**
• Hover to inspect variables
• Physics can be toggled
""")

# =====================================================
# Sidebar inputs
# =====================================================
st.sidebar.header("🔧 Inputs")

weakness_vec = [
    st.sidebar.slider("Apex", 0.0, 1.0, 0.6),
    st.sidebar.slider("Slalom", 0.0, 1.0, 0.3),
    st.sidebar.slider("Cornering", 0.0, 1.0, 0.1),
]

max_offset = st.sidebar.slider("Max Offset (m)", 0.5, 5.0, 3.0)

# =====================================================
# Graph specification
# =====================================================
NODES = {
    "ref_x": ("input", "Reference X coordinates"),
    "ref_y": ("input", "Reference Y coordinates"),
    "weakness_vec": ("input", f"{weakness_vec}"),
    "max_offset": ("input", f"{max_offset} m"),

    "s": ("process", "Arc length"),
    "s_norm": ("process", "Normalized arc length"),
    "kappa": ("process", "Curvature"),
    "nx_ny": ("process", "Normals"),

    "focus": ("decision", "argmax(weakness_vec)"),
    "mask": ("decision", "Curvature mask"),
    "strength": ("process", "Selected weakness"),

    "delta_raw": ("process", "Sinusoidal deformation"),
    "delta_smooth": ("process", "Smoothed deformation"),

    "new_x": ("output", "Deformed X"),
    "new_y": ("output", "Deformed Y"),
}

EDGES = [
    ("ref_x", "s"), ("ref_y", "s"),
    ("s", "s_norm"),

    ("ref_x", "kappa"), ("ref_y", "kappa"),
    ("ref_x", "nx_ny"), ("ref_y", "nx_ny"),

    ("weakness_vec", "focus"),
    ("weakness_vec", "strength"),

    ("kappa", "mask"),
    ("focus", "mask"),

    ("s_norm", "delta_raw"),
    ("strength", "delta_raw"),
    ("max_offset", "delta_raw"),
    ("mask", "delta_raw"),

    ("delta_raw", "delta_smooth"),
    ("delta_smooth", "new_x"),
    ("delta_smooth", "new_y"),
    ("nx_ny", "new_x"),
    ("nx_ny", "new_y"),
]

COLOR_MAP = {
    "input": "#97C2FC",
    "process": "#D3D3D3",
    "decision": "#FFB347",
    "output": "#77DD77",
}

# =====================================================
# Build PyVis network
# =====================================================
def build_network():
    net = Network(height="700px", width="100%", directed=True)
    net.toggle_physics(True)

    for node, (typ, desc) in NODES.items():
        net.add_node(
            node,
            label=node,
            title=f"{node}<br>{desc}",
            color=COLOR_MAP[typ],
            shape="box" if typ != "decision" else "diamond"
        )

    for src, dst in EDGES:
        net.add_edge(src, dst, arrows="to")

    return net

# =====================================================
# Render
# =====================================================
net = build_network()

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    net.save_graph(tmp.name)
    html_path = tmp.name

st.components.v1.html(
    open(html_path, "r", encoding="utf-8").read(),
    height=750,
    scrolling=True
)

os.unlink(html_path)

st.markdown("""
### ✅ Why PyVis?
- True drag & drop
- Physics-based layout
- Ideal for exploratory explainability
""")
