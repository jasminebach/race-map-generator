import streamlit as st
from pyvis.network import Network
import tempfile
import os

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Track Deformation Pipeline – Explainability Diagram")

st.markdown("""
This diagram visualizes the **full deformation pipeline**
derived from `deform_reference_track()`.

• Horizontal (left → right)  
• Nodes are draggable  
• Matches the handwritten design  
""")

# =====================================================
# Sidebar inputs (for context only)
# =====================================================
st.sidebar.header("🔧 Context Inputs")

weakness_vec = [
    st.sidebar.slider("Apex", 0.0, 1.0, 0.6),
    st.sidebar.slider("Slalom", 0.0, 1.0, 0.3),
    st.sidebar.slider("Cornering", 0.0, 1.0, 0.1),
]

max_offset = st.sidebar.slider("Max Offset", 0.5, 5.0, 3.0)

# =====================================================
# Build PyVis network
# =====================================================
def build_deformation_network():
    net = Network(
        height="720px",
        width="100%",
        directed=True
    )

    # physics OFF for clean horizontal layout
    net.toggle_physics(False)

    COLORS = {
        "input": "#97C2FC",
        "process": "#D3D3D3",
        "decision": "#FFB347",
        "output": "#77DD77",
    }

    # -------------------------------------------------
    # Nodes (fixed horizontal layout)
    # -------------------------------------------------
    nodes = [
        # Inputs
        ("ref_xy", "(ref_x, ref_y)", "input", 0, 180),
        ("weakness_vec", "weakness_vec", "input", 0, 420),
        ("max_offset", "max_offset", "input", 0, 260),

        # Geometry
        ("arc_length", "arc length (s)", "process", 300, 120),
        ("s_norm", "s_norm", "process", 600, 120),
        ("normals", "compute normals\n(nx, ny)", "process", 300, 260),
        ("kappa", "curvature\n(kappa)", "process", 300, 420),

        # Weakness logic
        ("argmax", "argmax", "decision", 300, 560),
        ("focus", "focus", "decision", 600, 560),
        ("strength", "strength", "process", 600, 420),

        ("apex_mask", "apex mask", "decision", 900, 360),
        ("slalom_mask", "slalom mask", "decision", 900, 440),
        ("corner_mask", "corner mask", "decision", 900, 520),

        ("mask", "mask", "decision", 1200, 440),

        # Deformation
        ("delta_raw", "delta_raw", "process", 900, 180),
        ("delta_smooth", "delta_smooth", "process", 1200, 180),

        # Outputs
        ("new_x", "new_x", "output", 1500, 120),
        ("new_y", "new_y", "output", 1500, 220),
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
    # Edges (exact mapping from your sketch)
    # -------------------------------------------------
    edges = [
        # Geometry
        ("ref_xy", "arc_length"),
        ("arc_length", "s_norm"),
        ("ref_xy", "normals"),
        ("ref_xy", "kappa"),

        # Weakness
        ("weakness_vec", "argmax"),
        ("argmax", "focus"),
        ("weakness_vec", "strength"),

        # Masking
        ("kappa", "apex_mask"),
        ("kappa", "slalom_mask"),
        ("kappa", "corner_mask"),

        ("focus", "apex_mask"),
        ("focus", "slalom_mask"),
        ("focus", "corner_mask"),

        ("apex_mask", "mask"),
        ("slalom_mask", "mask"),
        ("corner_mask", "mask"),

        # Deformation signal
        ("s_norm", "delta_raw"),
        ("strength", "delta_raw"),
        ("max_offset", "delta_raw"),
        ("mask", "delta_raw"),

        ("delta_raw", "delta_smooth"),

        # Projection
        ("delta_smooth", "new_x"),
        ("normals", "new_x"),
        ("delta_smooth", "new_y"),
        ("normals", "new_y"),
    ]

    for src, dst in edges:
        net.add_edge(src, dst, arrows="to")

    return net

# =====================================================
# Render diagram
# =====================================================
net = build_deformation_network()

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    net.save_graph(tmp.name)
    html_path = tmp.name

st.components.v1.html(
    open(html_path, "r", encoding="utf-8").read(),
    height=750,
    scrolling=True
)

os.unlink(html_path)

# =====================================================
# Explanation
# =====================================================
st.markdown("---")
st.markdown("""
### 🧩 How to Read This

• **Top lane**: geometry → deformation → projection  
• **Bottom lane**: weakness decision → masking  
• **Right**: final deformed trajectory  

This diagram exactly mirrors your handwritten pipeline.
""")
