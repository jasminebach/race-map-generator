import streamlit as st
from pyvis.network import Network
import tempfile
import os

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Deformation Generator — Explainability")

st.markdown("""
This page visualizes the system using **two complementary diagrams**:

**Diagram A** explains *how* the track is generated (continuous signal → geometry).  
**Diagram B** explains *why* specific regions are emphasized (discrete control logic).
""")

# =====================================================
# Sidebar (context only)
# =====================================================
st.sidebar.header("🔧 Contextual Inputs")

weakness_vec = [
    st.sidebar.slider("Apex Weakness", 0.0, 1.0, 0.6),
    st.sidebar.slider("Slalom Weakness", 0.0, 1.0, 0.3),
    st.sidebar.slider("Cornering Weakness", 0.0, 1.0, 0.1),
]

max_offset = st.sidebar.slider("Max Offset (m)", 0.5, 5.0, 3.0)

# =====================================================
# Shared color map
# =====================================================
COLORS = {
    "input": "#97C2FC",
    "signal": "#D3D3D3",
    "decision": "#FFB347",
    "output": "#77DD77",
}

# =====================================================
# Diagram A — Streamlined Deformation Generator
# =====================================================
st.subheader("📘 Diagram A — Streamlined Deformation Generator")
st.markdown("""
**Purpose:** Show how a **1D deformation signal** is generated and projected into **2D geometry**.  
No branching, no policy — only continuous computation.
""")

def build_diagram_A():
    net = Network(height="500px", width="100%", directed=True)
    net.toggle_physics(True)

    nodes = {
        "max_offset": ("input", "Max lateral offset"),
        "strength": ("input", "Selected weakness magnitude"),
        "s_norm": ("input", "Normalized arc length"),

        "delta_raw": ("signal", "Sinusoidal deformation Δ(s)"),
        "delta_smooth": ("signal", "Smoothed deformation field"),

        "nx_ny": ("signal", "Track normals"),

        "new_x": ("output", "Deformed X"),
        "new_y": ("output", "Deformed Y"),
    }

    edges = [
        ("max_offset", "delta_raw"),
        ("strength", "delta_raw"),
        ("s_norm", "delta_raw"),

        ("delta_raw", "delta_smooth"),

        ("delta_smooth", "new_x"),
        ("delta_smooth", "new_y"),
        ("nx_ny", "new_x"),
        ("nx_ny", "new_y"),
    ]

    for n, (typ, desc) in nodes.items():
        net.add_node(
            n,
            label=n,
            title=desc,
            color=COLORS[typ],
            shape="box"
        )

    for a, b in edges:
        net.add_edge(a, b, arrows="to")

    return net

net_A = build_diagram_A()

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    net_A.save_graph(tmp.name)
    html_A = tmp.name

st.components.v1.html(open(html_A, "r", encoding="utf-8").read(), height=520)
os.unlink(html_A)

st.markdown("""
**Interpretation**

- Δ(s) is a **1D learning signal**
- Geometry is modified only via **normal projection**
- Track topology and flow are preserved
""")

# =====================================================
# Diagram B — Weakness-Driven Control Logic
# =====================================================
st.subheader("📙 Diagram B — Weakness-Driven Control Logic")
st.markdown("""
**Purpose:** Explain *why* specific track regions are emphasized  
using **discrete policy decisions**.
""")

def build_diagram_B():
    net = Network(height="520px", width="100%", directed=True)
    net.toggle_physics(True)

    nodes = {
        "weakness_vec": ("input", f"Weakness vector {weakness_vec}"),
        "argmax": ("decision", "Select dominant weakness"),
        "focus": ("decision", "APEX / SLALOM / CORNERING"),

        "kappa": ("input", "Track curvature"),

        "apex_rule": ("decision", "κ > 75th percentile"),
        "slalom_rule": ("decision", "κ < 30th percentile"),
        "corner_rule": ("decision", "All regions"),

        "mask": ("output", "Region selection mask"),
        "strength": ("output", "Selected strength"),
    }

    edges = [
        ("weakness_vec", "argmax"),
        ("argmax", "focus"),

        ("focus", "apex_rule"),
        ("focus", "slalom_rule"),
        ("focus", "corner_rule"),

        ("kappa", "apex_rule"),
        ("kappa", "slalom_rule"),

        ("apex_rule", "mask"),
        ("slalom_rule", "mask"),
        ("corner_rule", "mask"),

        ("weakness_vec", "strength"),
    ]

    for n, (typ, desc) in nodes.items():
        net.add_node(
            n,
            label=n,
            title=desc,
            color=COLORS.get(typ, "#DDDDDD"),
            shape="diamond" if typ == "decision" else "box"
        )

    for a, b in edges:
        net.add_edge(a, b, arrows="to")

    return net

net_B = build_diagram_B()

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    net_B.save_graph(tmp.name)
    html_B = tmp.name

st.components.v1.html(open(html_B, "r", encoding="utf-8").read(), height=540)
os.unlink(html_B)

st.markdown("""
**Interpretation**

- Weakness vector defines **training policy**
- Curvature gates *where* deformation applies
- Output is a **mask + strength**, not geometry
""")

# =====================================================
# Closing explanation
# =====================================================
st.markdown("""
---

### 🧠 Why two diagrams?

- **Diagram A** shows *how* the deformation is generated (continuous, smooth)
- **Diagram B** shows *why* certain regions are emphasized (discrete, explainable)

Separating them improves interpretability and avoids misleading feedback loops.
""")


# import streamlit as st
# from pyvis.network import Network
# import tempfile
# import os

# # =====================================================
# # Streamlit setup
# # =====================================================
# st.set_page_config(layout="wide")
# st.title("🧠 Deformation Generator — Variable-Level Explainability")

# st.markdown("""
# This interactive diagram explains **how the training track is generated**.

# **Key idea**
# - Learning happens in **1D (along arc length s)**
# - Geometry happens in **2D (x–y projection)**

# Drag nodes to explore the data flow.
# """)

# # =====================================================
# # Sidebar inputs (for context only)
# # =====================================================
# st.sidebar.header("🔧 Inputs (Contextual)")

# weakness_vec = [
#     st.sidebar.slider("Apex Weakness", 0.0, 1.0, 0.6),
#     st.sidebar.slider("Slalom Weakness", 0.0, 1.0, 0.3),
#     st.sidebar.slider("Cornering Weakness", 0.0, 1.0, 0.1),
# ]

# max_offset = st.sidebar.slider("Max Offset (m)", 0.5, 5.0, 3.0)

# # =====================================================
# # Node definitions
# # =====================================================
# NODES = {
#     # -------- Inputs --------
#     "ref_x": ("input", "Reference X coordinates"),
#     "ref_y": ("input", "Reference Y coordinates"),
#     "weakness_vec": ("input", f"Weakness vector {weakness_vec}"),
#     "max_offset": ("input", f"Max deformation {max_offset} m"),

#     # -------- 1D Learning Signal --------
#     "s": ("signal", "Arc length parameter"),
#     "s_norm": ("signal", "Normalized arc length"),
#     "kappa": ("signal", "Track curvature"),
#     "focus": ("decision", "argmax(weakness_vec)"),
#     "mask": ("decision", "Region selection (percentile)"),
#     "strength": ("signal", "Selected weakness magnitude"),
#     "delta_raw": ("signal", "Sinusoidal deformation Δ(s)"),
#     "delta_smooth": ("signal", "Smoothed deformation field"),

#     # -------- 2D Geometry --------
#     "nx_ny": ("geometry", "Track normals"),
#     "new_x": ("output", "Deformed X coordinates"),
#     "new_y": ("output", "Deformed Y coordinates"),
# }

# # =====================================================
# # Edges (streamlined, explanation-focused)
# # =====================================================
# EDGES = [
#     # Arc-length space
#     ("ref_x", "s"), ("ref_y", "s"),
#     ("s", "s_norm"),

#     # Curvature & gating
#     ("ref_x", "kappa"), ("ref_y", "kappa"),
#     ("kappa", "mask"),
#     ("weakness_vec", "focus"),
#     ("focus", "mask"),
#     ("weakness_vec", "strength"),

#     # Deformation signal
#     ("s_norm", "delta_raw"),
#     ("strength", "delta_raw"),
#     ("max_offset", "delta_raw"),
#     ("mask", "delta_raw"),
#     ("delta_raw", "delta_smooth"),

#     # Geometry projection
#     ("ref_x", "nx_ny"), ("ref_y", "nx_ny"),
#     ("delta_smooth", "new_x"),
#     ("delta_smooth", "new_y"),
#     ("nx_ny", "new_x"),
#     ("nx_ny", "new_y"),
# ]

# # =====================================================
# # Visual configuration
# # =====================================================
# COLOR_MAP = {
#     "input": "#97C2FC",
#     "signal": "#D3D3D3",
#     "decision": "#FFB347",
#     "geometry": "#CFCFCF",
#     "output": "#77DD77",
# }

# Y_LEVELS = {
#     "input": -400,
#     "signal": -100,
#     "geometry": 200,
#     "output": 400,
# }

# GROUP_LEVEL = {
#     "ref_x": "input",
#     "ref_y": "input",
#     "weakness_vec": "input",
#     "max_offset": "input",

#     "s": "signal",
#     "s_norm": "signal",
#     "kappa": "signal",
#     "focus": "signal",
#     "mask": "signal",
#     "strength": "signal",
#     "delta_raw": "signal",
#     "delta_smooth": "signal",

#     "nx_ny": "geometry",
#     "new_x": "output",
#     "new_y": "output",
# }

# # =====================================================
# # Build PyVis network
# # =====================================================
# def build_network():
#     net = Network(height="750px", width="100%", directed=True)
#     net.toggle_physics(True)

#     # ---- Section labels (non-interactive) ----
#     net.add_node(
#         "LAYER_SIGNAL",
#         label="1D Deformation Signal\n(Learning Space)",
#         shape="box",
#         color="#EEEEEE",
#         physics=False,
#         x=0,
#         y=-250
#     )

#     net.add_node(
#         "LAYER_GEOM",
#         label="2D Geometry Application\n(Projection Space)",
#         shape="box",
#         color="#EEEEEE",
#         physics=False,
#         x=0,
#         y=300
#     )

#     # ---- Nodes ----
#     for node, (typ, desc) in NODES.items():
#         net.add_node(
#             node,
#             label=node,
#             title=f"{node}<br>{desc}",
#             color=COLOR_MAP[typ],
#             shape="diamond" if typ == "decision" else "box",
#             y=Y_LEVELS[GROUP_LEVEL[node]]
#         )

#     # ---- Edges ----
#     for src, dst in EDGES:
#         net.add_edge(src, dst, arrows="to")

#     return net

# # =====================================================
# # Render network
# # =====================================================
# net = build_network()

# with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#     net.save_graph(tmp.name)
#     html_path = tmp.name

# st.components.v1.html(
#     open(html_path, "r", encoding="utf-8").read(),
#     height=780,
#     scrolling=True
# )

# os.unlink(html_path)

# # =====================================================
# # Explanation
# # =====================================================
# st.markdown("""
# ### 🧠 How to read this diagram

# **Top → Bottom flow**

# 1. **Inputs**
#    - Track geometry, weakness vector, and max offset

# 2. **1D Learning Signal**
#    - A deformation field Δ(s) is generated along arc length
#    - Encodes *what to train* and *how strongly*

# 3. **2D Geometry Application**
#    - Δ(s) is projected sideways using track normals
#    - Track topology and flow are preserved

# This separation is central to the method’s explainability.
# """)


# import streamlit as st
# from pyvis.network import Network
# import tempfile
# import os

# # =====================================================
# # Streamlit setup
# # =====================================================
# st.set_page_config(layout="wide")
# st.title("🧠 Variable-Level Explainability (Draggable Graph)")

# st.markdown("""
# This interactive graph represents the **variable-level data flow**
# of `deform_reference_track()`.

# • Nodes are **draggable**
# • Hover to inspect variables
# • Physics can be toggled
# """)

# # =====================================================
# # Sidebar inputs
# # =====================================================
# st.sidebar.header("🔧 Inputs")

# weakness_vec = [
#     st.sidebar.slider("Apex", 0.0, 1.0, 0.6),
#     st.sidebar.slider("Slalom", 0.0, 1.0, 0.3),
#     st.sidebar.slider("Cornering", 0.0, 1.0, 0.1),
# ]

# max_offset = st.sidebar.slider("Max Offset (m)", 0.5, 5.0, 3.0)

# # =====================================================
# # Graph specification
# # =====================================================
# NODES = {
#     "ref_x": ("input", "Reference X coordinates"),
#     "ref_y": ("input", "Reference Y coordinates"),
#     "weakness_vec": ("input", f"{weakness_vec}"),
#     "max_offset": ("input", f"{max_offset} m"),

#     "s": ("process", "Arc length"),
#     "s_norm": ("process", "Normalized arc length"),
#     "kappa": ("process", "Curvature"),
#     "nx_ny": ("process", "Normals"),

#     "focus": ("decision", "argmax(weakness_vec)"),
#     "mask": ("decision", "Curvature mask"),
#     "strength": ("process", "Selected weakness"),

#     "delta_raw": ("process", "Sinusoidal deformation"),
#     "delta_smooth": ("process", "Smoothed deformation"),

#     "new_x": ("output", "Deformed X"),
#     "new_y": ("output", "Deformed Y"),
# }

# EDGES = [
#     ("ref_x", "s"), ("ref_y", "s"),
#     ("s", "s_norm"),

#     ("ref_x", "kappa"), ("ref_y", "kappa"),
#     ("ref_x", "nx_ny"), ("ref_y", "nx_ny"),

#     ("weakness_vec", "focus"),
#     ("weakness_vec", "strength"),

#     ("kappa", "mask"),
#     ("focus", "mask"),

#     ("s_norm", "delta_raw"),
#     ("strength", "delta_raw"),
#     ("max_offset", "delta_raw"),
#     ("mask", "delta_raw"),

#     ("delta_raw", "delta_smooth"),
#     ("delta_smooth", "new_x"),
#     ("delta_smooth", "new_y"),
#     ("nx_ny", "new_x"),
#     ("nx_ny", "new_y"),
# ]

# COLOR_MAP = {
#     "input": "#97C2FC",
#     "process": "#D3D3D3",
#     "decision": "#FFB347",
#     "output": "#77DD77",
# }

# # =====================================================
# # Build PyVis network
# # =====================================================
# def build_network():
#     net = Network(height="700px", width="100%", directed=True)
#     net.toggle_physics(True)

#     for node, (typ, desc) in NODES.items():
#         net.add_node(
#             node,
#             label=node,
#             title=f"{node}<br>{desc}",
#             color=COLOR_MAP[typ],
#             shape="box" if typ != "decision" else "diamond"
#         )

#     for src, dst in EDGES:
#         net.add_edge(src, dst, arrows="to")

#     return net

# # =====================================================
# # Render
# # =====================================================
# net = build_network()

# with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#     net.save_graph(tmp.name)
#     html_path = tmp.name

# st.components.v1.html(
#     open(html_path, "r", encoding="utf-8").read(),
#     height=750,
#     scrolling=True
# )

# os.unlink(html_path)

# st.markdown("""
# ### ✅ Why PyVis?
# - True drag & drop
# - Physics-based layout
# - Ideal for exploratory explainability
# """)
