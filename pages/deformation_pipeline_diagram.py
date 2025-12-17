# pages/weakness_extraction_explainability.py
import streamlit as st
from explainability.code_snippets import ANALYZE_CODE
from explainability.diagrams import render_pyvis
from explainability.diagrams import build_deformation_network

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Track Deformation Pipeline – Diagram & Code")

st.markdown("""
This page shows the **deformation pipeline** side by side with its
**actual Python implementation**.

• Left: interactive pipeline diagram  
• Right: source code  
""")

left, right = st.columns([3, 2])

with left:
    html = render_pyvis(build_deformation_network())
    st.components.v1.html(html, height=700)

with right:
    st.code(ANALYZE_CODE, language="python")

# =====================================================
# Footer explanation
# =====================================================
st.markdown("---")
st.markdown("""
### Why this layout works

• Diagram shows **conceptual flow**  
• Code shows **exact implementation**  
• One-to-one correspondence between nodes and variables  
• Ideal for debugging, teaching, and thesis figures  
""")


# import streamlit as st
# from pyvis.network import Network
# import tempfile
# import os

# # =====================================================
# # Streamlit setup
# # =====================================================
# st.set_page_config(layout="wide")
# st.title("🧠 Track Deformation Pipeline – Diagram & Code")

# st.markdown("""
# This page shows the **deformation pipeline** side by side with its
# **actual Python implementation**.

# • Left: interactive pipeline diagram  
# • Right: source code  
# """)

# # =====================================================
# # Sidebar inputs (context only)
# # =====================================================
# st.sidebar.header("🔧 Context Inputs")

# weakness_vec = [
#     st.sidebar.slider("Apex", 0.0, 1.0, 0.6),
#     st.sidebar.slider("Slalom", 0.0, 1.0, 0.3),
#     st.sidebar.slider("Cornering", 0.0, 1.0, 0.1),
# ]

# max_offset = st.sidebar.slider("Max Offset", 0.5, 5.0, 3.0)

# # =====================================================
# # Code to display (SOURCE OF TRUTH)
# # =====================================================
# DEFORM_CODE = """
# def deform_reference_track(ref_x, ref_y, weakness_vec, max_offset=3.0):
#     s = arc_length(ref_x, ref_y)
#     s_norm = s / (s[-1] + 1e-9)

#     kappa = np.abs(curvature(ref_x, ref_y))
#     nx, ny = compute_normals(ref_x, ref_y)

#     labels = ["apex", "slalom", "cornering"]
#     focus = labels[np.argmax(weakness_vec)]

#     delta = np.zeros_like(s)

#     if focus == "apex":
#         mask = kappa > np.percentile(kappa, 75)
#         strength = weakness_vec[0]
#     elif focus == "slalom":
#         mask = kappa < np.percentile(kappa, 30)
#         strength = weakness_vec[1]
#     else:
#         mask = np.ones_like(kappa, dtype=bool)
#         strength = weakness_vec[2]

#     delta[mask] = max_offset * strength * np.sin(2 * np.pi * 4 * s_norm[mask])
#     delta = np.convolve(delta, np.ones(15) / 15, mode="same")

#     new_x = ref_x + delta * nx
#     new_y = ref_y + delta * ny

#     return new_x, new_y, delta, focus
# """

# # =====================================================
# # Build PyVis network
# # =====================================================
# def build_deformation_network():
#     net = Network(height="720px", width="100%", directed=True)
#     net.toggle_physics(False)

#     COLORS = {
#         "input": "#97C2FC",
#         "process": "#D3D3D3",
#         "decision": "#FFB347",
#         "output": "#77DD77",
#     }

#     nodes = [
#         ("ref_xy", "(ref_x, ref_y)", "input", 0, 180),
#         ("weakness_vec", "weakness_vec", "input", 0, 420),
#         ("max_offset", "max_offset", "input", 0, 260),

#         ("arc_length", "arc length (s)", "process", 300, 120),
#         ("s_norm", "s_norm", "process", 600, 120),
#         ("normals", "compute normals\n(nx, ny)", "process", 300, 260),
#         ("kappa", "curvature\n(kappa)", "process", 300, 420),

#         ("argmax", "argmax", "decision", 300, 560),
#         ("focus", "focus", "decision", 600, 560),
#         ("strength", "strength", "process", 600, 420),

#         ("apex_mask", "apex mask", "decision", 900, 360),
#         ("slalom_mask", "slalom mask", "decision", 900, 440),
#         ("corner_mask", "corner mask", "decision", 900, 520),

#         ("mask", "mask", "decision", 1200, 440),

#         ("delta_raw", "delta_raw", "process", 900, 180),
#         ("delta_smooth", "delta_smooth", "process", 1200, 180),

#         ("new_x", "new_x", "output", 1500, 120),
#         ("new_y", "new_y", "output", 1500, 220),
#     ]

#     for node_id, label, typ, x, y in nodes:
#         net.add_node(
#             node_id,
#             label=label,
#             title=label,
#             x=x,
#             y=y,
#             fixed=True,
#             color=COLORS[typ],
#             shape="box" if typ != "decision" else "diamond",
#         )

#     edges = [
#         ("ref_xy", "arc_length"),
#         ("arc_length", "s_norm"),
#         ("ref_xy", "normals"),
#         ("ref_xy", "kappa"),

#         ("weakness_vec", "argmax"),
#         ("argmax", "focus"),
#         ("weakness_vec", "strength"),

#         ("kappa", "apex_mask"),
#         ("kappa", "slalom_mask"),
#         ("kappa", "corner_mask"),

#         ("focus", "apex_mask"),
#         ("focus", "slalom_mask"),
#         ("focus", "corner_mask"),

#         ("apex_mask", "mask"),
#         ("slalom_mask", "mask"),
#         ("corner_mask", "mask"),

#         ("s_norm", "delta_raw"),
#         ("strength", "delta_raw"),
#         ("max_offset", "delta_raw"),
#         ("mask", "delta_raw"),

#         ("delta_raw", "delta_smooth"),

#         ("delta_smooth", "new_x"),
#         ("normals", "new_x"),
#         ("delta_smooth", "new_y"),
#         ("normals", "new_y"),
#     ]

#     for src, dst in edges:
#         net.add_edge(src, dst, arrows="to")

#     return net

# # =====================================================
# # Layout: SIDE BY SIDE
# # =====================================================
# left, right = st.columns([3, 2])

# with left:
#     st.subheader("📊 Deformation Pipeline Diagram")

#     net = build_deformation_network()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#         net.save_graph(tmp.name)
#         html_path = tmp.name

#     st.components.v1.html(
#         open(html_path, "r", encoding="utf-8").read(),
#         height=750,
#         scrolling=True,
#     )

#     os.unlink(html_path)

# with right:
#     st.subheader("🧩 Python Implementation")
#     st.code(DEFORM_CODE, language="python")

# # =====================================================
# # Footer explanation
# # =====================================================
# st.markdown("---")
# st.markdown("""
# ### Why this layout works

# • Diagram shows **conceptual flow**  
# • Code shows **exact implementation**  
# • One-to-one correspondence between nodes and variables  
# • Ideal for debugging, teaching, and thesis figures  
# """)
