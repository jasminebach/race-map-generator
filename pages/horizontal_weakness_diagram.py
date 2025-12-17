# pages/weakness_extraction_explainability.py
import streamlit as st
from explainability.code_snippets import ANALYZE_CODE
from explainability.diagrams import render_pyvis
from explainability.diagrams import build_weakness_network


# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Weakness Extraction – Diagram & Code")

st.markdown("""
This page visualizes the **weakness extraction pipeline**
side by side with its **actual Python implementation**.

• Left: interactive pipeline diagram  
• Right: source code  
""")

left, right = st.columns([3, 2])

with left:
    html = render_pyvis(build_weakness_network())
    st.components.v1.html(html, height=700)

with right:
    st.code(ANALYZE_CODE, language="python")

# =====================================================
# Footer explanation
# =====================================================
st.markdown("---")
st.markdown("""
### How to read this

• **Top branch**: curvature-based apex analysis  
• **Middle branch**: distance alignment & indexing  
• **Bottom branch**: global cornering deviation  
• **Right**: normalized weakness vector  

Each diagram node corresponds directly to a variable in the code.
""")

# import streamlit as st
# from pyvis.network import Network
# import tempfile
# import os

# # =====================================================
# # Streamlit setup
# # =====================================================
# st.set_page_config(layout="wide")
# st.title("🧠 Weakness Extraction – Diagram & Code")

# st.markdown("""
# This page visualizes the **weakness extraction pipeline**
# side by side with its **actual Python implementation**.

# • Left: interactive pipeline diagram  
# • Right: source code  
# """)

# # =====================================================
# # Code to display (SOURCE OF TRUTH)
# # =====================================================
# ANALYZE_CODE = """
# def analyze_weakness(ref_x, ref_y, drv_x, drv_y):
#     dist, idx = compute_distance(ref_x, ref_y, drv_x, drv_y)
#     kappa = np.abs(curvature(ref_x, ref_y))

#     apex_mask = kappa > np.percentile(kappa, 75)

#     w_apex = np.mean(dist[np.isin(idx, np.where(apex_mask)[0])])
#     w_slalom = np.std(np.diff(dist))
#     w_corner = np.mean(dist)

#     w = np.array([w_apex, w_slalom, w_corner])
#     return w / (np.linalg.norm(w) + 1e-9)
# """

# # =====================================================
# # Build horizontal PyVis network
# # =====================================================
# def build_weakness_network():
#     net = Network(height="650px", width="100%", directed=True)

#     # Clean horizontal layout
#     net.toggle_physics(False)

#     COLORS = {
#         "input": "#97C2FC",
#         "process": "#D3D3D3",
#         "decision": "#FFB347",
#         "output": "#77DD77",
#     }

#     # -------------------------------------------------
#     # Nodes (fixed horizontal layout)
#     # -------------------------------------------------
#     nodes = [
#         # Inputs
#         ("ref_xy", "ref_x, ref_y", "input", 0, 180),
#         ("drv_xy", "drv_x, drv_y", "input", 0, 320),

#         # Geometry
#         ("distance", "compute distance\n(dist, idx)", "process", 300, 260),
#         ("kappa", "curvature\n(kappa)", "process", 300, 120),

#         # Mask
#         ("apex_mask", "apex mask\n(kappa > p75)", "decision", 600, 120),

#         # Weakness metrics
#         ("w_apex", "w_apex\nmean dist @ apex", "process", 900, 60),
#         ("w_slalom", "w_slalom\nstd(diff(dist))", "process", 900, 220),
#         ("w_corner", "w_corner\nmean(dist)", "process", 900, 380),

#         # Output
#         ("weakness_vec", "normalized\nweakness_vec", "output", 1200, 220),
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

#     # -------------------------------------------------
#     # Edges (exact mapping to code)
#     # -------------------------------------------------
#     edges = [
#         ("ref_xy", "distance"),
#         ("drv_xy", "distance"),

#         ("ref_xy", "kappa"),

#         ("kappa", "apex_mask"),

#         ("distance", "w_apex"),
#         ("apex_mask", "w_apex"),

#         ("distance", "w_slalom"),
#         ("distance", "w_corner"),

#         ("w_apex", "weakness_vec"),
#         ("w_slalom", "weakness_vec"),
#         ("w_corner", "weakness_vec"),
#     ]

#     for src, dst in edges:
#         net.add_edge(src, dst, arrows="to")

#     return net

# # =====================================================
# # Layout: SIDE BY SIDE
# # =====================================================
# left, right = st.columns([3, 2])

# with left:
#     st.subheader("📊 Weakness Extraction Diagram")

#     net = build_weakness_network()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#         net.save_graph(tmp.name)
#         html_path = tmp.name

#     st.components.v1.html(
#         open(html_path, "r", encoding="utf-8").read(),
#         height=700,
#         scrolling=True,
#     )

#     os.unlink(html_path)

# with right:
#     st.subheader("🧩 Python Implementation")
#     st.code(ANALYZE_CODE, language="python")

# # =====================================================
# # Footer explanation
# # =====================================================
# st.markdown("---")
# st.markdown("""
# ### How to read this

# • **Top branch**: curvature-based apex analysis  
# • **Middle branch**: distance alignment & indexing  
# • **Bottom branch**: global cornering deviation  
# • **Right**: normalized weakness vector  

# Each diagram node corresponds directly to a variable in the code.
# """)
