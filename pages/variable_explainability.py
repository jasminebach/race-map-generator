import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(
    page_title="Variable-Level Explainability",
    layout="wide"
)

st.title("🧠 Variable-Level Block Diagram Explorer")
st.markdown("""
This page visualizes the **computational graph** behind  
`deform_reference_track()`.

• **Nodes** = variables  
• **Edges** = data flow  
• **Colors** = variable roles  
""")

# =====================================================
# Sidebar – Inputs
# =====================================================
st.sidebar.header("🔧 Input Variables")

apex = st.sidebar.slider("Apex Weakness", 0.0, 1.0, 0.6)
slalom = st.sidebar.slider("Slalom Weakness", 0.0, 1.0, 0.3)
cornering = st.sidebar.slider("Cornering Weakness", 0.0, 1.0, 0.1)

weakness_vec = [apex, slalom, cornering]

max_offset = st.sidebar.slider(
    "Max Offset (m)",
    min_value=0.5,
    max_value=5.0,
    value=3.0,
    step=0.1
)

# =====================================================
# Graph definition (VARIABLE-LEVEL)
# =====================================================
GRAPH_NODES = {
    # Inputs
    "ref_x": {"type": "input", "label": "reference x"},
    "ref_y": {"type": "input", "label": "reference y"},
    "weakness_vec": {"type": "input", "label": str(weakness_vec)},
    "max_offset": {"type": "input", "label": f"{max_offset} m"},

    # Geometry
    "s": {"type": "process", "label": "arc_length"},
    "s_norm": {"type": "process", "label": "normalize"},
    "kappa": {"type": "process", "label": "curvature"},
    "nx_ny": {"type": "process", "label": "normals"},

    # Decision
    "focus": {"type": "decision", "label": "argmax"},
    "mask": {"type": "decision", "label": "percentile mask"},
    "strength": {"type": "process", "label": "selected weakness"},

    # Deformation
    "delta_raw": {"type": "process", "label": "sin(2π·4·s_norm)"},
    "delta_smooth": {"type": "process", "label": "low-pass filter"},

    # Outputs
    "new_x": {"type": "output", "label": "deformed x"},
    "new_y": {"type": "output", "label": "deformed y"},
}

GRAPH_EDGES = [
    ("ref_x", "s"),
    ("ref_y", "s"),
    ("s", "s_norm"),

    ("ref_x", "kappa"),
    ("ref_y", "kappa"),

    ("ref_x", "nx_ny"),
    ("ref_y", "nx_ny"),

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
    ("nx_ny", "new_x"),

    ("delta_smooth", "new_y"),
    ("nx_ny", "new_y"),
]

# =====================================================
# Layout computation (NO pygraphviz)
# =====================================================
def compute_layout(G):
    """
    Safe layout that works without Graphviz / pygraphviz.
    """
    layer_map = {
        "input": 0,
        "process": 1,
        "decision": 2,
        "output": 3,
    }

    for n in G.nodes:
        G.nodes[n]["layer"] = layer_map.get(
            G.nodes[n]["type"], 1
        )

    return nx.multipartite_layout(G, subset_key="layer")

# =====================================================
# Plot function
# =====================================================
def plot_block_diagram(nodes, edges):
    G = nx.DiGraph()

    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    for u, v in edges:
        G.add_edge(u, v)

    pos = compute_layout(G)

    color_map = []
    for n in G.nodes:
        t = G.nodes[n]["type"]
        if t == "input":
            color_map.append("#A7C7E7")     # blue
        elif t == "decision":
            color_map.append("#FFB347")     # orange
        elif t == "output":
            color_map.append("#77DD77")     # green
        else:
            color_map.append("#D3D3D3")     # grey

    labels = {
        n: f"{n}\n{G.nodes[n].get('label','')}"
        for n in G.nodes
    }

    fig, ax = plt.subplots(figsize=(20, 14))

    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=color_map,
        node_size=3600,
        font_size=9,
        arrowsize=18,
        ax=ax
    )

    ax.set_title("Variable-Level Block Diagram")
    ax.axis("off")
    return fig

# =====================================================
# Main layout
# =====================================================
left, right = st.columns([3, 2])

with left:
    st.subheader("📊 Computational Graph")
    fig = plot_block_diagram(GRAPH_NODES, GRAPH_EDGES)
    st.pyplot(fig)

with right:
    st.subheader("🧠 Decision Summary")

    labels = ["Apex", "Slalom", "Cornering"]
    focus_idx = max(range(3), key=lambda i: weakness_vec[i])
    focus = labels[focus_idx]

    st.success(f"**Active Focus Mode:** {focus}")

    st.markdown("### Current Inputs")
    st.json({
        "weakness_vec": weakness_vec,
        "max_offset": max_offset
    })

    st.markdown("""
    **Interpretation**
    - `focus` selects which curvature regions are exaggerated
    - `strength` scales deformation amplitude
    - deformation is applied along track normals
    """)

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption(
    "Explainable Track Deformation • Variable-Level DAG • Streamlit"
)
