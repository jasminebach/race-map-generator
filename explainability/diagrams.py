# explainability/diagrams.py
from pyvis.network import Network

# --------------------------------------------------
# Shared renderer
# --------------------------------------------------
def render_pyvis(net):
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()
    os.unlink(tmp.name)
    return html


# --------------------------------------------------
# Weakness extraction diagram
# --------------------------------------------------
def build_weakness_network():
    net = Network(height="650px", width="100%", directed=True)
    net.toggle_physics(False)

    COLORS = {
        "input": "#97C2FC",
        "process": "#D3D3D3",
        "decision": "#FFB347",
        "output": "#77DD77",
    }

    nodes = [
        ("ref_xy", "ref_x, ref_y", "input", 0, 180),
        ("drv_xy", "drv_x, drv_y", "input", 0, 320),

        ("distance", "compute distance\n(dist, idx)", "process", 300, 260),
        ("kappa", "curvature\n(kappa)", "process", 300, 120),

        ("apex_mask", "apex mask\n(kappa > p75)", "decision", 600, 120),

        ("w_apex", "w_apex\nmean(dist @ apex)", "process", 900, 60),
        ("w_slalom", "w_slalom\nstd(diff(dist))", "process", 900, 220),
        ("w_corner", "w_corner\nmean(dist)", "process", 900, 380),

        ("weakness_vec", "normalized\nweakness_vec", "output", 1200, 220),
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
            shape="box" if typ != "decision" else "diamond",
        )

    edges = [
        ("ref_xy", "distance"),
        ("drv_xy", "distance"),
        ("ref_xy", "kappa"),
        ("kappa", "apex_mask"),

        ("distance", "w_apex"),
        ("apex_mask", "w_apex"),

        ("distance", "w_slalom"),
        ("distance", "w_corner"),

        ("w_apex", "weakness_vec"),
        ("w_slalom", "weakness_vec"),
        ("w_corner", "weakness_vec"),
    ]

    for src, dst in edges:
        net.add_edge(src, dst, arrows="to")

    return net


def build_deformation_network():
    net = Network(height="720px", width="100%", directed=True)
    net.toggle_physics(False)

    COLORS = {
        "input": "#97C2FC",
        "process": "#D3D3D3",
        "decision": "#FFB347",
        "output": "#77DD77",
    }

    nodes = [
        ("ref_xy", "(ref_x, ref_y)", "input", 0, 180),
        ("weakness_vec", "weakness_vec", "input", 0, 420),
        ("max_offset", "max_offset", "input", 0, 260),

        ("arc_length", "arc length (s)", "process", 300, 120),
        ("s_norm", "s_norm", "process", 600, 120),
        ("normals", "compute normals\n(nx, ny)", "process", 300, 260),
        ("kappa", "curvature\n(kappa)", "process", 300, 420),

        ("argmax", "argmax", "decision", 300, 560),
        ("focus", "focus", "decision", 600, 560),
        ("strength", "strength", "process", 600, 420),

        ("apex_mask", "apex mask", "decision", 900, 360),
        ("slalom_mask", "slalom mask", "decision", 900, 440),
        ("corner_mask", "corner mask", "decision", 900, 520),

        ("mask", "mask", "decision", 1200, 440),

        ("delta_raw", "delta_raw", "process", 900, 180),
        ("delta_smooth", "delta_smooth", "process", 1200, 180),

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
            shape="box" if typ != "decision" else "diamond",
        )

    edges = [
        ("ref_xy", "arc_length"),
        ("arc_length", "s_norm"),
        ("ref_xy", "normals"),
        ("ref_xy", "kappa"),

        ("weakness_vec", "argmax"),
        ("argmax", "focus"),
        ("weakness_vec", "strength"),

        ("kappa", "apex_mask"),
        ("kappa", "slalom_mask"),
        ("kappa", "corner_mask"),

        ("focus", "apex_mask"),
        ("focus", "slalom_mask"),
        ("focus", "corner_mask"),

        ("apex_mask", "mask"),
        ("slalom_mask", "mask"),
        ("corner_mask", "mask"),

        ("s_norm", "delta_raw"),
        ("strength", "delta_raw"),
        ("max_offset", "delta_raw"),
        ("mask", "delta_raw"),

        ("delta_raw", "delta_smooth"),

        ("delta_smooth", "new_x"),
        ("normals", "new_x"),
        ("delta_smooth", "new_y"),
        ("normals", "new_y"),
    ]

    for src, dst in edges:
        net.add_edge(src, dst, arrows="to")

    return net