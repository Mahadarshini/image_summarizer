"""Generates docs/architecture_diagram.png using Graphviz."""

import os
from graphviz import Digraph

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
os.makedirs(OUT_DIR, exist_ok=True)

g = Digraph("architecture", format="png")
g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", splines="ortho")
g.attr("node", fontname="Helvetica", fontsize="11", shape="box", style="rounded,filled",
       fillcolor="#EEF3FF", color="#3B5BDB", penwidth="1.5")
g.attr("edge", fontname="Helvetica", fontsize="9", color="#495057")

g.node("ui", "User Interface\n(Streamlit app.py / CLI main.py)\nFR-1: Image Upload", fillcolor="#FFF3BF", color="#F08C00")

with g.subgraph(name="cluster_pre") as c:
    c.attr(label="Preprocessing  (src/preprocessing.py)", style="rounded", color="#868E96")
    c.node("validate", "Validate\n(format, integrity)")
    c.node("resize", "Resize to\ncommon resolution")
    c.node("align", "Align / Register\n(ORB + Homography)")
    c.node("normalize", "Normalize illumination\n(CLAHE)")
    c.edge("validate", "resize")
    c.edge("resize", "align")
    c.edge("align", "normalize")

with g.subgraph(name="cluster_detect") as c:
    c.attr(label="Difference Detection  (src/diff_detector.py)  — FR-3", style="rounded", color="#868E96")
    c.node("ssim", "SSIM +\nPixel Diff")
    c.node("thresh", "Threshold\n(Otsu / Fixed)")
    c.node("morph", "Morphological\nCleanup")
    c.node("contours", "Contour Detection\n-> Changed Regions")
    c.edge("ssim", "thresh")
    c.edge("thresh", "morph")
    c.edge("morph", "contours")

with g.subgraph(name="cluster_out") as c:
    c.attr(label="Outputs", style="rounded", color="#868E96")
    c.node("viz", "Visualization\n(src/visualize.py) — FR-4\nboxes / heatmap / mask / side-by-side")
    c.node("stats", "Statistics\n(src/statistics_module.py) — FR-5\ncounts, %, area, coords")
    c.node("summary", "AI Summary\n(src/summary.py) — FR-6\nrule-based or Claude API")

g.node("results", "Results Panel\n(images, stats.json, summary.txt)", fillcolor="#D3F9D8", color="#2F9E44")

g.edge("ui", "validate")
g.edge("normalize", "ssim")
g.edge("contours", "viz")
g.edge("contours", "stats")
g.edge("stats", "summary")
g.edge("viz", "results")
g.edge("summary", "results")
g.edge("stats", "results")

g.render(os.path.join(OUT_DIR, "architecture_diagram"), cleanup=True)
print("Wrote", os.path.join(OUT_DIR, "architecture_diagram.png"))
