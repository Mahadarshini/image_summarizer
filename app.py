"""
app.py — Streamlit web UI for the AI-Based Image Difference Detection system.

Run:
    streamlit run app.py
"""

import os
import sys
import tempfile

import streamlit as st
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocessing import preprocess_pair, SUPPORTED_EXTENSIONS
from diff_detector import detect_differences
from visualize import draw_bounding_boxes, make_overlay, make_mask_visual, make_side_by_side
from statistics_module import compute_statistics
from summary import generate_summary

st.set_page_config(page_title="AI Image Difference Detector", layout="wide")


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def save_upload(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}. Use JPG, JPEG, or PNG.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


st.title("🔍 AI-Based Image Difference Detector")
st.caption(
    "Upload two images to detect, visualize, and summarize the differences between them."
)

# --- FR-1: Upload ---
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Image A (Reference)", type=["jpg", "jpeg", "png"], key="a")
with col2:
    file_b = st.file_uploader("Image B (Comparison)", type=["jpg", "jpeg", "png"], key="b")

with st.sidebar:
    st.header("Settings")
    align = st.checkbox("Auto-align images (feature-based registration)", value=False,
                         help="Enable if the two photos were taken from slightly "
                              "different angles/positions. Leave off for screenshots, "
                              "scans, or already-aligned image pairs.")
    threshold_method = st.selectbox("Threshold method", ["otsu", "fixed"], index=0)
    fixed_threshold = st.slider("Fixed threshold", 10, 150, 30, disabled=(threshold_method == "otsu"))
    morph_kernel_size = st.slider("Noise-cleanup kernel size", 3, 15, 5, step=2)
    min_region_area = st.slider("Minimum region area (px)", 20, 3000, 150, step=10,
                                 help="Ignore changed blobs smaller than this many pixels.")
    use_llm = st.checkbox("Use Claude API for a richer summary", value=False,
                           help="Requires ANTHROPIC_API_KEY to be set in the environment.")

if file_a and file_b:
    if st.button("Compare Images", type="primary"):
        with st.spinner("Processing..."):
            path_a = save_upload(file_a)
            path_b = save_upload(file_b)

            try:
                img_a, img_b = preprocess_pair(path_a, path_b, do_alignment=align)
                result = detect_differences(
                    img_a, img_b,
                    min_region_area=min_region_area,
                    threshold_method=threshold_method,
                    fixed_threshold=fixed_threshold,
                    morph_kernel_size=morph_kernel_size,
                )
                stats = compute_statistics(result)
                summary_text = generate_summary(stats, use_llm=use_llm)

                boxed_b = draw_bounding_boxes(img_b, result)
                overlay = make_overlay(img_b, result)
                mask_visual = make_mask_visual(result)
                side_by_side = make_side_by_side(img_a, img_b)

                st.success(
                    f"Detected {stats.num_changed_regions} changed region(s) — "
                    f"{stats.percent_changed}% of the image."
                )

                # --- FR-6: AI Summary ---
                st.subheader("📝 AI-Generated Summary")
                st.info(summary_text)

                # --- FR-5: Statistics ---
                st.subheader("📊 Difference Statistics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Changed Regions", stats.num_changed_regions)
                m2.metric("% Image Changed", f"{stats.percent_changed}%")
                m3.metric("Changed Area (px)", f"{stats.total_changed_area_px:,}")
                m4.metric("SSIM Similarity", stats.ssim_similarity_score)

                if stats.regions:
                    st.dataframe(stats.regions, use_container_width=True)

                # --- FR-4: Visualizations ---
                st.subheader("🖼️ Visualizations")
                t1, t2, t3, t4 = st.tabs(
                    ["Side-by-Side", "Highlighted Regions", "Heatmap Overlay", "Difference Mask"]
                )
                with t1:
                    st.image(cv2_to_pil(side_by_side), use_container_width=True)
                with t2:
                    st.image(cv2_to_pil(boxed_b), use_container_width=True,
                              caption="Image B with bounding boxes around changes")
                with t3:
                    st.image(cv2_to_pil(overlay), use_container_width=True)
                with t4:
                    st.image(cv2_to_pil(mask_visual), use_container_width=True)

                st.download_button(
                    "Download Statistics (JSON)",
                    data=str(stats.to_dict()),
                    file_name="statistics.json",
                    mime="application/json",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error processing images: {exc}")
            finally:
                os.unlink(path_a)
                os.unlink(path_b)
else:
    st.info("Upload both images above, then click **Compare Images**.")
