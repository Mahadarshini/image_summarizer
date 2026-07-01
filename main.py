"""
main.py — CLI entry point for the AI-Based Image Difference Detection system.

Usage:
    python main.py --image_a path/to/A.png --image_b path/to/B.png --outdir outputs/

Produces (all required "Expected Outputs" from the spec):
    1. original_A.png
    2. original_B.png
    3. diff_visualization.png   (side-by-side)
    4. highlighted_regions.png  (Image B with bounding boxes)
    5. heatmap_overlay.png
    6. difference_mask.png
    7. summary_panel.png        (2x2 combined report image)
    8. statistics.json
    9. summary.txt              (AI-generated change summary paragraph)
"""

import argparse
import json
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cv2

from preprocessing import preprocess_pair
from diff_detector import detect_differences
from visualize import (
    draw_bounding_boxes,
    make_overlay,
    make_mask_visual,
    make_side_by_side,
    make_summary_panel,
)
from statistics_module import compute_statistics
from summary import generate_summary


def run_pipeline(image_a_path: str, image_b_path: str, outdir: str,
                  min_region_area: int = 150, align: bool = True,
                  use_llm_summary: bool = False, threshold_method: str = "otsu",
                  fixed_threshold: int = 30, morph_kernel_size: int = 5) -> dict:
    os.makedirs(outdir, exist_ok=True)

    # FR-1 + FR-2
    img_a, img_b = preprocess_pair(image_a_path, image_b_path, do_alignment=align)

    # FR-3
    result = detect_differences(
        img_a, img_b, min_region_area=min_region_area,
        threshold_method=threshold_method, fixed_threshold=fixed_threshold,
        morph_kernel_size=morph_kernel_size,
    )

    # FR-4
    boxed_b = draw_bounding_boxes(img_b, result)
    overlay = make_overlay(img_b, result)
    mask_visual = make_mask_visual(result)
    side_by_side = make_side_by_side(img_a, img_b)
    summary_panel = make_summary_panel(img_a, boxed_b, overlay, mask_visual)

    cv2.imwrite(os.path.join(outdir, "original_A.png"), img_a)
    cv2.imwrite(os.path.join(outdir, "original_B.png"), img_b)
    cv2.imwrite(os.path.join(outdir, "diff_visualization.png"), side_by_side)
    cv2.imwrite(os.path.join(outdir, "highlighted_regions.png"), boxed_b)
    cv2.imwrite(os.path.join(outdir, "heatmap_overlay.png"), overlay)
    cv2.imwrite(os.path.join(outdir, "difference_mask.png"), mask_visual)
    cv2.imwrite(os.path.join(outdir, "summary_panel.png"), summary_panel)

    # FR-5
    stats = compute_statistics(result)
    with open(os.path.join(outdir, "statistics.json"), "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    # FR-6
    summary_text = generate_summary(stats, use_llm=use_llm_summary)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(summary_text)

    return {"stats": stats.to_dict(), "summary": summary_text, "outdir": outdir}


def main():
    parser = argparse.ArgumentParser(description="AI-based image difference detection")
    parser.add_argument("--image_a", required=True, help="Path to reference image A")
    parser.add_argument("--image_b", required=True, help="Path to comparison image B")
    parser.add_argument("--outdir", default="outputs", help="Directory to write results to")
    parser.add_argument("--min_region_area", type=int, default=150,
                         help="Minimum contour area (px) to count as a real change")
    parser.add_argument("--no_align", action="store_true",
                         help="Disable automatic image alignment/registration")
    parser.add_argument("--threshold_method", choices=["otsu", "fixed"], default="otsu",
                         help="Thresholding method for the difference mask")
    parser.add_argument("--fixed_threshold", type=int, default=30,
                         help="Threshold value used when --threshold_method=fixed "
                              "(raise for noisy/high-res scans, e.g. 60-100)")
    parser.add_argument("--morph_kernel_size", type=int, default=5,
                         help="Morphological kernel size for noise cleanup (odd number)")
    parser.add_argument("--llm_summary", action="store_true",
                         help="Use Claude API for a more fluent summary (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    result = run_pipeline(
        args.image_a, args.image_b, args.outdir,
        min_region_area=args.min_region_area,
        align=not args.no_align,
        use_llm_summary=args.llm_summary,
        threshold_method=args.threshold_method,
        fixed_threshold=args.fixed_threshold,
        morph_kernel_size=args.morph_kernel_size,
    )

    print(json.dumps(result["stats"], indent=2))
    print("\n--- AI-Generated Summary ---")
    print(result["summary"])
    print(f"\nAll outputs written to: {result['outdir']}")


if __name__ == "__main__":
    main()
