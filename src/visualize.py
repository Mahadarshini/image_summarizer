"""
FR-4 Difference Visualization
-------------------------------
Generates all visual outputs: bounding boxes, highlighted mask/heatmap,
side-by-side comparison, and an overlay of detected changes.
"""

import cv2
import numpy as np

from diff_detector import DiffResult

BOX_COLOR = (0, 0, 255)       # red (BGR)
BOX_THICKNESS = 3
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bounding_boxes(img: np.ndarray, result: DiffResult) -> np.ndarray:
    """FR-4: Draw bounding boxes with region IDs around changed regions."""
    out = img.copy()
    for r in result.regions:
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), BOX_COLOR, BOX_THICKNESS)
        label = f"#{r.id}"
        (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, 0.7, 2)
        cv2.rectangle(out, (r.x, r.y - th - 8), (r.x + tw + 6, r.y), BOX_COLOR, -1)
        cv2.putText(out, label, (r.x + 3, r.y - 5), LABEL_FONT, 0.7, (255, 255, 255), 2)
    return out


def make_overlay(img_b: np.ndarray, result: DiffResult, alpha: float = 0.45) -> np.ndarray:
    """FR-4: Overlay the heatmap of detected changes on top of image B."""
    return cv2.addWeighted(result.heatmap, alpha, img_b, 1 - alpha, 0)


def make_mask_visual(result: DiffResult) -> np.ndarray:
    """Return the binary mask as a 3-channel white-on-black visual."""
    return cv2.cvtColor(result.diff_mask, cv2.COLOR_GRAY2BGR)


def make_side_by_side(img_a: np.ndarray, img_b: np.ndarray, gap: int = 12) -> np.ndarray:
    """FR-4: Side-by-side comparison of the two original images."""
    h = max(img_a.shape[0], img_b.shape[0])
    gap_col = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.hstack([_pad_to_height(img_a, h), gap_col, _pad_to_height(img_b, h)])


def _pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad = height - img.shape[0]
    return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def make_summary_panel(img_a: np.ndarray, img_b_boxed: np.ndarray, overlay: np.ndarray,
                        mask_visual: np.ndarray, gap: int = 10) -> np.ndarray:
    """
    Compose a single 2x2 grid image: [Original A | Original B w/ boxes]
                                       [Overlay/Heatmap | Difference Mask]
    Useful as one combined "report" image.
    """
    def label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(out, text, (8, 24), LABEL_FONT, 0.7, (255, 255, 255), 2)
        return out

    h, w = img_a.shape[:2]
    b = cv2.resize(img_b_boxed, (w, h))
    ov = cv2.resize(overlay, (w, h))
    mv = cv2.resize(mask_visual, (w, h))

    top = np.hstack([label(img_a, "Image A (Reference)"), np.full((h, gap, 3), 255, np.uint8),
                      label(b, "Image B (Boxes = Changes)")])
    bottom = np.hstack([label(ov, "Change Heatmap Overlay"), np.full((h, gap, 3), 255, np.uint8),
                         label(mv, "Difference Mask")])
    row_gap = np.full((gap, top.shape[1], 3), 255, np.uint8)
    return np.vstack([top, row_gap, bottom])
