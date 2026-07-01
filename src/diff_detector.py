"""
FR-3 Difference Detection
--------------------------
Compares two pre-processed images pixel-wise and structurally (SSIM),
produces a binary difference mask, filters out insignificant noise, and
extracts bounding boxes for each distinct changed region.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class ChangedRegion:
    """A single detected changed region."""
    id: int
    x: int
    y: int
    w: int
    h: int
    area_px: int
    location_label: str = ""  # e.g. "top-left", "center", "bottom-right"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class DiffResult:
    """Full result of a difference-detection pass."""
    ssim_score: float
    diff_mask: np.ndarray                  # binary mask, uint8 0/255
    heatmap: np.ndarray                     # color heatmap, BGR
    regions: List[ChangedRegion] = field(default_factory=list)
    total_changed_pixels: int = 0
    percent_changed: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)   # (height, width)


def _location_label(cx: int, cy: int, width: int, height: int) -> str:
    """Map a point to a coarse 3x3 grid label (top/center/bottom x left/center/right)."""
    col = "left" if cx < width / 3 else ("center" if cx < 2 * width / 3 else "right")
    row = "top" if cy < height / 3 else ("middle" if cy < 2 * height / 3 else "bottom")
    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def detect_differences(img_a: np.ndarray, img_b: np.ndarray,
                        min_region_area: int = 150,
                        blur_kernel: int = 3,
                        threshold_method: str = "otsu",
                        fixed_threshold: int = 30,
                        morph_kernel_size: int = 5) -> DiffResult:
    """
    FR-3: Compare img_a and img_b and return a DiffResult.

    Args:
        min_region_area: contours smaller than this (in px) are treated as
            noise and discarded (raise this for noisy/high-res scans).
        blur_kernel: Gaussian blur kernel size applied before thresholding.
            Keep this low for drawings because thin circles and text can fade
            after heavy blur.
        threshold_method: "otsu" (automatic) or "fixed".
        fixed_threshold: threshold value used when threshold_method="fixed".
            Raise this (e.g. 60-100) for images with sub-pixel rendering
            noise (e.g. two different exports/scans of the same document).
        morph_kernel_size: size of the morphological close/dilate kernel used
            to connect nearby changed pixels. Raise this for noisy scans, but
            keep it low for thin annotations.

    Steps:
      1. Convert to grayscale and LAB color.
      2. Compute SSIM (structural similarity) map -> catches structural /
         perceptual differences robustly, not just raw pixel deltas.
      3. Combine SSIM, grayscale pixel difference, and color difference.
      4. Threshold the combined map to get a binary change mask.
      5. Morphological cleanup to remove noise / merge nearby blobs without
         erasing thin annotation strokes.
      6. Contour detection -> individual changed regions with bounding boxes,
         filtered by `min_region_area` to ignore insignificant noise.
    """
    height, width = img_a.shape[:2]

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray_a, gray_b, full=True)
    diff = (diff * 255).astype("uint8")
    # Pixel-wise grayscale difference catches dark/light content changes.
    abs_diff = cv2.absdiff(gray_a, gray_b)

    # Color difference catches red/colored annotations that may be weak after
    # grayscale conversion, such as circles, arrows, and markup text.
    lab_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_delta = np.linalg.norm(lab_a - lab_b, axis=2)
    color_delta = np.clip(color_delta * 1.35, 0, 255).astype("uint8")

    inv_ssim_diff = 255 - diff
    combined = cv2.max(cv2.max(inv_ssim_diff, abs_diff), color_delta)

    if blur_kernel > 1:
        combined = cv2.GaussianBlur(combined, (blur_kernel, blur_kernel), 0)

    if threshold_method == "otsu":
        _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(combined, fixed_threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: close small gaps and slightly thicken thin marks.
    # Use a tiny opening kernel only for isolated speckles; a large opening
    # erases circular/text annotations in drawings.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_size = max(3, morph_kernel_size | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, small_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[ChangedRegion] = []
    region_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_region_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        label = _location_label(cx, cy, width, height)
        regions.append(ChangedRegion(id=region_id, x=x, y=y, w=w, h=h,
                                      area_px=int(area), location_label=label))
        region_id += 1

    # Sort regions by area (largest/most significant first).
    regions.sort(key=lambda r: r.area_px, reverse=True)
    for i, r in enumerate(regions, start=1):
        r.id = i

    total_changed_pixels = int(np.count_nonzero(mask))
    percent_changed = round((total_changed_pixels / (width * height)) * 100, 2)

    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)

    return DiffResult(
        ssim_score=round(float(score), 4),
        diff_mask=mask,
        heatmap=heatmap,
        regions=regions,
        total_changed_pixels=total_changed_pixels,
        percent_changed=percent_changed,
        image_shape=(height, width),
    )
