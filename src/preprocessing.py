"""
FR-1 Image Upload & FR-2 Image Preprocessing
---------------------------------------------
Handles validation, loading, resizing to a common resolution, and
alignment/registration of two input images before comparison.
"""

import os
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_file(path: str) -> None:
    """FR-1: Validate that an uploaded file exists and is a supported format."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"File '{path}' could not be read as an image (corrupt or invalid).")


def load_image(path: str) -> np.ndarray:
    """Load an image as a BGR numpy array (OpenCV convention)."""
    validate_image_file(path)
    img = cv2.imread(path)
    return img


def resize_to_common_resolution(img_a: np.ndarray, img_b: np.ndarray):
    """
    FR-2: Resize both images to a common resolution if their sizes differ.
    The smaller of the two is used as the target so we never upsample
    (which would introduce blur/artifacts that could be mistaken for changes).
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    if (h_a, w_a) == (h_b, w_b):
        return img_a, img_b

    target_h = min(h_a, h_b)
    target_w = min(w_a, w_b)

    img_a_resized = cv2.resize(img_a, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_b_resized = cv2.resize(img_b, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return img_a_resized, img_b_resized


def align_images(img_a: np.ndarray, img_b: np.ndarray, max_features: int = 5000,
                  good_match_percent: float = 0.15) -> np.ndarray:
    """
    FR-2: Align img_b onto img_a using ORB feature matching + homography.
    Corrects for slight positional/rotational/scale differences (e.g. photos
    taken from slightly different angles). Falls back to the original image
    if not enough good matches are found (e.g. near-identical inputs).
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
        return img_b  # not enough features to align reliably; skip

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(desc_a, desc_b, None))
    if len(matches) < 10:
        return img_b

    matches.sort(key=lambda m: m.distance)
    num_good = max(int(len(matches) * good_match_percent), 10)
    matches = matches[:num_good]

    points_a = np.zeros((len(matches), 2), dtype=np.float32)
    points_b = np.zeros((len(matches), 2), dtype=np.float32)
    for i, m in enumerate(matches):
        points_a[i, :] = kp_a[m.queryIdx].pt
        points_b[i, :] = kp_b[m.trainIdx].pt

    h_matrix, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC)
    if h_matrix is None:
        return img_b

    height, width = img_a.shape[:2]
    aligned_b = cv2.warpPerspective(img_b, h_matrix, (width, height))
    return aligned_b


def preprocess_pair(path_a: str, path_b: str, do_alignment: bool = True,
                     normalize: bool = True):
    """
    Full FR-1 + FR-2 pipeline: validate, load, resize, optionally align,
    optionally normalize brightness/contrast so lighting differences alone
    don't trigger false positives.

    Returns (img_a, img_b) as same-sized BGR numpy arrays ready for diffing.
    """
    img_a = load_image(path_a)
    img_b = load_image(path_b)

    img_a, img_b = resize_to_common_resolution(img_a, img_b)

    if do_alignment:
        img_b = align_images(img_a, img_b)

    if normalize:
        img_a = _normalize_illumination(img_a)
        img_b = _normalize_illumination(img_b)

    return img_a, img_b


def _normalize_illumination(img: np.ndarray) -> np.ndarray:
    """Normalize brightness/contrast using CLAHE on the L channel (LAB space)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
