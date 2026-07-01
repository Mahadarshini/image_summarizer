"""
generate_synthetic_samples.py

Creates a clean, controlled "before/after" image pair with known,
unambiguous differences (an added shape, a removed shape, a moved shape,
and a recolored shape). Useful for quickly verifying that the detection
pipeline is working correctly, independent of noisy real-world inputs.

Run:
    python scripts/generate_synthetic_samples.py
"""

import os
import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


def make_base_scene() -> np.ndarray:
    img = np.full((600, 900, 3), 245, dtype=np.uint8)  # light gray background

    # ground / horizon line
    cv2.rectangle(img, (0, 420), (900, 600), (210, 235, 210), -1)

    # a "building" (rectangle)
    cv2.rectangle(img, (100, 220), (260, 420), (160, 120, 90), -1)
    cv2.rectangle(img, (100, 220), (260, 420), (90, 70, 50), 3)

    # a "tree" (circle on a trunk) - top-left area - will be REMOVED in v2
    cv2.rectangle(img, (60, 360), (75, 420), (60, 90, 130), -1)
    cv2.circle(img, (67, 340), 35, (60, 150, 60), -1)

    # a "sun" (circle) - top-right - will change COLOR in v2
    cv2.circle(img, (780, 90), 45, (60, 220, 250), -1)

    # a "car" (rectangle + circles) - bottom-center - will MOVE in v2
    cv2.rectangle(img, (430, 380), (560, 430), (50, 50, 200), -1)
    cv2.circle(img, (455, 430), 18, (20, 20, 20), -1)
    cv2.circle(img, (535, 430), 18, (20, 20, 20), -1)

    return img


def make_modified_scene(base: np.ndarray) -> np.ndarray:
    img = base.copy()

    # 1) REMOVE the tree (paint over with background/ground color)
    cv2.rectangle(img, (15, 295, ), (115, 425), (245, 245, 245), -1)
    cv2.rectangle(img, (15, 420), (115, 425), (210, 235, 210), -1)

    # 2) CHANGE the sun's color (yellow -> orange/red, e.g. sunset)
    cv2.circle(img, (780, 90), 45, (40, 100, 240), -1)

    # 3) MOVE the car to the right
    cv2.rectangle(img, (430, 380), (560, 430), (245, 245, 245), -1)
    cv2.rectangle(img, (430, 420), (560, 430), (210, 235, 210), -1)
    cv2.rectangle(img, (620, 380), (750, 430), (50, 50, 200), -1)
    cv2.circle(img, (645, 430), 18, (20, 20, 20), -1)
    cv2.circle(img, (725, 430), 18, (20, 20, 20), -1)

    # 4) ADD a new object: a "cloud" top-center
    cv2.circle(img, (400, 80), 30, (240, 240, 240), -1)
    cv2.circle(img, (430, 70), 35, (240, 240, 240), -1)
    cv2.circle(img, (465, 85), 28, (240, 240, 240), -1)

    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = make_base_scene()
    modified = make_modified_scene(base)

    path_a = os.path.join(OUT_DIR, "synthetic_A.png")
    path_b = os.path.join(OUT_DIR, "synthetic_B.png")
    cv2.imwrite(path_a, base)
    cv2.imwrite(path_b, modified)
    print(f"Wrote: {path_a}")
    print(f"Wrote: {path_b}")


if __name__ == "__main__":
    main()
