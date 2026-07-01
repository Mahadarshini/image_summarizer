# AI-Based Image Difference Detection, Visualization, and Automated Change Summarization

An AI-powered system that accepts two images, detects and localizes visual
differences between them, produces annotated visualizations, computes
change statistics, and generates a human-readable summary paragraph
describing what changed.

Built in Python using OpenCV, scikit-image (SSIM), and Streamlit.

---

## Features (maps to the project spec)

| Requirement | Implementation |
|---|---|
| **FR-1** Image Upload | `app.py` (Streamlit uploader) / `main.py --image_a --image_b`. Validates format (JPG/JPEG/PNG) and file integrity. |
| **FR-2** Preprocessing | `src/preprocessing.py` — resize to common resolution, optional ORB+homography alignment, CLAHE illumination normalization. |
| **FR-3** Difference Detection | `src/diff_detector.py` — SSIM structural-similarity map + pixel-wise diff, adaptive/fixed thresholding, morphological noise cleanup, contour-based region extraction. |
| **FR-4** Visualization | `src/visualize.py` — bounding boxes, heatmap overlay, difference mask, side-by-side comparison, combined summary panel. |
| **FR-5** Statistics | `src/statistics_module.py` — region count, % area changed, per-region bounding boxes/coordinates, SSIM score. |
| **FR-6** AI Summary | `src/summary.py` — rule-based natural-language generator (default, offline) with an optional Claude-API-enhanced mode (`--llm_summary`). |

---

## Project Structure

```
project/
├── app.py                          # Streamlit web UI (FR-1 upload + all outputs)
├── main.py                         # CLI entry point (full pipeline)
├── requirements.txt
├── README.md
├── src/
│   ├── preprocessing.py            # FR-1, FR-2
│   ├── diff_detector.py            # FR-3
│   ├── visualize.py                # FR-4
│   ├── statistics_module.py        # FR-5
│   └── summary.py                  # FR-6
├── scripts/
│   ├── generate_synthetic_samples.py   # builds a clean test image pair
│   └── generate_architecture_diagram.py
├── docs/
│   └── architecture_diagram.png
└── sample_data/
    ├── synthetic_A.png / synthetic_B.png       # clean controlled test pair
    ├── image_A.png / image_B.png               # real-world test pair (engineering drawing revisions)
    ├── sample_output_synthetic/                # pre-generated outputs for the synthetic pair
    └── sample_output_bridge/                   # pre-generated outputs for the real-world pair
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The `anthropic` package is only required if you use `--llm_summary` /
"Use Claude API" — everything else works fully offline.

---

## Usage

### 1. Web app (recommended)

```bash
streamlit run app.py
```

Upload two images, tune sensitivity in the sidebar, click **Compare Images**,
and view the summary, statistics, and visualizations directly in the browser.

### 2. Command line

```bash
python main.py --image_a path/to/A.png --image_b path/to/B.png --outdir outputs
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--no_align` | Skip alignment (use for already-registered images, e.g. screenshots, scans, PDF renders) |
| `--threshold_method fixed --fixed_threshold 70` | Use a fixed sensitivity instead of automatic Otsu thresholding — raise the value to ignore rendering/compression noise |
| `--morph_kernel_size 9` | Larger kernel = more aggressive noise cleanup (good for scanned/re-rendered documents) |
| `--min_region_area 500` | Ignore changed blobs smaller than N pixels |
| `--llm_summary` | Use the Claude API for a more fluent summary (requires `ANTHROPIC_API_KEY`) |

Outputs written to `--outdir`:
`original_A.png`, `original_B.png`, `diff_visualization.png` (side-by-side),
`highlighted_regions.png` (boxes), `heatmap_overlay.png`, `difference_mask.png`,
`summary_panel.png` (2×2 combined report), `statistics.json`, `summary.txt`.

---

## Sample Data — where to get test images

You don't need an external dataset to try this out. Two ready-made options
are included:

1. **`sample_data/synthetic_A.png` / `synthetic_B.png`** — a clean,
   programmatically generated scene with four unambiguous changes (an
   object added, one removed, one moved, one recolored). Regenerate or
   customize with:
   ```bash
   python scripts/generate_synthetic_samples.py
   ```
   This is the best pair to sanity-check the pipeline end-to-end, since the
   ground truth is known exactly.

2. **`sample_data/image_A.png` / `image_B.png`** — two real revisions of an
   engineering drawing (rendered from the uploaded `v1.pdf`/`v2.pdf`), where
   v2 adds weep-hole details, updates the legend, and bumps the revision
   number. This is a good real-world test of noisy, high-detail documents.
   Note: two independent renders of the same vector drawing can differ by a
   few sub-pixels everywhere (font hinting/anti-aliasing), which shows up as
   widespread low-magnitude noise. Use `--threshold_method fixed
   --fixed_threshold 70 --morph_kernel_size 9 --min_region_area 600` (as
   used for `sample_data/sample_output_bridge/`) to suppress that noise and
   keep only real content changes.

For your own testing, any two images of the same scene/document work —
photos, screenshots, scans, or product images before/after a change.

---

## How the detection works

1. **Preprocess**: validate → resize to common resolution → (optional)
   align via ORB feature matching + homography → normalize illumination
   with CLAHE.
2. **Detect**: compute an SSIM structural-similarity map (catches
   perceptual/structural changes, not just raw pixel deltas) combined with
   a pixel-wise absolute difference; threshold (Otsu automatic or a fixed
   value); clean up with morphological open/close to remove speckle noise;
   extract contours as individual changed regions, filtered by minimum
   area.
3. **Visualize**: draw bounding boxes, build a JET heatmap overlay, render
   the binary mask, and compose a side-by-side and a 2×2 summary panel.
4. **Quantify**: region count, % of image changed, per-region area/
   coordinates/location label, overall SSIM similarity score.
5. **Summarize**: a deterministic template turns the statistics into a
   paragraph (overall result, major regions, approximate locations,
   severity, confidence) — matching the format in the spec. If
   `ANTHROPIC_API_KEY` is set and `--llm_summary` is passed, the statistics
   are instead handed to Claude for a more fluent paragraph, with automatic
   fallback to the rule-based summary on any failure.

---

## Extending this project

- Swap the SSIM+threshold detector for a learned change-detection model
  (e.g. a Siamese CNN) if you need to handle more complex real-world photo
  pairs (shadows, lighting, camera shake).
- Add object-aware descriptions (e.g. "a car appeared") by running an
  object detector on each region crop before generating the summary.
- Persist results to a database for tracking changes over time (e.g.
  construction progress monitoring).
