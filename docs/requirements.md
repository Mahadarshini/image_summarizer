# Requirements Document — AI-Based Image Difference Detection System

## 1. Purpose
Provide an automated system to compare two images, detect and localize
visual differences, produce annotated visualizations, and generate a
natural-language summary of the detected changes — reducing the manual,
error-prone process of eyeballing image pairs (e.g. for QA/inspection,
document version control, surveillance, or construction progress checks).

## 2. Scope
A Python application (CLI + Streamlit web UI) that:
- Accepts two uploaded images (JPG/JPEG/PNG).
- Preprocesses them for a fair comparison (resize, optional alignment,
  illumination normalization).
- Detects differences using structural similarity (SSIM) and pixel-wise
  comparison.
- Produces visual outputs (boxes, heatmap, mask, side-by-side).
- Computes quantitative statistics.
- Generates a human-readable summary paragraph.

Out of scope: video comparison, real-time streaming diffing, multi-image
(>2) batch comparison (though the modules are reusable for that
extension), and training a custom deep-learning change-detection model.

## 3. Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-1 | Upload two images (JPG/JPEG/PNG); validate before processing | ✅ Implemented |
| FR-2 | Resize to common resolution, align if needed, normalize quality | ✅ Implemented |
| FR-3 | Pixel-wise / CV-based comparison, detect added/removed/modified regions, ignore noise, produce a diff mask | ✅ Implemented |
| FR-4 | Visual output: bounding boxes, heatmap/mask, side-by-side, overlay | ✅ Implemented |
| FR-5 | Statistics: region count, % changed, area, coordinates | ✅ Implemented |
| FR-6 | AI-generated natural-language change summary | ✅ Implemented (rule-based, optional LLM-enhanced) |

## 4. Non-Functional Requirements
- **Usability**: no-code web UI (Streamlit) alongside a scriptable CLI.
- **Configurability**: detection sensitivity (threshold method/value,
  morphological kernel size, minimum region area) is exposed as parameters
  rather than hardcoded, since optimal settings vary by input type (clean
  synthetic images vs. noisy scans/photos).
- **Offline-first**: the default summary generator requires no external API
  or network access; an LLM-enhanced summary is opt-in.
- **Portability**: pure Python + OpenCV/scikit-image, no GPU required.

## 5. Inputs / Outputs

**Inputs**: Image A (reference), Image B (comparison) — JPG/JPEG/PNG.

**Outputs**:
1. Original Image A
2. Original Image B
3. Difference visualization (side-by-side)
4. Highlighted changed regions (bounding boxes)
5. Difference statistics (JSON: region count, % changed, area, coordinates, SSIM score)
6. AI-generated summary paragraph (text)

## 6. Deliverables Checklist
- [x] Source code (`src/`, `app.py`, `main.py`)
- [x] Project documentation (`README.md`, this document)
- [x] Requirements document (this document)
- [x] System architecture diagram (`docs/architecture_diagram.png`)
- [x] Sample input and output images (`sample_data/`)
- [ ] Demonstration video (optional — not produced)
- [x] README with setup instructions
