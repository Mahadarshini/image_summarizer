"""
Flask web UI for AI-Based Image Difference Detection.

Run:
    python app.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file, session, url_for
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
STATIC_RUNS_DIR = BASE_DIR / "static" / "runs"
UPLOAD_DIR = BASE_DIR / "uploads"

sys.path.insert(0, str(SRC_DIR))

from preprocessing import SUPPORTED_EXTENSIONS  # noqa: E402
from main import run_pipeline  # noqa: E402


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "image-diff-dev")


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    STATIC_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def _save_upload(file_storage, run_upload_dir: Path, prefix: str) -> Path:
    if not file_storage or not file_storage.filename:
        raise ValueError("Both Image A and Image B are required.")

    original_name = secure_filename(file_storage.filename)
    if not _allowed(original_name):
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Please upload {allowed} files.")

    ext = Path(original_name).suffix.lower()
    saved_path = run_upload_dir / f"{prefix}{ext}"
    file_storage.save(saved_path)
    return saved_path


def _has_upload(name: str) -> bool:
    file_storage = request.files.get(name)
    return bool(file_storage and file_storage.filename)


def _current_pair() -> dict | None:
    pair = session.get("current_pair")
    if not pair:
        return None

    image_a = Path(pair["image_a_path"])
    image_b = Path(pair["image_b_path"])
    if not image_a.exists() or not image_b.exists():
        session.pop("current_pair", None)
        return None

    return pair


def _save_or_reuse_pair() -> tuple[Path, Path, dict]:
    pair = _current_pair()
    has_a = _has_upload("image_a")
    has_b = _has_upload("image_b")

    if not pair and (not has_a or not has_b):
        raise ValueError("Upload both images once. After that, you can change only the parameters and compare again.")

    if pair:
        pair_id = pair["pair_id"]
        pair_dir = UPLOAD_DIR / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
    else:
        pair_id = uuid.uuid4().hex[:12]
        pair_dir = UPLOAD_DIR / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair = {"pair_id": pair_id}

    if has_a:
        image_a = _save_upload(request.files.get("image_a"), pair_dir, "image_a")
        pair["image_a_path"] = str(image_a)
        pair["image_a_name"] = secure_filename(request.files["image_a"].filename)

    if has_b:
        image_b = _save_upload(request.files.get("image_b"), pair_dir, "image_b")
        pair["image_b_path"] = str(image_b)
        pair["image_b_name"] = secure_filename(request.files["image_b"].filename)

    session["current_pair"] = pair
    session.modified = True
    return Path(pair["image_a_path"]), Path(pair["image_b_path"]), pair


def _coerce_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(request.form.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _build_artifacts(run_id: str, stats: dict, summary: str) -> dict:
    image_names = {
        "original_a": "original_A.png",
        "original_b": "original_B.png",
        "difference": "diff_visualization.png",
        "highlighted": "highlighted_regions.png",
        "heatmap": "heatmap_overlay.png",
        "mask": "difference_mask.png",
        "panel": "summary_panel.png",
    }


def _pair_for_template(pair: dict | None) -> dict | None:
    if not pair:
        return None
    pair_id = pair["pair_id"]
    return {
        "image_a_name": pair.get("image_a_name", "Image A"),
        "image_b_name": pair.get("image_b_name", "Image B"),
        "image_a_url": url_for("uploaded_image", pair_id=pair_id, image_key="a"),
        "image_b_url": url_for("uploaded_image", pair_id=pair_id, image_key="b"),
    }


def _form_values() -> dict:
    return {
        "align": request.form.get("align") == "on",
        "threshold_method": request.form.get("threshold_method", "otsu"),
        "fixed_threshold": _coerce_int("fixed_threshold", 30, 1, 255),
        "min_region_area": _coerce_int("min_region_area", 150, 10, 10000),
        "morph_kernel_size": _coerce_int("morph_kernel_size", 5, 3, 31) | 1,
    }
    return {
        "run_id": run_id,
        "stats": stats,
        "summary": summary,
        "images": {
            key: url_for("static", filename=f"runs/{run_id}/{filename}")
            for key, filename in image_names.items()
        },
        "pdf_url": url_for("download_report", run_id=run_id),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    _ensure_dirs()
    error = None
    result = None
    form_values = _form_values()

    if request.method == "POST":
        run_id = uuid.uuid4().hex[:12]
        run_output_dir = STATIC_RUNS_DIR / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            image_a, image_b, _ = _save_or_reuse_pair()

            threshold_method = form_values["threshold_method"]
            if threshold_method not in {"otsu", "fixed"}:
                threshold_method = "otsu"

            pipeline_result = run_pipeline(
                str(image_a),
                str(image_b),
                str(run_output_dir),
                min_region_area=form_values["min_region_area"],
                align=form_values["align"],
                use_llm_summary=False,
                threshold_method=threshold_method,
                fixed_threshold=form_values["fixed_threshold"],
                morph_kernel_size=form_values["morph_kernel_size"],
            )

            _create_pdf_report(
                run_output_dir,
                pipeline_result["stats"],
                pipeline_result["summary"],
            )
            result = _build_artifacts(
                run_id,
                pipeline_result["stats"],
                pipeline_result["summary"],
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            shutil.rmtree(run_output_dir, ignore_errors=True)

    return render_template(
        "index.html",
        result=result,
        error=error,
        current_pair=_pair_for_template(_current_pair()),
        form_values=form_values,
    )


@app.route("/download/<run_id>/report.pdf")
def download_report(run_id: str):
    if not run_id.replace("-", "").isalnum():
        abort(404)
    report_path = STATIC_RUNS_DIR / run_id / "image_difference_report.pdf"
    if not report_path.exists():
        abort(404)
    return send_file(report_path, as_attachment=True, download_name="image_difference_report.pdf")


@app.route("/uploaded/<pair_id>/<image_key>")
def uploaded_image(pair_id: str, image_key: str):
    if not pair_id.isalnum() or image_key not in {"a", "b"}:
        abort(404)

    pair = _current_pair()
    if not pair or pair.get("pair_id") != pair_id:
        abort(404)

    path_key = "image_a_path" if image_key == "a" else "image_b_path"
    image_path = Path(pair[path_key])
    if not image_path.exists():
        abort(404)

    return send_file(image_path)


def _create_pdf_report(outdir: Path, stats: dict, summary: str) -> Path:
    report_path = outdir / "image_difference_report.pdf"
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#172033"),
        alignment=TA_LEFT,
    )
    section = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#23436f"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#273244"),
    )

    story = [
        Paragraph("AI-Based Image Difference Detection Report", title),
        Paragraph("Original images, visual difference outputs, statistics, and automated summary.", body),
        Spacer(1, 0.12 * inch),
    ]

    image_sections = [
        ("1. Original Image A", "original_A.png"),
        ("2. Original Image B", "original_B.png"),
        ("3. Difference Visualization", "diff_visualization.png"),
        ("4. Highlighted Changed Regions", "highlighted_regions.png"),
    ]
    for heading, filename in image_sections:
        story.append(Paragraph(heading, section))
        story.append(_pdf_image(outdir / filename, max_width=6.7 * inch, max_height=3.0 * inch))

    story.append(Paragraph("5. Difference Statistics", section))
    metrics = [
        ["Changed regions", stats.get("num_changed_regions", 0)],
        ["Image changed", f"{stats.get('percent_changed', 0)}%"],
        ["Changed area", f"{stats.get('total_changed_area_px', 0):,} px"],
        ["Total image area", f"{stats.get('total_image_area_px', 0):,} px"],
        ["SSIM similarity", stats.get("ssim_similarity_score", "n/a")],
    ]
    table = Table(metrics, colWidths=[2.3 * inch, 3.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf3fb")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172033")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c9d4e4")),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)

    if stats.get("regions"):
        rows = [["ID", "Location", "Box x,y,w,h", "Area", "% of image"]]
        for region in stats["regions"][:18]:
            rows.append(
                [
                    region["id"],
                    region["location"],
                    ", ".join(str(v) for v in region["bbox_xywh"]),
                    f"{region['area_px']:,}",
                    region["percent_of_image"],
                ]
            )
        region_table = Table(rows, colWidths=[0.45 * inch, 1.2 * inch, 1.75 * inch, 1.05 * inch, 1.1 * inch])
        region_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d8e0ec")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(Spacer(1, 0.12 * inch))
        story.append(region_table)

    story.append(Paragraph("6. AI-Generated Summary Paragraph", section))
    story.append(Paragraph(summary, body))
    story.append(Paragraph("Supporting Heatmap and Difference Mask", section))
    story.append(_pdf_image(outdir / "summary_panel.png", max_width=6.7 * inch, max_height=4.1 * inch))

    with (outdir / "report_payload.json").open("w", encoding="utf-8") as fh:
        json.dump({"stats": stats, "summary": summary}, fh, indent=2)

    doc.build(story)
    return report_path


def _pdf_image(path: Path, max_width: float, max_height: float) -> PdfImage:
    image = PdfImage(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


if __name__ == "__main__":
    _ensure_dirs()
    app.run(debug=True, host="127.0.0.1", port=5000)
