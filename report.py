"""
report.py
==============================================
Automated PDF export of batch inspection results.

Satisfies the assignment's extra-effort requirement:
    "Reporting: Automated export of results and findings into PDF
     format."

Takes the list of per-image result dicts produced by
colorDetection.inspect_image() (as assembled by app.py across a batch
upload) and renders a PDF with:
    - a title page / run summary (counts per fruit type / quality
      across every detected fruit, average size, calibration status)
    - a bar chart of fruit-type and quality distribution
    - one section per image: annotated (every fruit's bbox+contour)
      thumbnail, plus a per-fruit table (type, quality, confidence,
      measurements) for every fruit detected in that photo

Uses reportlab (pure-Python, no external binary dependency) so it runs
anywhere the rest of the pipeline runs, including inside the Streamlit
app's sandbox.
"""

import io
from collections import Counter
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _cv2_to_reportlab_image(bgr_image, max_width_cm=8.0):
    """Encodes a BGR numpy image as PNG bytes and wraps it in a
    reportlab Image flowable sized to fit the page."""
    success, buf = cv2.imencode(".png", bgr_image)
    if not success:
        return None
    bio = io.BytesIO(buf.tobytes())
    h, w = bgr_image.shape[:2]
    width = max_width_cm * cm
    height = width * (h / w)
    return RLImage(bio, width=width, height=height)


def _distribution_chart(labels, title, color_map=None):
    """Returns a reportlab Image flowable containing a bar chart of
    label counts, or None if there's nothing to plot."""
    counts = Counter(l for l in labels if l)
    if not counts:
        return None

    fig, ax = plt.subplots(figsize=(5, 3))
    classes = list(counts.keys())
    values = [counts[c] for c in classes]
    color_map = color_map or {}
    bar_colors = [color_map.get(c, "#607D8B") for c in classes]
    ax.bar(classes, values, color=bar_colors)
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()

    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=150)
    plt.close(fig)
    bio.seek(0)
    return RLImage(bio, width=12 * cm, height=7.2 * cm)


def generate_report(results, output_path="inspection_report.pdf", batch_name="Fruit Quality Inspection"):
    """
    results: list of dicts, each the return value of
        colorDetection.inspect_image() plus a "filename" key. Key
        fields used here:
        "filename": str
        "objects": list of per-fruit dicts (index, fruit_type,
            fruit_type_confidence, label, confidence, width_cm/
            height_cm/area_cm2 or *_px fallback)
        "summary": {"Apple": {"Fresh": 2, "Rotten": 1}, ...}
        "count": int, number of fruit detected in this image
        "annotated": np.ndarray (BGR image with every fruit's
            bbox/contour drawn)
        "calibration_method": str
    output_path: where to write the PDF.

    Returns output_path.
    """
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    small_style = ParagraphStyle("small", parent=body_style, fontSize=9, textColor=colors.grey)

    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    # ---- Title / summary page ----
    story.append(Paragraph(batch_name, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small_style))
    story.append(Spacer(1, 0.5 * cm))

    all_objects = [obj for r in results for obj in r.get("objects", [])]
    fruit_types = [obj.get("fruit_type") for obj in all_objects]
    labels = [obj.get("label") for obj in all_objects]
    counts = Counter(l for l in labels if l)
    total = len(results)
    total_fruit = len(all_objects)
    classified = sum(1 for obj in all_objects if obj.get("label"))

    summary_rows = [["Metric", "Value"]]
    summary_rows.append(["Total images inspected", str(total)])
    summary_rows.append(["Total fruit detected", str(total_fruit)])
    summary_rows.append(["Successfully classified", str(classified)])
    for fruit_type, n in Counter(f for f in fruit_types if f).items():
        summary_rows.append([f"  {fruit_type} (fruit type)", str(n)])
    for cls, n in counts.items():
        pct = (n / classified * 100) if classified else 0
        summary_rows.append([f"  {cls} (quality)", f"{n} ({pct:.1f}%)"])

    calibrated = [r for r in results if r.get("calibration_method") not in (None, "none")]
    summary_rows.append(["Images with physical calibration", f"{len(calibrated)} / {total}"])

    areas = [obj.get("area_cm2") for obj in all_objects if obj.get("area_cm2") is not None]
    if areas:
        summary_rows.append(["Average fruit area", f"{sum(areas)/len(areas):.2f} cm^2"])

    table = Table(summary_rows, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.7 * cm))

    fruit_chart = _distribution_chart(fruit_types, "Fruit Type Distribution")
    if fruit_chart:
        story.append(fruit_chart)
        story.append(Spacer(1, 0.4 * cm))

    quality_chart = _distribution_chart(
        labels, "Quality Distribution",
        color_map={"Fresh": "#4CAF50", "Unripe": "#FFC107", "Rotten": "#795548"},
    )
    if quality_chart:
        story.append(quality_chart)

    story.append(PageBreak())

    # ---- Per-image sections ----
    for i, r in enumerate(results, start=1):
        objects = r.get("objects", [])
        count = r.get("count", len(objects))
        summary_text = ", ".join(
            f"{n} {fruit_type} ({', '.join(f'{c} {q}' for q, c in qualities.items())})"
            for fruit_type, qualities in r.get("summary", {}).items()
            for n in [sum(qualities.values())]
        ) or "No fruit detected"
        story.append(Paragraph(f"{i}. {r.get('filename', f'image_{i}')} — {count} fruit detected", heading_style))
        story.append(Paragraph(summary_text, small_style))

        if r.get("annotated") is not None:
            img_flowable = _cv2_to_reportlab_image(r["annotated"])
            if img_flowable:
                story.append(img_flowable)
        story.append(Spacer(1, 0.3 * cm))

        # Each fruit cropped out on its own (background + neighboring
        # fruit removed via its mask) — the "separated" view, next to
        # the whole-photo annotated image above.
        crops = [obj.get("crop_isolated") for obj in objects if obj.get("crop_isolated") is not None]
        if crops:
            cell_w = min(4.0 * cm, 16.0 * cm / max(1, len(crops)))
            img_row, caption_row = [], []
            for obj in objects:
                crop = obj.get("crop_isolated")
                if crop is None:
                    continue
                h, w = crop.shape[:2]
                cell_h = cell_w * (h / w) if w else cell_w
                success, buf = cv2.imencode(".png", crop)
                img_row.append(RLImage(io.BytesIO(buf.tobytes()), width=cell_w, height=cell_h) if success else "")
                caption_row.append(f"#{obj.get('index', 0) + 1} {obj.get('fruit_type') or '?'} / {obj.get('label') or '?'}")
            crop_table = Table([img_row, caption_row], colWidths=[cell_w] * len(img_row))
            crop_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 1), (-1, 1), 7.5),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
            ]))
            story.append(crop_table)
            story.append(Spacer(1, 0.3 * cm))

        if objects:
            # Stem column is only added when at least one object in this
            # image actually carries a "stem" key (i.e. Stem Detection was
            # selected for this run) — keeps the report unchanged for
            # runs/teammates that don't use that module.
            has_stem_data = any("stem" in obj for obj in objects)
            header = ["#", "Fruit Type", "Quality", "Type Conf.", "Quality Conf.", "Size"]
            if has_stem_data:
                header += ["Stem", "Stem Length"]
            rows = [header]
            for obj in objects:
                fruit_type = obj.get("fruit_type") or "Not detected"
                label = obj.get("label") or "Not classified"
                type_conf = obj.get("fruit_type_confidence", 0.0)
                conf = obj.get("confidence", 0.0)
                if obj.get("width_cm") is not None:
                    size = f"{obj['width_cm']:.1f} x {obj['height_cm']:.1f} cm"
                else:
                    size = f"{obj.get('width_px', 0):.0f} x {obj.get('height_px', 0):.0f} px"
                row = [
                    str(obj.get("index", 0) + 1), fruit_type, label,
                    f"{type_conf * 100:.1f}%", f"{conf * 100:.1f}%", size,
                ]
                if has_stem_data:
                    stem = obj.get("stem")
                    if stem is not None and stem.found:
                        row.append(stem.condition or "Found")
                        row.append(f"{stem.length_cm:.1f} cm" if stem.length_cm is not None else f"{stem.length_px:.0f} px")
                    else:
                        row.append("Not found")
                        row.append("-")
                rows.append(row)

            col_widths = [1 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm, 3 * cm]
            if has_stem_data:
                col_widths += [2 * cm, 2.4 * cm]
            detail_table = Table(rows, colWidths=col_widths)
            detail_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#546E7A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]))
            story.append(detail_table)
        else:
            story.append(Paragraph("No fruit detected in this image.", body_style))

        story.append(Paragraph(f"Calibration method: {r.get('calibration_method', 'none')}", small_style))
        story.append(Spacer(1, 0.6 * cm))

        if i < len(results):
            story.append(PageBreak())

    doc.build(story)
    return output_path