"""
PDF report generation for the overall pipeline. Adapted from ASS's own
report.py (same libraries, same bar-chart-plus-table structure), but built
against overall.py's actual FruitResult fields instead of ASS's dict-shaped
per-object results.

Usage (see app.py):
    pdf_bytes = generate_report(results, batch_name="...")
    st.download_button("Download PDF report", pdf_bytes, "report.pdf", "application/pdf")

`results` is the same list of dicts app.py already builds in
st.session_state["overall_results"]: each one has "filename", "image"
(the resized BGR frame fr.bbox is relative to), "fruits" (list of
FruitResult), "calibration" (CalibrationResult), "cm_per_pixel_x/y".
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
from reportlab.lib.units import cm as CM
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from overall import draw_annotations, BOX_COLOURS

_QUALITY_HEX = {name: "#%02x%02x%02x" % (b, g, r) for name, (b, g, r) in BOX_COLOURS.items()}


def _cv2_to_reportlab_image(bgr_image, max_width_cm=16.0):
    success, buf = cv2.imencode(".png", bgr_image)
    if not success:
        return None
    bio = io.BytesIO(buf.tobytes())
    h, w = bgr_image.shape[:2]
    width = max_width_cm * CM
    height = width * (h / w) if w else width
    return RLImage(bio, width=width, height=height)


def distribution_chart(labels, title, color_map=None, figsize=(5, 3)):
    """Bar chart of value counts. Returns a matplotlib Figure, or None if there's no data."""
    counts = Counter(l for l in labels if l)
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig


def _fig_to_rl_image(fig, width_cm=12.0, height_cm=7.2):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=150)
    plt.close(fig)
    bio.seek(0)
    return RLImage(bio, width=width_cm * CM, height=height_cm * CM)


def _fruit_size_cm(fr, r):
    calibration = r.get("calibration")
    if calibration is None or calibration.method == "none":
        return None
    _x, _y, w, h = fr.bbox
    return w * r["cm_per_pixel_x"], h * r["cm_per_pixel_y"]


def generate_report(results, batch_name="Fruit Quality Inspection"):
    """Build the PDF in memory and return it as bytes."""
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    small_style = ParagraphStyle("small", parent=body_style, fontSize=9, textColor=colors.grey)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * CM, bottomMargin=1.5 * CM)
    story = []

    story.append(Paragraph(batch_name, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small_style))
    story.append(Spacer(1, 0.5 * CM))

    all_fruits = [fr for r in results for fr in r["fruits"]]
    species_list = [fr.species for fr in all_fruits]
    quality_list = [fr.final_quality for fr in all_fruits]

    total_photos = len(results)
    total_fruit = len(all_fruits)
    classified = sum(1 for fr in all_fruits if fr.species)

    summary_rows = [["Metric", "Value"]]
    summary_rows.append(["Total images inspected", str(total_photos)])
    summary_rows.append(["Total fruit detected", str(total_fruit)])
    summary_rows.append(["Successfully classified", str(classified)])
    for species, n in Counter(s for s in species_list if s).items():
        summary_rows.append([f"  {species} (species)", str(n)])
    for quality, n in Counter(q for q in quality_list if q).items():
        pct = (n / total_fruit * 100) if total_fruit else 0
        summary_rows.append([f"  {quality} (quality)", f"{n} ({pct:.1f}%)"])

    calibrated = [r for r in results if r.get("calibration") is not None and r["calibration"].method != "none"]
    summary_rows.append(["Images with physical calibration", f"{len(calibrated)} / {total_photos}"])

    table = Table(summary_rows, colWidths=[8 * CM, 8 * CM])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.7 * CM))

    species_fig = distribution_chart(species_list, "Species Distribution")
    if species_fig:
        story.append(_fig_to_rl_image(species_fig))
        story.append(Spacer(1, 0.4 * CM))

    quality_fig = distribution_chart(quality_list, "Quality Distribution", color_map=_QUALITY_HEX)
    if quality_fig:
        story.append(_fig_to_rl_image(quality_fig))

    story.append(PageBreak())

    for i, r in enumerate(results, start=1):
        fruits = r["fruits"]
        story.append(Paragraph(f"{i}. {r['filename']} -- {len(fruits)} fruit(s) detected", heading_style))

        if r.get("image") is not None:
            annotated = draw_annotations(r["image"], fruits) if fruits else r["image"]
            img_flowable = _cv2_to_reportlab_image(annotated)
            if img_flowable:
                story.append(img_flowable)
        story.append(Spacer(1, 0.3 * CM))

        crops = [fr for fr in fruits if fr.crop is not None]
        if crops:
            cell_w = min(4.0 * CM, 16.0 * CM / max(1, len(crops)))
            img_row, caption_row = [], []
            for fr in crops:
                h, w = fr.crop.shape[:2]
                cell_h = cell_w * (h / w) if w else cell_w
                success, buf = cv2.imencode(".png", fr.crop)
                img_row.append(RLImage(io.BytesIO(buf.tobytes()), width=cell_w, height=cell_h) if success else "")
                caption_row.append(f"{fr.species or '?'} / {fr.final_quality or '?'}")
            crop_table = Table([img_row, caption_row], colWidths=[cell_w] * len(img_row))
            crop_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 1), (-1, 1), 7.5),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
            ]))
            story.append(crop_table)
            story.append(Spacer(1, 0.3 * CM))

        if fruits:
            header = ["#", "Species", "Quality", "Species Conf.", "Quality Conf.", "Defect %", "Stem", "Size"]
            rows = [header]
            for idx, fr in enumerate(fruits, start=1):
                size_cm = _fruit_size_cm(fr, r)
                if size_cm:
                    size_str = f"{size_cm[0]:.1f} x {size_cm[1]:.1f} cm"
                else:
                    size_str = f"{fr.bbox[2]:.0f} x {fr.bbox[3]:.0f} px"
                rows.append([
                    str(idx),
                    fr.species or "-",
                    fr.final_quality or "-",
                    f"{fr.species_confidence * 100:.1f}%",
                    f"{fr.final_quality_confidence * 100:.1f}%" if fr.final_quality else "-",
                    f"{fr.defect_percentage:.1f}%" if fr.defect_percentage is not None else "-",
                    "Yes" if fr.stem_detected else "No",
                    size_str,
                ])
            col_widths = [1 * CM, 2.2 * CM, 2.2 * CM, 2.2 * CM, 2.2 * CM, 2 * CM, 1.6 * CM, 3 * CM]
            detail_table = Table(rows, colWidths=col_widths)
            detail_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#546E7A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]))
            story.append(detail_table)
            story.append(Spacer(1, 0.3 * CM))

            # Full per-fruit breakdown -- same information shown in the
            # Streamlit fruit cards (species voting, quality basis,
            # colour/ripeness/defect/stem, morphological+texture), not just
            # the compact table above.
            detail_style = ParagraphStyle("fruitdetail", parent=body_style, fontSize=8.5, leading=11.5)
            for idx, fr in enumerate(fruits, start=1):
                lines = [f"<b>Fruit #{idx}: {fr.species or '?'}</b> "
                         f"({fr.species_confidence * 100:.0f}%) -- won by: {fr.species_source or '-'}"]
                lines.append(
                    f"1) YOLO+CNN: {fr.own_species or '-'} ({fr.own_confidence * 100:.0f}%) | "
                    f"2) YOLO raw: {fr.yolo_species or '-'} ({fr.yolo_confidence * 100:.0f}%) | "
                    f"3) Morph: {fr.morph_fruit_type or 'no match'}"
                    + (f" ({fr.morph_fruit_type_confidence * 100:.0f}%)" if fr.morph_fruit_type else "")
                )
                lines.append(f"CNN raw guess (feeds into #1): {fr.cnn_species or '-'} ({fr.cnn_confidence * 100:.0f}%)")
                lines.append(
                    f"<b>Final quality: {fr.final_quality or 'N/A'} "
                    f"({fr.final_quality_confidence * 100:.0f}%)</b>"
                )
                if fr.quality_note:
                    lines.append(f"<i>{fr.quality_note}</i>")
                lines.append(
                    f"Colour quality (own KNN): {fr.colour_quality or 'N/A'} ({fr.colour_confidence * 100:.0f}%)"
                )
                lines.append(
                    f"Ripeness (defect rule): {fr.defect_ripeness or 'N/A'} "
                    f"({fr.defect_ripeness_confidence * 100:.0f}%)"
                )
                if fr.defect_percentage is not None:
                    lines.append(f"Defect: {fr.defect_percentage:.1f}%")
                else:
                    lines.append(f"Defect: {fr.defect_note or '-'}")
                lines.append(
                    f"Stem: {'detected' if fr.stem_detected else 'not detected'} ({fr.stem_confidence * 100:.0f}%)"
                )
                if fr.morph_aspect_ratio is not None:
                    lines.append(
                        f"Morphological: size={fr.morph_size_class or 'N/A'}, "
                        f"aspect={fr.morph_aspect_ratio:.2f}, circularity={fr.morph_circularity:.2f}, "
                        f"extent={fr.morph_extent:.2f}"
                    )
                    lines.append(
                        f"Texture (GLCM): contrast={fr.tex_contrast:.1f}, energy={fr.tex_energy:.3f}, "
                        f"homogeneity={fr.tex_homogeneity:.3f}, entropy={fr.tex_entropy:.2f}"
                    )
                elif fr.morphological_note:
                    lines.append(fr.morphological_note)
                story.append(Paragraph("<br/>".join(lines), detail_style))
                story.append(Spacer(1, 0.25 * CM))
        else:
            story.append(Paragraph("No fruit detected in this image.", body_style))

        calibration = r.get("calibration")
        story.append(Paragraph(
            f"Calibration method: {calibration.method if calibration else 'none'}", small_style
        ))
        story.append(Spacer(1, 0.6 * CM))

        if i < len(results):
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()
