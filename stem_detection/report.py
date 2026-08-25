"""PDF report generation.

Extra Effort requirement: "Reporting: Automated export of results and
findings into PDF format."
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def _bgr_to_rl_image(bgr_image: np.ndarray, max_width_cm: float = 16.0) -> RLImage:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Failed to encode image for PDF embedding.")
    h, w = bgr_image.shape[:2]
    width = max_width_cm * cm
    height = width * h / w
    return RLImage(io.BytesIO(buf.tobytes()), width=width, height=height)


def generate_report(
    results: list[dict],
    output_path: str | Path,
    benchmark_summary: Optional[pd.DataFrame] = None,
    title: str = "Fruit Stem Detection Report",
    analysis_text: Optional[str] = None,
) -> None:
    """Build a PDF summarizing detection results (and, optionally, the
    Traditional/YOLO/Hybrid benchmark comparison) across one or more images.

    `results` is a list of per-image dicts, one per processed photo:
        {
            "filename": str,
            "annotated": np.ndarray (BGR),   # image with boxes/contours drawn
            "detections": list[Detection],
            "fruit_type": str,
            "method": str,
            "calibration": CalibrationResult | None,
        }
    This mirrors what app.py builds per uploaded image, so app.py can pass
    its results list straight through.
    """
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(
        f"{len(results)} image(s) processed. "
        f"{sum(len(r['detections']) for r in results)} stem(s) detected in total.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.5 * cm))

    # Overall results table
    table_data = [["Image", "Method", "Stems found", "Best confidence", "Size (cm)"]]
    for r in results:
        detections = r["detections"]
        best_conf = f"{max((d.confidence for d in detections), default=0):.0%}" if detections else "-"
        size_str = "-"
        calibration = r.get("calibration")
        if detections and calibration is not None and calibration.is_calibrated:
            x1, y1, x2, y2 = detections[0].bbox
            w_cm = calibration.px_to_cm(x2 - x1)
            h_cm = calibration.px_to_cm(y2 - y1)
            if w_cm is not None:
                size_str = f"{w_cm:.1f} x {h_cm:.1f}"
        table_data.append([
            r["filename"], r.get("method", "-"),
            str(len(detections)), best_conf, size_str,
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.7 * cm))

    if benchmark_summary is not None and not benchmark_summary.empty:
        story.append(Paragraph("Method comparison (Traditional vs YOLO vs Hybrid)", styles["Heading2"]))
        bench_data = [list(benchmark_summary.columns)]
        for _, row in benchmark_summary.iterrows():
            bench_data.append([
                f"{v:.3f}" if isinstance(v, float) else str(v) for v in row
            ])
        bench_table = Table(bench_data, repeatRows=1)
        bench_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(bench_table)
        story.append(Spacer(1, 0.7 * cm))

        if analysis_text:
            story.append(Paragraph("Analysis", styles["Heading2"]))
            for paragraph in analysis_text.split("\n\n"):
                html_paragraph = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", paragraph)
                story.append(Paragraph(html_paragraph, styles["Normal"]))
                story.append(Spacer(1, 0.2 * cm))
            story.append(Spacer(1, 0.5 * cm))

    # Per-image annotated photos
    for r in results:
        story.append(PageBreak())
        story.append(Paragraph(r["filename"], styles["Heading2"]))
        story.append(Paragraph(
            f"Method: {r.get('method', '-')} | Stems found: {len(r['detections'])}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(_bgr_to_rl_image(r["annotated"]))

    doc.build(story)
