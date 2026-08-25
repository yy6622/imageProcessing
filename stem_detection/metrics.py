"""Quantitative comparison of the Traditional / YOLO / Hybrid stem-detection
techniques: detection rate, mean confidence, and mean processing time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .detector import StemDetector


@dataclass
class BenchmarkRow:
    filename: str
    fruit_type: str
    method: str
    detected: bool
    confidence: float
    processing_ms: float


def run_benchmark(
    images: list[tuple[str, str, np.ndarray]],
    methods: tuple[str, ...] = ("Traditional", "YOLO", "Hybrid"),
    detector: Optional[StemDetector] = None,
    yolo_confidence: float = 0.25,
) -> pd.DataFrame:
    """Run every method against every image; one row per (image, method).

    images: list of (filename, fruit_type, bgr_image) tuples.
    YOLO/Hybrid are skipped with a zero row if no model is loaded.
    """
    detector = detector or StemDetector()
    rows: list[BenchmarkRow] = []

    for filename, fruit_type, image in images:
        for method in methods:
            if method in ("YOLO", "Hybrid") and not detector.yolo_ready:
                rows.append(BenchmarkRow(filename, fruit_type, method, False, 0.0, 0.0))
                continue
            try:
                if method == "Traditional":
                    detections, elapsed, _ = detector.detect_traditional(image, fruit_type)
                elif method == "YOLO":
                    detections, elapsed = detector.detect_yolo(image, fruit_type, yolo_confidence)
                else:
                    detections, elapsed = detector.detect_hybrid(image, fruit_type, yolo_confidence)
            except FileNotFoundError:
                rows.append(BenchmarkRow(filename, fruit_type, method, False, 0.0, 0.0))
                continue

            if not detections:
                rows.append(BenchmarkRow(filename, fruit_type, method, False, 0.0, elapsed * 1000))
                continue

            best = max(detections, key=lambda d: d.confidence)
            rows.append(BenchmarkRow(filename, fruit_type, method, True, best.confidence, elapsed * 1000))

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-image benchmark rows into one row per method."""
    summary = df.groupby("method").agg(
        detected=("detected", "mean"),
        confidence=("confidence", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
        processing_ms=("processing_ms", "mean"),
    ).reset_index()
    summary = summary.rename(columns={
        "detected": "detection_rate",
        "confidence": "mean_confidence",
        "processing_ms": "mean_processing_ms",
    })
    return summary


def generate_analysis_text(summary: pd.DataFrame) -> str:
    """Turn the aggregated benchmark summary into a short written discussion."""
    if summary.empty:
        return "No benchmark results available."

    lines = []
    # Ties broken by speed, not alphabetically.
    top_rate = summary["detection_rate"].max()
    tied_for_top = summary[summary["detection_rate"] == top_rate]
    best_detection = tied_for_top.loc[tied_for_top["mean_processing_ms"].idxmin()]
    fastest = summary.loc[summary["mean_processing_ms"].idxmin()]

    if len(tied_for_top) > 1:
        other_tied = [m for m in tied_for_top["method"] if m != best_detection["method"]]
        lines.append(
            f"**{best_detection['method']}** and {', '.join(other_tied)} tied for the highest "
            f"detection rate ({top_rate:.0%}), but **{best_detection['method']}** did it faster "
            f"({best_detection['mean_processing_ms']:.0f} ms/image vs. the others), making it the "
            f"better result of the tie."
        )
    else:
        lines.append(
            f"**{best_detection['method']}** had the highest detection rate "
            f"({best_detection['detection_rate']:.0%}), while **{fastest['method']}** was fastest "
            f"({fastest['mean_processing_ms']:.0f} ms/image on average)."
        )

    if "mean_confidence" in summary.columns:
        top_confidence = summary["mean_confidence"].max()
        tied_for_confidence = summary[summary["mean_confidence"] == top_confidence]
        most_confident = tied_for_confidence.loc[tied_for_confidence["mean_processing_ms"].idxmin()]
        tie_note = " (tied with others, fastest shown)" if len(tied_for_confidence) > 1 else ""
        lines.append(
            f"**{most_confident['method']}** produced the most confident detections on average "
            f"({most_confident['mean_confidence']:.0%}){tie_note}."
        )

    if best_detection["method"] == fastest["method"]:
        lines.append(
            f"{best_detection['method']} led on both detection rate and speed in this run, "
            f"making it the clear choice for this dataset."
        )
    else:
        lines.append(
            f"This is an accuracy/speed trade-off: {best_detection['method']} finds more stems, "
            f"but {fastest['method']} responds faster — the right choice depends on whether the "
            f"deployment prioritises coverage or latency (e.g. a real-time sorting line vs. an "
            f"offline batch inspection)."
        )

    return "\n\n".join(lines)
