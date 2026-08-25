
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import calibration as calib
from colorDetection import inspect_image_yolo
import report as report_mod

try:
    import ultralytics  # noqa: F401  (only used to check availability up front)
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch  # noqa: F401  (only used to check availability up front)
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

CNN_MODELS_DIR = "cnn_quality_models"
try:
    from train_cnn_quality import CLASS_FOLDERS as _CLASS_FOLDERS
    CNN_FRUIT_TYPES = tuple(sorted({ftype for _, (ftype, _q) in _CLASS_FOLDERS.items()}))
except Exception:
    CNN_FRUIT_TYPES = ("Apple", "Banana", "Orange", "Mango", "Strawberry")
_cnn_models_present = (
    os.path.isdir(CNN_MODELS_DIR)
    and any(os.path.isfile(os.path.join(CNN_MODELS_DIR, f"{ft}.pt")) for ft in CNN_FRUIT_TYPES)
)
CNN_AVAILABLE = _TORCH_AVAILABLE and _cnn_models_present

st.set_page_config(page_title="Fruit Quality Inspection Dashboard", layout="wide")

DEFAULT_ERODE_PIXELS = 10
DEFAULT_YOLO_CONFIDENCE = 0.25
# Only used by the Stem Detection section's own ArUco-marker calibration —
# the main pipeline's calibration no longer uses marker size (manual
# cm-per-pixel only).
DEFAULT_MARKER_SIZE_CM = 5.0

DENOISE_METHOD = "median"
ENHANCE_METHOD = "clahe"


# ======================================================
# Helpers
# ======================================================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload_to_bgr(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def infer_freshness_from_filename(filename):
    """Best-effort Fresh/Rotten label from filename (e.g. "FreshApple (12).jpg")."""
    lower = filename.lower()
    if "fresh" in lower:
        return "Fresh"
    if "rotten" in lower or "spoiled" in lower or "stale" in lower:
        return "Rotten"
    return None


def stem_quality_tier(confidence):
    """Translate a raw confidence score into a plain-language tier."""
    if not confidence or confidence <= 0:
        return "No Reliable Stem Detected"
    if confidence >= 0.5:
        return "High Confidence"
    return "Review Recommended"


def summary_to_text(summary):
    parts = []
    for fruit_type, qualities in summary.items():
        quality_text = ", ".join(f"{n} {q}" for q, n in qualities.items())
        parts.append(f"{fruit_type}: {quality_text}")
    return "; ".join(parts) if parts else "No fruit detected"


# ======================================================
# Sidebar — minimal configuration
# ======================================================
st.sidebar.title("Inspection Settings")

IMPLEMENTED_TECHNIQUE = "Colour Feature Extraction"
TECHNIQUE_OPTIONS = [
    IMPLEMENTED_TECHNIQUE,
    "Morphological and Texture Feature Extraction",
    "Defect Detection",
    "Stem Detection",
    "All of the above",
]
selected_technique = st.sidebar.selectbox(
    "Technique / method (report section)",
    TECHNIQUE_OPTIONS,
    index=0,
    help="Matches the report's methodology sections. Only Colour Feature "
         "Extraction is implemented in this app (LAB chroma-distance "
         "segmentation + YOLO + CNN) — the others are placeholders for "
         "teammates' modules and don't change what actually runs below.",
)
if selected_technique not in (IMPLEMENTED_TECHNIQUE, "Stem Detection"):
    st.sidebar.caption(
        f"ℹ️ “{selected_technique}” isn't implemented in this module yet — "
        f"running the same {IMPLEMENTED_TECHNIQUE} pipeline (LAB + YOLO + CNN) below."
    )

# ======================================================
# Stem Detection — self-contained section, runs instead of the Colour
# Feature Extraction pipeline below.
# ======================================================
if selected_technique == "Stem Detection":
    from stem_detection import calibration as stem_calib
    from stem_detection import metrics as stem_metrics
    from stem_detection import preprocessing as stem_pp
    from stem_detection import report as stem_report
    from stem_detection.detector import StemDetector

    st.sidebar.subheader("Stem Detection settings")
    stem_fruit_type = "Fruit"
    stem_method = "Automatic"

    # Prefer the locally trained V4 weights, fall back to V3 then models/best.pt.
    _stem_model_candidates = [
        "runs/detect/fruit_stem_detector_v4/weights/best.pt",
        "runs/detect/fruit_stem_detector_v3/weights/best.pt",
        "models/best.pt",
    ]
    stem_model_path = next((p for p in _stem_model_candidates if os.path.isfile(p)), _stem_model_candidates[0])
    stem_yolo_confidence = 0.10
    if not YOLO_AVAILABLE:
        st.sidebar.warning(
            "YOLO is unavailable, so automatic stem detection will use the "
            "traditional image-processing fallback only."
        )

    @st.cache_resource
    def _get_stem_detector(model_path: str) -> StemDetector:
        return StemDetector(model_path=model_path)

    stem_detector = _get_stem_detector(stem_model_path)
    if YOLO_AVAILABLE and not stem_detector.yolo_ready:
        st.sidebar.warning(
            f"No YOLO stem model found at `{stem_model_path}`. Automatic mode will "
            "continue with the traditional image-processing fallback."
        )

    stem_denoise_method = "median"
    stem_enhance_method = "clahe"

    st.sidebar.divider()
    stem_want_calibration = st.sidebar.checkbox("Measure physical size (cm)", value=False, key="stem_want_calib")
    stem_calib_mode = None
    stem_marker_cm = DEFAULT_MARKER_SIZE_CM
    stem_manual_ratio = None
    if stem_want_calibration:
        stem_calib_mode = st.sidebar.radio(
            "How is scale determined?",
            ["Auto-detect ArUco marker in photo", "I know my cm-per-pixel ratio"],
            key="stem_calib_mode",
        )
        if stem_calib_mode.startswith("Auto"):
            stem_marker_cm = st.sidebar.number_input(
                "Marker side length (cm)", value=DEFAULT_MARKER_SIZE_CM, min_value=0.1, step=0.5, key="stem_marker_cm",
            )
        else:
            stem_manual_ratio = st.sidebar.number_input(
                "cm per pixel", value=0.02, min_value=0.0001, step=0.001, format="%.4f", key="stem_manual_ratio",
            )

    def _get_stem_calibration(image):
        if not stem_want_calibration:
            return stem_calib.uncalibrated()
        if stem_calib_mode.startswith("Auto"):
            return stem_calib.calibrate(image, marker_size_cm=stem_marker_cm)
        return stem_calib.manual_scale(stem_manual_ratio)

    st.title("Stem Detection")
    st.caption(
        "Automatically localises apple, banana and orange stems/crowns/calyxes using "
        "a combined YOLO + classical image-processing pipeline."
    )

    stem_tab_images, stem_tab_benchmark = st.tabs(
        ["Image inspection", "Method benchmark"]
    )

    # --- Image inspection ---
    with stem_tab_images:
        stem_uploaded_files = st.file_uploader(
            "Select one or more images", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True, key="stem_image_uploader",
        )
        stem_images_to_process = []
        if stem_uploaded_files:
            for f in stem_uploaded_files:
                img = read_upload_to_bgr(f)
                if img is not None:
                    stem_images_to_process.append((f.name, img))

        # Automatic mode can still run when the YOLO model is unavailable,
        # because detector.py contains a traditional fallback.
        stem_run_disabled = len(stem_images_to_process) == 0
        stem_run_button = st.button(
            "Run detection", type="primary", disabled=stem_run_disabled, key="stem_run_images",
        )

        if "stem_image_results" not in st.session_state:
            st.session_state["stem_image_results"] = []

        if stem_run_button:
            stem_results = []
            stem_progress = st.progress(0.0, text="Running detection...")
            for i, (name, img) in enumerate(stem_images_to_process):
                processed = stem_pp.preprocess(img, stem_denoise_method, stem_enhance_method)
                calibration_result = _get_stem_calibration(img)
                detections, elapsed, method_used = stem_detector.detect(
                    processed, stem_fruit_type, stem_method, stem_yolo_confidence, skip_preprocess=True,
                )
                annotated = stem_detector.annotate(processed, detections)
                if calibration_result.marker_corners is not None:
                    annotated = stem_calib.draw_marker_overlay(annotated, calibration_result)
                stem_results.append({
                    "filename": name, "original": img, "processed": processed, "annotated": annotated,
                    "detections": detections, "fruit_type": stem_fruit_type, "method": stem_method,
                    "method_used": method_used,
                    "candidate_mask": stem_detector.traditional_mask(processed),
                    "calibration": calibration_result, "processing_ms": elapsed * 1000,
                })
                stem_progress.progress((i + 1) / len(stem_images_to_process), text=f"Processed {name}")
            stem_progress.empty()
            st.session_state["stem_image_results"] = stem_results

        stem_image_results = st.session_state["stem_image_results"]

        if stem_image_results:
            st.divider()
            st.header("Summary")
            stem_total = sum(len(r["detections"]) for r in stem_image_results)
            stem_found_in = sum(1 for r in stem_image_results if r["detections"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Photos inspected", len(stem_image_results))
            c2.metric("Stems detected", stem_total)
            c3.metric("Photos with a stem found", f"{stem_found_in}/{len(stem_image_results)}")
            c4.metric("Avg. processing time", f"{np.mean([r['processing_ms'] for r in stem_image_results]):.0f} ms")

            stem_table_rows = []
            for r in stem_image_results:
                best = max(r["detections"], key=lambda d: d.confidence) if r["detections"] else None
                size_cm = "-"
                if best is not None and r["calibration"].is_calibrated:
                    x1, y1, x2, y2 = best.bbox
                    w_cm = r["calibration"].px_to_cm(x2 - x1)
                    h_cm = r["calibration"].px_to_cm(y2 - y1)
                    size_cm = f"{w_cm:.1f} x {h_cm:.1f}"
                stem_table_rows.append({
                    "Image": r["filename"], "Method": r["method"],
                    "Detected via": r["method_used"].capitalize(),
                    "Stems found": len(r["detections"]),
                    "Best confidence": f"{best.confidence:.0%}" if best else "-",
                    "Quality": stem_quality_tier(best.confidence if best else None),
                    "Size (cm)": size_cm, "Time (ms)": f"{r['processing_ms']:.0f}",
                })
            stem_df = pd.DataFrame(stem_table_rows)
            st.dataframe(stem_df, use_container_width=True)

            stem_csv_bytes = stem_df.to_csv(index=False).encode("utf-8")
            stem_dl1, stem_dl2 = st.columns(2)
            stem_dl1.download_button("Download results as CSV", stem_csv_bytes, "stem_detection_results.csv", "text/csv")

            if stem_dl2.button("Generate PDF report", key="stem_gen_pdf"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    stem_report.generate_report(stem_image_results, output_path=tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.session_state["stem_pdf_bytes"] = f.read()
                os.unlink(tmp.name)
            if "stem_pdf_bytes" in st.session_state:
                stem_dl2.download_button(
                    "Download PDF report", st.session_state["stem_pdf_bytes"],
                    "stem_detection_report.pdf", "application/pdf",
                )

            stem_freshness_rows = []
            for r in stem_image_results:
                freshness = infer_freshness_from_filename(r["filename"])
                if freshness is None:
                    continue
                best = max(r["detections"], key=lambda d: d.confidence) if r["detections"] else None
                stem_freshness_rows.append({
                    "freshness": freshness,
                    "stem_detected": bool(r["detections"]),
                    "best_confidence": best.confidence if best else 0.0,
                })

            if stem_freshness_rows:
                st.divider()
                st.header("Stem visibility vs. freshness")
                st.caption("Freshness inferred from filename (e.g. \"FreshApple\"). Images without a hint are excluded.")
                stem_fresh_df = pd.DataFrame(stem_freshness_rows)
                stem_fresh_summary = stem_fresh_df.groupby("freshness").agg(
                    stem_detection_rate=("stem_detected", "mean"),
                    mean_confidence=("best_confidence", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
                    photos=("stem_detected", "size"),
                ).reset_index()
                st.dataframe(stem_fresh_summary, use_container_width=True)

                fresh_chart1, fresh_chart2 = st.columns(2)
                fresh_chart1.caption("Stem detection rate by freshness")
                fresh_chart1.bar_chart(stem_fresh_summary.set_index("freshness")["stem_detection_rate"])
                fresh_chart2.caption("Mean stem confidence by freshness")
                fresh_chart2.bar_chart(stem_fresh_summary.set_index("freshness")["mean_confidence"])

                fresh_row = stem_fresh_summary[stem_fresh_summary["freshness"] == "Fresh"]
                rotten_row = stem_fresh_summary[stem_fresh_summary["freshness"] == "Rotten"]
                if not fresh_row.empty and not rotten_row.empty:
                    rate_diff = fresh_row["stem_detection_rate"].iloc[0] - rotten_row["stem_detection_rate"].iloc[0]
                    if abs(rate_diff) >= 0.1:
                        direction = "higher" if rate_diff > 0 else "lower"
                        st.info(f"Stem detection rate was {abs(rate_diff):.0%} {direction} on Fresh vs Rotten photos.")
                    else:
                        st.info("Stem detection rate was similar between Fresh and Rotten photos.")

            st.divider()
            st.header("Per-image detail")
            for r in stem_image_results:
                title = f"{r['filename']} - {len(r['detections'])} stem(s) found" if r["detections"] \
                    else f"{r['filename']} - No visible stem detected"
                with st.expander(title):
                    best = max(r["detections"], key=lambda d: d.confidence) if r["detections"] else None
                    quality = stem_quality_tier(best.confidence if best else None)
                    if quality == "High Confidence":
                        st.success(f"{quality} — detected via {r['method_used'].capitalize()}")
                    elif quality == "Review Recommended":
                        st.warning(f"{quality} — detected via {r['method_used'].capitalize()}")
                    else:
                        st.error(quality)

                    ic1, ic2 = st.columns(2)
                    ic1.image(bgr_to_rgb(r["original"]), caption="Original", use_container_width=True)
                    ic2.image(bgr_to_rgb(r["annotated"]), caption="Stem detection (bbox + contour)",
                               use_container_width=True)
                    if r["calibration"].is_calibrated:
                        st.caption(f"Calibration: {r['calibration'].method} - {r['calibration'].confidence}")

                    if st.toggle("View Detection Details (pipeline steps)", key=f"stem_pipeline_{r['filename']}"):
                        pc1, pc2, pc3, pc4 = st.columns(4)
                        pc1.image(bgr_to_rgb(r["original"]), caption="1. Original", use_container_width=True)
                        pc2.image(bgr_to_rgb(r["processed"]), caption="2. Preprocessed (denoise + enhance)",
                                   use_container_width=True)
                        pc3.image(r["candidate_mask"], caption="3. Candidate regions (stem-colour mask)",
                                   use_container_width=True)
                        pc4.image(bgr_to_rgb(r["annotated"]), caption="4. Final detection",
                                   use_container_width=True)
        else:
            st.info("Upload one or more images, then click **Run detection**.")

    # --- Method benchmark (Mode A comparative evaluation) ---
    with stem_tab_benchmark:
        st.write(
            "Compares three stem-detection techniques on the same images — Traditional "
            "(classical image processing), YOLO (deep learning), and Hybrid (both combined) "
            "— to show which one performs best."
        )

        stem_bench_images = [(r["filename"], "Fruit", r["original"]) for r in stem_image_results]

        if not stem_bench_images:
            st.info("Upload and run images in **Image inspection** first, then come back here to benchmark them.")
        else:
            if st.button("Run benchmark", type="primary", key="stem_run_bench"):
                with st.spinner("Running all methods..."):
                    stem_bench_df = stem_metrics.run_benchmark(
                        stem_bench_images, methods=("Traditional", "YOLO", "Hybrid"),
                        detector=stem_detector, yolo_confidence=stem_yolo_confidence,
                    )
                    stem_bench_visuals = []
                    for stem_bv_name, stem_bv_fruit, stem_bv_img in stem_bench_images:
                        stem_bv_trad, _, _ = stem_detector.detect_traditional(stem_bv_img, stem_bv_fruit)
                        stem_bv_yolo, _ = stem_detector.detect_yolo(stem_bv_img, stem_bv_fruit, stem_yolo_confidence)
                        stem_bv_hybrid, _ = stem_detector.detect_hybrid(stem_bv_img, stem_bv_fruit, stem_yolo_confidence)
                        stem_bench_visuals.append({
                            "filename": stem_bv_name,
                            "Traditional": stem_detector.annotate(stem_bv_img, stem_bv_trad),
                            "YOLO": stem_detector.annotate(stem_bv_img, stem_bv_yolo),
                            "Hybrid": stem_detector.annotate(stem_bv_img, stem_bv_hybrid),
                        })
                st.session_state["stem_bench_df"] = stem_bench_df
                st.session_state["stem_bench_summary"] = stem_metrics.summarize(stem_bench_df)
                st.session_state["stem_bench_visuals"] = stem_bench_visuals

            if "stem_bench_summary" in st.session_state:
                st.subheader("Method comparison")
                stem_summary = st.session_state["stem_bench_summary"]
                st.dataframe(stem_summary, use_container_width=True)

                stem_chart1, stem_chart2 = st.columns(2)
                stem_chart1.caption("Detection rate by method")
                stem_chart1.bar_chart(stem_summary.set_index("method")["detection_rate"])
                stem_chart2.caption("Mean processing time (ms) by method")
                stem_chart2.bar_chart(stem_summary.set_index("method")["mean_processing_ms"])

                stem_top_rate = stem_summary["detection_rate"].max()
                stem_tied_top = stem_summary[stem_summary["detection_rate"] == stem_top_rate]
                stem_best_method = stem_tied_top.loc[stem_tied_top["mean_processing_ms"].idxmin(), "method"]
                st.success(f"Highest detection rate on this set: **{stem_best_method}**")

                stem_analysis_text = stem_metrics.generate_analysis_text(stem_summary)
                st.subheader("Analysis")
                st.markdown(stem_analysis_text)

                st.subheader("Visual comparison")
                for stem_bv in st.session_state.get("stem_bench_visuals", []):
                    with st.expander(stem_bv["filename"]):
                        stem_bv_c1, stem_bv_c2, stem_bv_c3 = st.columns(3)
                        stem_bv_c1.image(bgr_to_rgb(stem_bv["Traditional"]), caption="Traditional",
                                          use_container_width=True)
                        stem_bv_c2.image(bgr_to_rgb(stem_bv["YOLO"]), caption="YOLO", use_container_width=True)
                        stem_bv_c3.image(bgr_to_rgb(stem_bv["Hybrid"]), caption="Hybrid", use_container_width=True)

                with st.expander("Per-image, per-method detail"):
                    st.dataframe(
                        st.session_state["stem_bench_df"].drop(columns=["fruit_type"]),
                        use_container_width=True,
                    )

                if st.button("Generate benchmark PDF report", key="stem_bench_pdf_btn"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        stem_report.generate_report(
                            [], output_path=tmp.name, benchmark_summary=stem_summary,
                            title="Stem Detection Method Benchmark", analysis_text=stem_analysis_text,
                        )
                        with open(tmp.name, "rb") as f:
                            stem_bench_pdf_bytes = f.read()
                    os.unlink(tmp.name)
                    st.download_button(
                        "Download benchmark PDF", stem_bench_pdf_bytes,
                        "stem_detection_benchmark.pdf", "application/pdf", key="stem_bench_pdf_dl",
                    )

    st.stop()

st.sidebar.divider()

if not YOLO_AVAILABLE:
    st.sidebar.error(
        "ultralytics (YOLO) isn't installed — detection can't run. "
        "Run `pip install ultralytics` and restart the app."
    )

if not CNN_AVAILABLE:
    if _TORCH_AVAILABLE:
        st.sidebar.warning(
            "No cnn_quality_models/*.pt found. Run `python train_cnn_quality.py` to train one per fruit type — "
            "until then, Quality will show as unavailable."
        )
    else:
        st.sidebar.warning(
            "PyTorch not installed. Run `pip install torch torchvision`, then `python train_cnn_quality.py` — "
            "until then, Quality will show as unavailable."
        )

st.sidebar.divider()
want_measurements = st.sidebar.checkbox("Measure physical size (cm)", value=False)
manual_cm_per_pixel = None
if want_measurements:
    manual_cm_per_pixel = st.sidebar.number_input(
        "cm per pixel", value=0.02, min_value=0.0001, step=0.001, format="%.4f",
        help="Known scale for your camera setup (e.g. derived once from a ruler photo).",
    )

with st.sidebar.expander("Perspective rectification (optional)"):
    st.caption(
        "Straightens the image plane before measurement if the camera isn't "
        "perfectly perpendicular to the inspection surface."
    )
    want_rectify = st.checkbox("Enable perspective rectification", value=False)
    rectify_mode = "Auto-detect (largest rectangle in photo)"
    rectify_points = None
    if want_rectify:
        rectify_mode = st.radio("How to find the 4 corners?",
                                 ["Auto-detect (largest rectangle in photo)", "Enter coordinates manually"])
        if rectify_mode.startswith("Auto"):
            st.caption(
                "Looks for the largest 4-sided shape in each photo (e.g. a tray, "
                "card, or table edge) and uses it as the reference. Less reliable "
                "than a coded marker — no unique identity to lock onto, so it can "
                "pick the wrong shape in a cluttered frame. Falls back to no "
                "rectification for a photo if nothing suitable is found."
            )
        else:
            st.caption(
                "Pixel coordinates of the 4 corners of a flat reference region "
                "(e.g. the tray/table edges), any order."
            )
            rc1, rc2 = st.columns(2)
            x1 = rc1.number_input("Corner 1 — x", value=0, step=1)
            y1 = rc2.number_input("Corner 1 — y", value=0, step=1)
            x2 = rc1.number_input("Corner 2 — x", value=100, step=1)
            y2 = rc2.number_input("Corner 2 — y", value=0, step=1)
            x3 = rc1.number_input("Corner 3 — x", value=100, step=1)
            y3 = rc2.number_input("Corner 3 — y", value=100, step=1)
            x4 = rc1.number_input("Corner 4 — x", value=0, step=1)
            y4 = rc2.number_input("Corner 4 — y", value=100, step=1)
            rectify_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

with st.sidebar.expander("Advanced settings"):
    erode_pixels = st.slider("Mask erosion (px)", 0, 30, DEFAULT_ERODE_PIXELS,
                              help="Trims mixed-color boundary patches between fruit and background.")
    yolo_confidence = st.slider("YOLO confidence threshold", 0.05, 0.9, DEFAULT_YOLO_CONFIDENCE, step=0.05,
                                 help="Lower catches more heavily-occluded fruit at the cost of more false positives.")


def get_calibration(image):
    if not want_measurements:
        return calib.uncalibrated()
    return calib.manual_scale(manual_cm_per_pixel)


# ======================================================
# Main — image ingestion
# ======================================================
st.title("Fruit Quality Inspection Dashboard")
st.caption("Detects every fruit in each photo, classifies its type and quality, and reports the results.")

images_to_process = []  # list of (name, bgr_image)

uploaded_files = st.file_uploader(
    "Select one or more images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True
)
if uploaded_files:
    for f in uploaded_files:
        img = read_upload_to_bgr(f)
        if img is not None:
            images_to_process.append((f.name, img))

run_button = st.button(
    "Run Inspection", type="primary",
    disabled=(len(images_to_process) == 0 or not YOLO_AVAILABLE),
)
if not YOLO_AVAILABLE and images_to_process:
    st.warning("Can't run — ultralytics (YOLO) isn't installed. See the sidebar for install instructions.")

if "results" not in st.session_state:
    st.session_state["results"] = []

if run_button:
    results = []
    progress = st.progress(0.0, text="Running pipeline...")
    for i, (name, img) in enumerate(images_to_process):
        if want_rectify:
            if rectify_mode.startswith("Auto"):
                auto_quad = calib.detect_reference_quad(img)
                if auto_quad is not None:
                    img = calib.rectify_perspective(img, auto_quad)
            elif rectify_points is not None:
                img = calib.rectify_perspective(img, rectify_points)
        calibration_result = get_calibration(img)
        out = inspect_image_yolo(
            img,
            calibration=calibration_result,
            denoise_method=DENOISE_METHOD,
            enhance_method=ENHANCE_METHOD,
            erode_pixels=erode_pixels,
            yolo_confidence=yolo_confidence,
        )
        out["filename"] = name
        results.append(out)
        progress.progress((i + 1) / len(images_to_process), text=f"Processed {name}")
    progress.empty()
    st.session_state["results"] = results

results = st.session_state["results"]

# ======================================================
# Dashboard — summary
# ======================================================
if results:
    st.divider()
    st.header("Summary")

    all_objects = [obj for r in results for obj in r["objects"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Photos inspected", len(results))
    col2.metric("Fruits detected", len(all_objects))
    fresh_n = sum(1 for o in all_objects if o.get("label") == "Fresh")
    rotten_n = sum(1 for o in all_objects if o.get("label") == "Rotten")
    col3.metric("Fresh", fresh_n)
    col4.metric("Rotten", rotten_n)

    chart_col1, chart_col2 = st.columns(2)
    fruit_type_counts = pd.Series([o.get("fruit_type") or "Unknown" for o in all_objects]).value_counts()
    chart_col1.caption("By fruit type")
    chart_col1.bar_chart(fruit_type_counts)
    label_counts = pd.Series([o.get("label") or "Unclassified" for o in all_objects]).value_counts()
    chart_col2.caption("By quality")
    chart_col2.bar_chart(label_counts)

    st.subheader("Every fruit detected")
    table_rows = []
    for r in results:
        for obj in r["objects"]:
            table_rows.append({
                "Image": r["filename"],
                "#": obj["index"] + 1,
                "Fruit Type": obj.get("fruit_type") or "—",
                "Type Conf": f"{obj.get('fruit_type_confidence', 0) * 100:.1f}%" if obj.get("fruit_type") else "—",
                "Quality": obj.get("label") or "—",
                "Quality Conf": f"{obj.get('confidence', 0) * 100:.1f}%" if obj.get("label") else "—",
                "Wound %": f"{obj.get('defect_fraction', 0) * 100:.1f}%",
                "Width (cm)": f"{obj['width_cm']:.2f}" if obj.get("width_cm") is not None else "—",
                "Height (cm)": f"{obj['height_cm']:.2f}" if obj.get("height_cm") is not None else "—",
                "Area (cm^2)": f"{obj['area_cm2']:.2f}" if obj.get("area_cm2") is not None else "—",
                "Width (px)": f"{obj.get('width_px', 0):.0f}",
                "Height (px)": f"{obj.get('height_px', 0):.0f}",
            })
    df = pd.DataFrame(table_rows)
    st.dataframe(df, width="stretch")

    csv = df.to_csv(index=False).encode("utf-8")
    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button("Download results as CSV", csv, "inspection_results.csv", "text/csv")

    if dl_col2.button("Generate PDF report"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            report_mod.generate_report(results, output_path=tmp.name)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
        st.session_state["pdf_bytes"] = pdf_bytes

    if "pdf_bytes" in st.session_state:
        dl_col2.download_button(
            "Download PDF report", st.session_state["pdf_bytes"],
            "inspection_report.pdf", "application/pdf",
        )

    st.divider()
    st.header("Per-photo detail")
    for r in results:
        with st.expander(f"{r['filename']}  —  {r['count']} fruit(s): {summary_to_text(r['summary'])}"):
            c1, c2 = st.columns(2)
            c1.image(bgr_to_rgb(r["original"]), caption="Original", width="stretch")
            c2.image(bgr_to_rgb(r["annotated"]), caption="Detected fruits (bbox + contour + label)", width="stretch")

            if not r["objects"]:
                st.warning("No fruit detected in this photo.")
                continue

            st.markdown(f"**Separated fruit ({r['count']}):**")
            crop_cols = st.columns(len(r["objects"]))
            for obj, col in zip(r["objects"], crop_cols):
                caption = f"#{obj['index'] + 1} {obj.get('fruit_type') or '?'} {obj.get('label') or '?'}"
                col.image(bgr_to_rgb(obj["crop_isolated"]), caption=caption, width="stretch")

            for obj in r["objects"]:
                st.markdown(f"**Fruit #{obj['index'] + 1}**")
                cols = st.columns(4)
                cls = obj.get("classification")

                cols[0].write(f"Type: **{obj.get('fruit_type') or 'N/A'}** "
                               f"({obj.get('fruit_type_confidence', 0) * 100:.0f}%, YOLO)")
                cols[1].write(f"Quality: **{obj.get('label') or 'N/A'}** "
                               f"({obj.get('confidence', 0) * 100:.0f}%, CNN)")
                if obj.get("width_cm") is not None:
                    cols[2].write(f"Size: {obj['width_cm']:.1f} × {obj['height_cm']:.1f} cm")
                    cols[3].write(f"Area: {obj['area_cm2']:.1f} cm²")
                else:
                    cols[2].write(f"Size: {obj.get('width_px', 0):.0f} × {obj.get('height_px', 0):.0f} px")
                    cols[3].write(f"Area: {obj.get('area_px', 0):.0f} px²")

                if cls is not None and cls.error:
                    st.caption(f"⚠️ {cls.error}")
                st.divider()

            st.caption(f"Calibration: {r.get('calibration_method')} ({r.get('calibration_confidence')})")
else:
    st.info("Upload images or point to a folder, then click **Run Inspection**.")