"""
Run with:
    streamlit run app.py
"""

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
CNN_FRUIT_TYPES = ("Apple", "Banana", "Orange")
_cnn_models_present = (
    os.path.isdir(CNN_MODELS_DIR)
    and any(os.path.isfile(os.path.join(CNN_MODELS_DIR, f"{ft}.pt")) for ft in CNN_FRUIT_TYPES)
)
CNN_AVAILABLE = _TORCH_AVAILABLE and _cnn_models_present

st.set_page_config(page_title="Fruit Quality Inspection Dashboard", layout="wide")

# Backend defaults for everything that isn't exposed in the main UI.
# Exposed only inside the "Advanced settings" expander below.
DEFAULT_ERODE_PIXELS = 10
DEFAULT_YOLO_CONFIDENCE = 0.25

# Denoise/enhance are fixed rather than user-selectable — median +
# CLAHE cover the two jobs (salt-and-pepper noise removal, local
# contrast boost) well enough for this pipeline that exposing every
# alternative in the sidebar just added clutter without a real
# use case for switching them at inspection time.
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


def summary_to_text(summary):
    """{"Apple": {"Fresh": 2, "Rotten": 1}} -> 'Apple: 2 Fresh, 1 Rotten'"""
    parts = []
    for fruit_type, qualities in summary.items():
        quality_text = ", ".join(f"{n} {q}" for q, n in qualities.items())
        parts.append(f"{fruit_type}: {quality_text}")
    return "; ".join(parts) if parts else "No fruit detected"


# ======================================================
# Sidebar — minimal configuration
# ======================================================
st.sidebar.title("Inspection Settings")

# Methodology-section selector — matches the report's 2.1.x technique
# breakdown (each teammate owns one section). Only 2.1.1 Colour Feature
# Extraction is wired up in this module (LAB chroma-distance
# segmentation + YOLO + CNN, below); the other sections belong to
# teammates' modules and aren't implemented here yet. Selecting one of
# them does NOT change what actually runs — the pipeline below always
# executes the same Colour Feature Extraction path regardless of this
# choice. This is a label/display selector, not a functional switch.
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
if selected_technique != IMPLEMENTED_TECHNIQUE:
    st.sidebar.caption(
        f"ℹ️ “{selected_technique}” isn't implemented in this module yet — "
        f"running the same {IMPLEMENTED_TECHNIQUE} pipeline (LAB + YOLO + CNN) below."
    )

st.sidebar.divider()

# Detection + fruit TYPE: YOLOv8 only — no classical/SVM fallback.
# Quality: CNN only — no SVM fallback either. Both stay silent in the
# sidebar when working normally (no reassuring "it's fine" banner for
# steady-state operation) — only surfaced here as a warning/error when
# something's actually missing, since that's the only time the user
# needs to act on it.
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
ref_width_cm = None
ref_width_px = None
scale_mode = "I know my cm-per-pixel ratio"
if want_measurements:
    scale_mode = st.sidebar.radio(
        "How is scale determined?",
        ["I know my cm-per-pixel ratio", "I know a reference object's width (cm + px)"],
    )
    if scale_mode.startswith("I know my cm"):
        manual_cm_per_pixel = st.sidebar.number_input(
            "cm per pixel", value=0.02, min_value=0.0001, step=0.001, format="%.4f",
            help="Known scale for your camera setup (e.g. derived once from a ruler photo).",
        )
    else:
        ref_width_cm = st.sidebar.number_input(
            "Reference object width (cm)", value=8.56, min_value=0.01, step=0.1,
            help="Real-world width of a reference object visible in the photo (e.g. a card).",
        )
        ref_width_px = st.sidebar.number_input(
            "Reference object width in photo (px)", value=240.0, min_value=1.0, step=1.0,
            help="How wide that same reference object measures in the photo, in pixels.",
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
    if scale_mode.startswith("I know my cm"):
        return calib.manual_scale(manual_cm_per_pixel)
    return calib.manual_reference(ref_width_cm, ref_width_px)


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
