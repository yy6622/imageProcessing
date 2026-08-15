"""
app.py
==============================================
Streamlit dashboard for the fruit quality inspection system.

Ties together every other module into one GUI:
    - preprocessing.py   (noise removal + enhancement)
    - calibration.py     (pixel -> physical unit scaling)
    - colorDetection.py  (YOLO detection/fruit-type + CNN quality classification)
    - report.py          (PDF export)

Satisfies:
    - "Data Analysis Dashboard: An interface to summarise object
      properties and inspection/enhancement results."
    - (Extra effort) "GUI: Support for bulk image ingestion, such as
      folder-based uploads or multi-image selection."
    - (Extra effort) "Reporting: Automated export ... into PDF format."
      (wired to report.generate_report via a download button)

ARCHITECTURE NOTE (as of this version): the original SVM-based pipeline
(patch+majority-vote classifiers for both fruit TYPE and QUALITY,
classical distance-transform/Hough/watershed detection) has been
removed entirely, per an explicit decision to drop SVM from this
project — not just disconnected from this app, but deleted from
colorDetection.py (inspect_image(), classify_fruit_type(),
classify_segmented_image(), classify_segmented_image_known_type(),
detect_objects(), and friends are all gone from that file now). This
app now runs YOLOv8 (pretrained on COCO — apple/banana/orange are
already 3 of its 80 classes, zero extra training needed) for detection
+ fruit type, and a CNN (train_cnn_quality.py, one MobileNetV2 model
per fruit type) for quality — an explicit choice, not a default: the
SVM quality path had a confirmed structural blind spot (patch-vote
majority washes out small localized decay) and measured ~73% held-out
accuracy vs. the CNN's ~98%; YOLO recovers meaningfully more instances
than the classical splitter on densely packed/overlapping fruit. There
is no SVM fallback anymore — if ultralytics or a trained CNN model
isn't available, this app says so plainly rather than silently
degrading. segmentation.py has since been trimmed the same
way: its SVM/RandomForest/KNN training functions, the classical
multi-fruit watershed/Hough splitter, and every hand-engineered
feature extractor were all deleted outright (unused code, not kept
"just in case") — see that file's module docstring. All that's left
there is segmentation_mask_and_contour(), which colorDetection.py
still imports for local re-segmentation within each YOLO box.

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
DEFAULT_MARKER_SIZE_CM = 5.0


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
marker_size_cm = DEFAULT_MARKER_SIZE_CM
manual_cm_per_pixel = None
calib_mode = "None (pixels only)"
if want_measurements:
    calib_mode = st.sidebar.radio(
        "How is scale determined?",
        ["Auto-detect ArUco marker in photo", "I know my cm-per-pixel ratio"],
    )
    if calib_mode.startswith("Auto"):
        marker_size_cm = st.sidebar.number_input("Marker side length (cm)", value=DEFAULT_MARKER_SIZE_CM, min_value=0.1, step=0.5)
        st.sidebar.caption("Falls back to pixels-only for any photo without a detected marker.")
    else:
        manual_cm_per_pixel = st.sidebar.number_input(
            "cm per pixel", value=0.02, min_value=0.0001, step=0.001, format="%.4f"
        )

with st.sidebar.expander("Advanced settings"):
    st.caption(
        "Denoise/enhance here only affect image display quality and the local "
        "re-segmentation used to build each fruit's mask/crop within its YOLO box — "
        "neither YOLO nor the CNN quality model needs these to match any particular "
        "training preprocessing (unlike the old SVM pipeline)."
    )
    denoise_method = st.selectbox("Denoise method", ["median", "gaussian", "bilateral"], index=0)
    enhance_method = st.selectbox("Enhance method", ["clahe", "histogram_equalize", "contrast_stretch", "none"], index=0)
    erode_pixels = st.slider("Mask erosion (px)", 0, 30, DEFAULT_ERODE_PIXELS,
                              help="Trims mixed-color boundary patches between fruit and background.")
    yolo_confidence = st.slider("YOLO confidence threshold", 0.05, 0.9, DEFAULT_YOLO_CONFIDENCE, step=0.05,
                                 help="Lower catches more heavily-occluded fruit at the cost of more false positives.")


def get_calibration(image):
    if not want_measurements:
        return calib.uncalibrated()
    if calib_mode.startswith("Auto"):
        return calib.calibrate(image, marker_size_cm=marker_size_cm, manual_cm_per_pixel=None)
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
        calibration_result = get_calibration(img)
        out = inspect_image_yolo(
            img,
            calibration=calibration_result,
            denoise_method=denoise_method,
            enhance_method=enhance_method,
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
                "Width (cm)": f"{obj['width_cm']:.2f}" if obj.get("width_cm") is not None else "—",
                "Height (cm)": f"{obj['height_cm']:.2f}" if obj.get("height_cm") is not None else "—",
                "Area (cm^2)": f"{obj['area_cm2']:.2f}" if obj.get("area_cm2") is not None else "—",
                "Width (px)": f"{obj.get('width_px', 0):.0f}",
                "Height (px)": f"{obj.get('height_px', 0):.0f}",
            })
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True)

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
            c1.image(bgr_to_rgb(r["original"]), caption="Original", use_container_width=True)
            c2.image(bgr_to_rgb(r["annotated"]), caption="Detected fruits (bbox + contour + label)", use_container_width=True)

            if not r["objects"]:
                st.warning("No fruit detected in this photo.")
                continue

            st.markdown(f"**Separated fruit ({r['count']}):**")
            crop_cols = st.columns(len(r["objects"]))
            for obj, col in zip(r["objects"], crop_cols):
                caption = f"#{obj['index'] + 1} {obj.get('fruit_type') or '?'} {obj.get('label') or '?'}"
                col.image(bgr_to_rgb(obj["crop_isolated"]), caption=caption, use_container_width=True)

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
