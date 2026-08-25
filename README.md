# Stem Detection - Comparative & Enhancement Study (Mode A)

Individual technique module: **Stem Detection** for apple, banana, and
orange, comparing three approaches (Traditional CV / YOLO / Hybrid) as
required by Mode A.

## Integration with the shared team app

`app.py` is the **shared team dashboard** (Fruit Quality Inspection
Dashboard - fruit type + freshness via YOLO/CNN, from the Colour Feature
Extraction section). Per the file-responsibility spec, all Stem Detection
logic lives in the `stem_detection/` package and is imported by `app.py`;
the interface itself stays in `app.py`.

Selecting **"Stem Detection"** from the sidebar's technique dropdown
switches the whole page to a dedicated Stem Detection section (own file
uploader, own settings, own results/PDF/CSV) built entirely from
`stem_detection/*` - it does not touch or depend on the Colour Feature
Extraction pipeline (`colorDetection.py`, the CNN quality models, or the
top-level `calibration.py`/`report.py` those use), since I don't have
visibility into those teammates' implementations and didn't want to risk
breaking them. If another teammate's section needs to reuse anything here
(e.g. the calibration or PDF logic), it's all plain, independently
importable functions in `stem_detection/`.

## Project structure

```
stem_detection/
  detector.py        Traditional / YOLO / Hybrid stem detection (bbox + contour)
  preprocessing.py    Noise removal + enhancement (Core Requirement)
  calibration.py       ArUco-marker spatial calibration (Core Requirement)
  metrics.py             Detection rate/confidence/speed/IoU + precision/recall/F1-score
  report.py              PDF report generation (Extra Effort)
  video.py                 Video frame-by-frame processing (Extra Effort)
  __init__.py
app.py                Shared team dashboard; Stem Detection section added, gated behind
                        the technique selector, everything else untouched
train_stem_yolo.py    Script to train the YOLO model used by detect_yolo/detect_hybrid
csv_to_yolo_labels.py Converts pixel-box CSV annotations into YOLO label format
dataset/               YOLO training data folder (data.yaml + images/labels)
requirements.txt
```

## Why Mode A fits this module

Mode A asks each member to implement a *different contemporary technique*
for the same sub-problem, then benchmark them quantitatively. Here that's:

- **Traditional**: HSV colour thresholding + morphological analysis + shape
  metrics (an isoperimetric "thinness" measure that stays accurate even when
  a stem curves, not just a straight bounding-box aspect ratio). No trained
  model needed; fully explainable.
- **YOLO**: a trained object detector (`detect_yolo`), learns stem
  appearance directly from labelled examples instead of hand-tuned
  thresholds - meant to handle cases the traditional method structurally
  can't (heavy bruising, cluttered/multi-fruit scenes).
- **Hybrid**: takes YOLO's boxes and only keeps ones whose region also
  looks stem-coloured under the traditional colour mask - a cheap way to
  cut YOLO false positives without extra training data.

`stem_detection/metrics.py` runs all three against the same image set and
reports detection rate, mean confidence, and mean processing time per
method (plus mean IoU if you supply ground-truth boxes) - this is the
"Experimental Evaluation: compare techniques using quantitative metrics"
requirement, and it's live in the dashboard's "Method benchmark" tab, not
just a one-off script.

## Requirement -> code mapping

| Requirement | Where |
|---|---|
| Preprocessing (noise removal + enhancement) | `stem_detection/preprocessing.py`; selectable in the app sidebar |
| Image Calibration (spatial scaling) | `stem_detection/calibration.py`; ArUco auto-detect or manual ratio, toggle in sidebar |
| Object Detection (bbox + contour) | `stem_detection/detector.py`; `Detection.bbox` and `Detection.contour`, both drawn by `annotate()` |
| Data Analysis Dashboard | `app.py`; summary metrics, per-image detail, charts |
| Extra: Reporting (PDF) | `stem_detection/report.py`; download button in app.py |
| Extra: Video Processing | `stem_detection/video.py`; "Video processing" tab in app.py |
| Extra: GUI bulk upload | `app.py`; `st.file_uploader(..., accept_multiple_files=True)` |
| Extra: Supplemental | `stem_detection/metrics.py`; Traditional/YOLO/Hybrid benchmark - detection rate, confidence, speed, IoU, and precision/recall/F1-score against ground truth, in the app's benchmark tab |

## Running it

**Important:** `app.py` unconditionally imports `calibration`, `colorDetection`,
and `report` at the top (your teammates' modules) - that's inherited from the
existing shared file, not something I added. This means `app.py` will fail
to even start if those files aren't present alongside it, *regardless* of
which technique you select. If you want to test just the Stem Detection
section in isolation before your teammates' files are ready, either:
- run against the full team repo once those files exist, or
- temporarily comment out the three `import calibration` / `from
  colorDetection import ...` / `import report` lines (and the code that
  uses them further down) while testing locally.

```
pip install -r requirements.txt
streamlit run app.py
```

Once running, select **"Stem Detection"** from the sidebar's technique
dropdown. That section works fully in Traditional mode with no extra setup.
YOLO/Hybrid need a trained model at `models/best.pt` - see
`train_stem_yolo.py` / `csv_to_yolo_labels.py` for how to train one; the
sidebar tells you clearly if the model file is missing rather than failing
silently.

## Known, documented limitations (worth stating in the report, not hiding)

- The Traditional method can't localise a stem that's recessed in a shallow
  calyx dimple with nothing protruding past the fruit's silhouette (no
  colour/shape rule reliably separates that from ordinary skin texture).
  This is exactly the kind of case the YOLO/Hybrid comparison is meant to
  address - it's a legitimate example for the report's "why compare
  techniques" discussion, not a bug to paper over.
- Cluttered, multi-fruit, no-plain-background scenes (fruit still on the
  tree, or piled together) are a different problem than single-fruit
  product photography and need a properly trained YOLO model with that kind
  of photo in its training set; the traditional background-subtraction
  approach fundamentally assumes a plain backdrop to sample.
- Heavily bruised/rotten fruit makes bruise-vs-stem/crown disambiguation
  genuinely hard by colour alone; same conclusion.

These are worth citing directly as the *motivation* for comparing multiple
techniques rather than shipping only one - which is the actual point of
Mode A.
