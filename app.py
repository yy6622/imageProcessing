
import base64
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from overall import run_overall_pipeline, BOX_COLOURS, draw_annotations  # adds common/ to sys.path
import calibration as calib          # common/calibration.py
import pdf_report as report_module   # common/pdf_report.py


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload_to_bgr(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def fruit_size_cm(fr, r):
    """(width_cm, height_cm, area_cm2) from fr.bbox, or None if uncalibrated."""
    calibration = r.get("calibration")
    if calibration is None or calibration.method == "none":
        return None
    _x, _y, w, h = fr.bbox
    width_cm = w * r["cm_per_pixel_x"]
    height_cm = h * r["cm_per_pixel_y"]
    return width_cm, height_cm, width_cm * height_cm


def bgr_to_thumb_html(bgr_img, size=110, badge="✔"):
    """Square center-cropped thumbnail as a data-URI <img>, with a small
    circular badge in the corner -- used for upload previews / summary cards."""
    h, w = bgr_img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    square = bgr_img[y0:y0 + side, x0:x0 + side]
    thumb = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    badge_html = f'<span class="fq-thumb-badge">{badge}</span>' if badge else ""
    return (
        f'<div class="fq-thumb" style="width:{size}px;height:{size}px;">'
        f'<img src="data:image/jpeg;base64,{b64}"/>{badge_html}</div>'
    )


# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
QUALITY_ICONS = {"Fresh": "🟢", "Unripe": "🟠", "Rotten": "🔴", "Uncertain": "🟡"}

st.set_page_config(page_title="Fruit Quality Inspection", layout="wide", page_icon="🍓")

st.markdown(
    """
    <style>
      :root {
          --fq-green: #2E7D32; --fq-orange: #EF6C00; --fq-red: #C62828;
          --fq-ink: #1F2A24; --fq-line: #E3E8E1; --fq-tint: #EEF4EA;
      }

      /* ---- overall canvas ---- */
      header[data-testid="stHeader"] { background: #FFFFFF; }
      .block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1200px; }
      h1, h2, h3, h4 { font-weight: 700; color: var(--fq-ink); }

      /* ---- plain header ---- */
      .fq-title { font-size: 2rem; font-weight: 800; color: var(--fq-green); margin: 0; }
      .fq-subtitle { color: #6b776e; font-size: 1rem; margin: 0.3rem 0 1.4rem 0; }

      /* ---- section titles ---- */
      h4 {
          margin-top: 0.4rem; padding-left: 0.65rem;
          border-left: 4px solid var(--fq-green); font-size: 1.05rem;
      }
      .fq-section {
          display: flex; align-items: center; gap: 0.55rem; margin: 1.6rem 0 0.8rem 0;
          padding-left: 0.7rem; border-left: 4px solid var(--fq-orange);
      }
      .fq-section h3 { margin: 0; border: none; padding: 0; font-size: 1.2rem; }

      /* ---- upload dropzone card ----
         Streamlit renders each element into its own DOM node, so a raw
         <div> opened in one st.markdown call can't wrap later widgets.
         st.container(..., key="upload_dropzone") gives the wrapper a
         stable ".st-key-upload_dropzone" class we can target directly
         (more reliable across browsers than :has()). */
      .st-key-upload_dropzone [data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--fq-tint) !important; border: 2px dashed #b8cdae !important;
          border-radius: 20px !important;
      }
      .st-key-upload_dropzone { text-align: center; }
      .fq-drop-icon { font-size: 2.6rem; margin-bottom: 0.3rem; }
      .fq-drop-heading { margin: 0 0 0.2rem 0; color: var(--fq-green); font-size: 1.3rem; font-weight: 700; }
      .fq-drop-caption { color: #6b776e; margin: 0 0 1rem 0; font-size: 0.9rem; }
      .st-key-upload_dropzone [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
      .st-key-upload_dropzone [data-testid="stFileUploaderDropzone"] {
          background: transparent; border: none; justify-content: center; width: auto;
      }
      .st-key-upload_dropzone [data-testid="stBaseButton-secondary"] {
          background: white; border: 1px solid var(--fq-line); border-radius: 10px; font-weight: 600;
          padding: 0.5rem 1.4rem;
      }

      /* ---- preview thumbnails ---- */
      .fq-thumb-row { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; padding-bottom: 1.4rem; }
      .fq-thumb { position: relative; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
      .fq-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
      .fq-thumb-badge {
          position: absolute; top: 5px; right: 5px; width: 20px; height: 20px; border-radius: 50%;
          background: var(--fq-green); color: white; font-size: 0.68rem; font-weight: 700;
          display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 5px rgba(0,0,0,0.3);
      }

      /* ---- metrics ---- */
      div[data-testid="stMetric"] {
          background: #f7faf6; border: 1px solid var(--fq-line); border-radius: 14px;
          padding: 0.9rem 1.1rem;
      }
      div[data-testid="stMetric"] label { color: #55645a; }

      /* ---- containers / cards / expanders ---- */
      div[data-testid="stVerticalBlockBorderWrapper"] {
          border-radius: 14px !important; border-color: var(--fq-line) !important;
      }
      [data-testid="stExpander"] {
          border-radius: 14px; border: 1px solid var(--fq-line); overflow: hidden;
      }
      [data-testid="stExpander"] summary { font-weight: 600; }

      /* ---- tabs ---- */
      button[data-baseweb="tab"] {
          font-size: 1rem; font-weight: 600; padding: 0.5rem 1.1rem;
      }
      div[data-baseweb="tab-highlight"] { background-color: var(--fq-green) !important; height: 3px; }
      div[data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--fq-line); }

      /* ---- buttons ---- */
      .stButton > button {
          border-radius: 10px; font-weight: 600; padding: 0.5rem 1.3rem;
          box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      }
      .stDownloadButton > button {
          border-radius: 10px; font-weight: 600;
      }

      /* ---- sidebar ---- */
      section[data-testid="stSidebar"] {
          border-right: 1px solid var(--fq-line); background: var(--fq-tint);
      }
      section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
      section[data-testid="stSidebar"] h2 {
          font-size: 1.05rem; display: flex; align-items: center; gap: 0.4rem;
          padding-bottom: 0.5rem; margin-bottom: 0.2rem;
      }
      section[data-testid="stSidebar"] [data-testid="stExpander"] {
          background: white; border: 1px solid var(--fq-line);
      }
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.4rem; }
      .fq-side-caption { color: #6b776e; font-size: 0.85rem; margin-bottom: 0.8rem; }
      .fq-side-label { font-weight: 600; font-size: 0.8rem; color: #6b776e; text-transform: uppercase;
          letter-spacing: 0.03em; margin: 0.6rem 0 0.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Polished product UI layer. This overrides only presentation; the image
# processing, calibration and report logic below stays unchanged.
st.markdown(
    """
    <style>
      :root {
        --fq-green: #155a3a;
        --fq-green-hover: #0f492f;
        --fq-mint: #eef5e8;
        --fq-lime: #9fbe53;
        --fq-orange: #ee8a22;
        --fq-red: #c85148;
        --fq-ink: #17231c;
        --fq-muted: #6e7972;
        --fq-line: #e5e7df;
        --fq-surface: #ffffff;
        --fq-canvas: #fbfaf6;
      }

      html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .stApp { background: var(--fq-canvas); color: var(--fq-ink); }
      header[data-testid="stHeader"] { height: 2.5rem; background: rgba(251,250,246,.88); backdrop-filter: blur(12px); }
      #MainMenu, footer { visibility: hidden; }
      .block-container { max-width: 1240px; padding: 2.15rem 2.5rem 3.5rem; }
      h1, h2, h3, h4, p { letter-spacing: -.01em; }

      /* Product header */
      .fq-header { display:flex; align-items:flex-start; justify-content:space-between; gap:2rem; margin:0 0 1.6rem; }
      .fq-eyebrow { color:var(--fq-orange); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin:0 0 .45rem; }
      .fq-title { color:var(--fq-green); font-size:2.25rem; line-height:1.05; font-weight:820; letter-spacing:-.045em; margin:0; }
      .fq-subtitle { max-width:760px; color:var(--fq-muted); font-size:.98rem; line-height:1.55; margin:.65rem 0 0; }
      .fq-header-chip { display:flex; align-items:center; gap:.55rem; flex:0 0 auto; margin-top:.3rem; padding:.58rem .85rem; color:var(--fq-green); background:#fff; border:1px solid var(--fq-line); border-radius:999px; box-shadow:0 4px 14px rgba(26,61,41,.05); font-size:.79rem; font-weight:700; }
      .fq-live-dot { width:8px; height:8px; border-radius:50%; background:#59a44d; box-shadow:0 0 0 4px #e7f3e2; }

      /* Sidebar */
      section[data-testid="stSidebar"] { width:335px !important; background:#f5f6f0; border-right:1px solid var(--fq-line); }
      section[data-testid="stSidebar"] > div { width:335px !important; }
      section[data-testid="stSidebar"] .block-container { padding:2.2rem 1.8rem; }
      section[data-testid="stSidebar"] h2 { color:var(--fq-ink); font-size:1.15rem; line-height:1.3; margin:0 0 .3rem; padding:0; }
      .fq-side-caption { color:var(--fq-muted); font-size:.84rem; line-height:1.55; margin:.55rem 0 2rem; }
      .fq-side-label { color:var(--fq-green); font-size:.73rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin:0 0 .65rem; }
      section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#38453d; font-size:.9rem; font-weight:600; }
      section[data-testid="stSidebar"] [data-testid="stExpander"] { background:rgba(255,255,255,.72); border:1px solid #e2e5dc; border-radius:12px; margin-top:.55rem; box-shadow:none; }
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary { min-height:48px; padding:.6rem .8rem; font-size:.9rem; }
      section[data-testid="stSidebar"] [data-testid="stToggle"] { padding:.45rem .1rem .8rem; }
      section[data-testid="stSidebar"] hr { border-color:var(--fq-line); }
      .st-key-calibration_expander { transition:opacity .18s ease; }

      /* Upload workspace */
      .st-key-upload_dropzone { text-align:center; }
      .st-key-upload_dropzone [data-testid="stVerticalBlockBorderWrapper"] {
        min-height:410px;
        background:radial-gradient(circle at 50% 30%, #fff 0%, #f7faef 52%, #f0f5e8 100%) !important;
        border:2px dashed #a8c584 !important;
        border-radius:22px !important;
        box-shadow:0 16px 42px rgba(45,78,50,.045);
      }
      .st-key-upload_dropzone [data-testid="stVerticalBlock"] { align-items:center; justify-content:center; gap:.35rem; }
      .fq-drop-icon { width:82px; height:82px; display:grid; place-items:center; margin:.2rem auto .55rem; border-radius:24px; background:linear-gradient(145deg,#fff,#edf5dd); box-shadow:0 12px 28px rgba(65,103,54,.12), inset 0 0 0 1px rgba(142,176,87,.2); font-size:2.8rem; transform:rotate(-3deg); }
      .fq-drop-heading { color:var(--fq-green); font-size:1.75rem; line-height:1.15; font-weight:800; letter-spacing:-.035em; margin:.1rem 0 .35rem; }
      .fq-drop-caption { color:var(--fq-muted); font-size:.91rem; margin:0 0 1.05rem; }
      .st-key-upload_dropzone [data-testid="stFileUploader"] { width:min(100%, 520px); margin-inline:auto; }
      .st-key-upload_dropzone [data-testid="stFileUploaderDropzone"] { min-height:0; padding:0; background:transparent; border:0; }
      .st-key-upload_dropzone [data-testid="stFileUploaderDropzoneInstructions"],
      .st-key-upload_dropzone [data-testid="stFileUploaderFile"] { display:none; }
      /* Recolour the uploader's button(s) green -- but only colour, never
         force a size. Forcing min-width/padding on EVERY button in this
         scope also hit the small icon-only remove/add-more buttons that
         Streamlit shows once a file is picked, blowing them up into giant
         pills with a stray white square (the icon, un-scaled) inside.
         Colour-only changes are safe regardless of which button it is. */
      .st-key-upload_dropzone [data-testid="stFileUploaderDropzone"] button {
        border:0 !important; border-radius:10px !important;
        background:linear-gradient(180deg,#216d49,#155a3a) !important;
        color:#fff !important; font-weight:700 !important;
      }
      .st-key-upload_dropzone [data-testid="stFileUploaderDropzone"] button:hover {
        background:linear-gradient(180deg,#1c5f3f,#0f492f) !important;
      }
      .st-key-upload_dropzone [data-testid="stFileUploaderDropzone"] button svg {
        fill:#fff !important;
      }
      .fq-thumb-row { gap:14px; padding:1.15rem 0 .15rem; }
      .fq-thumb { border-radius:14px; border:3px solid #fff; box-shadow:0 4px 14px rgba(33,62,42,.15); }
      .fq-thumb-badge { width:25px; height:25px; top:6px; right:6px; border:2px solid white; background:#138448; font-size:.72rem; }

      /* Primary action -- just spacing. Deliberately no custom width/colour
         rules here: it should render exactly like the "View full analysis"
         button below it (same width="stretch" + type="primary" + theme
         defaults), so the two always match without fighting each other. */
      .st-key-analysis_action { margin:1rem 0 1.2rem; }

      /* Results and cards */
      .fq-section { border:0; padding:0; margin:1.5rem 0 .8rem; }
      .fq-section h3 { color:var(--fq-ink); font-size:1.05rem; font-weight:780; }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--fq-line) !important; border-radius:16px !important; background:#fff; box-shadow:0 5px 20px rgba(25,43,32,.035); }
      div[data-testid="stMetric"] { min-height:105px; display:flex; justify-content:center; background:#fff; border:1px solid var(--fq-line); border-radius:14px; padding:1rem 1.1rem; box-shadow:0 4px 16px rgba(25,43,32,.03); }
      div[data-testid="stMetric"] label { color:var(--fq-muted); font-size:.82rem; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:var(--fq-green); font-weight:800; letter-spacing:-.035em; }
      [data-testid="stExpander"] { border-color:var(--fq-line); border-radius:14px; overflow:hidden; background:#fff; }

      /* Tabs, tables and secondary controls */
      div[data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--fq-line); }
      /* Every tab's main card -- Summary's results table, both Charts
         cards, and the PDF export card -- share one min-height so the
         page doesn't visibly shrink/jump when you switch tabs. Matched via
         stable keys instead of :has(), which isn't reliably supported in
         every renderer this app runs in. */
      .st-key-tabcard_summary [data-testid="stVerticalBlockBorderWrapper"],
      .st-key-tabcard_chart_species [data-testid="stVerticalBlockBorderWrapper"],
      .st-key-tabcard_chart_quality [data-testid="stVerticalBlockBorderWrapper"],
      .st-key-tabcard_pdf [data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 480px;
      }
      /* Per-photo detail has no single big card (just collapsed expanders
         per photo), so give the whole tab panel the same floor height too
         -- that keeps ALL FOUR tabs occupying the same vertical space. */
      div[data-baseweb="tab-panel"] { min-height: 620px; }
      button[data-baseweb="tab"] {
        color:var(--fq-muted) !important; font-size:.9rem !important; font-weight:650 !important;
        padding:.7rem 1rem !important; margin:0 !important; border:0 !important;
        outline:none !important; box-shadow:none !important; background:transparent !important;
        transition: color .18s ease;
      }
      /* Clicking a tab gives the <button> browser focus, which by default
         draws its own outline/ring box -- that's what made the just-clicked
         tab look like a different size from the others. Kill it everywhere. */
      button[data-baseweb="tab"]:focus,
      button[data-baseweb="tab"]:focus-visible,
      button[data-baseweb="tab"]:active {
        outline:none !important; box-shadow:none !important; border:0 !important;
      }
      button[data-baseweb="tab"][aria-selected="true"] { color:var(--fq-green) !important; }
      /* Smoothly slide the underline between tabs instead of snapping --
         short labels (Charts) vs long ones (Per-photo detail) otherwise
         made the indicator visibly "jump" in width/position. */
      div[data-baseweb="tab-highlight"] {
        height:3px; border-radius:3px 3px 0 0; background:var(--fq-green) !important;
        transition: left .22s cubic-bezier(.4,0,.2,1), width .22s cubic-bezier(.4,0,.2,1) !important;
      }
      .stDownloadButton > button, .stButton > button { border-radius:10px; font-weight:680; }
      [data-testid="stDataFrame"] { border:1px solid var(--fq-line); border-radius:14px; overflow:hidden; }
      [data-testid="stAlert"] { border-radius:13px; }

      .fq-empty { display:flex; align-items:center; gap:.85rem; padding:1rem 1.1rem; color:var(--fq-muted); background:#fff; border:1px solid var(--fq-line); border-radius:14px; box-shadow:0 4px 14px rgba(25,43,32,.025); }
      .fq-empty-icon { display:grid; place-items:center; width:38px; height:38px; flex:0 0 38px; border-radius:11px; color:var(--fq-green); background:var(--fq-mint); font-size:1rem; }

      @media (max-width: 900px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .fq-header { flex-direction:column; gap:.8rem; }
        .fq-title { font-size:1.8rem; }
        .st-key-upload_dropzone [data-testid="stVerticalBlockBorderWrapper"] { min-height:350px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="fq-header">
      <div>
        <p class="fq-eyebrow">Computer vision workspace</p>
        <p class="fq-title">Fruit Quality Inspection</p>
        <p class="fq-subtitle">AI-powered grading for ripeness, defects and morphology — species,
        colour, stem and texture combined into one clear verdict per fruit.</p>
      </div>
      <div class="fq-header-chip"><span class="fq-live-dot"></span> Inspection engine ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

QUALITY_BADGE_COLOURS = {"Fresh": "green", "Unripe": "orange", "Rotten": "red", "Uncertain": "yellow"}


def quality_badge(container, label, confidence=None):
    """Native st.badge, colour-matched to our theme's semantic palette."""
    text = label or "N/A"
    if confidence is not None:
        text += f" ({confidence * 100:.0f}%)"
    container.badge(text, icon=QUALITY_ICONS.get(label, "⚪"), color=QUALITY_BADGE_COLOURS.get(label, "gray"))


# Morphological size ranking: XL/L = best (full-grown, well-formed fruit),
# M = ok, S = worst (under-developed / shrivelled).
_MORPH_SIZE_RANK = {"XL": 1.0, "L": 1.0, "M": 0.3, "S": -1.0}

BASE_WEIGHT = 2.0       # colour+defect fused result carries the biggest weight
STEM_WEIGHT = 0.4
MORPH_WEIGHT = 0.4


def compute_sub_final(fr):
    """Sub Final = weighted vote across the colour+defect fused verdict
    ("Final quality" on the detail card, itself already colour-dominant and
    defect-aware), plus stem and morphological as secondary nudges.

    We deliberately reuse fr.final_quality as the base signal instead of
    re-deriving colour+defect fusion from scratch: it already properly
    weighs colour vs defect % (e.g. a fruit that still looks fresh in
    colour isn't flipped to Rotten just because of a moderate defect %),
    so a standalone "defect % >= X -> Rotten" rule would just fight it and
    produce a different answer than what's shown as Final quality elsewhere.
    Stem and morphological -- which final_quality ignores -- are added on
    top so Sub Final can still disagree with Final quality when those
    secondary signals are strong enough.
    """
    votes = {"Fresh": 0.0, "Unripe": 0.0, "Rotten": 0.0}

    if fr.final_quality in votes:
        votes[fr.final_quality] += max(fr.final_quality_confidence, 0.05) * BASE_WEIGHT

    # Stem: having a stem is better than not having one.
    if fr.stem_detected:
        votes["Fresh"] += STEM_WEIGHT
    else:
        votes["Unripe"] += STEM_WEIGHT * 0.5
        votes["Rotten"] += STEM_WEIGHT * 0.5

    # Morphological size class: XL/L best, M ok, S worst.
    size = (fr.morph_size_class or "").strip().upper()
    rank = _MORPH_SIZE_RANK.get(size)
    if rank is not None:
        if rank >= 0:
            votes["Fresh"] += MORPH_WEIGHT * rank
        else:
            votes["Unripe"] += MORPH_WEIGHT * abs(rank) * 0.5
            votes["Rotten"] += MORPH_WEIGHT * abs(rank) * 0.5

    total = sum(votes.values())
    if total <= 0:
        return "Uncertain", 0.0

    best_label = max(votes, key=votes.get)
    best_weight = votes[best_label]
    confidence = best_weight / total
    if confidence < 0.35:
        return "Uncertain", confidence
    return best_label, confidence


# Sub Final is shown as a letter grade rather than the Fresh/Unripe/Rotten
# wording -- same A-D scale as the batch-level "Grade" metric.
_SUB_FINAL_GRADE = {"Fresh": "A", "Unripe": "B", "Rotten": "D", "Uncertain": "C"}


def sub_final_grade(fr):
    """(grade_letter, confidence) -- grade-letter view of compute_sub_final()."""
    label, confidence = compute_sub_final(fr)
    return _SUB_FINAL_GRADE.get(label, "C"), confidence


# ---------------------------------------------------------------------------
# Sidebar -- calibration & rectification
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ◉ &nbsp; Calibration & Rectification")
st.sidebar.markdown(
    '<p class="fq-side-caption">Ensure accurate measurements and consistent results.</p>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<p class="fq-side-label">Image setup</p>', unsafe_allow_html=True)
want_measurements = st.sidebar.toggle("Measure physical size (cm)", value=False)

if not want_measurements:
    # Locked look -- Calibration only makes sense once "Measure physical
    # size" is on, so grey it out and block interaction until then instead
    # of leaving it clickable with nothing useful inside.
    st.sidebar.markdown(
        '<style>.st-key-calibration_expander { opacity: 0.45; pointer-events: none; }</style>',
        unsafe_allow_html=True,
    )

with st.sidebar.expander("🔒  Calibration" if not want_measurements else "▣  Calibration",
                          expanded=False, key="calibration_expander"):
    manual_cm_per_pixel = None
    if want_measurements:
        manual_cm_per_pixel = st.number_input(
            "cm per pixel (measured against your uploaded photo)",
            value=0.02, min_value=0.0001, step=0.001, format="%.4f",
            help="Known scale for your camera setup (e.g. derived once from a ruler photo). "
                 "The analysis pipeline now keeps the original image resolution.",
        )
    else:
        st.caption("Turn on \"Measure physical size\" above to unlock this.")

with st.sidebar.expander("⊞ Perspective correction", expanded=False):
    st.caption(
        "Straightens the image plane before detection if the camera isn't "
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
                "card, or table edge) and uses it as the reference. Falls back to "
                "no rectification for a photo if nothing suitable is found."
            )
        else:
            st.caption("Pixel coordinates of the 4 corners of a flat reference region, any order.")
            rc1, rc2 = st.columns(2)
            x1 = rc1.number_input("Corner 1 -- x", value=0, step=1)
            y1 = rc2.number_input("Corner 1 -- y", value=0, step=1)
            x2 = rc1.number_input("Corner 2 -- x", value=100, step=1)
            y2 = rc2.number_input("Corner 2 -- y", value=0, step=1)
            x3 = rc1.number_input("Corner 3 -- x", value=100, step=1)
            y3 = rc2.number_input("Corner 3 -- y", value=100, step=1)
            x4 = rc1.number_input("Corner 4 -- x", value=0, step=1)
            y4 = rc2.number_input("Corner 4 -- y", value=100, step=1)
            rectify_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)


def get_calibration():
    if not want_measurements:
        return calib.uncalibrated()
    return calib.manual_scale(manual_cm_per_pixel)


# ---------------------------------------------------------------------------
# Upload & run
# ---------------------------------------------------------------------------
if "overall_results" not in st.session_state:
    st.session_state["overall_results"] = []
if "view" not in st.session_state:
    st.session_state["view"] = "upload"

results = st.session_state["overall_results"]
if not results:
    # Nothing to show a results page for -- always land on the upload page.
    st.session_state["view"] = "upload"

if st.session_state["view"] == "upload":
    with st.container(border=True, key="upload_dropzone", horizontal_alignment="center"):
        st.markdown(
            """
            <div class="fq-drop-icon">🍊</div>
            <p class="fq-drop-heading">Drop fruit photos here</p>
            <p class="fq-drop-caption">JPG, PNG or BMP &middot; upload one or multiple images</p>
            """,
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Select one or more images", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if uploaded_files:
            thumbs = "".join(bgr_to_thumb_html(read_upload_to_bgr(f), size=124) for f in uploaded_files)
            st.markdown(f'<div class="fq-thumb-row">{thumbs}</div>', unsafe_allow_html=True)

    with st.container(key="analysis_action"):
        run_button = st.button(
            f"✦  Analyze {len(uploaded_files)} photo{'s' if len(uploaded_files) != 1 else ''}" if uploaded_files
            else "✦  Analyze photos",
            type="primary", disabled=not uploaded_files, width="stretch",
        )

    if run_button:
        all_results = []
        calibration_result = get_calibration()
        progress = st.progress(0.0, text="Running overall pipeline...")
        for i, f in enumerate(uploaded_files):
            img = read_upload_to_bgr(f)
            if img is None:
                continue

            if want_rectify:
                if rectify_mode.startswith("Auto"):
                    auto_quad = calib.detect_reference_quad(img)
                    if auto_quad is not None:
                        img = calib.rectify_perspective(img, auto_quad)
                elif rectify_points is not None:
                    img = calib.rectify_perspective(img, rectify_points)

            # The overall pipeline now runs at the uploaded image's original
            # resolution, so returned bounding boxes stay in the same pixel
            # coordinate space as the uploaded image.
            scale_x = 1.0
            scale_y = 1.0

            fruit_results, display_img = run_overall_pipeline(img)
            all_results.append({
                "filename": f.name,
                "image": display_img,
                "fruits": fruit_results,
                "calibration": calibration_result,
                "cm_per_pixel_x": calibration_result.cm_per_pixel * scale_x,
                "cm_per_pixel_y": calibration_result.cm_per_pixel * scale_y,
            })
            progress.progress((i + 1) / len(uploaded_files), text=f"Processed {f.name}")
        progress.empty()
        st.session_state["overall_results"] = all_results
        st.session_state["last_run_time"] = datetime.now()
        st.session_state.pop("pdf_report_bytes", None)
        results = st.session_state["overall_results"]

    if not results:
        st.markdown(
            """
            <div class="fq-empty">
              <span class="fq-empty-icon">✓</span>
              <span><strong>Ready for inspection.</strong> Upload fruit photos above, then start the analysis.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        all_fruits_preview = [fr for r in results for fr in r["fruits"]]
        # Use the same Sub Final signal (colour+defect fusion + stem +
        # morphological) as the per-fruit results table, not the raw
        # colour+defect-only final_quality.
        sub_final_labels = [compute_sub_final(fr)[0] for fr in all_fruits_preview]
        fresh_pct = (
            100 * sum(1 for lbl in sub_final_labels if lbl == "Fresh") / len(all_fruits_preview)
            if all_fruits_preview else 0.0
        )
        needs_attention = sum(1 for lbl in sub_final_labels if lbl in ("Rotten", "Uncertain"))

        # Grade is the average of each fruit's own Sub Final grade (GPA-style:
        # A=4 .. D=1), not "% of fruits that are exactly Fresh" -- that way
        # e.g. one A fruit + one B fruit lands the batch around A-/B, instead
        # of being dragged down to C just because only one of two is "Fresh".
        _GRADE_POINTS = {"A": 4, "B": 3, "C": 2, "D": 1}
        fruit_grades = [sub_final_grade(fr)[0] for fr in all_fruits_preview]
        avg_points = (
            sum(_GRADE_POINTS[g] for g in fruit_grades) / len(fruit_grades)
            if fruit_grades else 0.0
        )
        grade = "A" if avg_points >= 3.5 else "B" if avg_points >= 2.5 else "C" if avg_points >= 1.5 else "D"

        st.markdown('<div class="fq-section"><h3>🗂️ Latest analysis</h3></div>', unsafe_allow_html=True)
        with st.container(border=True):
            info_col, m1, m2, m3 = st.columns([2.2, 1, 1, 1])
            with info_col:
                thumb_c, text_c = st.columns([1, 2.2])
                thumb_c.markdown(bgr_to_thumb_html(results[0]["image"], size=64), unsafe_allow_html=True)
                run_time = st.session_state.get("last_run_time")
                text_c.badge("Completed", icon="✅", color="green")
                if run_time:
                    text_c.caption(run_time.strftime("%b %d, %Y  ·  %I:%M %p"))
                text_c.caption(f"{len(results)} photo{'s' if len(results) != 1 else ''} analyzed")
            m1.metric("🍃 Fresh", f"{fresh_pct:.0f}%")
            m2.metric("🐛 Needs attention", needs_attention)
            m3.metric("🏅 Grade", grade)

        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        if st.button("🔍  View full analysis  →", key="go_to_detail", type="primary", width="stretch"):
            st.session_state["view"] = "detail"
            st.rerun()

else:
    back_col, _spacer = st.columns([1, 4])
    if back_col.button("←  Back to upload", key="back_to_upload", width="stretch"):
        st.session_state["view"] = "upload"
        st.rerun()

    st.markdown('<div class="fq-section"><h3>🗂️ Analysis results</h3></div>', unsafe_allow_html=True)
    all_fruits = [(r["filename"], fr, r) for r in results for fr in r["fruits"]]

    tab_summary, tab_charts, tab_pdf, tab_detail = st.tabs(
        ["📊 Summary", "📈 Charts", "📄 PDF Report", "🔍 Per-photo detail"]
    )

    # ------------------------------------------------------------------
    # Summary tab
    # ------------------------------------------------------------------
    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📷 Photos inspected", len(results))
        col2.metric("🍎 Fruits detected", len(all_fruits))
        fresh_n = sum(1 for _n, fr, _r in all_fruits if compute_sub_final(fr)[0] == "Fresh")
        rotten_n = sum(1 for _n, fr, _r in all_fruits if compute_sub_final(fr)[0] == "Rotten")
        col3.metric("🟢 Fresh", fresh_n)
        col4.metric("🔴 Rotten", rotten_n)

        with st.container(border=True, key="tabcard_summary"):
            st.markdown("####   Results table")
            table_rows = []
            for idx, (filename, fr, r) in enumerate(all_fruits, start=1):
                size_cm = fruit_size_cm(fr, r)
                size_str = f"{size_cm[0]:.1f} x {size_cm[1]:.1f}" if size_cm else "-"
                defect_str = f"{fr.defect_percentage:.1f}%" if fr.defect_percentage is not None else (fr.defect_note or "-")
                stem_str = "Yes" if fr.stem_detected else "No"
                fruit_label = f"{filename} -- #{idx} ({fr.species or '?'})"
                colour_str = (
                    f"{fr.colour_quality} ({fr.colour_confidence * 100:.0f}%)" if fr.colour_quality else "-"
                )

                morph_str = fr.morph_size_class or fr.morphological_note or "-"
                sub_final_label, sub_final_conf = sub_final_grade(fr)
                sub_final_str = f"{sub_final_label} ({sub_final_conf * 100:.0f}%)"

                table_rows.append({
                    "Fruit": fruit_label,
                    "Colour": colour_str,
                    "Defect %": defect_str,
                    "Stem": stem_str,
                    "Morphological": morph_str,
                    "Size (cm)": size_str,
                    "Sub Final": sub_final_str,
                })
            df = pd.DataFrame(table_rows)
            st.dataframe(df, width="stretch", hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download results as CSV", csv, "overall_results.csv", "text/csv")

    # ------------------------------------------------------------------
    # Charts tab
    # ------------------------------------------------------------------
    with tab_charts:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1.container(border=True, key="tabcard_chart_species"):
            st.markdown("**Species distribution**")
            species_fig = report_module.distribution_chart(
                [fr.species for _n, fr, _r in all_fruits], "Species Distribution"
            )
            if species_fig is not None:
                st.pyplot(species_fig, width="stretch")
            else:
                st.caption("No classified species to chart.")
        with chart_col2.container(border=True, key="tabcard_chart_quality"):
            st.markdown("**Quality distribution**")
            quality_fig = report_module.distribution_chart(
                [fr.final_quality for _n, fr, _r in all_fruits], "Quality Distribution",
                color_map={k: "#%02x%02x%02x" % (b, g, r) for k, (b, g, r) in BOX_COLOURS.items()},
            )
            if quality_fig is not None:
                st.pyplot(quality_fig, width="stretch")
            else:
                st.caption("No quality verdicts to chart.")

    # ------------------------------------------------------------------
    # PDF tab
    # ------------------------------------------------------------------
    with tab_pdf:
        with st.container(border=True, key="tabcard_pdf"):
            st.markdown("**Export a full inspection report**")
            st.caption(
                "Includes the summary table, both distribution charts, and every photo "
                "with annotated boxes, per-fruit crops, and the full detail breakdown."
            )
            if st.button("🖨️ Generate PDF report"):
                with st.spinner("Building PDF..."):
                    pdf_bytes = report_module.generate_report(results, batch_name="Fruit Quality Inspection")
                st.session_state["pdf_report_bytes"] = pdf_bytes
            if "pdf_report_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download PDF report", st.session_state["pdf_report_bytes"],
                    "fruit_inspection_report.pdf", "application/pdf",
                )

    # ------------------------------------------------------------------
    # Per-photo detail tab
    # ------------------------------------------------------------------
    FRUIT_GRID_COLUMNS = 3

    def render_fruit_card(col, fr, r=None):
        with col.container(border=True):
            if r is not None:
                size_cm = fruit_size_cm(fr, r)
                if size_cm is not None:
                    st.caption(f"📏 {size_cm[0]:.1f} x {size_cm[1]:.1f} cm (area {size_cm[2]:.1f} cm²)")
            images = [(fr.crop, f"{fr.species or '?'}")]
            if fr.defect_marked_crop is not None:
                images.append((fr.defect_marked_crop, "Defect marked"))
            if fr.stem_crop is not None:
                images.append((fr.stem_crop, "Stem detection"))
            if len(images) == 1:
                st.image(bgr_to_rgb(images[0][0]), caption=images[0][1], width=200)
            else:
                for img_col, (img, caption) in zip(st.columns(len(images)), images):
                    img_col.image(bgr_to_rgb(img), caption=caption, width=180)

            st.markdown(
                f"**Species:** {fr.species} ({fr.species_confidence*100:.0f}%) "
                f"&mdash; won by: `{fr.species_source}`", unsafe_allow_html=True,
            )
            st.caption(
                f"1) YOLO: {fr.yolo_species or 'no result'} "
                f"({fr.yolo_confidence*100:.0f}%)  |  "
                f"2) CNN + Rules: {fr.cnn_rule_species or 'no result'} "
                f"({fr.cnn_rule_confidence*100:.0f}%)  |  "
                f"3) Morph: {fr.morph_fruit_type or 'no match'}"
                + (
                    f" ({fr.morph_fruit_type_confidence*100:.0f}%)"
                    if fr.morph_fruit_type
                    else ""
                )
            )
            #st.caption(f"CNN raw guess (feeds into #1): {fr.cnn_species} ({fr.cnn_confidence*100:.0f}%)")

            st.write("**Final quality:**")
            quality_badge(st, fr.final_quality, fr.final_quality_confidence)
            #if fr.quality_note:
            #    st.caption(fr.quality_note)

            st.write(f"**Colour quality:** {fr.colour_quality or 'N/A'} "
                     f"({fr.colour_confidence*100:.0f}%)")
            st.write(f"**Ripeness:** {fr.defect_ripeness or 'N/A'} "
                     f"({fr.defect_ripeness_confidence*100:.0f}%)")
            if fr.defect_percentage is not None:
                st.write(f"**Defect:** {fr.defect_percentage:.1f}%")
            else:
                st.write(f"**Defect:** {fr.defect_note}")
            st.write(f"**Stem:** {'✅ detected' if fr.stem_detected else '❌ not detected'} "
                     f"({fr.stem_confidence*100:.0f}%)")

            if fr.morph_aspect_ratio is not None:
                with st.expander("Morphology & texture"):
                    st.write(f"**Morphological:** size={fr.morph_size_class or 'N/A'}, "
                             f"aspect={fr.morph_aspect_ratio:.2f}, circularity={fr.morph_circularity:.2f}, "
                             f"extent={fr.morph_extent:.2f}")
                    st.write(f"**Texture (GLCM):** contrast={fr.tex_contrast:.1f}, "
                             f"energy={fr.tex_energy:.3f}, homogeneity={fr.tex_homogeneity:.3f}, "
                             f"entropy={fr.tex_entropy:.2f}")
            else:
                st.caption(fr.morphological_note)

    with tab_detail:
        for r in results:
            with st.expander(f"🖼️ {r['filename']} -- {len(r['fruits'])} fruit(s)"):
                annotated = draw_annotations(r["image"], r["fruits"]) if r["fruits"] else r["image"]
                st.image(bgr_to_rgb(annotated), caption="Detected fruits", width=420)
                if not r["fruits"]:
                    st.warning("No fruit detected in this photo.")
                    continue
                st.caption(
                    "Only species this project supports (Apple / Banana / Orange / Mango / "
                    "Strawberry) get a box -- other objects in the photo are skipped by design."
                )
                fruits = r["fruits"]
                for row_start in range(0, len(fruits), FRUIT_GRID_COLUMNS):
                    row_fruits = fruits[row_start:row_start + FRUIT_GRID_COLUMNS]
                    cols = st.columns(FRUIT_GRID_COLUMNS)
                    for fr, col in zip(row_fruits, cols):
                        render_fruit_card(col, fr, r)
