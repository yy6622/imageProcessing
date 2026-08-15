# Fruit Quality Inspection System

A computer vision system that separates every individual fruit in a
photo — even several touching/overlapping apples in one shot — detects
each one's **fruit type** (Apple / Banana / Orange) and classifies its
quality as **Fresh / Unripe / Rotten**. It measures each fruit, and
reports quantity + type + quality per photo through a dashboard and
PDF export.

**Live pipeline (what `app.py` actually runs):** detection + fruit
TYPE come from **YOLOv8**, pretrained on COCO (apple/banana/orange are
3 of its 80 classes already, zero extra training needed) — this
handles densely packed/overlapping fruit far better than classical
splitting. Quality comes from a **CNN** (`train_cnn_quality.py`, one
MobileNetV2 transfer-learning model per fruit type, looking at the
whole fruit at once) — measured ~98% held-out accuracy vs. an earlier
SVM approach's ~73%. Neither has a fallback to the other/older approach
anymore; both are explicit choices made after testing, not defaults.

**Original approach (deleted entirely, not kept around):** a two-stage
SVM pipeline — one patch+color-histogram SVM classifier for fruit
type, a second per-species SVM for quality — with a distance-transform
+ Hough-assisted watershed splitter for separating touching fruit
without any object detector. All of it, including the classical
detection/splitting code and every hand-engineered shape/color-
histogram/color-moment/texture feature extractor, has since been
removed from the codebase on request ("no unused code"), not just
disconnected from the live pipeline. The ONE exception:
`segmentation_mask_and_contour()` in `colorDetection_Train.py`,
because it's a genuine live dependency — `colorDetection.py` still
calls it for local re-segmentation within each YOLO box. The "Known
limitations" and "Notes / design decisions" sections below still
describe the original design and the reasoning/bugs behind it — kept
for historical/report context even though the code itself is gone; see
each entry's note on whether it still applies to the live pipeline.

## Files

| File | Assignment requirement it satisfies |
|---|---|
| `colorDetection_Train.py` | Down to just LAB chroma-distance background segmentation (`segmentation_mask_and_contour()`) — the one piece of the original classical pipeline still actually called by anything (`colorDetection.py` imports it for local re-segmentation within each YOLO box). Everything else that used to live here — SVM/RandomForest/KNN classifier training, the distance-transform + Hough-assisted watershed splitter for touching fruit, and every hand-engineered feature extractor (shape/color-histogram/color-moment/LBP-texture) — has been deleted outright, per an explicit "no unused code" decision, not just disconnected. See this file's module docstring. |
| `train_cnn_quality.py` | Trains the CNN quality classifier (MobileNetV2 transfer learning, one model per fruit type) that `app.py` actually uses for Fresh/Unripe/Rotten — see the "CNN quality classification" section below for the full story (including a data-leakage bug found and fixed along the way). |
| `preprocessing.py` | **Preprocessing** — Gaussian / median / bilateral denoising, plus contrast stretching / CLAHE / histogram equalization for enhancement. |
| `calibration.py` | **Image Calibration** — spatial scaling from pixels to cm, either automatically via an ArUco marker placed in the photo, or manually via a known cm-per-pixel ratio. Also includes 4-point perspective rectification. |
| `colorDetection.py` | **Object Detection** + inference pipeline. `inspect_image_yolo()` (the only pipeline function left, and what `app.py` calls) detects/counts/types fruit via YOLOv8 and classifies quality via the CNN. The original SVM-based `inspect_image()`, `classify_fruit_type()`, and their classical multi-instance detection helpers have been deleted entirely — see this file's module docstring. |
| `app.py` | **Data Analysis Dashboard** + **GUI (bulk ingestion)** — Streamlit app: multi-image upload or folder-based bulk ingestion, per-photo fruit counts/breakdown (e.g. "3 apples: 2 Fresh, 1 Rotten") plus per-fruit detail, aggregate results, CSV/PDF export. Runs YOLO + CNN only (see ARCHITECTURE NOTE in its docstring); settings are minimal by default with everything else tucked behind an "Advanced settings" expander. |
| `report.py` | **Reporting** — automated PDF export (summary stats, fruit-type and quality distribution charts across every detected fruit, per-image annotated thumbnail + a per-fruit measurements table) via `reportlab`. |

## Setup

```bash
pip install -r requirements.txt
```

`opencv-contrib-python` is required (not plain `opencv-python`) because
ArUco marker detection lives in the contrib module. If you already have
plain `opencv-python` installed, uninstall it first to avoid a conflict:

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## 1. Train the quality model

Detection + fruit TYPE need no training at all — YOLOv8 is used
pretrained, straight off COCO (apple/banana/orange are already 3 of
its 80 classes). The only thing to train is quality (Fresh/Unripe/
Rotten), via `train_cnn_quality.py`:

```bash
pip install torch torchvision
python train_cnn_quality.py --dataset dataset/train
```

This trains one CNN (MobileNetV2 transfer learning) per fruit type
found in `CLASS_FOLDERS` at the top of that script, prints per-epoch
train/validation accuracy, and saves each as
`cnn_quality_models/<FruitType>.pt`. Edit `CLASS_FOLDERS` there to
match your actual dataset folder names/fruit species if they differ.

Two things worth knowing before you run this on your own dataset (see
the "CNN quality classification" section below for the full story):
first, the train/validation split is grouped by *original source
photo* (not by individual file) specifically to stop augmented copies
of one photo leaking across both sides — a real bug that inflated a
first run's accuracy to a fake 99-100%; second, any leftover
augmented-file duplicates in your dataset with untraceable filenames
(this project's had `aug_*`-prefixed files with no link back to their
source image) should be deleted rather than left in, since they can't
be reliably grouped by filename.

## 2. Run the dashboard

```bash
streamlit run app.py
```

The sidebar tells you plainly which backend is active (YOLO for
detection/type, CNN for quality) and warns clearly if either is
missing (`pip install ultralytics` / no `cnn_quality_models/*.pt`
found yet) — there's no model-file path to configure, and no fallback
to a weaker approach if something's missing. The only genuine decision
surfaced is:

**Measure physical size (cm)?** — off by default (pixel sizes
only). If turned on, choose either:
- **Auto-detect ArUco marker** — print a 4x4 ArUco marker (any
  marker generator, e.g. OpenCV's `cv2.aruco.generateImageMarker`),
  note its printed side length in cm, place it in-frame next to
  the fruit when photographing, and set that length in the
  sidebar. The app detects the marker automatically per photo.
- **I know my cm-per-pixel ratio** — enter a fixed ratio directly
  if you already have one (e.g. from a one-time ruler photo).

Everything else — denoise/enhance method, mask erosion, YOLO
confidence threshold — is tucked behind an **Advanced settings**
expander with defaults that already work.

Then upload images (drag-and-drop multiple files) or enter a
server-side folder path for bulk ingestion, and click
**Run Inspection**.

Every photo can contain multiple fruit — even touching ones — and
each is detected, measured, and classified separately. The dashboard
shows: aggregate counts and fruit-type/quality distribution charts
across every detected fruit, a per-fruit results table, and a
per-photo detail view (original + annotated image showing every
fruit's bounding box/contour/label, plus a breakdown card per fruit
with its type, quality, confidence, and size). Buttons let you
download the results as CSV or as a formatted PDF report.

## Known limitations

- **Non-plain / textured backgrounds (wood tables, patterned surfaces)
  are not reliably supported yet.** Segmentation (`_foreground_contours`
  in `colorDetection_Train.py`) estimates ONE background color by
  sampling a thin strip along the image border, then thresholds by LAB
  chroma distance from it. That assumes a plain, roughly uniform
  background — a wood table's grain lines/knots/shadows between planks
  vary enough that parts of the table can pass the same threshold as
  the fruit and get pulled into the mask, dragging that background
  color into the patches fed to the classifier. Confirmed directly on a
  real photo (a banana bunch on a wood table): the detected contour
  visibly traced individual plank grooves, and the fruit-type
  prediction came out wrong (Orange instead of Banana) as a result.
  A fix (a second, larger morphological-opening pass to strip the
  jagged leaked regions) was tried and **rejected** — it broke two
  other already-verified real-photo cases (apple+leaf went from 1
  detection back to 2; a 3-touching-apples photo dropped from 3
  detected down to 1), so it was reverted rather than shipped. Proper
  support needs a smarter background model (e.g. GrabCut seeded from
  the border sample) rather than a single global threshold — real
  design work, not a quick parameter tweak. Until then: photos with a
  plain, fairly uniform background behind the fruit are the
  supported/tested case.
- **Fixed: fruit packed edge-to-edge, touching or nearly touching all 4
  sides of the frame, used to break background estimation entirely.**
  `estimate_background_chroma()` samples a thin strip along the image
  border assuming it's pure background. When fruit fills the whole
  shot (a tray of apples with almost no visible tray/backdrop margin),
  most of that border strip is actually fruit, not background — the
  old median-based estimate got dragged toward the fruit's own color,
  so the segmentation mistook most of the fruit for background and only
  kept the few odd corners that still looked different enough (fruit
  fragments right at the image edge) as "foreground." Confirmed on a
  real apple-tray photo: only 2 tiny edge slivers were ever detected,
  not one of the dozen-plus whole apples filling the rest of the frame.
  Fixed by taking the MODE of the border pixels' (a, b) chroma (the
  densest cluster in a coarse 2D histogram, refined by the median of
  just that cluster) instead of the median of everything — even when
  true background pixels are a minority of the border, they form one
  tight color cluster, while contaminating fruit pixels (varied red/
  green hues) don't win any single bin. Verified on a synthetic
  reproduction of the same edge-to-edge layout: detection went from 1
  (the old fix's very first attempt) up to 7 out of 14 true apples, and
  every previously-verified regression test (leaf, 3-apples, touching-
  cluster cases, the dense-pile tests from the earlier fix) still
  passes unchanged. Still doesn't recover every last apple in a
  packed tray — same underlying "how densely can this classical-CV
  approach split touching same-color round fruit" ceiling described
  below — but detection at least happens now instead of misfiring
  almost everywhere.
- **A cluster of several same-species fruit posed touching/overlapping
  (a bunch of bananas, a pile of apples) may be detected as one object
  instead of split into individuals**, even on a plain background, if
  the watershed splitting logic in `_split_touching_cluster` doesn't
  find a confident split for that particular arrangement. The whole
  cluster's AGGREGATE shape (elongation, extent, solidity — see
  `extract_shape_features`) also won't look like a single fruit's
  shape, which can bias the fruit-type prediction even when the color
  is otherwise correct — confirmed on the same wood-table banana-bunch
  photo, where the cluster's compact aggregate outline read as "not
  elongated" and pushed the classification toward round-fruit types.
  Same root cause also corrupts the QUALITY prediction for the
  un-split cluster, not just the count: the shadowed gaps between
  touching fruit are genuinely part of the fruit's own surface (dark,
  but still fruit-colored), so they correctly stay in the mask — but
  with the cluster un-split, those dark seam patches get pooled
  together with every other fruit's seams into one aggregate quality
  vote, and dark patches are the same signal the model uses for
  Rotten. Confirmed on a real photo of a basket of ~15 oranges: mostly
  healthy-looking fruit, still voted "Rotten" once treated as one
  object.
  **Partially improved (still not solved) for dense same-species
  piles specifically** (many round fruit, not just 2-3): added a
  per-instance plausibility filter (`_filter_plausible_instances`) so
  a split isn't discarded entirely just because a few heavily-occluded
  pieces don't pass on their own, plus a second Hough pass
  (`_hough_dense_seed_candidates`) that estimates a per-fruit radius
  from the median of several initial seeds instead of the whole blob's
  own (in a deep pile, inflated) distance-transform maximum. Tested
  against two synthetic dense-pile images plus every previously-
  verified real-photo case (no regressions): a moderately-packed pile
  of 6 went from 2 detected instances to 5; a very densely packed pile
  of 14 went from 3 to 6. Real improvement, but still well short of
  the true count at high density — distance-transform local maxima and
  Hough circle arcs both run out of usable geometric evidence once
  overlap gets heavy enough, which is a structural limit of this
  classical-CV approach, not a remaining parameter to tune. Reliably
  counting 10+ densely piled fruit would need a different technique
  entirely (a trained object detector, e.g. YOLO), out of scope here.
- **Fixed: Fresh vs. Rotten confusion for Apple had a structural cause,
  not a feature-tuning one — now solved with a CNN, integrated and
  active.** The patch+majority-vote SVM quality model classifies each
  ~32x32 patch independently, then votes across ~15-20 patches per
  fruit. Confirmed directly: adding a synthetic dark spot covering up
  to 25% of a genuinely Fresh apple's surface never flipped the vote to
  Rotten, because most patches still look normal — and the same
  mechanism runs the other way too (a few odd-looking patches, e.g.
  water droplets, could outvote an otherwise Fresh apple). Four
  separate feature-engineering attempts this session (more/fewer hue
  bins, more texture bins, dropping the V channel, a per-patch "hue
  deviation from the fruit's own average hue" feature meant to separate
  shadow from real rot) all landed at the same ~65-73% held-out
  accuracy — confirming the bottleneck was the patch+vote architecture
  itself, not which features fed it.

  `train_cnn_quality.py` trains a CNN (MobileNetV2 transfer learning)
  that looks at the WHOLE fruit at once instead of tiling+voting, one
  model per fruit type — trained and validated on the user's own
  machine (PyTorch can't be installed in this sandbox; see the
  environment-limitation note further below). The FIRST run showed a
  suspicious 99-100% validation accuracy — traced to validation-split
  leakage: augmented copies of a training photo (rotated/flipped/noised
  duplicates, plus randomly-named `aug_*` files with no filename link
  back to their source) were landing on both sides of the train/val
  split, so the model was partly "grading its own homework." Fixed in
  `train_cnn_quality.py` by grouping samples by a recovered
  original-photo identity (`_base_image_key` strips known augmentation
  prefixes) before splitting, so every copy of one source photo stays
  on one side. A perceptual-hash-based near-duplicate merge was also
  tried, to catch the untraceable `aug_*` files by image content rather
  than filename — rejected after testing: too loose a threshold chains
  unrelated photos into one giant false cluster (47 images falsely
  merged at threshold=6), too strict misses real duplicates. The
  practical fix shipped instead: delete the `aug_*` files (same
  treatment as the other augmentation types), same reasoning as the
  rejected background-cleanup fix earlier in this file — an unreliable
  fix isn't worth shipping just because a reliable one is more manual.
  After that cleanup, the grouped split re-ran with train/val ratios
  matching the real file count almost exactly (e.g. Banana: 1111 files
  -> 1111 distinct photo families, meaning ~zero leftover duplicates),
  and validation accuracy landed at a believable 98.6-99.5% — consistent
  with published results for this dataset lineage, and a real,
  structural improvement over the SVM's ~73%, not an artifact.

  **Integrated into the live pipeline**, in `colorDetection.py` —
  and, per an explicit follow-up decision, the SVM quality path was
  then removed from that pipeline entirely rather than kept as a
  fallback. First cut: `classify_quality_cnn()` loaded whichever
  `cnn_quality_models/<FruitType>.pt` files existed and overrode the
  SVM's quality label per fruit, falling back silently to the SVM's own
  quality guess if no CNN model existed for that type. That fallback
  caused a real, confusing bug: `app.py`'s "Quality votes" caption still
  showed the SVM's raw per-patch vote breakdown (e.g. `{'Rotten': 4,
  'Fresh': 1, 'Unripe': 2}`) even on fruits where the CNN had overridden
  the final label to something the votes disagreed with (Fresh) —
  fixed by adding `ClassificationResult.quality_backend` and clearing
  `vote_counts` whenever the CNN produced the label, so the displayed
  votes can never contradict the shown result.

  After that fix, on request, the SVM quality path was removed as a
  fallback altogether: `inspect_image()` / `inspect_image_yolo()`
  (both still existed at this point) called a fruit-TYPE-only SVM
  classifier (`classify_fruit_type()`) plus `classify_quality_cnn()` —
  no code path silently substituted the weaker ~73%-accuracy SVM
  quality guess anymore. If a detected fruit's type had no
  `cnn_quality_models/<Type>.pt` (or PyTorch wasn't installed), that
  object's quality came back as unavailable with a specific
  `ClassificationResult.error` message instead. `app.py`'s sidebar
  reflected this: it warned explicitly (rather than a neutral
  "classical" caption) when no CNN model was available.

  Then, on a further explicit request to drop SVM from the project
  entirely (not just disconnect it), `app.py` was cut over to call
  ONLY `inspect_image_yolo()` (no model-file sidebar input, no
  fallback to the classical pipeline if `ultralytics` isn't
  installed — the Run button is disabled with a clear message
  instead), and `colorDetection.py` itself was rewritten to physically
  remove `inspect_image()`, `classify_fruit_type()`,
  `classify_segmented_image()`, `classify_segmented_image_known_type()`,
  `detect_objects()`/`detect_object()`, and every other classical
  multi-fruit-detection helper — not just stop calling them.
  `ClassificationResult` was also simplified (dropped `fruit_type_votes`/
  `vote_counts`/`n_patches`, which only ever meant anything for the SVM
  patch-vote path). `colorDetection_Train.py` got the same treatment for
  its half of the SVM code: `build_model()`/`train_model()`/
  `train_hierarchical_model()`/`compare_models()`, `VotingEnsemble`,
  `evaluate_majority_vote()`, and every `save_*`/`load_*` model
  function were deleted, cutting the file from 2200 lines to about
  1200. What's left there is scoped deliberately narrower than "not
  SVM" — it's specifically the classical *detection* algorithms
  (segmentation, watershed/Hough splitting) and feature extractors,
  kept because they're genuinely separate CV techniques from the SVM
  classifier that used to consume their output, not because they're
  still used by anything (only `segmentation_mask_and_contour()`
  actually still is, by `colorDetection.py`). The standalone
  `yolo_hybrid_detect.py` prototype script (an earlier, since-
  superseded attempt at YOLO+SVM hybrid detection) was deleted outright
  for the same reason — it was dead weight duplicating what
  `inspect_image_yolo()` already does, using the SVM approach being
  removed.
- **Real, non-augmented training data was limited across every quality
  class, not just Unripe** — since resolved by deleting leftover
  augmented files and adding more real photos (see above); this entry
  is kept for context. Stripping augmentation-prefixed filenames from
  `dataset/train` originally showed true unique-photo counts of roughly
  141-327 per class (e.g. freshapples: 1010 files -> 141 unique;
  rottenapples: 2342 -> 327). This is the main reason
  `train_cnn_quality.py` uses transfer learning (a backbone pretrained
  on ImageNet) rather than training a CNN from scratch: transfer
  learning is specifically suited to getting usable results from a few
  hundred labeled images per class instead of the thousands a
  from-scratch CNN would typically want.

## Notes / design decisions

- **Why two-stage fruit-type + per-fruit quality classification?**
  (Historical — describes the original SVM pipeline; `train_hierarchical_
  model()`, `classify_segmented_image()`, and `save_hierarchical_model()`/
  `load_hierarchical_model()` mentioned below have since been deleted
  from the codebase, see "CNN quality classification" above. Kept here
  because the underlying reasoning — quality must be learned per fruit
  species, not shared — is still exactly why the live YOLO+CNN pipeline
  also trains a separate CNN per fruit type in `train_cnn_quality.py`.)

  Color meaning is fruit-specific: a yellow patch means "unripe" on an
  apple but "ripe" on a banana. The original design used one shared
  Fresh/Unripe/Rotten classifier across every fruit, which forced it to
  learn a blurred, self-contradictory color->ripeness mapping (the same
  yellow color would need to mean two different things). Fixed by
  training two stages (`train_hierarchical_model()` in
  `colorDetection_Train.py`): stage 1 classifies fruit type from every
  patch regardless of species; stage 2 is a *separate* quality
  classifier per fruit type, so "yellow" is only ever interpreted using
  the color->ripeness mapping learned from that species' own training
  images. `colorDetection.py`'s `classify_segmented_image()` mirrored
  this at inference: classify fruit type by majority vote, then route
  to that fruit type's quality model for a second majority vote. Model
  files were saved/loaded as one bundle (`save_hierarchical_model()` /
  `load_hierarchical_model()`) containing the fruit-type model plus a
  dict of per-fruit quality models.
- **Why does the fruit-type stage also use shape, not just color?**
  Different fruit species often share near-identical colors at the
  Unripe/Rotten stages (e.g. two species that are both green-when-
  unripe, or both brown/black-when-rotten) — color alone can't reliably
  tell them apart there. `extract_shape_features()` in
  `colorDetection_Train.py` computes a 4-number whole-object descriptor
  once per image from its detected contour — bounding-box aspect ratio,
  extent, solidity, and ellipse elongation — and appends it to every
  patch's color feature ONLY for the fruit-type stage (the quality
  stage stays color-only, since ripeness genuinely is a color
  phenomenon once the species is known). This is a cheap, color-blind
  signal: an elongated banana looks nothing like a round apple/orange
  by shape regardless of what color state it's in. In a deliberately
  color-ambiguous test (apple and banana sharing identical Unripe/
  Rotten colors, differing only in shape), adding shape features raised
  fruit-type patch accuracy from 85% to 99.7%. The bundle records
  `"uses_shape_features"` so `colorDetection.py` knows whether to
  append shape before calling the fruit-type model — old model files
  saved without it will need retraining.
- **Why does every patch also get a texture histogram, not just
  color?** Color alone struggles specifically on Apple vs Orange:
  both are round (shape doesn't help there) and under bright, even
  studio lighting their red/orange hues can land close enough that
  the color histogram stops separating them — confirmed directly on a
  real photo, where two normally-lit, unoccluded apples split ~50/50
  Apple/Orange despite looking nothing alike to a person.
  `extract_texture_histogram()` in `colorDetection_Train.py` adds a
  16-bin Local Binary Pattern (LBP) histogram to every patch's color
  feature (521-dim -> 537-dim) — LBP encodes, per pixel, whether each
  of its 8 neighbors is brighter or darker than it, a lighting-
  brightness-invariant description of local surface micro-texture. It
  captures what color can't: an apple's skin is smooth and glossy, an
  orange's peel is finely dimpled all over, a banana's skin has soft
  ribbing. In a test built specifically so color statistics matched
  exactly between two classes (only spatial texture differed), color-
  only features got 79% accuracy; adding the texture histogram raised
  it to 100%. Used for BOTH stages (unlike shape, which is fruit-type-
  only) since texture also plausibly helps quality (e.g. a bruised or
  rotten spot has a different surface texture than healthy skin).
  Changing what `extract_patch_feature()` returns changes the
  dimensionality every model was trained on, so `save_hierarchical_model()`
  now records `"patch_feature_dim"`, and `load_hierarchical_model()`
  checks it (falling back to reading a quality-stage scaler's own
  `n_features_in_` for older files that predate this field) and raises
  a clear "retrain with the current script" error instead of a raw
  scikit-learn dimension-mismatch traceback if it doesn't match.
- **Why can a leaf get detected as its own "fruit"?** It doesn't
  anymore, but here's why it could: a leaf attached to a fruit by a
  thin stalk often doesn't survive the mask-building step as a
  bridge, so the leaf turns up as its own disjoint blob — exactly the
  same mechanism that correctly finds a second, physically separate
  fruit elsewhere in the photo. A leaf's flat, curved, veined shape
  then reads as elongated enough to get shape-classified as Banana.
  `_looks_like_fruit()` in `colorDetection_Train.py` rejects any
  detected object whose outline fills under 40% of its bounding box
  AND falls well below a round fruit's convexity AT THE SAME TIME —
  confirmed on a real apple-with-leaf photo that a leaf fails both at
  once, while even a genuinely elongated fruit or a partially-hidden
  fruit slice was confirmed to only ever fail one of the two. Some
  leaves curve gently enough that they pass this shape check anyway
  (not concave enough to trip the convexity threshold), so a second,
  independent filter — `_is_attached_debris()` — also rejects any
  small object whose bounding box sits substantially inside a much
  larger fruit's bounding box in the same photo, since a leaf/stem
  grows out of the same point as its fruit while a genuinely separate
  second fruit does not. It only compares objects that came from
  different original blobs, so it never rejects one apple in a
  touching/overlapping cluster just because it's the smallest.
- **Preprocessing must match between training and inference.** This
  classifier's features are pure color histograms/moments, so denoise/
  enhance settings change what it sees. `colorDetection_Train.py`
  applies `preprocessing.preprocess_image()` to every training image
  before segmenting it (`--denoise`/`--enhance` flags, defaults
  `median`/`clahe`), and saves those settings inside the `.pkl` file.
  `colorDetection.py`'s `load_model_and_preprocessing()` reads them
  back out so `app.py` classifies with the same preprocessing
  automatically — don't override the sidebar's advanced checkbox
  unless you retrain with matching `--denoise`/`--enhance` values too.
- **Why patch-based classification, not whole-image?** The training
  script classifies 32×32 color patches and aggregates by majority
  vote per image — this is more robust to partial occlusion, uneven
  lighting across the fruit surface, and localized rot spots than a
  single whole-image color histogram would be.
- **Why split touching fruit into separate instances, and count fruit
  that aren't touching at all?** A single-largest-contour approach has
  two separate failure modes: it treats several touching/overlapping
  fruit as one blob (a photo of 3 apples in contact got measured as
  one oversized, oddly-shaped "fruit" and even misclassified), AND it
  silently drops any fruit that isn't part of the single largest
  connected region (a second apple sitting elsewhere in the same
  photo, not touching anything, would just vanish). `find_object_instances()`
  in `colorDetection_Train.py` fixes both: it looks at every foreground
  contour above the noise-size threshold (not just the largest), and
  for each one, `_split_touching_cluster()` decides whether it's one
  fruit or several fused together —
    1. Distance-transform + watershed on that blob's mask: local maxima
       of the distance-to-background transform become seeds (one per
       fruit "core"), with a greedy radius-based merge (`SEED_MERGE_FRAC`)
       to discard spurious nearby peaks. This alone handles tangent
       contact and moderate overlap (up to ~1/3 radius) correctly.
    2. For heavier overlap, the touching dip between fruits gets too
       shallow to register as a separate peak at all, so the distance
       transform collapses the whole cluster to a single seed (this was
       reported directly: 3 real, heavily overlapping apples detected
       as 1). `cv2.HoughCircles` runs as a rescue in that specific case
       — it votes up each fruit's own round outline instead of needing
       a dip between fruits, so it can still recover all 3.
    3. Before trusting either technique's proposed split, the blob's
       contour must show a genuine concave "seam" via
       `cv2.convexityDefects` (`_has_genuine_seam()`) — real proof that
       two round shapes are fused together, not just a single fruit's
       ordinary boundary noise (a stem dimple, a soft drop-shadow).
       This is what stops Hough's occasional false positive — it was
       observed mistaking one apple's lighting gradient for a second
       circle — from ever splitting a genuinely single fruit.
  (Historical — `detect_objects()`/`inspect_image()`, which used to
  call this splitter, have since been deleted from `colorDetection.py`;
  the live pipeline gets per-fruit separation from YOLO's own detection
  boxes instead. `find_object_instances()`/`_split_touching_cluster()`
  themselves are still here, just unused — see the "CNN quality
  classification" section above.) They used to return one
  classification per fruit, not per photo — plus a `crop`/
  `crop_isolated` image per fruit (background and neighboring fruit
  blacked out) so each one could be looked at on its own, which the
  dashboard and PDF report both displayed. Detection ran on a
  denoise-only version of the image (skipping the enhancement/CLAHE
  step) because CLAHE's local contrast remapping introduces blocky
  artifacts that disrupt the distance-transform peaks; classification
  patches still come from the fully preprocessed image, since color
  statistics are what CLAHE was added to help with.
- **Why erode the segmentation mask?** Patches straddling the
  fruit/background edge contain a mix of black background and fruit
  color that the model never saw in training and that tends to get
  misread as "Rotten" (dark, low-saturation). Eroding the mask by a
  configurable margin removes that ring of contaminated patches.
- **Why LAB chroma-distance segmentation, not a fixed HSV threshold?**
  `segment_fruit()` estimates the background color by sampling a strip
  along the image border, then keeps a pixel as foreground only if its
  LAB (a, b) chroma is far enough from that background color —
  deliberately ignoring lightness (L). A soft drop-shadow under the
  fruit is the same background surface, just darker, so it keeps the
  background's chroma even though it's visibly darker; a threshold on
  HSV saturation/value can't reliably tell "background, but darker"
  apart from "genuinely different-colored object," so shadows used to
  leak into the mask and get fed to the classifier as if they were
  fruit pixels. Tunable via `SEGMENTATION_CHROMA_THRESHOLD` (raise if
  background still isn't fully excluded; lower if pale fruit is being
  cut out) and `SEGMENTATION_BORDER_FRAC` in `colorDetection_Train.py`.
  `colorDetection.py`'s `build_foreground_mask()` mirrors this exact
  logic so the bounding box/contour shown in the dashboard always
  matches the region actually fed to the classifier.
- **Calibration confidence**: `calibration.py` reports `"ok"` when the
  scale came from a directly-measured source (ArUco marker or a
  manually supplied ratio), `"fallback"` when auto-detection failed
  and a manual value was substituted, and `"uncalibrated"` when no
  scale is available at all — the dashboard and PDF report both
  surface this so measurements are never presented as more reliable
  than they are.
