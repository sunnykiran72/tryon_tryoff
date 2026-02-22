# Garment Detection & Analysis Enhancements Plan

This document tracks edge cases where the `/analyze` API fails, specifically focusing on multi-garment detection, dress routing, and confidence scoring.

## Issue 1: The "Dress Hallmark" False Positive (Crop Top Sets)

### **Image Evidence (Gray Crop Set)**

- **Original:** `crop_top_set_gray.png` (Gray Crop Top + Joggers)
- **Failure State:** The system detects the gray crop top as a "Dress" (specifically mislabeled as "Sweatshirt Dress").

### **The Problem (Type Conflict)**

In the latest logs, the user tried to select the **"top"** from a two-piece set. However, the system rejected the request with `INVALID_SELECTION_TYPE`:

- **YOLO Label:** Correctly identified the item as `long sleeved shirt`.
- **Heuristic Over-Correction:** Our recent "massive item" check in `garment_detector.py` saw the bounding box was large and overrode the label to **`dress`**.
- **The Result:** Even though the user asked for `top`, the system only had a `dress` in its memory, causing a 400 error.

### **Technical Root Cause (Crop Set)**

Historically, the `h_ratio_local >= 0.60` check in `garment_detector.py` was too aggressive for crop-top photos where the person is zoomed in. Because the crop top + partial pants took up most of the frame, the code assumed it was a dress.

### **Resolution Strategy (Crop Set)**

- **Tighten Dress Override:** Increase the height ratio required for the dress override or add a check for skin/gap presence before force-labeling as a dress.
  - **Current code status:** The active guard in `src/garment_detector.py` is `h_ratio_local >= 0.75` and only overrides `top -> dress` when the box spans near full-body vertical extent (`y0 < 0.15h` and `y1 > 0.85h`). This was tightened to reduce crop-top false positives.
- **Support Type Fallbacks:** If a user selects `top` but we found a `dress` that covers the top half, we should allow the VTON process to continue rather than rejecting it.

---

## Issue 2: Misclassified "Multiple Items" on Solid Dresses

### **Image Evidence (Black Dress)**

- **Original:** `black_dress_buttons.jpeg` (Black dress with waist tabs/buttons)
- **Failure State:** Returned `MULTI_ITEM_SELECTION_REQUIRED`.

### **The Problem (Native Split)**

The high-accuracy YOLO model sees the visual break (buttons/tabs) and predicts two separate pieces (a top and a bottom) instead of one continuous piece.

### **Resolution Strategy (Black Dress)**

- **Color/Texture Consistency Check:** We implemented a bridge in `main.py` to compare the top and bottom pieces. If the color and texture match perfectly, they are merged back into a dress.
- **Tuning:** We need to continue tuning these thresholds to ensure patterned dresses aren't accidentally split.

---

## Issue 3: Low Confidence Crashes on High-Detail Items

### **Image Evidence (Maxi Dress)**

- **Original:** `maxi_patterned_dress.png`
- **Failure State:** `LOW_CONFIDENCE` (CLIP score < 0.18).

### **The Problem (CLIP Variance)**

Detailed patterns sometimes confuse the CLIP classifier when it is restricted to a specific type (e.g., asking if a complex pattern is a "shirt").

### **Resolution Strategy (Maxi Dress)**

- **Adaptive Thresholds:** Lower the CLIP requirement slightly if the YOLO detector confidence is very high (>0.80).
- **Geometric Sanity:** Trust the bounding box size as a proxy for garment presence if the visual model is uncertain.

---

## Issue 4: Strapless Gown with Accessory Occlusion (Fur Stole)

### **Image Evidence (Black Strapless Gown)**

- **Original:** `black_strapless_gown.png`
- **Failure State:** `MULTI_ITEM_SELECTION_REQUIRED` (2 items found: "bottom/trousers" and a "top").

### **The Problem (Occlusion Split)**

The user is wearing a single continuous black strapless gown, but has a large fur stole draped over her arms.

- **The Split:** The fur stole physically cuts across the dress silhouette. The YOLO model sees the bottom half as a single object (mislabeling it as "trousers") and the top bodice as a separate object.
- **The Result:** The system fails to "see" the connection through the arms/accessories, resulting in a 400 rejection for Multiple Items.

### **Technical Root Cause (Occlusion)**

- **Occlusion:** Large accessories (like fur coats, stoles, or bags) that sever the visual continuity of a garment silhouette.
- **Low Confidence Variance:** The log shows a `detector_conf` of `0.2065`. At such low confidence levels, minor variations in image scaling can cause the "seed-like" behavior where results differ slightly between runs.

### **Resolution Strategy (Occlusion)**

- **Proactive Hue Matching:** Enhance the merge logic to trust color continuity across visual gaps if the pieces are perfectly centered.
- **Occlusion Bridge:** Increased `DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_PX` to 64px and ratio to 0.08 to bridge visual gaps caused by accessories/arms. (Fixed in main.py).
- **Lower Confidence Filtering:** Increased `DETECT_YOLO_MIN_CONF` to 0.25 to stabilize detections and ignore "ghost" objects that cause variance between test runs. (Fixed in main.py).

---

## Issue 5: Cut-Out Dress with Fragmentation and Duplication

### **Image Evidence (Sage Green Cut-Out Dress)**

- **Original:** `sage_green_cutout_dress.png`
- **Failure State:** `MULTI_ITEM_SELECTION_REQUIRED` with duplicate previews of the same component (the skirt).

### **The Problem (Fragmentation & Duplication)**

The dress has a significant cut-out at the waist, creating two separate pixel "islands" (the one-shoulder top and the ruched skirt).

- **The Duplication:** The system returned two identical previews of the skirt part. This happens when deduplication (IoU) fails to catch two nearly-identical bounding boxes.
- **The Fragmentation:** The top bodice part was likely either ignored or detected separately, preventing a single "Dress" result.
- **The Result:** User sees two "Bottom" choices that are actually the same physical item, and the top is missing.

### **Technical Root Cause (Cut-Outs)**

- **Deduplication Threshold:** The IoU threshold of 0.90 might be too high for items with very similar but non-identical masks (e.g., YOLO vs Parser versions of the same thing).
- **Island Connectivity:** Standard dress logic expects a continuous silhouette. Large waist-level cut-outs break the "Single Item" detection.

### **Resolution Strategy (Cut-Outs)**

- **Aggressive Deduplication:** Lower the deduplication IoU threshold to 0.85 to catch near-duplicates.
- **Centric Priority:** Prioritize the largest centered component and attempt to merge nearby fragments of the same color/texture.
- **Label Smoothing:** Force merger of disjointed islands if they share identical color histograms and vertical alignment.
  - **Current code status:** Implemented at detector-mask level (`IoU >= 0.85` dedup) plus bbox-level dedup in analyzer post-processing.

---

## Issue 6: Zoom-Dependent Detection Variance (Crop Top Cropping)

### **Image Evidence (Crochet Crop Top)**

- **Original:** `crochet_crop_top_full.png` (Wide shot: 2 items found, correct).
- **Failure State:** `crochet_crop_top_zoom.png` (Zoomed shot: `INVALID_SELECTION_TYPE` when requesting "top").

### **The Problem (Zoom-Induced Over-Classification)**

The same outfit behaves differently depending on the image crop.

- **Full Shot:** The system correctly sees a "top" and a "bottom" (Multiple Items).
- **Zoomed Shot:** The crop top fills a large vertical percentage of the cropped frame. The system overrides the YOLO "shirt" label to **"dress"** due to the height ratio.
- **The Result:** When the user selects "top", the API returns a 400 error because it only "sees" a dress, even though it's clearly the top the user wants.

### **Technical Root Cause (Context Loss)**

In zoomed photos, the "image height" is no longer a reliable proxy for "person height." A 12-inch crop top in a 15-inch photo looks like a "massive item" (0.80 ratio), triggering the dress override logic.

### **Resolution Strategy (Zoom Consistency)**

- **Cross-Type Satisfaction:** Modify `flow_handlers.py` to allow a `dress` result to satisfy a `top` or `bottom` request if it's the only valid item found and the original YOLO class was a shirt/pants.
- **Improved Center-of-Mass Check:** Only override to "dress" if the item spans both the upper and lower quadrants of the detected person region.

---

## Issue 7: Recursive Zoom and Context Compounding (Split Previews)

### **Image Evidence (Crochet Crop Top / Patterned Dress)**

- **Original:** `test1.jpeg`, `test2.png`
- **Failure State:** Previews in "Multiple Items Found" response were sometimes truncated or "bottom-half" only, even for top candidates.

### **The Problem (Sub-Crop Splitting)**

When the system was uncertain, it attempted to split a candidate into top/bottom previews. However, it was splitting from a previously-trimmed sub-crop (the "selected item crop") rather than the global source image. If the sub-crop was already biased (e.g., just the waist-down region), the resulting "top" preview was actually just a slice of the pants.

### **Resolution Strategy (Global Split)**

- **Global Context:** Modified `flow_handlers.py` to always use the `source_image` (full upload) for synthesizing previews in the low-confidence split flow.
- **Geometry-Aware Cut:** Switched to `_split_crop_for_forced_type_from_items` which uses all detected bounding boxes to calculate the mathematically optimal horizon line for the split.
- **Dress Protection:** Added a height-based guard (`h_ratio >= 0.58`) to prevent prominent single pieces (like midi dresses) from being forced into a split-type response.

---

## Commit Review Notes (2026-02-22)

Verified commit chain:

- `d83b403`: introduced top->dress geometry override for very tall "top" detections.
- `f7a6374`: added single-piece merge logic for native YOLO split outputs (color/texture bridge).
- `fefe4b8`: stabilized dress routing and multi-item behavior.
- `36850e0`: introduced this enhancement tracking document.
- `56aae74`: added duplication-focused improvements.
- `c8b864b`: zoom stability and cross-type routing adjustments.
- `330649d`: dynamic waist split and high-contrast tuck bypass in `yolo_cropper`.
- `577c803`: prevented forced single-piece merge when split signal is strong.

### Audit Findings

- Dress override guard is currently `0.75` with vertical-span constraints (`y0 < 0.15h`, `y1 > 0.85h`) in `src/garment_detector.py`.
- "Fixed in main.py" is partially true for merge logic, but related routing now also lives in `src/api_features/flow_handlers.py` and detector behavior in `src/garment_detector.py`.
- Bottom metadata specificity (`jeans`, `trousers`, etc.) depends on detector class-name mapping and forced-type routing logic in `flow_handlers.py`; generic `Bottom` still appears when style confidence is weak or fallback style is used.
- Current repository state has a coherent commit sequence but a non-clean working tree (many untracked files), so any production release should use a cleanup/curation pass before tagging.
