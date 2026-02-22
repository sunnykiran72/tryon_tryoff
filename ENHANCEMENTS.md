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

The `h_ratio_local >= 0.60` check in `garment_detector.py` is too aggressive for crop-top photos where the person is zoomed in. Because the crop top + partial pants took up most of the frame, the code assumed it was a dress.

### **Resolution Strategy (Crop Set)**

- **Tighten Dress Override:** Increase the height ratio required for the dress override or add a check for skin/gap presence before force-labeling as a dress. (Fixed: Raised to 0.75 ratio).
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
