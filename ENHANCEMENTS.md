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
