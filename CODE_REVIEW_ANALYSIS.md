# Parserless Garment Split Refactoring Plan

## 1. The Core Problem

The current `/analyze` API fails to identify obvious two-piece outfits (like a pink tucked-in shirt with black pants, or a striped crop top with black pants) when running without a heavy human parsing model (`ANALYZE_USE_HUMAN_PARSER = false`).

It falsely forces these images into a single "Dress" or generic garment path due to brittle, rigid heuristics inside `src/api_features/yolo_cropper.py`.

## 2. Root Cause Analysis

The failures in the `_likely_two_piece_from_person_region` function stem from two major logic flaws:

### Flaw A: Geometry Clamps on Tucked-in Shirts (Image 1 Failure)

- **The Issue:** The code requires a *poor* `waist_x_alignment` to validate a color-based split (`waist_x_alignment <= 0.86`).
- **The Result:** If a user wears a perfectly tucked-in shirt, the left and right hip contours align seamlessly. The algorithmic `waist_x_alignment` scores near `0.95`. The code sees this perfect silhouette, ignores the massive color contrast between the shirt and pants, and falsely assumes it is a single color-blocked dress.

### Flaw B: Static Height Split Assumptions (Image 2 Failure)

- **The Issue:** The `_compute_split_geometry` function assumes a person's waist is always located at exactly `45%`, `49%`, or `54%` down their bounding box. It then samples a very narrow 10% vertical band at that static location.
- **The Result:** For high-waisted pants, crop tops, or photos cropped above the knees, the true "waist" line shifts. If the hardcoded 45% mark lands entirely over the black pants, the algorithm sees only black pixels above and below the line. It detects zero color difference and zero skin, completely missing the crop top located slightly higher up the torso.

---

## 3. Proposed Refactoring Solution

We must abandon static assumptions and rigid geometry clamps. Instead, we need to implement a **Dynamic Waist Search** and **Perfect Tuck Detection**.

### Initiative 1: Dynamic Split Line Search

Instead of guessing where the waist is based on a rigid bounding box ratio, the algorithm must dynamically scan the torso region to find the true separation line.

1. **Scan the Torso Matrix:** Evaluate horizontal pixel slices across the middle 50% of the bounding box (e.g., from `y = 25%` down to `y = 75%`).
2. **Find the Contrast Peak:** Calculate the color variance and edge density for each horizontal slice. The true boundary between a top and bottom will register as the slice with the maximum vertical color gradient.
3. **Anchor the Split:** Set `split_y` to this dynamically discovered peak contrast line, ensuring we are actually comparing the top garment to the bottom garment, regardless of high-waisted pants or crop-top proportions.

### Initiative 2: Perfect Tuck Tolerance

We must decouple massive color differences from silhouette geometry constraints.

1. **Bypass Geometry Clamps for High Contrast:** If the `color_dist` is exceptionally high (e.g., `> 0.25`, like bright pink vs deep black), the system should instantly classify it as a two-piece, even if the `waist_x_alignment` is a flawless `1.0`.
2. **Reserve Geometry for Ambiguity:** The `waist_x_alignment` check should only be enforced when the color difference is ambiguous or low (e.g., a navy top with black jeans). In those cases, a bulky overhang (poor alignment) serves securely as tie-breaker evidence of two separate pieces.

### Initiative 3: Robust Skin Span Detection

For crop tops where color contrast might be lower but skin is clearly visible:

1. **Continuous Horizontal Skin Tracking:** Instead of just measuring the overall percentage of skin in a region (`skin_ratio`), measure the longest continuous horizontal contiguous run of skin pixels (`horizontal_skin_span`).
2. **The Crop Top Rule:** If a horizontal skin band completely severs the torso from left to right (spanning >80% of the width), it is definitively a crop-top and bottom combination. This distinguishes it successfully from a single dress with a deep V-neck or side-cutouts (which do not sever the horizontal axis).

## 4. Expected Code Changes

**Target File:** `src/api_features/yolo_cropper.py`

- **Modify `_compute_split_geometry`:** Remove the static `aspect >= 2.8` hardcoded ratios. Replace with a dynamic vertical scan returning the highest-contrast `y` coordinate.
- **Modify `_likely_two_piece_from_person_region`:**
  - Implement the "Perfect Tuck" bypass rule for extreme `color_dist`.
  - Ensure `waist_x_alignment` is only checked as a secondary fallback constraint.
  - Enhance the `horizontal_skin_span` logic to require a full horizontal severance to trigger the crop-top rule.
