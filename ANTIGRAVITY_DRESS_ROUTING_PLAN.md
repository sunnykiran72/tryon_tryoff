# Antigravity Review: Dress-First Routing Plan

## 1) Problem Summary

We are seeing false `MULTI_ITEM_SELECTION_REQUIRED` for true single-dress images.

Root cause:

- The current parserless split heuristic can over-trigger on person masks (especially exposed skin around waist/chest), causing `yolo_person_split`.
- Once split occurs, response goes to multi-item flow before single-dress VTON path can complete.

Business expectation:

- If image is truly a single dress, route directly to VTON and return final extracted garment.
- Multi-item flow should trigger only when there is strong evidence of real top+bottom separation.

## 2) Current Relevant Code Paths

- Detector + class/type inference:
  - `/Users/kiran/Documents/github/clients/glamify/tryon_tryoff_v2/src/garment_detector.py`
- Parserless split heuristic:
  - `/Users/kiran/Documents/github/clients/glamify/tryon_tryoff_v2/src/api_features/yolo_cropper.py` (`_likely_two_piece_from_person_region`, `_split_single_component_top_bottom`)
- Single-piece merge logic:
  - `/Users/kiran/Documents/github/clients/glamify/tryon_tryoff_v2/main.py` (`_postprocess_single_piece_candidates`)
- Low-confidence split branch:
  - `/Users/kiran/Documents/github/clients/glamify/tryon_tryoff_v2/src/api_features/flow_handlers.py`

## 3) Target Decision Policy

1. Dress-first decision in no-type mode:
   - If evidence strongly indicates single dress, force single-item dress route.
2. Multi-item only when two-piece evidence is strong.
3. Person presence is not a business-level condition:
   - It can be used as geometric fallback signal only.

## 4) Proposed Identification Logic

### Stage A: Candidate Build

- Use current YOLO masks and bbox candidates.
- Preserve current filters: min area, min overlap, minimum pixel constraints.

### Stage B: Compute `single_dress_likelihood`

Use these features:

- `bbox_height_ratio` (dress tends to be tall). *Note: Must be correlated with the person's absolute height in frame to handle mini dresses/cropped photos.*
- `mask_continuity` (single connected region favors dress). *Note: Must include a generous dilation tolerance to gracefully handle cutouts, sheer mesh panels, plunging necklines.*
- `x_overlap/top-bottom continuity`
- `waist_x_alignment`: Check if the vertical alignment of the left and right edges at the split line matches. This is a very strong indicator of a single connected garment and can override color/texture differences.
- `vertical gap` between hypothetical upper/lower regions (small favors dress)

### Stage C: Compute `two_piece_evidence`

Use combined signals only:

- `waist valley strength`
  - *Constraint:* Check for `occluder_bridge_check` (belt/hand/bag detection around the waist). If an occluder label is bridging the gap, do not increase two-piece evidence.
- `upper/lower color distance`
- `upper/lower texture delta`
  - *Constraint:* Color and texture deltas should ONLY vote towards a split if the geometry also implies a split. If `waist_x_alignment` is perfect, ignore color/texture delta (e.g. for color-blocked dresses).
- `horizontal_skin_span`: Measure if the skin completely severs the horizontal axis (spanning 80-100% of the torso width) to distinguish between a true crop-top/skirt combination vs. a dress with a side cut-out.

Important:

- Skin ratio by itself must not force split.
- Split should require multi-signal confirmation.

### Stage D: Route Decision

- If `single_dress_likelihood >= T_dress` and `two_piece_evidence < T_split`:
  - Force `garment_type = dress`
  - Route to single-item VTON flow directly.
  - *Note: `T_dress` must be weighted heavily. If there is ambiguity, err on the side of a single dress to preserve the UX, rather than incorrectly splitting a dress.*
- Else if `two_piece_evidence >= T_split`:
  - Multi-item selection response (`top` / `bottom`).
- Else:
  - Low-confidence reject (no forced split).

## 5) Planned Code Changes

1. `src/api_features/yolo_cropper.py`
   - Make split strength explicit (strong vs weak).
   - Only strong split signals can block later dress merge.
   - Add dress-lock gate before committing split.

2. `main.py` (`_postprocess_single_piece_candidates`)
   - Keep merge enabled for weak split outcomes.
   - Block merge only when split evidence is strong and consistent.

3. `src/api_features/flow_handlers.py`
   - Prevent low-confidence auto-split when dress-lock is true.
   - Keep explicit `type=top|bottom` behavior unchanged.

## 6) Debug Fields To Expose

For each analyzed item (internal + response debug):

- `decision_path` (`single_dress_direct_vton` | `multi_selection` | `low_conf_reject`)
- `single_dress_likelihood`
- `two_piece_evidence`
- `split_reason_codes`
- `detector_source`
- `bbox_height_ratio`
- `x_overlap`
- `waist_x_alignment`
- `vertical_gap_px`
- `color_dist`
- `texture_delta`
- `skin_ratio`
- `horizontal_skin_span`
- `occluder_bridge_check`

## 7) Acceptance Criteria

1. Single-dress cases (no `type` sent):
   - Return `200`
   - `selected_type = dress`
   - Output from VTON path

2. True top+bottom multi-garment cases:
   - Return `400 MULTI_ITEM_SELECTION_REQUIRED` on first pass
   - Re-run with `type=top` or `type=bottom` returns `200`

3. No regression:
   - Existing explicit `type` route quality remains stable
   - Bottom extraction quality should not degrade from known good baselines

## 8) Regression Suite (Minimum)

Run all with `/analyze` and no `type` first:

- Known single dresses: must return single dress (`200`)
- Known two-piece sets: must return multi-select (`400`)
- For multi-select, run second pass with `type=top` and `type=bottom`: both must return `200`

Store:

- request image
- response metadata json
- output image urls
- local artifacts for side-by-side comparison

## 9) Why This Plan Is Safer

- It aligns with product behavior (single dress should not be forced through selection UX).
- It limits person-signal influence to fallback geometry only.
- It is robust to modern fashion complexities (color-blocking, cutouts, mesh panels, mini dresses).
- It introduces measurable debug metrics so failures are diagnosable, not heuristic-black-box.
