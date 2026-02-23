# Pipeline Stage Testing Guide

This guide lets you test each stage independently using URLs or files, without running the full `/analyze` decision logic every time.

## 1) What to test separately

- **Stage A: YOLO cropper/object detection + split logic only**
- **Stage B: VTON tryon (`/vton-v15/cloth-only`) only**
- **Stage C: Full `/analyze` integration (optional regression pass)**

## 2) Prerequisites

Run commands from repository root:

```bash
cd /Users/kiran/Documents/github/clients/glamify/tryon_tryoff_v2
```

Services:

- Analyze API: `http://127.0.0.1:8000`
- VTON API: `http://127.0.0.1:8004/vton-v15/cloth-only`

If you want deterministic model behavior, ensure model paths and env vars are loaded before starting services.

## 3) Stage A: YOLO Cropper only

Script:

- `final-testing/run_yolo_cropper_stage_tests.py`

What it calls internally:

- `main._build_yolo_item_breakdown_from_image(...)`
- By default this runner uses `--skip-clip` (stubbed CLIP) so you can debug detection/split/crop without HF/CLIP dependency.

### 3.1 Run with URLs

```bash
python3 final-testing/run_yolo_cropper_stage_tests.py \
  --url "https://images.unsplash.com/photo-1733310925495-0b0c596945bc?q=80&w=987&auto=format&fit=crop" \
  --url "https://www.glamifyfashion.com/cdn/shop/files/20oct-17.png?v=1760949660" \
  --pretrim-empty-border \
  --symmetric-padding-only \
  --min-padding-px 8 \
  --min-padding-ratio 0.02 \
  --parser-disabled
```

### 3.2 Run with local files

```bash
python3 final-testing/run_yolo_cropper_stage_tests.py \
  --file "/Users/kiran/Downloads/dom-hill-nimElTcTNyY-unsplash.jpg" \
  --file "/Users/kiran/Downloads/test1.jpeg"
```

### 3.3 Run with manifest

Sample manifest file:

- `final-testing/manifests/yolo_stage_urls.sample.json`

Run:

```bash
python3 final-testing/run_yolo_cropper_stage_tests.py \
  --manifest final-testing/manifests/yolo_stage_urls.sample.json
```

To force full CLIP scoring in stage-A:

```bash
python3 final-testing/run_yolo_cropper_stage_tests.py \
  --manifest final-testing/manifests/yolo_stage_urls.sample.json \
  --use-clip
```

### 3.4 Stage A outputs

Each run creates:

- `final-testing/yolo_cropper_stage_<timestamp>/summary.json`
- `final-testing/yolo_cropper_stage_<timestamp>/detailed.json`
- `final-testing/yolo_cropper_stage_<timestamp>/items/*.png`

Key debug fields to inspect:

- `detection_source`
- `raw_instances_count`
- `candidate_instances_count`
- `valid_instances_count`
- `detector_debug`
- `candidate_postprocess_debug`
- `pretrim_debug`
- each item's `bbox_original`, `bbox_crop`, `garment_type`, `style`, `preview_local_path`

Useful Stage-A knobs in this runner:

- `--pretrim-empty-border` / `--no-pretrim-empty-border`
- `--pretrim-diff-threshold`
- `--pretrim-min-content-ratio`
- `--symmetric-padding-only` / `--type-aware-padding`
- `--min-padding-px`
- `--min-padding-ratio`
- `--parser-disabled` / `--parser-enabled`

## 4) Stage B: Tryon only

Script:

- `final-testing/run_tryon_stage_tests.py`

What it does:

- Calls only VTON endpoint (`/vton-v15/cloth-only`)
- Does **not** run yolo-cropper or selection logic
- Saves result JSON and downloads output images locally by default

### 4.1 Run with URL garments

```bash
python3 final-testing/run_tryon_stage_tests.py \
  --person-image-url "file:///workspace/Any2anyTryon/inputs/fashn_showroom_model.png" \
  --garment-url "https://www.glamifyfashion.com/cdn/shop/files/31_666b7329-6cfe-4968-8077-f4a8c6fd287a.png?v=1740136682" \
  --garment-url "https://www.glamifyfashion.com/cdn/shop/files/20oct-17.png?v=1760949660" \
  --category one-pieces \
  --timesteps 35 \
  --guidance-scale 3.2 \
  --cutout-enabled \
  --cutout-background transparent \
  --zoom-enabled
```

### 4.2 Run with manifest

Sample manifest file:

- `final-testing/manifests/tryon_stage_urls.sample.json`

Run:

```bash
python3 final-testing/run_tryon_stage_tests.py \
  --manifest final-testing/manifests/tryon_stage_urls.sample.json \
  --person-image-url "file:///workspace/Any2anyTryon/inputs/fashn_showroom_model.png" \
  --category one-pieces \
  --timesteps 35 \
  --guidance-scale 3.2
```

### 4.3 Stage B outputs

Each run creates:

- `final-testing/vton_stage_<timestamp>/results.json`
- `final-testing/vton_stage_<timestamp>/outputs/*` (downloaded output images)

Key fields:

- `http_status`
- `output_url`
- `metrics`
- `input_debug`
- `downloaded_output_path`

## 5) Stage C: full `/analyze` optional

Use existing scripts:

- `final-testing/run_analyze_regression.py`
- `final-testing/run_requested_cases.py`

Example:

```bash
python3 final-testing/run_requested_cases.py
```

This is useful after Stage A/B are validated.

## 6) Fast debugging workflow

1. Run Stage A first and verify split/type accuracy.
2. Run Stage B on exactly those crops/URLs and verify cloth-only quality.
3. Run full `/analyze` only after A and B are stable.

## 7) Common failure signatures

- `detection_source: foreground_fallback`
  - Usually YOLO model unavailable or not loaded.
- `LOW_CONFIDENCE` in full `/analyze`
  - Detector/CLIP disagreement or weak crop context.
- Wrong top/bottom output in tryon stage
  - Category mismatch (`tops|bottoms|one-pieces`) or insufficient crop context.

## 8) What to share in a new chat

For quick handoff, share:

- latest Stage A `summary.json` + `detailed.json`
- latest Stage B `results.json`
- 2–3 representative preview/output images
- current `.env` detector + analyze threshold block
