#!/usr/bin/env python3
"""
Stage-specific test runner for YOLO cropper logic used by /analyze.

This script calls the same internal function used by analyze:
  main._build_yolo_item_breakdown_from_image(...)

It accepts URL/file inputs and writes:
  - summary.json (high-level result per case)
  - detailed.json (full debug payload per case)
  - copied crop previews under items/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as app_main


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON array")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(data, start=1):
        if isinstance(row, str):
            out.append({"id": f"case_{idx:02d}", "image": row})
            continue
        if not isinstance(row, dict):
            continue
        image = row.get("image") or row.get("image_url") or row.get("url") or row.get("file")
        if not image:
            continue
        out.append(
            {
                "id": str(row.get("id") or row.get("name") or f"case_{idx:02d}"),
                "image": str(image),
                "max_items": row.get("max_items"),
                "crop_padding_px": row.get("crop_padding_px"),
                "min_item_pixels": row.get("min_item_pixels"),
            }
        )
    return out


def _normalize_source(value: str) -> str:
    src = str(value).strip()
    if src.startswith(("http://", "https://", "file://")):
        return src
    p = Path(src)
    if p.exists():
        return f"file://{p.resolve()}"
    return src


def _iter_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    if args.manifest:
        cases.extend(_load_manifest(Path(args.manifest)))

    for idx, url in enumerate(args.url or [], start=1):
        cases.append({"id": f"url_{idx:02d}", "image": str(url)})

    for idx, file_path in enumerate(args.file or [], start=1):
        cases.append({"id": f"file_{idx:02d}", "image": str(file_path)})

    if not cases:
        raise ValueError("no input cases provided. Use --url/--file/--manifest")
    return cases


def _calc_default_min_pixels(image: Image.Image) -> int:
    image_area = int(image.width * image.height)
    by_ratio = max(128, int(image_area * max(0.0005, app_main.ANALYZE_MIN_COMPONENT_AREA_RATIO)))
    return max(int(app_main.DETECT_MIN_ITEM_PIXELS), by_ratio)


def _copy_preview(preview_path: Optional[str], dest: Path) -> Optional[str]:
    if not preview_path:
        return None
    src = Path(preview_path)
    if not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)


def _install_clip_stub() -> None:
    style_map = {
        "top": "Top",
        "bottom": "Bottom",
        "dress": "Dress",
        "outer": "Outerwear",
    }

    def _stub(image: Image.Image, garment_type: str) -> tuple[str, float]:
        _ = image
        gt = str(garment_type or "").strip().lower()
        return style_map.get(gt, "Garment"), 0.5

    # main._yolo_cropper_deps() binds run_clip_classification_fn from this symbol.
    # Overriding it avoids CLIP/HF dependency during stage-A split/crop testing.
    app_main._run_clip_classification = _stub


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage tests for yolo_cropper only.")
    parser.add_argument("--url", action="append", help="Input image URL. Repeatable.")
    parser.add_argument("--file", action="append", help="Input local image file path. Repeatable.")
    parser.add_argument("--manifest", help="JSON manifest path (array of string/dict entries).")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--crop-padding-px", type=int, default=12)
    parser.add_argument("--min-item-pixels", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--pretrim-empty-border", dest="pretrim_empty_border", action="store_true")
    parser.add_argument("--no-pretrim-empty-border", dest="pretrim_empty_border", action="store_false")
    parser.set_defaults(pretrim_empty_border=True)
    parser.add_argument("--pretrim-diff-threshold", type=int, default=18)
    parser.add_argument("--pretrim-min-content-ratio", type=float, default=0.02)
    parser.add_argument("--symmetric-padding-only", dest="symmetric_padding_only", action="store_true")
    parser.add_argument("--type-aware-padding", dest="symmetric_padding_only", action="store_false")
    parser.set_defaults(symmetric_padding_only=True)
    parser.add_argument("--min-padding-px", type=int, default=8)
    parser.add_argument("--min-padding-ratio", type=float, default=0.02)
    parser.add_argument("--parser-enabled", dest="parser_enabled", action="store_true")
    parser.add_argument("--parser-disabled", dest="parser_enabled", action="store_false")
    parser.set_defaults(parser_enabled=False)
    parser.add_argument("--skip-clip", dest="skip_clip", action="store_true")
    parser.add_argument("--use-clip", dest="skip_clip", action="store_false")
    parser.set_defaults(skip_clip=True)
    args = parser.parse_args()

    # Stage-A test toggles: keep this script focused on cropper behavior only.
    app_main.ANALYZE_USE_HUMAN_PARSER = bool(args.parser_enabled)
    app_main.ANALYZE_PRETRIM_EMPTY_BORDER = bool(args.pretrim_empty_border)
    app_main.ANALYZE_PRETRIM_DIFF_THRESHOLD = int(args.pretrim_diff_threshold)
    app_main.ANALYZE_PRETRIM_MIN_CONTENT_RATIO = float(args.pretrim_min_content_ratio)
    app_main.ANALYZE_SYMMETRIC_PADDING_ONLY = bool(args.symmetric_padding_only)
    app_main.ANALYZE_MIN_PADDING_PX = int(args.min_padding_px)
    app_main.ANALYZE_MIN_PADDING_RATIO = float(args.min_padding_ratio)

    if args.skip_clip:
        _install_clip_stub()

    out_dir = Path(args.out_dir) if args.out_dir else Path("final-testing") / f"yolo_cropper_stage_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items_dir = out_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    cases = _iter_cases(args)
    summary: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []

    for row in cases:
        case_id = str(row["id"])
        source = _normalize_source(str(row["image"]))
        max_items = int(row.get("max_items") or args.max_items)
        crop_padding_px = int(row.get("crop_padding_px") or args.crop_padding_px)

        record: Dict[str, Any] = {
            "id": case_id,
            "source": source,
            "ok": False,
        }

        try:
            t0 = time.perf_counter()
            image, _ = app_main._download_image(source)
            min_item_pixels = int(
                row.get("min_item_pixels")
                or args.min_item_pixels
                or _calc_default_min_pixels(image)
            )
            yolo = app_main._build_yolo_item_breakdown_from_image(
                source_image=image,
                max_items=max_items,
                crop_padding_px=crop_padding_px,
                min_item_pixels=min_item_pixels,
                include_upload_url=False,
                include_base64=False,
            )
            elapsed = round(time.perf_counter() - t0, 4)

            item_breakdown = list(yolo.get("item_breakdown") or [])
            preview_items: List[Dict[str, Any]] = []
            for item in item_breakdown:
                rank = int(item.get("rank", 0))
                p_local = item.get("image_local_path")
                copied = _copy_preview(
                    p_local,
                    items_dir / f"{case_id}_item{rank}.png",
                )
                preview_items.append(
                    {
                        "rank": rank,
                        "garment_type": item.get("garment_type"),
                        "style": item.get("style"),
                        "bbox_original": item.get("bbox_original"),
                        "bbox_crop": item.get("bbox_crop"),
                        "source": item.get("source"),
                        "crop_size": item.get("crop_size"),
                        "garment_pixels": item.get("garment_pixels"),
                        "preview_local_path": copied,
                    }
                )

            record.update(
                {
                    "ok": True,
                    "elapsed_s": elapsed,
                    "image_size": {"width": image.width, "height": image.height},
                    "settings": {
                        "max_items": max_items,
                        "crop_padding_px": crop_padding_px,
                        "min_item_pixels": min_item_pixels,
                        "skip_clip": bool(args.skip_clip),
                        "parser_enabled": bool(args.parser_enabled),
                        "pretrim_empty_border": bool(args.pretrim_empty_border),
                        "pretrim_diff_threshold": int(args.pretrim_diff_threshold),
                        "pretrim_min_content_ratio": float(args.pretrim_min_content_ratio),
                        "symmetric_padding_only": bool(args.symmetric_padding_only),
                        "min_padding_px": int(args.min_padding_px),
                        "min_padding_ratio": float(args.min_padding_ratio),
                    },
                    "counts": {
                        "raw_instances_count": yolo.get("raw_instances_count"),
                        "candidate_instances_count": yolo.get("candidate_instances_count"),
                        "valid_instances_count": yolo.get("valid_instances_count"),
                        "parser_category_count": yolo.get("parser_category_count"),
                    },
                    "detection_source": yolo.get("detection_source"),
                    "pretrim_debug": yolo.get("pretrim_debug"),
                    "detector_debug": yolo.get("detector_debug"),
                    "parser_category_debug": yolo.get("parser_category_debug"),
                    "candidate_postprocess_debug": yolo.get("candidate_postprocess_debug"),
                    "items": preview_items,
                }
            )
        except Exception as exc:
            record["error"] = str(exc)

        detailed.append(record)
        summary.append(
            {
                "id": record["id"],
                "ok": record["ok"],
                "elapsed_s": record.get("elapsed_s"),
                "detection_source": record.get("detection_source"),
                "pretrim_applied": bool((record.get("pretrim_debug") or {}).get("applied", False)),
                "valid_instances_count": (record.get("counts") or {}).get("valid_instances_count"),
                "items": [
                    {
                        "rank": i.get("rank"),
                        "garment_type": i.get("garment_type"),
                        "style": i.get("style"),
                        "crop_size": i.get("crop_size"),
                        "preview_local_path": i.get("preview_local_path"),
                    }
                    for i in (record.get("items") or [])
                ],
                "error": record.get("error"),
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "detailed.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "cases": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
