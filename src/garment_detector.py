import os
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


_yolo_model = None
_yolo_model_path = None
_yolo_lock = Lock()

_TOP_KEYWORDS = {
    "top",
    "shirt",
    "tshirt",
    "sweatshirt",
    "crewneck",
    "tee",
    "blouse",
    "hoodie",
    "sweater",
    "pullover",
    "crop",
    "tank",
    "jersey",
    "camisole",
    "vest",
}
_BOTTOM_KEYWORDS = {
    "bottom",
    "pant",
    "pants",
    "trouser",
    "trousers",
    "jean",
    "jeans",
    "jogger",
    "legging",
    "leggings",
    "short",
    "shorts",
    "skirt",
}
_DRESS_KEYWORDS = {"dress", "gown", "jumpsuit", "romper", "onepiece", "one-piece", "overall"}
_OUTER_KEYWORDS = {"coat", "jacket", "blazer", "cardigan", "outer", "outerwear", "windbreaker"}
_PERSON_KEYWORDS = {"person", "human", "man", "woman", "people", "body", "model"}


def _normalize_label_name(name: str) -> str:
    normalized = name.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in normalized)
    return " ".join(normalized.split())


def _label_matches_keywords(label: str, keywords: set) -> bool:
    if not label:
        return False
    tokens = label.split()
    for kw in keywords:
        if kw in label:
            return True
        if kw in tokens:
            return True
    return False


def _infer_garment_type_from_class_name(class_name: str) -> Optional[str]:
    label = _normalize_label_name(class_name)
    if not label:
        return None
    if _label_matches_keywords(label, _DRESS_KEYWORDS):
        return "dress"
    if _label_matches_keywords(label, _OUTER_KEYWORDS):
        return "outer"
    if _label_matches_keywords(label, _TOP_KEYWORDS):
        return "top"
    if _label_matches_keywords(label, _BOTTOM_KEYWORDS):
        return "bottom"
    return None


def _is_garment_like_class(class_name: str) -> bool:
    label = _normalize_label_name(class_name)
    if not label:
        return False
    keyword_groups = (_TOP_KEYWORDS, _BOTTOM_KEYWORDS, _DRESS_KEYWORDS, _OUTER_KEYWORDS)
    return any(_label_matches_keywords(label, group) for group in keyword_groups)


def _infer_garment_type_from_bbox(y0: int, y1: int, image_height: int) -> str:
    h = max(1, y1 - y0)
    y_center_ratio = float((y0 + y1) / max(1, 2 * image_height))
    h_ratio = float(h / max(1, image_height))
    if h_ratio >= 0.62:
        return "dress"
    if y_center_ratio <= 0.50:
        return "top"
    return "bottom"


def _is_person_like_class(class_name: str, class_id: int) -> bool:
    label = _normalize_label_name(class_name)
    _ = class_id
    return bool(label and _label_matches_keywords(label, _PERSON_KEYWORDS))


def _is_generic_garment_candidate(
    bbox: Tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    area_ratio: float,
) -> bool:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    w_ratio = float(bw / max(1, image_width))
    h_ratio = float(bh / max(1, image_height))
    cx = float((x0 + x1) / max(1, 2 * image_width))
    cy = float((y0 + y1) / max(1, 2 * image_height))

    # Keep generic classes only when they look like a central garment/body region.
    # This preserves older "works despite wrong class-name" behavior without
    # allowing tiny/random background detections.
    if area_ratio < 0.03:
        return False
    if w_ratio < 0.18 or h_ratio < 0.24:
        return False
    if cx < 0.08 or cx > 0.92:
        return False
    if cy < 0.10 or cy > 0.95:
        return False
    return True


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _bbox_x_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, _, ax1, _ = [int(v) for v in a]
    bx0, _, bx1, _ = [int(v) for v in b]
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    min_width = max(1, min(max(1, ax1 - ax0), max(1, bx1 - bx0)))
    return float(inter / min_width)


def _bbox_vertical_gap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    _, ay0, _, ay1 = [int(v) for v in a]
    _, by0, _, by1 = [int(v) for v in b]
    if ay0 <= by0:
        return max(0, by0 - ay1)
    return max(0, ay0 - by1)


def _resolve_detector_model_paths(explicit_path: Optional[str] = None) -> List[str]:
    candidates = [
        explicit_path or "",
        os.getenv("DETECT_YOLO_MODEL_PATH", ""),
        os.getenv("DETECT_YOLO_FALLBACK_MODEL_PATH", ""),
        "/workspace/clothing-analysis/fashion_unified.pt",
        "/workspace/Any2anyTryon/models/fashion_unified.pt",
        "/workspace/fashion_unified.pt",
        "/workspace/clothing-analysis/yolov8n-seg.pt",
        "/workspace/Any2anyTryon/models/yolov8n-seg.pt",
        "/workspace/yolov8n-seg.pt",
    ]
    resolved: List[str] = []
    seen = set()
    for path in candidates:
        norm = (path or "").strip()
        if not norm:
            continue
        abs_norm = os.path.abspath(norm)
        if abs_norm in seen:
            continue
        seen.add(abs_norm)
        if os.path.exists(abs_norm):
            resolved.append(abs_norm)
    return resolved


def _ensure_detector_loaded(model_path: str):
    global _yolo_model, _yolo_model_path
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")

    if _yolo_model is not None and _yolo_model_path == model_path:
        return _yolo_model

    with _yolo_lock:
        if _yolo_model is not None and _yolo_model_path == model_path:
            return _yolo_model
        _yolo_model = YOLO(model_path, task="segment")
        _yolo_model_path = model_path
        return _yolo_model


def _resize_mask_to_image(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    if cv2 is not None:
        return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    pil = Image.fromarray((mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    pil = pil.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def detect_garment_instances(
    image: Image.Image,
    max_items: int,
    min_conf: float,
    min_area_ratio: float,
    iou: float,
    min_area_pixels: int = 0,
    model_path: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    model_paths = _resolve_detector_model_paths(model_path)
    if not model_paths:
        return [], {"enabled": False, "reason": "model_not_found"}

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    img_area = float(max(1, h * w))
    min_area_pixels = max(0, int(min_area_pixels))
    attempts: List[Dict[str, str]] = []

    for path in model_paths:
        try:
            model = _ensure_detector_loaded(path)
        except Exception as exc:
            attempts.append({"model_path": path, "status": f"model_load_failed: {exc}"})
            continue

        try:
            result = model(rgb, verbose=False, conf=min_conf, iou=iou)[0]
        except Exception as exc:
            attempts.append({"model_path": path, "status": f"inference_failed: {exc}"})
            continue

        if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
            attempts.append({"model_path": path, "status": "no_instances"})
            continue

        mask_tensor = result.masks.data.detach().cpu().numpy()
        conf_tensor = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else None
        cls_tensor = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else None
        names = {}
        if hasattr(result, "names") and isinstance(result.names, dict):
            names = result.names
        elif hasattr(model, "names") and isinstance(model.names, dict):
            names = model.names

        strict_instances: List[Dict[str, object]] = []
        generic_fallback_instances: List[Dict[str, object]] = []
        # Common detector misspelling aliases from custom models.
        class_aliases = {
            "outwear": "outerwear",
            "long sleeved outwear": "long sleeved outerwear",
            "short sleeved outwear": "short sleeved outerwear",
        }
        for idx, raw_mask in enumerate(mask_tensor):
            mask_resized = _resize_mask_to_image(raw_mask, w, h)
            mask_bool = mask_resized > 0.5
            area = int(mask_bool.sum())
            area_ratio = float(area / img_area)
            if area_ratio < min_area_ratio or area < min_area_pixels:
                continue
            ys, xs = np.where(mask_bool)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
            conf = float(conf_tensor[idx]) if conf_tensor is not None and idx < len(conf_tensor) else 0.0
            class_id = int(cls_tensor[idx]) if cls_tensor is not None and idx < len(cls_tensor) else -1
            class_name = _normalize_label_name(str(names.get(class_id, "")))
            class_name = class_aliases.get(class_name, class_name)
            is_person = _is_person_like_class(class_name, class_id)
            is_garment_class = _is_garment_like_class(class_name)
            garment_type = _infer_garment_type_from_class_name(class_name)
            h_ratio_local = float(max(1, y1 - y0) / max(1, h))
            if garment_type is None and is_person:
                garment_type = _infer_garment_type_from_bbox(y0, y1, h)
            elif garment_type == "top" and h_ratio_local >= 0.68:
                # Guard against full-length/one-piece dresses mislabeled as "shirt".
                # Keep strict lower-coverage requirements so true tops/crop-tops are not remapped.
                spans_upper_body = y0 < h * 0.32
                reaches_lower_body = y1 > h * 0.78
                large_area = area_ratio >= 0.14
                if spans_upper_body and reaches_lower_body and large_area:
                    garment_type = "dress"
            instance_payload = {
                "mask": mask_bool,
                "bbox": (x0, y0, x1, y1),
                "area": area,
                "area_ratio": round(area_ratio, 6),
                "component_id": idx + 1,
                "detector_conf": round(conf, 4),
                "class_id": class_id,
                "class_name": class_name,
                "garment_type": garment_type,
                "is_garment_class": bool(is_garment_class),
                "is_person": bool(is_person),
                "source": "yolo",
            }

            if is_person or is_garment_class:
                strict_instances.append(instance_payload)
                continue

            if _is_generic_garment_candidate(
                (x0, y0, x1, y1),
                image_width=w,
                image_height=h,
                area_ratio=area_ratio,
            ):
                instance_payload["garment_type"] = garment_type or _infer_garment_type_from_bbox(y0, y1, h)
                instance_payload["source"] = "yolo_generic_fallback"
                generic_fallback_instances.append(instance_payload)

        strict_garments = [inst for inst in strict_instances if not bool(inst.get("is_person"))]
        person_instances = [inst for inst in strict_instances if bool(inst.get("is_person"))]

        # Start with strict outputs. If only generic outputs exist, keep generic path.
        instances = list(strict_instances) if strict_instances else list(generic_fallback_instances)

        merged_generic_count = 0
        rejected_generic_count = 0
        if strict_instances and generic_fallback_instances:
            person_mask: Optional[np.ndarray] = None
            if person_instances:
                person_mask = np.zeros((h, w), dtype=bool)
                for p in person_instances:
                    person_mask |= np.asarray(p.get("mask", np.zeros((h, w), dtype=bool))).astype(bool)
            accepted_generics: List[Dict[str, object]] = []
            for candidate in generic_fallback_instances:
                c_mask = np.asarray(candidate.get("mask", np.zeros((h, w), dtype=bool))).astype(bool)
                c_bbox = tuple(candidate.get("bbox", (0, 0, 0, 0)))
                c_area = max(1, int(candidate.get("area", 0)))
                c_area_ratio = float(candidate.get("area_ratio", c_area / img_area))
                c_conf = float(candidate.get("detector_conf", 0.0))

                if c_conf < max(0.20, min_conf * 0.70):
                    rejected_generic_count += 1
                    continue

                # Reject unusually huge generic blobs when strict garments already exist.
                if strict_garments and c_area_ratio > 0.48:
                    rejected_generic_count += 1
                    continue

                # Avoid duplicates against strict/accepted candidates.
                duplicate = False
                compare_pool = strict_garments + accepted_generics
                for kept in compare_pool:
                    if _mask_iou(c_mask, np.asarray(kept.get("mask", np.zeros((h, w), dtype=bool))).astype(bool)) >= 0.82:
                        duplicate = True
                        break
                if duplicate:
                    rejected_generic_count += 1
                    continue

                # Clothing-only spatial guard:
                # - if strict garments exist, candidate must be spatially tied to them
                # - otherwise, if person exists, candidate must overlap person region
                spatial_ok = True
                if strict_garments:
                    max_x = 0.0
                    min_gap = h
                    for kept in strict_garments:
                        k_bbox = tuple(kept.get("bbox", (0, 0, 0, 0)))
                        max_x = max(max_x, _bbox_x_overlap_ratio(c_bbox, k_bbox))
                        min_gap = min(min_gap, _bbox_vertical_gap(c_bbox, k_bbox))
                    spatial_ok = (max_x >= 0.18) and (min_gap <= int(h * 0.30))
                elif person_mask is not None and int(person_mask.sum()) > 0:
                    person_overlap = float((c_mask & person_mask).sum() / max(1, c_area))
                    spatial_ok = person_overlap >= 0.08

                if not spatial_ok:
                    rejected_generic_count += 1
                    continue

                accepted_generics.append(candidate)
                merged_generic_count += 1

            if accepted_generics:
                instances.extend(accepted_generics)
        
        # If we have specific garment boxes, suppress the redundant 'person' boxes
        # which often just wrap the same area and cause MULTI_ITEM prompts.
        has_garments = any(inst.get("is_garment_class") for inst in instances)
        if has_garments:
            instances = [inst for inst in instances if not inst.get("is_person")]

        instances.sort(key=lambda item: (int(item["area"]), float(item.get("detector_conf", 0.0))), reverse=True)
        if len(instances) > 1:
            deduped_instances: List[Dict[str, object]] = []
            for instance in instances:
                duplicate = False
                for kept in deduped_instances:
                    iou = _mask_iou(instance["mask"], kept["mask"])
                    # Aggressive deduplication: drop near-identical masks regardless of label
                    # if they overlap by more than 85%.
                    if iou >= 0.85:
                        duplicate = True
                        break
                if not duplicate:
                    deduped_instances.append(instance)
            instances = deduped_instances
        instances = instances[: max(1, max_items)]
        if instances:
            attempts.append({"model_path": path, "status": "ok"})
            return instances, {
                "enabled": True,
                "reason": "ok",
                "model_path": path,
                "attempted_models": attempts,
                "raw_instances": int(len(mask_tensor)),
                "strict_kept_instances": int(len(strict_instances)),
                "generic_fallback_kept_instances": int(len(generic_fallback_instances)),
                "generic_merged_into_output": int(merged_generic_count),
                "generic_rejected_after_merge_checks": int(rejected_generic_count),
                "kept_instances": int(len(instances)),
            }

        attempts.append({"model_path": path, "status": "filtered_no_instances"})

    return [], {
        "enabled": True,
        "reason": "no_instances",
        "attempted_models": attempts,
    }
