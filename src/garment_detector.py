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
            is_person = _is_person_like_class(class_name, class_id)
            is_garment_class = _is_garment_like_class(class_name)
            garment_type = _infer_garment_type_from_class_name(class_name)
            h_ratio_local = float(max(1, y1 - y0) / max(1, h))
            if garment_type is None and is_person:
                garment_type = _infer_garment_type_from_bbox(y0, y1, h)
            elif garment_type == "top" and h_ratio_local >= 0.60:
                # Override YOLO classification if the geometry explicitly screams dress
                # but YOLO mistakenly predicted "shirt" (e.g. for long-sleeved dresses)
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

        instances = strict_instances if strict_instances else generic_fallback_instances
        instances.sort(key=lambda item: (int(item["area"]), float(item.get("detector_conf", 0.0))), reverse=True)
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
                "kept_instances": int(len(instances)),
            }

        attempts.append({"model_path": path, "status": "filtered_no_instances"})

    return [], {
        "enabled": True,
        "reason": "no_instances",
        "attempted_models": attempts,
    }
