import base64
import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class YoloCropperConfig:
    analyze_use_human_parser: bool
    analyze_min_component_area_ratio: float
    detect_min_item_width_px: int
    detect_min_item_height_px: int
    detect_min_garment_overlap_ratio: float
    detect_use_yolo: bool
    analyze_yolo_min_conf: float
    analyze_yolo_min_area_ratio: float
    analyze_yolo_iou: float
    detect_enable_waist_split: bool
    detect_parsing_type_override_min_score: float
    analyze_multi_item_isolate_preview: bool
    analyze_multi_item_preview_keep_occluders: bool
    analyze_multi_item_isolate_min_mask_ratio: float
    analyze_multi_item_isolate_trim_padding_px: int


@dataclass
class YoloCropperDeps:
    cv2_module: Any
    binary_open_fn: Callable[[np.ndarray, int], np.ndarray]
    binary_close_fn: Callable[[np.ndarray, int], np.ndarray]
    build_soft_alpha_fn: Callable[[np.ndarray], np.ndarray]
    save_local_analyze_image_fn: Callable[..., Optional[str]]

    detect_garment_instances_fn: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]
    run_human_parsing_fn: Callable[[Image.Image], np.ndarray]
    resolve_label_ids_from_model_fn: Callable[[str], Tuple[set, set, Dict[str, Any]]]
    refine_garment_mask_fn: Callable[[np.ndarray, str], Tuple[np.ndarray, Dict[str, Any]]]
    build_parser_category_components_fn: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]
    build_parsing_category_masks_fn: Callable[[np.ndarray], Dict[str, np.ndarray]]
    max_component_overlap_ratio_fn: Callable[[List[Dict[str, Any]], np.ndarray], float]
    attempt_waist_split_components_fn: Callable[..., List[Dict[str, Any]]]
    extract_component_sections_fn: Callable[..., List[Dict[str, Any]]]
    infer_component_garment_type_from_parsing_fn: Callable[[np.ndarray, Dict[str, np.ndarray]], Tuple[str, float, Dict[str, float]]]
    postprocess_single_piece_candidates_fn: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]
    crop_with_padding_fn: Callable[..., Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]]
    estimate_occlusion_from_parsing_masks_fn: Callable[[np.ndarray, np.ndarray], Dict[str, Any]]

    run_clip_classification_fn: Callable[[Image.Image, str], Tuple[str, float]]
    normalize_item_garment_type_fn: Callable[[str], str]
    wardrobe_category_from_garment_type_fn: Callable[[str, Optional[str]], Dict[str, str]]
    upload_to_azure_fn: Callable[..., str]


def crop_from_bbox_with_padding(
    *,
    image: Image.Image,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_px: int,
    fallback_crop_fn: Callable[..., Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]],
) -> Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    pad = max(0, int(padding_px))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.width, x1 + pad)
    y1 = min(image.height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return fallback_crop_fn(image, mask, padding_px=padding_px)
    cropped_image = image.crop((x0, y0, x1, y1))
    cropped_mask = mask[y0:y1, x0:x1]
    return cropped_image, cropped_mask, (x0, y0, x1, y1)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _compute_dynamic_split_y(crop_rgb: np.ndarray, crop_mask: np.ndarray) -> int:
    h, w = crop_rgb.shape[:2]
    aspect = float(h / max(1, w))
    if aspect >= 2.8:
        base_split = 0.45
    elif aspect >= 2.2:
        base_split = 0.49
    else:
        base_split = 0.54
    best_y = int(h * base_split)
    
    if h < 80 or w < 40:
        return best_y
        
    start_y = int(h * 0.25)
    end_y = int(h * 0.65)
    
    max_gradient = -1.0
    window = max(2, int(h * 0.04))
    
    for y in range(start_y + window, end_y - window):
        top_mask = crop_mask[y-window:y, :]
        bot_mask = crop_mask[y:y+window, :]
        
        if top_mask.sum() > window * w * 0.15 and bot_mask.sum() > window * w * 0.15:
            top_pixels = crop_rgb[y-window:y, :][top_mask]
            bot_pixels = crop_rgb[y:y+window, :][bot_mask]
            
            if len(top_pixels) > 0 and len(bot_pixels) > 0:
                top_mean = np.mean(top_pixels, axis=0)
                bot_mean = np.mean(bot_pixels, axis=0)
                diff = float(np.linalg.norm(top_mean - bot_mean) / 255.0)
                if diff > max_gradient:
                    max_gradient = diff
                    best_y = y
                
    return best_y


def _compute_split_geometry(
    x0: int, y0: int, x1: int, y1: int,
    crop_rgb: Optional[np.ndarray] = None,
    crop_mask: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int]:
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    
    if crop_rgb is not None and crop_mask is not None:
        split_y = y0 + _compute_dynamic_split_y(crop_rgb, crop_mask)
    else:
        aspect = float(h / max(1, w))
        if aspect >= 2.8:
            split_ratio = 0.45
        elif aspect >= 2.2:
            split_ratio = 0.49
        else:
            split_ratio = 0.54
        split_y = int(y0 + (split_ratio * h))
        
    split_y = max(y0 + 1, min(y1 - 1, split_y))
    overlap_px = max(10, min(52, int(h * 0.07)))
    top_end = min(y1, split_y + overlap_px)
    bottom_start = max(y0, split_y - overlap_px)
    return split_y, overlap_px, top_end, bottom_start


def _likely_two_piece_from_person_region(
    *,
    source_image: Image.Image,
    person_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[bool, Dict[str, float]]:
    """
    Fast parserless heuristic:
    split only when upper/lower garment regions look materially different.
    This prevents false multi-item responses for single-piece dresses.
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    debug: Dict[str, float] = {
        "color_dist": 0.0,
        "texture_delta": 0.0,
        "skin_ratio": 0.0,
        "horizontal_skin_span": 0.0,
        "waist_x_alignment": 0.0,
        "decision": 0.0,
        "reason_code": 0.0,
    }

    if x1 <= x0 or y1 <= y0:
        debug["reason_code"] = 1.0
        return False, debug

    mask_bool = np.asarray(person_mask).astype(bool)
    crop_mask = mask_bool[y0:y1, x0:x1]
    if not crop_mask.any():
        debug["reason_code"] = 2.0
        return False, debug

    rgb = np.asarray(source_image.convert("RGB"), dtype=np.float32)
    crop_rgb = rgb[y0:y1, x0:x1, :]
    h = crop_rgb.shape[0]
    if h < 80:
        debug["reason_code"] = 3.0
        return False, debug

    split_y, _, _, _ = _compute_split_geometry(x0, y0, x1, y1, crop_rgb=crop_rgb, crop_mask=crop_mask)
    split_local = int(split_y - y0)
    band = max(10, min(48, int(h * 0.10)))

    top_y0 = max(0, split_local - band)
    top_y1 = max(0, split_local)
    bottom_y0 = min(h, split_local)
    bottom_y1 = min(h, split_local + band)
    if top_y1 <= top_y0 or bottom_y1 <= bottom_y0:
        debug["reason_code"] = 4.0
        return False, debug

    top_mask = crop_mask[top_y0:top_y1, :]
    bottom_mask = crop_mask[bottom_y0:bottom_y1, :]
    if int(top_mask.sum()) < 128 or int(bottom_mask.sum()) < 128:
        debug["reason_code"] = 5.0
        return False, debug

    top_pixels = crop_rgb[top_y0:top_y1, :][top_mask]
    bottom_pixels = crop_rgb[bottom_y0:bottom_y1, :][bottom_mask]
    if len(top_pixels) < 128 or len(bottom_pixels) < 128:
        debug["reason_code"] = 6.0
        return False, debug

    top_med = np.median(top_pixels, axis=0)
    bottom_med = np.median(bottom_pixels, axis=0)
    color_dist = float(np.linalg.norm(top_med - bottom_med) / 255.0)
    debug["color_dist"] = round(color_dist, 6)

    top_std = float(np.std(top_pixels) / 255.0)
    bottom_std = float(np.std(bottom_pixels) / 255.0)
    texture_delta = abs(top_std - bottom_std)
    debug["texture_delta"] = round(texture_delta, 6)

    # Detect exposed midriff around waist split. Keep the band center-focused
    # to avoid sleeves/arms at the edges inflating skin ratio.
    x0c = int(crop_rgb.shape[1] * 0.32)
    x1c = int(crop_rgb.shape[1] * 0.68)
    mid_rgb = crop_rgb[top_y0:bottom_y1, x0c:x1c, :]
    mid_mask = crop_mask[top_y0:bottom_y1, x0c:x1c]
    skin_ratio = 0.0
    horizontal_skin_span = 0.0
    valid_mid = int(mid_mask.sum())
    if valid_mid >= 128:
        r = mid_rgb[..., 0].astype(np.int32)
        g = mid_rgb[..., 1].astype(np.int32)
        b = mid_rgb[..., 2].astype(np.int32)
        mx = np.maximum.reduce([r, g, b])
        mn = np.minimum.reduce([r, g, b])
        skin = (
            (r > 95)
            & (g > 40)
            & (b > 20)
            & ((mx - mn) > 15)
            & (np.abs(r - g) > 15)
            & (r > g)
            & (r > b)
        )
        skin_on_mid = skin & mid_mask
        skin_ratio = float(skin_on_mid.sum() / max(1, valid_mid))

        # Estimate whether skin forms a broad horizontal severing band.
        # This helps distinguish real crop-top + bottom from side cutouts.
        col_den = np.maximum(mid_mask.sum(axis=0), 1)
        col_skin_ratio = skin_on_mid.sum(axis=0) / col_den
        skin_cols = col_skin_ratio >= 0.45
        if skin_cols.size > 0:
            mean_span = float(skin_cols.mean())
            max_run = 0
            run = 0
            for flag in skin_cols.tolist():
                if flag:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 0
            contiguous_span = float(max_run / max(1, int(skin_cols.size)))
            horizontal_skin_span = max(mean_span, contiguous_span)
    debug["skin_ratio"] = round(skin_ratio, 6)
    debug["horizontal_skin_span"] = round(horizontal_skin_span, 6)

    # Compare top/bottom silhouette alignment near split line.
    waist_x_alignment = 0.0
    try:
        top_rows = np.where(top_mask.any(axis=1))[0]
        bottom_rows = np.where(bottom_mask.any(axis=1))[0]
        if len(top_rows) >= 3 and len(bottom_rows) >= 3:
            top_l = []
            top_r = []
            for ry in top_rows:
                xs = np.where(top_mask[ry])[0]
                if xs.size > 0:
                    top_l.append(int(xs.min()))
                    top_r.append(int(xs.max()))
            bot_l = []
            bot_r = []
            for ry in bottom_rows:
                xs = np.where(bottom_mask[ry])[0]
                if xs.size > 0:
                    bot_l.append(int(xs.min()))
                    bot_r.append(int(xs.max()))
            if top_l and top_r and bot_l and bot_r:
                d_left = abs(float(np.median(top_l)) - float(np.median(bot_l)))
                d_right = abs(float(np.median(top_r)) - float(np.median(bot_r)))
                w_local = max(1.0, float(crop_mask.shape[1]))
                misalign = min(1.0, (d_left + d_right) / (2.0 * w_local))
                waist_x_alignment = 1.0 - misalign
    except Exception:
        waist_x_alignment = 0.0
    debug["waist_x_alignment"] = round(waist_x_alignment, 6)

    # Strong exposed horizontal skin band -> likely top+bottom.
    # Require a minimal appearance break to avoid false splits on
    # single dresses where skin detector over-fires.
    if (
        skin_ratio >= 0.30
        and horizontal_skin_span >= 0.75
        and (color_dist >= 0.07 or texture_delta >= 0.02)
    ):
        debug["decision"] = 1.0
        debug["reason_code"] = 10.0
        return True, debug
        
    # [NEW] Initiative 2: Perfect Tuck Tolerance (Bypass alignment geometry for high contrast)
    if color_dist >= 0.25:
        debug["decision"] = 1.0
        debug["reason_code"] = 11.0 # High contrast bypass
        return True, debug

    # Strong geometry-aware color/texture split.
    if color_dist >= 0.15 and texture_delta >= 0.04 and waist_x_alignment <= 0.88:
        debug["decision"] = 1.0
        debug["reason_code"] = 12.0
        return True, debug
    if color_dist >= 0.12 and texture_delta >= 0.05 and waist_x_alignment <= 0.82:
        debug["decision"] = 1.0
        debug["reason_code"] = 13.0
        return True, debug
    # Moderate skin + color evidence (still requires broad skin span).
    if skin_ratio >= 0.22 and horizontal_skin_span >= 0.60 and color_dist >= 0.07:
        debug["decision"] = 1.0
        debug["reason_code"] = 14.0
        return True, debug

    debug["reason_code"] = 20.0
    return False, debug


def _split_single_component_top_bottom(
    component: Dict[str, Any],
    source_image: Image.Image,
    *,
    min_area_pixels: int,
) -> List[Dict[str, Any]]:
    mask = component.get("mask")
    if mask is None:
        return []
    mask_bool = np.asarray(mask).astype(bool)
    bbox = _bbox_from_mask(mask_bool)
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    split_signal_ok, split_signal_debug = _likely_two_piece_from_person_region(
        source_image=source_image,
        person_mask=mask_bool,
        bbox=bbox,
    )
    if not split_signal_ok:
        return []

    rgb = np.asarray(source_image.convert("RGB"), dtype=np.float32)
    crop_rgb = rgb[y0:y1, x0:x1, :]
    crop_mask = mask_bool[y0:y1, x0:x1]
    
    _, _, top_end, bottom_start = _compute_split_geometry(x0, y0, x1, y1, crop_rgb=crop_rgb, crop_mask=crop_mask)

    top_mask = np.zeros_like(mask_bool, dtype=bool)
    bottom_mask = np.zeros_like(mask_bool, dtype=bool)
    top_mask[y0:top_end, x0:x1] = mask_bool[y0:top_end, x0:x1]
    bottom_mask[bottom_start:y1, x0:x1] = mask_bool[bottom_start:y1, x0:x1]

    top_area = int(top_mask.sum())
    bottom_area = int(bottom_mask.sum())
    if top_area < min_area_pixels or bottom_area < min_area_pixels:
        return []

    top_bbox = _bbox_from_mask(top_mask)
    bottom_bbox = _bbox_from_mask(bottom_mask)
    if top_bbox is None or bottom_bbox is None:
        return []

    base_conf = float(component.get("detector_conf", 0.0) or 0.0)
    base_class_id = int(component.get("class_id", -1))
    base_class_name = str(component.get("class_name", "person"))
    reason_code = int(round(float(split_signal_debug.get("reason_code", 0.0) or 0.0)))
    # Keep explicit multi-item only for strong waist-skin evidence.
    # Color/texture-only split signals are allowed to merge back into dress later.
    prevent_single_piece_merge = reason_code in {10, 13}
    split_signal_strength = "strong_multi" if prevent_single_piece_merge else "weak_multi"
    return [
        {
            "mask": top_mask,
            "bbox": top_bbox,
            "area": top_area,
            "component_id": 1,
            "detector_conf": base_conf,
            "class_id": base_class_id,
            "class_name": base_class_name,
            "garment_type": "top",
            "source": "yolo_person_split",
            "split_signal": dict(split_signal_debug),
            "split_signal_strength": split_signal_strength,
            "prevent_single_piece_merge": prevent_single_piece_merge,
        },
        {
            "mask": bottom_mask,
            "bbox": bottom_bbox,
            "area": bottom_area,
            "component_id": 2,
            "detector_conf": base_conf,
            "class_id": base_class_id,
            "class_name": base_class_name,
            "garment_type": "bottom",
            "source": "yolo_person_split",
            "split_signal": dict(split_signal_debug),
            "split_signal_strength": split_signal_strength,
            "prevent_single_piece_merge": prevent_single_piece_merge,
        },
    ]


def _infer_garment_type_from_bbox_for_parserless(
    bbox: Tuple[int, int, int, int],
    image_height: int,
) -> str:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    _ = x0, x1
    h = max(1, y1 - y0)
    y_center_ratio = float((y0 + y1) / max(1, 2 * image_height))
    h_ratio = float(h / max(1, image_height))
    if h_ratio >= 0.62:
        return "dress"
    if y_center_ratio <= 0.50:
        return "top"
    return "bottom"


def _build_foreground_fallback_component(
    *,
    source_image: Image.Image,
    min_area_pixels: int,
    cv2_module: Any,
) -> Optional[Dict[str, Any]]:
    """
    Parser-free fallback when detector returns no instances.
    Builds a single dominant foreground component from color contrast
    against border background samples.
    """
    rgb = np.asarray(source_image.convert("RGB"))
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return None

    border_h = max(4, int(h * 0.06))
    border_w = max(4, int(w * 0.06))
    border_pixels = np.concatenate(
        [
            rgb[:border_h, :, :].reshape(-1, 3),
            rgb[h - border_h :, :, :].reshape(-1, 3),
            rgb[:, :border_w, :].reshape(-1, 3),
            rgb[:, w - border_w :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_rgb = np.median(border_pixels.astype(np.float32), axis=0)
    diff = np.linalg.norm(rgb.astype(np.float32) - bg_rgb[None, None, :], axis=2)
    base_mask = diff > 26.0

    if cv2_module is not None:
        cv2 = cv2_module
        mask_u8 = (base_mask.astype(np.uint8) * 255)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return None
        best_id = None
        best_area = 0
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_id = idx
        if best_id is None or best_area < max(min_area_pixels, int(h * w * 0.05)):
            return None
        mask_bool = labels == best_id
    else:
        mask_bool = base_mask
        area = int(mask_bool.sum())
        if area < max(min_area_pixels, int(h * w * 0.05)):
            return None

    bbox = _bbox_from_mask(mask_bool)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    area = int(mask_bool.sum())

    # Keep this fallback conservative: require a substantial, body-like region.
    if area < max(min_area_pixels, int(h * w * 0.08)):
        return None
    if bh < int(h * 0.45) or bw < int(w * 0.22):
        return None

    h_ratio = float(bh / max(1, h))
    top_ratio = float(y0 / max(1, h))
    bottom_ratio = float(y1 / max(1, h))
    if h_ratio >= 0.48 and top_ratio <= 0.25 and bottom_ratio >= 0.58:
        garment_type = "dress"
    else:
        garment_type = _infer_garment_type_from_bbox_for_parserless((x0, y0, x1, y1), h)

    if garment_type == "dress":
        # If detector-free fallback found only a partial central blob,
        # widen the crop bbox to preserve full-length dress context.
        target_h = int(round(h * 0.82))
        target_w = int(round(w * 0.52))
        if bh < target_h:
            delta_h = target_h - bh
            y0 = max(0, y0 - int(delta_h * 0.30))
            y1 = min(h, y1 + int(delta_h * 0.70))
        if bw < target_w:
            delta_w = target_w - bw
            left_expand = delta_w // 2
            right_expand = delta_w - left_expand
            x0 = max(0, x0 - left_expand)
            x1 = min(w, x1 + right_expand)
        # Keep bbox valid after expansion/clamping.
        if y1 <= y0:
            y0, y1 = 0, h
        if x1 <= x0:
            x0, x1 = 0, w
        bh = max(1, y1 - y0)
        bw = max(1, x1 - x0)

    return {
        "mask": mask_bool,
        "bbox": (x0, y0, x1, y1),
        "area": area,
        "component_id": 1,
        "detector_conf": 0.1,
        "class_id": -1,
        "class_name": "foreground",
        "garment_type": garment_type,
        "source": "foreground_fallback",
    }


def build_isolated_preview_image(
    crop_image: Image.Image,
    crop_mask: np.ndarray,
    *,
    config: YoloCropperConfig,
    deps: YoloCropperDeps,
) -> Tuple[Image.Image, Dict[str, object]]:
    mask = np.asarray(crop_mask).astype(bool)
    total_pixels = int(mask.sum())
    refined = mask.copy()

    if total_pixels > 0:
        if deps.cv2_module is not None:
            cv2 = deps.cv2_module
            refined_u8 = refined.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_u8, connectivity=8)
            if num_labels > 1:
                best_id = None
                best_area = 0
                for idx in range(1, num_labels):
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    if area > best_area:
                        best_area = area
                        best_id = idx
                if best_id is not None and best_area > 0:
                    refined = labels == best_id
                else:
                    refined = refined_u8.astype(bool)
            else:
                refined = refined_u8.astype(bool)
        else:
            refined = deps.binary_open_fn(refined, 3)
            refined = deps.binary_close_fn(refined, 3)

    refined_pixels = int(refined.sum())
    min_keep_pixels = max(48, int(max(0.0, config.analyze_multi_item_isolate_min_mask_ratio) * max(total_pixels, 1)))
    fallback_to_raw_mask = refined_pixels < min_keep_pixels
    if fallback_to_raw_mask:
        refined = mask.copy()
        refined_pixels = int(refined.sum())

    alpha = deps.build_soft_alpha_fn(refined)
    if int((alpha > 0).sum()) == 0 and refined_pixels > 0:
        alpha = (refined.astype(np.uint8) * 255)

    rgb = np.asarray(crop_image.convert("RGB"))
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    isolated = Image.fromarray(rgba, mode="RGBA")

    trim_padding = max(0, int(config.analyze_multi_item_isolate_trim_padding_px))
    if refined_pixels > 0:
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0 and len(ys) > 0:
            tx0 = max(int(xs.min()) - trim_padding, 0)
            tx1 = min(int(xs.max()) + trim_padding + 1, isolated.width)
            ty0 = max(int(ys.min()) - trim_padding, 0)
            ty1 = min(int(ys.max()) + trim_padding + 1, isolated.height)
            isolated = isolated.crop((tx0, ty0, tx1, ty1))

    debug = {
        "enabled": True,
        "input_mask_pixels": total_pixels,
        "refined_mask_pixels": refined_pixels,
        "min_keep_pixels": min_keep_pixels,
        "fallback_to_raw_mask": fallback_to_raw_mask,
        "trim_padding_px": trim_padding,
    }
    return isolated, debug


def build_component_preview_asset(
    *,
    crop_image: Image.Image,
    crop_mask: np.ndarray,
    rank: int,
    config: YoloCropperConfig,
    deps: YoloCropperDeps,
    logger,
) -> Dict[str, object]:
    raw_buffer = io.BytesIO()
    crop_image.convert("RGB").save(raw_buffer, format="PNG")
    raw_bytes = raw_buffer.getvalue()
    raw_local_path = deps.save_local_analyze_image_fn(raw_bytes, prefix=f"analyze_item_{rank}", ext="png")

    isolated_bytes: Optional[bytes] = None
    isolated_local_path: Optional[str] = None
    isolation_debug: Dict[str, object] = {"enabled": False}

    if config.analyze_multi_item_isolate_preview:
        try:
            isolated_image, isolation_debug = build_isolated_preview_image(
                crop_image,
                crop_mask,
                config=config,
                deps=deps,
            )
            iso_buffer = io.BytesIO()
            isolated_image.save(iso_buffer, format="PNG")
            isolated_bytes = iso_buffer.getvalue()
            isolated_local_path = deps.save_local_analyze_image_fn(
                isolated_bytes,
                prefix=f"analyze_item_{rank}_isolated",
                ext="png",
            )
        except Exception as exc:
            isolation_debug = {"enabled": True, "error": str(exc)}
            logger.warning("Failed to build isolated preview for item %s: %s", rank, exc)

    return {
        "raw_bytes": raw_bytes,
        "raw_local_path": raw_local_path,
        "isolated_bytes": isolated_bytes,
        "isolated_local_path": isolated_local_path,
        "isolation_debug": isolation_debug,
    }


def build_yolo_item_breakdown_from_image(
    *,
    source_image: Image.Image,
    max_items: int,
    crop_padding_px: int,
    min_item_pixels: int,
    include_upload_url: bool,
    include_base64: bool,
    banned_clothing: set,
    config: YoloCropperConfig,
    deps: YoloCropperDeps,
    logger,
) -> Dict[str, object]:
    max_items = max(1, min(5, int(max_items)))
    image_area = int(source_image.width * source_image.height)
    min_component_area_ratio = max(0.0005, min(0.2, config.analyze_min_component_area_ratio))
    min_area_pixels_by_ratio = max(128, int(image_area * max(0.0005, min_component_area_ratio)))
    min_item_pixels = max(1, int(min_item_pixels), min_area_pixels_by_ratio)

    crop_padding_px = max(0, int(crop_padding_px))
    min_item_width_px = max(0, int(config.detect_min_item_width_px))
    min_item_height_px = max(0, int(config.detect_min_item_height_px))
    min_garment_overlap_ratio = max(0.0, min(1.0, float(config.detect_min_garment_overlap_ratio)))
    internal_max_candidates = max(10, max_items * 4)

    detection_source = "parser_fallback"
    detector_debug: Dict[str, object] = {"enabled": False, "reason": "disabled_by_config"}
    raw_instances: List[Dict[str, object]] = []
    components: List[Dict[str, object]] = []

    if config.detect_use_yolo:
        raw_instances, detector_debug = deps.detect_garment_instances_fn(
            image=source_image,
            max_items=internal_max_candidates,
            min_conf=max(0.01, min(0.95, config.analyze_yolo_min_conf)),
            min_area_ratio=max(0.0005, min(0.2, config.analyze_yolo_min_area_ratio)),
            iou=max(0.05, min(0.95, config.analyze_yolo_iou)),
            min_area_pixels=min_item_pixels,
        )
        components = list(raw_instances)
        if components:
            detection_source = "yolo"

    parsing: Optional[np.ndarray] = None
    garment_mask = np.zeros((source_image.height, source_image.width), dtype=bool)
    occluder_mask = np.zeros_like(garment_mask)
    parser_category_components: List[Dict[str, object]] = []
    parser_category_debug: Dict[str, object] = {
        "enabled": bool(config.analyze_use_human_parser),
        "reason": "disabled_by_config" if not config.analyze_use_human_parser else "enabled",
    }
    category_masks: Dict[str, np.ndarray] = {
        "top": np.zeros_like(garment_mask),
        "bottom": np.zeros_like(garment_mask),
        "dress": np.zeros_like(garment_mask),
        "outer": np.zeros_like(garment_mask),
    }

    if config.analyze_use_human_parser:
        parsing = deps.run_human_parsing_fn(source_image)
        garment_labels, occluder_labels, _ = deps.resolve_label_ids_from_model_fn("all")

        garment_mask = np.isin(parsing, list(garment_labels))
        garment_mask = deps.binary_open_fn(garment_mask, 3)
        garment_mask = deps.binary_close_fn(garment_mask, 5)
        garment_mask, _ = deps.refine_garment_mask_fn(garment_mask, "all")

        occluder_mask = np.isin(parsing, list(occluder_labels))
        parser_category_components, parser_category_debug = deps.build_parser_category_components_fn(
            parsing=parsing,
            min_area_pixels=min_item_pixels,
            max_items=internal_max_candidates,
        )
        category_masks = deps.build_parsing_category_masks_fn(parsing)

        if components:
            non_person_components = [c for c in components if not bool(c.get("is_person", False))]
            if non_person_components:
                detector_debug["person_filtered"] = len(components) - len(non_person_components)
                components = non_person_components

        if parser_category_components:
            yolo_class_ids = {int(c.get("class_id", -1)) for c in components} if components else set()
            yolo_has_type = any(str(c.get("garment_type", "")) in {"top", "bottom", "dress", "outer"} for c in components)
            yolo_is_generic = len(components) <= 1 and (not yolo_has_type or not yolo_class_ids or yolo_class_ids <= {0, -1})
            parser_has_multi = len(parser_category_components) > 1
            yolo_best_overlap = deps.max_component_overlap_ratio_fn(components, garment_mask) if components else 0.0
            parser_best_overlap = deps.max_component_overlap_ratio_fn(parser_category_components, garment_mask)
            yolo_low_overlap = (
                bool(components)
                and yolo_best_overlap < min_garment_overlap_ratio
                and parser_best_overlap >= max(0.08, min_garment_overlap_ratio * 0.5)
            )
            if detection_source == "parser_fallback" or (parser_has_multi and yolo_is_generic) or yolo_low_overlap:
                components = parser_category_components
                detection_source = "parser_category_overlap_fallback" if yolo_low_overlap else "parser_category"
                parser_category_debug["yolo_overlap_fallback"] = {
                    "applied": bool(yolo_low_overlap),
                    "yolo_best_overlap": round(float(yolo_best_overlap), 4),
                    "parser_best_overlap": round(float(parser_best_overlap), 4),
                    "threshold": round(float(min_garment_overlap_ratio), 4),
                }

        if (
            config.detect_enable_waist_split
            and len(components) == 1
            and detection_source in {"yolo", "parser_fallback"}
            and int(components[0].get("class_id", -1)) in {0, -1}
        ):
            primary_mask = components[0]["mask"].astype(bool)
            split_components = deps.attempt_waist_split_components_fn(primary_mask, min_area_pixels=min_item_pixels)
            if len(split_components) >= 2:
                components = split_components
                detection_source = "waist_split"

        if not components:
            components = deps.extract_component_sections_fn(
                garment_mask=garment_mask,
                max_items=internal_max_candidates,
                min_area_pixels=min_item_pixels,
            )
    else:
        non_person_components = [c for c in components if not bool(c.get("is_person", False))]
        person_components = [c for c in components if bool(c.get("is_person", False))]
        if non_person_components:
            components = non_person_components
            detection_source = "yolo_non_person"
        elif person_components:
            components = person_components
            if config.detect_enable_waist_split and len(components) == 1:
                primary_mask = np.asarray(components[0].get("mask")).astype(bool)
                split_components = deps.attempt_waist_split_components_fn(primary_mask, min_area_pixels=min_item_pixels)
                if len(split_components) >= 2:
                    components = split_components
                    detection_source = "waist_split"
                else:
                    fallback_split = _split_single_component_top_bottom(
                        components[0],
                        source_image=source_image,
                        min_area_pixels=min_item_pixels,
                    )
                    if len(fallback_split) == 2:
                        components = fallback_split
                        detection_source = "person_top_bottom_split"
            elif len(components) > 1:
                components.sort(
                    key=lambda item: (int(item.get("area", 0)), float(item.get("detector_conf", 0.0))),
                    reverse=True,
                )
                components = components[:1]
                fallback_split = _split_single_component_top_bottom(
                    components[0],
                    source_image=source_image,
                    min_area_pixels=min_item_pixels,
                )
                if len(fallback_split) == 2:
                    components = fallback_split
                    detection_source = "person_top_bottom_split"
                else:
                    detection_source = "yolo_person"

        if not components:
            fallback_component = _build_foreground_fallback_component(
                source_image=source_image,
                min_area_pixels=min_item_pixels,
                cv2_module=deps.cv2_module,
            )
            if fallback_component is not None:
                components = [fallback_component]
                detection_source = "foreground_fallback"
                cmask = np.asarray(fallback_component.get("mask", np.zeros_like(garment_mask))).astype(bool)
                garment_mask = cmask.copy()
                ctype = str(fallback_component.get("garment_type", "")).lower().strip()
                if ctype in category_masks:
                    category_masks[ctype] |= cmask

        if components:
            aggregate_mask = np.zeros_like(garment_mask)
            for component in components:
                cmask = np.asarray(component.get("mask", np.zeros_like(garment_mask))).astype(bool)
                aggregate_mask |= cmask
                ctype = str(component.get("garment_type", "")).lower().strip()
                if ctype in category_masks:
                    category_masks[ctype] |= cmask
            garment_mask = aggregate_mask

    item_breakdown: List[Dict[str, object]] = []
    internal_image_url_by_component_id: Dict[int, str] = {}
    internal_image_url_by_rank: Dict[int, str] = {}
    internal_isolated_image_url_by_component_id: Dict[int, str] = {}
    internal_isolated_image_url_by_rank: Dict[int, str] = {}
    size_filtered_components: List[Dict[str, object]] = []
    preview_assets: List[Dict[str, object]] = []

    for component in components:
        bbox = component.get("bbox", (0, 0, 0, 0))
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        width = max(0, x1 - x0)
        height = max(0, y1 - y0)
        component_mask = component["mask"].astype(bool)
        area = int(component_mask.sum())

        if area < min_item_pixels:
            continue
        if width < min_item_width_px or height < min_item_height_px:
            continue

        overlap_pixels = int((component_mask & garment_mask).sum())
        overlap_ratio = float(overlap_pixels / max(area, 1))
        if overlap_ratio < min_garment_overlap_ratio:
            continue

        component["area"] = area
        component["garment_overlap_pixels"] = overlap_pixels
        component["garment_overlap_ratio"] = round(overlap_ratio, 4)

        current_type = str(component.get("garment_type", "")).lower().strip()
        has_valid_current = current_type in {"top", "bottom", "dress", "outer"}
        if parsing is not None:
            inferred_type, inferred_score, overlap_by_type = deps.infer_component_garment_type_from_parsing_fn(
                component_mask=component_mask,
                category_masks=category_masks,
            )
            has_valid_inferred = inferred_type in {"top", "bottom", "dress", "outer"}

            if has_valid_inferred and inferred_score >= max(0.0, min(1.0, config.detect_parsing_type_override_min_score)):
                if (not has_valid_current) or (current_type != inferred_type):
                    component["garment_type"] = inferred_type
                    component["garment_type_source"] = "parsing_override" if has_valid_current else "parsing_inferred"
            elif not has_valid_current:
                component["garment_type"] = inferred_type if has_valid_inferred else "all"
                component["garment_type_source"] = "parsing_fallback"

            component["garment_type_score"] = round(inferred_score, 4)
            component["garment_type_overlap"] = {k: round(v, 4) for k, v in overlap_by_type.items()}
        else:
            if not has_valid_current:
                bbox = tuple(component.get("bbox", (0, 0, source_image.width, source_image.height)))
                inferred_type = _infer_garment_type_from_bbox_for_parserless(bbox, source_image.height)
                component["garment_type"] = inferred_type
                component["garment_type_source"] = "bbox_parserless"
            component["garment_type_score"] = round(float(component.get("detector_conf", 0.0) or 0.0), 4)
            component["garment_type_overlap"] = {}
        size_filtered_components.append(component)

    size_filtered_components.sort(
        key=lambda item: (int(item.get("area", 0)), float(item.get("detector_conf", 0.0))),
        reverse=True,
    )
    size_filtered_components, candidate_postprocess_debug = deps.postprocess_single_piece_candidates_fn(
        size_filtered_components,
        image_height=source_image.height,
        min_item_pixels=min_item_pixels,
        parsing=parsing,
    )
    selected_components = size_filtered_components[:max_items]

    for rank, inst in enumerate(selected_components, start=1):
        component_mask = inst["mask"].astype(bool)
        component_padding_px = int(crop_padding_px)
        # Foreground fallback often under-crops long dresses. Use an adaptive
        # larger padding so the downstream VTON step receives more full-garment
        # context and preserves shape/detail better.
        if str(inst.get("source", detection_source)) == "foreground_fallback":
            component_padding_px = max(
                component_padding_px,
                int(max(16, round(0.06 * max(source_image.width, source_image.height)))),
            )
        crop_image, crop_mask, crop_bbox = crop_from_bbox_with_padding(
            image=source_image,
            mask=component_mask,
            bbox=tuple(inst.get("bbox", (0, 0, source_image.width, source_image.height))),
            padding_px=component_padding_px,
            fallback_crop_fn=deps.crop_with_padding_fn,
        )
        x0, y0, x1, y1 = crop_bbox
        crop_occluder = occluder_mask[y0:y1, x0:x1]
        crop_occlusion_debug = deps.estimate_occlusion_from_parsing_masks_fn(crop_mask, crop_occluder)
        occlusion_ratio = float(crop_occlusion_debug.get("proxy_ratio", 0.0))

        garment_type = deps.normalize_item_garment_type_fn(str(inst.get("garment_type", "all")))
        style_name, clip_conf = deps.run_clip_classification_fn(crop_image, garment_type=garment_type)
        category_meta = deps.wardrobe_category_from_garment_type_fn(garment_type, style=style_name)

        detector_conf = float(inst.get("detector_conf", 0.0) or 0.0)
        combined_score = (detector_conf * 0.3) + (clip_conf * 0.7)
        is_safe = not any(b in style_name.lower() for b in banned_clothing)
        area = int(inst.get("area", component_mask.sum()))

        payload: Dict[str, object] = {
            "rank": rank,
            "component_id": int(inst.get("component_id", rank)),
            "garment_type": garment_type,
            "type": garment_type,
            "style": category_meta["style"],
            "primary_category_key": category_meta["primary_category_key"],
            "category_key": category_meta["category_key"],
            "confidence": round(combined_score, 4),
            "detector_conf": round(detector_conf, 4),
            "clip_confidence": round(clip_conf, 4),
            "combined_score": round(combined_score, 4),
            "occlusion_ratio": round(occlusion_ratio, 4),
            "is_safe": is_safe,
            "source": str(inst.get("source", detection_source)),
            "bbox_original": {
                "x0": int(inst.get("bbox", (0, 0, 0, 0))[0]),
                "y0": int(inst.get("bbox", (0, 0, 0, 0))[1]),
                "x1": int(inst.get("bbox", (0, 0, 0, 0))[2]),
                "y1": int(inst.get("bbox", (0, 0, 0, 0))[3]),
            },
            "bbox_crop": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)},
            "crop_size": {"width": int(crop_image.width), "height": int(crop_image.height)},
            "garment_pixels": area,
            "class_id": int(inst.get("class_id", -1)),
            "class_name": str(inst.get("class_name", "")),
        }
        if "garment_type_score" in inst:
            payload["garment_type_score"] = float(inst.get("garment_type_score", 0.0))
        if "garment_type_overlap" in inst:
            payload["garment_type_overlap"] = dict(inst.get("garment_type_overlap", {}))
        if "garment_overlap_pixels" in inst:
            payload["garment_overlap_pixels"] = int(inst.get("garment_overlap_pixels", 0))
        if "garment_overlap_ratio" in inst:
            payload["garment_overlap_ratio"] = float(inst.get("garment_overlap_ratio", 0.0))
        if "split_signal" in inst:
            payload["split_signal"] = dict(inst.get("split_signal", {}))
        if "split_signal_strength" in inst:
            payload["split_signal_strength"] = str(inst.get("split_signal_strength"))

        preview_asset = build_component_preview_asset(
            crop_image=crop_image,
            crop_mask=crop_mask,
            rank=rank,
            config=config,
            deps=deps,
            logger=logger,
        )
        isolation_debug = dict(preview_asset.get("isolation_debug", {}))

        payload["preview_debug"] = {"isolation": isolation_debug}
        preview_assets.append(
            {
                "rank": rank,
                "component_id": int(payload["component_id"]),
                "raw_bytes": preview_asset.get("raw_bytes"),
                "raw_local_path": preview_asset.get("raw_local_path"),
                "isolated_bytes": preview_asset.get("isolated_bytes"),
                "isolated_local_path": preview_asset.get("isolated_local_path"),
            }
        )
        item_breakdown.append(payload)

    use_isolated_preview = (
        config.analyze_multi_item_isolate_preview
        and len(item_breakdown) > 1
        and not config.analyze_multi_item_preview_keep_occluders
    )

    binary_parts: List[Dict[str, object]] = []
    internal_image_url_by_component_id.clear()
    internal_image_url_by_rank.clear()
    internal_isolated_image_url_by_component_id.clear()
    internal_isolated_image_url_by_rank.clear()

    for payload, assets in zip(item_breakdown, preview_assets):
        rank = int(payload.get("rank", assets.get("rank", 0)))
        component_id = int(payload.get("component_id", assets.get("component_id", 0)))

        chosen_isolated = bool(use_isolated_preview and assets.get("isolated_bytes") is not None)
        chosen_bytes = bytes(assets.get("isolated_bytes")) if chosen_isolated else bytes(assets.get("raw_bytes"))
        raw_bytes = bytes(assets.get("raw_bytes"))
        raw_local_path = str(assets.get("raw_local_path") or "").strip()
        isolated_bytes_raw = assets.get("isolated_bytes")
        isolated_bytes = bytes(isolated_bytes_raw) if isolated_bytes_raw is not None else b""
        isolated_local_path_raw = str(assets.get("isolated_local_path") or "").strip()

        chosen_local_path = str(assets.get("isolated_local_path") or "") if chosen_isolated else str(assets.get("raw_local_path") or "")
        chosen_local_path = chosen_local_path or None

        payload["preview_mode"] = "isolated" if chosen_isolated else "crop"
        if chosen_local_path:
            payload["image_local_path"] = chosen_local_path
            payload["image_local_file_url"] = f"file://{chosen_local_path}"
        else:
            payload.pop("image_local_path", None)
            payload.pop("image_local_file_url", None)

        if include_base64:
            payload["image_base64_png"] = base64.b64encode(chosen_bytes).decode("utf-8")
        else:
            payload.pop("image_base64_png", None)

        image_url: Optional[str] = None
        if include_upload_url:
            try:
                image_url = deps.upload_to_azure_fn(chosen_bytes, extension="png", content_type="image/png")
            except Exception as exc:
                logger.warning("Failed to upload preview for component %s: %s", component_id, exc)
        elif chosen_local_path:
            image_url = f"file://{chosen_local_path}"

        if image_url:
            payload["image_url"] = image_url
        else:
            payload.pop("image_url", None)

        # Internal extraction path should always use the raw crop to avoid
        # over-trimmed isolated previews degrading bottom/top reconstruction.
        internal_image_url: Optional[str] = None
        if raw_local_path:
            internal_image_url = f"file://{raw_local_path}"
        elif image_url and not chosen_isolated:
            internal_image_url = image_url
        elif include_upload_url and raw_bytes:
            try:
                internal_image_url = deps.upload_to_azure_fn(raw_bytes, extension="png", content_type="image/png")
            except Exception as exc:
                logger.warning("Failed to upload raw internal preview for component %s: %s", component_id, exc)
                internal_image_url = image_url
        else:
            internal_image_url = image_url

        if internal_image_url:
            internal_image_url_by_component_id[component_id] = internal_image_url
            internal_image_url_by_rank[rank] = internal_image_url

        isolated_internal_image_url: Optional[str] = None
        if isolated_local_path_raw:
            isolated_internal_image_url = f"file://{isolated_local_path_raw}"
        elif include_upload_url and isolated_bytes:
            try:
                isolated_internal_image_url = deps.upload_to_azure_fn(
                    isolated_bytes,
                    extension="png",
                    content_type="image/png",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to upload isolated internal preview for component %s: %s",
                    component_id,
                    exc,
                )
        if isolated_internal_image_url:
            internal_isolated_image_url_by_component_id[component_id] = isolated_internal_image_url
            internal_isolated_image_url_by_rank[rank] = isolated_internal_image_url

        binary_parts.append(
            {
                "name": f"item_{rank}",
                "filename": f"item_{rank}.png",
                "content_type": "image/png",
                "bytes": chosen_bytes,
            }
        )

    return {
        "raw_instances_count": len(raw_instances),
        "candidate_instances_count": len(components),
        "valid_instances_count": len(size_filtered_components),
        "parser_category_count": len(parser_category_components),
        "detection_source": detection_source,
        "detector_debug": detector_debug,
        "parser_category_debug": parser_category_debug,
        "candidate_postprocess_debug": candidate_postprocess_debug,
        "item_breakdown": item_breakdown,
        "binary_parts": binary_parts,
        "raw_debug": detector_debug,
        "size_filters": {
            "min_item_pixels": min_item_pixels,
            "min_item_width_px": min_item_width_px,
            "min_item_height_px": min_item_height_px,
            "min_garment_overlap_ratio": min_garment_overlap_ratio,
        },
        "internal_image_url_by_component_id": internal_image_url_by_component_id,
        "internal_image_url_by_rank": internal_image_url_by_rank,
        "internal_isolated_image_url_by_component_id": internal_isolated_image_url_by_component_id,
        "internal_isolated_image_url_by_rank": internal_isolated_image_url_by_rank,
    }
