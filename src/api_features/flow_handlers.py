from __future__ import annotations

# Flow handlers extracted from runtime main.py
# Execute against main module context passed in `ctx`.

def _normalize_selection_type(selected_type) -> str:
    if selected_type is None:
        return ""
    raw = str(selected_type).strip().lower()
    if not raw:
        return ""
    if raw in {"null", "none", "undefined", "(null)", "nil"}:
        return ""
    if raw in {"auto", "any"}:
        return "all"
    if raw in {"top", "bottom", "dress", "outer", "all"}:
        return raw
    return ""


def _item_matches_selected_type(item: Dict[str, object], selected_type: str) -> bool:
    if not selected_type or selected_type == "all":
        return True
    item_type = _normalize_item_garment_type(str(item.get("garment_type", item.get("type", "all"))))
    return bool(_garment_type_matches_requested(selected_type, item_type))


def _select_best_item_by_type(items: List[Dict[str, object]], selected_type: str) -> Optional[Dict[str, object]]:
    if not items:
        return None
    filtered = [item for item in items if _item_matches_selected_type(item, selected_type)]
    if not filtered:
        return None
    filtered.sort(
        key=lambda item: (
            float(item.get("combined_score", item.get("confidence", 0.0)) or 0.0),
            int(item.get("garment_pixels", 0) or 0),
            -float(item.get("occlusion_ratio", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return filtered[0]


def _crop_from_bbox_with_padding(
    *,
    image: Image.Image,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_px: int,
) -> Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    pad = max(0, int(padding_px))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.width, x1 + pad)
    y1 = min(image.height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return crop_with_padding(image, mask, padding_px=padding_px)
    cropped_image = image.crop((x0, y0, x1, y1))
    cropped_mask = mask[y0:y1, x0:x1]
    return cropped_image, cropped_mask, (x0, y0, x1, y1)


def _split_crop_for_forced_type(image: Image.Image, forced_type: str) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    aspect = float(h / max(1, w))
    if aspect >= 2.8:
        split_ratio = 0.45
    elif aspect >= 2.2:
        split_ratio = 0.49
    else:
        split_ratio = 0.54
    split_y = max(1, min(h - 1, int(h * split_ratio)))
    top_overlap = max(6, min(28, int(h * 0.03)))
    bottom_overlap = max(10, min(52, int(h * 0.07)))
    top_end = min(h, split_y + top_overlap)
    bottom_start = max(0, split_y - bottom_overlap)

    kind = (forced_type or "").strip().lower()
    if kind == "top":
        return image.crop((0, 0, w, top_end))
    if kind == "bottom":
        return image.crop((0, bottom_start, w, h))
    return image


def _infer_style_from_detector_class_name(class_name: str, garment_type: str) -> Optional[str]:
    name = str(class_name or "").strip().lower()
    gtype = str(garment_type or "").strip().lower()
    if not name:
        return None

    if gtype == "bottom":
        if "jean" in name or "denim" in name:
            return "jeans"
        if "track" in name or "jogger" in name:
            return "track pants"
        if "legging" in name or "tights" in name:
            return "leggings"
        if "short" in name:
            return "shorts"
        if "skirt" in name:
            return "skirt"
        if any(token in name for token in ("trouser", "pant", "slack")):
            return "trousers"
    return None


async def handle_detect_garments(request, ctx):
    globals().update(ctx)
    try:
        timings: Dict[str, float] = {}
        overall_start = time.perf_counter()

        max_items = max(1, min(5, int(request.max_items)))
        min_area_ratio = max(0.0005, min(0.2, float(request.min_component_area_ratio)))
        detector_min_area_ratio = max(
            0.0005,
            min(
                0.2,
                float(request.detector_min_area_ratio if request.detector_min_area_ratio is not None else DETECT_YOLO_MIN_AREA_RATIO),
            ),
        )
        use_yolo = DETECT_USE_YOLO if request.use_yolo is None else bool(request.use_yolo)
        yolo_min_conf = max(0.01, min(0.95, float(request.yolo_min_conf if request.yolo_min_conf is not None else DETECT_YOLO_MIN_CONF)))
        yolo_iou = max(0.05, min(0.95, float(request.yolo_iou if request.yolo_iou is not None else DETECT_YOLO_IOU)))
        min_item_width_px = max(0, int(request.min_item_width_px if request.min_item_width_px is not None else DETECT_MIN_ITEM_WIDTH_PX))
        min_item_height_px = max(0, int(request.min_item_height_px if request.min_item_height_px is not None else DETECT_MIN_ITEM_HEIGHT_PX))
        min_garment_overlap_ratio = max(
            0.0,
            min(
                1.0,
                float(
                    request.min_garment_overlap_ratio
                    if request.min_garment_overlap_ratio is not None
                    else DETECT_MIN_GARMENT_OVERLAP_RATIO
                ),
            ),
        )
        enable_waist_split = DETECT_ENABLE_WAIST_SPLIT if request.enable_waist_split is None else bool(request.enable_waist_split)
        crop_padding = max(0, int(request.crop_padding_px))

        step_start = time.perf_counter()
        source_image, download_bytes = _download_image(request.image_url)
        timings["download_s"] = round(time.perf_counter() - step_start, 4)
        timings["download_mb"] = round(download_bytes / (1024 * 1024), 4)
        image_area = int(source_image.width * source_image.height)
        min_area_pixels_by_ratio = max(128, int(image_area * min_area_ratio))
        min_item_pixels = max(
            int(request.min_item_pixels if request.min_item_pixels is not None else DETECT_MIN_ITEM_PIXELS),
            min_area_pixels_by_ratio,
        )
        internal_max_candidates = max(10, max_items * 4)

        detection_source = "parser"
        detector_debug: Dict[str, object] = {"enabled": False, "reason": "disabled_by_request"}
        components: List[Dict[str, object]] = []
        if use_yolo:
            step_start = time.perf_counter()
            components, detector_debug = detect_garment_instances(
                image=source_image,
                max_items=internal_max_candidates,
                min_conf=yolo_min_conf,
                min_area_ratio=detector_min_area_ratio,
                iou=yolo_iou,
                min_area_pixels=min_item_pixels,
                model_path=request.yolo_model_path,
            )
            timings["detector_s"] = round(time.perf_counter() - step_start, 4)
            if components:
                detection_source = "yolo"
            else:
                detection_source = "parser_fallback"

        step_start = time.perf_counter()
        parsing = _run_human_parsing(source_image)
        timings["parsing_s"] = round(time.perf_counter() - step_start, 4)

        step_start = time.perf_counter()
        garment_labels, occluder_labels, label_debug = _resolve_label_ids_from_model("all")
        garment_mask = np.isin(parsing, list(garment_labels))
        garment_mask = _binary_open(garment_mask, 3)
        garment_mask = _binary_close(garment_mask, 5)
        garment_mask, refinement_debug = _refine_garment_mask(garment_mask, "all")
        occluder_mask = np.isin(parsing, list(occluder_labels))
        timings["mask_build_s"] = round(time.perf_counter() - step_start, 4)
        step_start = time.perf_counter()
        parser_category_components, parser_category_debug = _build_parser_category_components(
            parsing=parsing,
            min_area_pixels=min_item_pixels,
            max_items=internal_max_candidates,
        )
        timings["parser_category_s"] = round(time.perf_counter() - step_start, 4)
        step_start = time.perf_counter()
        category_masks = _build_parsing_category_masks(parsing)
        timings["category_masks_s"] = round(time.perf_counter() - step_start, 4)

        if components:
            non_person_components = [c for c in components if not bool(c.get("is_person", False))]
            if non_person_components:
                detector_debug["person_filtered"] = len(components) - len(non_person_components)
                components = non_person_components

        if not garment_mask.any() and not components:
            return {
                "status": 200,
                "message": "No garment found in the image",
                "data": {
                    "items": [],
                    "total_detected": 0,
                    "detection_source": detection_source,
                    "detector_debug": detector_debug,
                    "parser_category_debug": parser_category_debug,
                    "timings": {**timings, "total_s": round(time.perf_counter() - overall_start, 4)},
                },
            }

        # If YOLO returns only a broad person-like detection, parser categories
        # usually split top/bottom more accurately for wardrobe selection.
        if parser_category_components:
            yolo_class_ids = {int(c.get("class_id", -1)) for c in components} if components else set()
            yolo_has_type = any(str(c.get("garment_type", "")) in {"top", "bottom", "dress", "outer"} for c in components)
            yolo_is_generic = len(components) <= 1 and (not yolo_has_type or not yolo_class_ids or yolo_class_ids <= {0, -1})
            parser_has_multi = len(parser_category_components) > 1
            yolo_best_overlap = _max_component_overlap_ratio(components, garment_mask) if components else 0.0
            parser_best_overlap = _max_component_overlap_ratio(parser_category_components, garment_mask)
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
            enable_waist_split
            and len(components) == 1
            and detection_source in {"yolo", "parser_fallback"}
            and int(components[0].get("class_id", -1)) in {0, -1}
        ):
            primary_mask = components[0]["mask"].astype(bool)
            split_components = _attempt_waist_split_components(primary_mask, min_area_pixels=min_item_pixels)
            if len(split_components) >= 2:
                components = split_components
                detection_source = "waist_split"

        if request.force_preview_split_single and len(components) == 1:
            base_mask = components[0]["mask"].astype(bool)
            split_components = _attempt_waist_split_components(base_mask, min_area_pixels=min_item_pixels)
            if len(split_components) < 2:
                split_components = _force_split_single_component(base_mask, min_area_pixels=min_item_pixels)
            if len(split_components) >= 2:
                components = split_components
                detection_source = "forced_split"

        if not components:
            components = extract_component_sections(
                garment_mask=garment_mask,
                max_items=internal_max_candidates,
                min_area_pixels=min_item_pixels,
            )

        size_filtered_components: List[Dict[str, object]] = []
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
            inferred_type, inferred_score, overlap_by_type = _infer_component_garment_type_from_parsing(
                component_mask=component_mask,
                category_masks=category_masks,
            )
            current_type = str(component.get("garment_type", "")).lower()
            if current_type not in {"top", "bottom", "dress", "outer"}:
                component["garment_type"] = inferred_type
            component["garment_type_score"] = round(inferred_score, 4)
            component["garment_type_overlap"] = {k: round(v, 4) for k, v in overlap_by_type.items()}
            size_filtered_components.append(component)

        size_filtered_components.sort(
            key=lambda item: (int(item.get("area", 0)), float(item.get("detector_conf", 0.0))),
            reverse=True,
        )
        size_filtered_components, candidate_postprocess_debug = _postprocess_single_piece_candidates(
            size_filtered_components,
            image_height=source_image.height,
            min_item_pixels=min_item_pixels,
            parsing=parsing,
        )
        components = size_filtered_components[:max_items]

        if not components:
            timings["total_s"] = round(time.perf_counter() - overall_start, 4)
            return {
                "status": 200,
                "message": "No garment items passed minimum size constraints",
                "data": {
                    "items": [],
                    "total_detected": 0,
                    "max_items": max_items,
                    "detection_source": detection_source,
                    "detector_debug": detector_debug,
                    "parser_category_debug": parser_category_debug,
                    "candidate_postprocess_debug": candidate_postprocess_debug,
                    "size_filters": {
                        "min_item_pixels": min_item_pixels,
                        "min_item_width_px": min_item_width_px,
                        "min_item_height_px": min_item_height_px,
                        "min_garment_overlap_ratio": min_garment_overlap_ratio,
                        "candidate_count": len(size_filtered_components),
                    },
                    "timings": timings,
                },
            }

        items: List[Dict[str, object]] = []
        step_start = time.perf_counter()
        for idx, component in enumerate(components, start=1):
            component_mask = component["mask"].astype(bool)
            crop_image, crop_mask, crop_bbox = _crop_from_bbox_with_padding(
                image=source_image,
                mask=component_mask,
                bbox=tuple(component.get("bbox", (0, 0, source_image.width, source_image.height))),
                padding_px=crop_padding,
            )
            x0, y0, x1, y1 = crop_bbox
            crop_occluder = occluder_mask[y0:y1, x0:x1]
            crop_occlusion_debug = _estimate_occlusion_from_parsing_masks(crop_mask, crop_occluder)
            garment_pixels = int(crop_occlusion_debug.get("garment_pixels", int(crop_mask.sum())))
            occluded_pixels = int(crop_occlusion_debug.get("occluded_pixels", 0))
            occlusion_ratio = float(crop_occlusion_debug.get("proxy_ratio", 0.0))

            if request.isolate_garment:
                rgb = np.asarray(crop_image.convert("RGB"))
                alpha = _build_soft_alpha(crop_mask)[:, :, None]
                isolated_rgba = np.concatenate([rgb, alpha], axis=2)
                output_image = Image.fromarray(isolated_rgba, mode="RGBA")
                output_mode = "isolated"
            else:
                output_image = crop_image.convert("RGB")
                output_mode = "crop"

            item_payload: Dict[str, object] = {
                "rank": idx,
                "component_id": int(component["component_id"]),
                "bbox_original": {
                    "x0": int(component["bbox"][0]),
                    "y0": int(component["bbox"][1]),
                    "x1": int(component["bbox"][2]),
                    "y1": int(component["bbox"][3]),
                },
                "bbox_crop": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "crop_size": {"width": crop_image.width, "height": crop_image.height},
                "garment_pixels": garment_pixels,
                "occluded_pixels": occluded_pixels,
                "occlusion_ratio": occlusion_ratio,
                "source": component.get("source", "parser"),
                "output_mode": output_mode,
                "garment_overlap_pixels": int(component.get("garment_overlap_pixels", garment_pixels)),
                "garment_overlap_ratio": float(component.get("garment_overlap_ratio", 1.0)),
            }
            if "garment_type" in component:
                item_payload["garment_type"] = str(component["garment_type"])
                item_payload["type"] = str(component["garment_type"])
            if "detector_conf" in component:
                item_payload["detector_conf"] = float(component["detector_conf"])
            if "class_id" in component:
                item_payload["class_id"] = int(component["class_id"])
            if "class_name" in component:
                item_payload["class_name"] = str(component["class_name"])
            if "garment_type_score" in component:
                item_payload["garment_type_score"] = float(component["garment_type_score"])
            if "garment_type_overlap" in component:
                item_payload["garment_type_overlap"] = dict(component["garment_type_overlap"])

            if request.include_base64:
                item_payload["image_base64_png"] = image_to_base64_png(output_image)

            if request.include_upload_url:
                buf = io.BytesIO()
                output_image.save(buf, format="PNG")
                item_payload["image_url"] = upload_to_azure(
                    buf.getvalue(),
                    extension="png",
                    content_type="image/png",
                )

            items.append(item_payload)
        timings["crop_build_s"] = round(time.perf_counter() - step_start, 4)
        timings["total_s"] = round(time.perf_counter() - overall_start, 4)

        return {
            "status": 200,
            "message": f"Detected {len(items)} garment item(s)",
            "data": {
                "total_detected": len(items),
                "max_items": max_items,
                "items": items,
                "label_debug": label_debug,
                "refinement_debug": refinement_debug,
                "detection_source": detection_source,
                "detector_debug": detector_debug,
                "parser_category_debug": parser_category_debug,
                "candidate_postprocess_debug": candidate_postprocess_debug,
                "size_filters": {
                    "min_item_pixels": min_item_pixels,
                    "min_item_width_px": min_item_width_px,
                    "min_item_height_px": min_item_height_px,
                    "min_garment_overlap_ratio": min_garment_overlap_ratio,
                },
                "timings": timings,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("detect_garments failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

async def handle_extract_cloth(request, ctx):
    globals().update(ctx)
    try:
        timings = {}
        overall_start = time.perf_counter()
        prefilter_enabled = (
            EXTRACT_OCCLUSION_PREFILTER_ENABLED
            if request.use_occlusion_prefilter is None
            else bool(request.use_occlusion_prefilter)
        )
        fallback_enabled = (
            EXTRACT_VTON_FALLBACK_ENABLED
            if request.enable_vton_fallback is None
            else bool(request.enable_vton_fallback)
        )
        proxy_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    request.occlusion_proxy_threshold
                    if request.occlusion_proxy_threshold is not None
                    else EXTRACT_OCCLUSION_PROXY_THRESHOLD
                ),
            ),
        )
        prefilter_debug: Dict[str, object] = {"enabled": prefilter_enabled, "status": "disabled"}
        fallback_debug: Dict[str, object] = {
            "enabled": fallback_enabled,
            "used": False,
            "triggered": bool(request.force_vton_fallback),
            "trigger_reason": "force_vton_fallback" if request.force_vton_fallback else None,
            "occlusion_proxy_threshold": proxy_threshold,
        }

        step_start = time.perf_counter()
        source_image, download_bytes = _download_image(request.image_url)
        timings["download_s"] = round(time.perf_counter() - step_start, 4)
        timings["download_mb"] = round(download_bytes / (1024 * 1024), 4)

        cloth_labels = GARMENT_LABEL_MAP.get(request.garment_type)
        if not cloth_labels:
            raise HTTPException(status_code=400, detail=f"Unsupported garment_type: {request.garment_type}")

        if prefilter_enabled:
            step_start = time.perf_counter()
            min_prefilter_pixels = max(
                96,
                int(source_image.width * source_image.height * max(0.0005, EXTRACT_PREFILTER_YOLO_MIN_AREA_RATIO)),
            )
            prefilter_debug, _ = _run_occlusion_prefilter(
                source_image=source_image,
                garment_type=request.garment_type,
                min_area_pixels=min_prefilter_pixels,
            )
            timings["occlusion_prefilter_s"] = round(time.perf_counter() - step_start, 4)

            proxy_ratio = float(prefilter_debug.get("proxy_occlusion_ratio", 0.0))
            threshold_hit = bool(prefilter_debug.get("status") == "ok" and proxy_ratio >= proxy_threshold)
            if threshold_hit and not request.force_vton_fallback:
                fallback_debug["triggered"] = True
                fallback_debug["trigger_reason"] = "high_proxy_occlusion"
            fallback_debug["proxy_occlusion_ratio"] = round(proxy_ratio, 4)
            fallback_debug["prefilter_status"] = prefilter_debug.get("status")

        if fallback_enabled and bool(fallback_debug.get("triggered")):
            fallback_result = _maybe_execute_vton_fallback(
                request_image_url=request.image_url,
                request_garment_type=request.garment_type,
                source_image=source_image,
                prefilter_debug=prefilter_debug,
                fallback_debug=fallback_debug,
                timings=timings,
                overall_start=overall_start,
            )
            if fallback_result is not None:
                return fallback_result

        step_start = time.perf_counter()
        parsing = _run_human_parsing(source_image)
        timings["parsing_s"] = round(time.perf_counter() - step_start, 4)

        step_start = time.perf_counter()
        resolved_cloth_labels, resolved_occluder_labels, label_debug = _resolve_label_ids_from_model(request.garment_type)
        garment_mask = np.isin(parsing, list(resolved_cloth_labels))
        garment_mask = _binary_open(garment_mask, 3)
        garment_mask = _binary_close(garment_mask, 5)
        garment_mask, refinement_debug = _refine_garment_mask(garment_mask, request.garment_type)

        occluder_mask = np.isin(parsing, list(resolved_occluder_labels))
        occlusion_debug = _estimate_occlusion_from_parsing_masks(garment_mask, occluder_mask)
        occlusion_mask = np.asarray(occlusion_debug.get("occlusion_mask", np.zeros_like(garment_mask, dtype=bool))).astype(bool)
        timings["mask_build_s"] = round(time.perf_counter() - step_start, 4)

        if garment_mask.sum() == 0:
            fallback_debug["parser_coverage_ratio"] = 0.0
            fallback_debug["parser_occlusion_ratio"] = 1.0
            fallback_debug["parser_occlusion_breakdown"] = {
                "direct_ratio": 0.0,
                "structural_ratio": 0.0,
                "hole_ratio": 0.0,
                "mid_ratio": 0.0,
                "core_ratio": 0.0,
                "weighted_hole_ratio": 0.0,
                "weighted_mid_ratio": 0.0,
                "weighted_core_ratio": 0.0,
                "kernel_size": int(occlusion_debug.get("kernel_size", 0)),
            }
            if fallback_enabled and not bool(fallback_debug.get("used", False)):
                fallback_debug["triggered"] = True
                fallback_debug["trigger_reason"] = "no_parser_garment_pixels"
                fallback_result = _maybe_execute_vton_fallback(
                    request_image_url=request.image_url,
                    request_garment_type=request.garment_type,
                    source_image=source_image,
                    prefilter_debug=prefilter_debug,
                    fallback_debug=fallback_debug,
                    timings=timings,
                    overall_start=overall_start,
                )
                if fallback_result is not None:
                    return fallback_result
            raise HTTPException(status_code=422, detail="No garment region detected for the requested garment_type")

        garment_pixels = int(garment_mask.sum())
        parser_coverage_ratio = float(garment_pixels / max(1, source_image.width * source_image.height))
        fallback_debug["parser_coverage_ratio"] = round(parser_coverage_ratio, 4)
        fallback_debug["parser_min_coverage_ratio"] = EXTRACT_PARSER_MIN_COVERAGE_RATIO
        parser_occlusion_ratio = float(occlusion_debug.get("proxy_ratio", 0.0))
        fallback_debug["parser_occlusion_ratio"] = round(parser_occlusion_ratio, 4)
        fallback_debug["parser_occlusion_breakdown"] = {
            "direct_ratio": float(occlusion_debug.get("direct_ratio", 0.0)),
            "structural_ratio": float(occlusion_debug.get("structural_ratio", 0.0)),
            "hole_ratio": float(occlusion_debug.get("hole_ratio", 0.0)),
            "mid_ratio": float(occlusion_debug.get("mid_ratio", 0.0)),
            "core_ratio": float(occlusion_debug.get("core_ratio", 0.0)),
            "weighted_hole_ratio": float(occlusion_debug.get("weighted_hole_ratio", 0.0)),
            "weighted_mid_ratio": float(occlusion_debug.get("weighted_mid_ratio", 0.0)),
            "weighted_core_ratio": float(occlusion_debug.get("weighted_core_ratio", 0.0)),
            "kernel_size": int(occlusion_debug.get("kernel_size", 0)),
        }
        if (
            fallback_enabled
            and not bool(fallback_debug.get("used", False))
            and parser_occlusion_ratio >= proxy_threshold
            and not bool(fallback_debug.get("triggered", False))
        ):
            fallback_debug["triggered"] = True
            fallback_debug["trigger_reason"] = "high_parser_occlusion"
            fallback_result = _maybe_execute_vton_fallback(
                request_image_url=request.image_url,
                request_garment_type=request.garment_type,
                source_image=source_image,
                prefilter_debug=prefilter_debug,
                fallback_debug=fallback_debug,
                timings=timings,
                overall_start=overall_start,
            )
            if fallback_result is not None:
                return fallback_result
        if (
            fallback_enabled
            and not bool(fallback_debug.get("used", False))
            and parser_coverage_ratio < EXTRACT_PARSER_MIN_COVERAGE_RATIO
        ):
            fallback_debug["triggered"] = True
            fallback_debug["trigger_reason"] = "low_parser_coverage"
            fallback_result = _maybe_execute_vton_fallback(
                request_image_url=request.image_url,
                request_garment_type=request.garment_type,
                source_image=source_image,
                prefilter_debug=prefilter_debug,
                fallback_debug=fallback_debug,
                timings=timings,
                overall_start=overall_start,
            )
            if fallback_result is not None:
                return fallback_result

        work_image = source_image
        work_garment_mask = garment_mask
        work_occlusion_mask = occlusion_mask
        # Temporarily keep full-frame garment output (no tight final crop).
        if request.crop_to_garment and not EXTRACT_DISABLE_FINAL_CROP:
            work_image, work_garment_mask, work_occlusion_mask = _crop_image_and_masks(
                work_image,
                work_garment_mask,
                work_occlusion_mask,
                padding_px=max(0, request.crop_padding_px),
            )

        step_start = time.perf_counter()
        rgba = np.asarray(work_image.convert("RGB"))
        alpha = _build_soft_alpha(work_garment_mask)[:, :, None]
        cloth_rgba = np.concatenate([rgba, alpha], axis=2)
        cloth_image = Image.fromarray(cloth_rgba, mode="RGBA")

        cloth_buffer = io.BytesIO()
        cloth_image.save(cloth_buffer, format="PNG")
        cloth_url = upload_to_azure(cloth_buffer.getvalue(), extension="png", content_type="image/png")
        timings["upload_cloth_s"] = round(time.perf_counter() - step_start, 4)

        occlusion_url = None
        if request.include_occlusion_mask:
            step_start = time.perf_counter()
            occ_image = Image.fromarray((work_occlusion_mask.astype(np.uint8) * 255), mode="L")
            occ_buffer = io.BytesIO()
            occ_image.save(occ_buffer, format="PNG")
            occlusion_url = upload_to_azure(occ_buffer.getvalue(), extension="png", content_type="image/png")
            timings["upload_occlusion_s"] = round(time.perf_counter() - step_start, 4)

        garment_pixels = int(work_garment_mask.sum())
        occluded_pixels = int(work_occlusion_mask.sum())
        occlusion_ratio = round(float(occlusion_debug.get("proxy_ratio", 0.0)), 4)

        timings["total_s"] = round(time.perf_counter() - overall_start, 4)

        return {
            "status": "success",
            "cloth_url": cloth_url,
            "occlusion_mask_url": occlusion_url,
            "metrics": {
                "garment_type": request.garment_type,
                "extraction_path": "parser",
                "image_size": {"width": work_image.width, "height": work_image.height},
                "coverage": {
                    "garment_pixels": garment_pixels,
                    "occluded_pixels": occluded_pixels,
                    "occlusion_ratio": occlusion_ratio,
                },
                "label_debug": label_debug,
                "refinement_debug": refinement_debug,
                "occlusion_debug": {
                    "direct_ratio": float(occlusion_debug.get("direct_ratio", 0.0)),
                    "structural_ratio": float(occlusion_debug.get("structural_ratio", 0.0)),
                    "hole_ratio": float(occlusion_debug.get("hole_ratio", 0.0)),
                    "mid_ratio": float(occlusion_debug.get("mid_ratio", 0.0)),
                    "core_ratio": float(occlusion_debug.get("core_ratio", 0.0)),
                    "weighted_hole_ratio": float(occlusion_debug.get("weighted_hole_ratio", 0.0)),
                    "weighted_mid_ratio": float(occlusion_debug.get("weighted_mid_ratio", 0.0)),
                    "weighted_core_ratio": float(occlusion_debug.get("weighted_core_ratio", 0.0)),
                    "proxy_ratio": float(occlusion_debug.get("proxy_ratio", 0.0)),
                    "kernel_size": int(occlusion_debug.get("kernel_size", 0)),
                },
                "prefilter_debug": prefilter_debug,
                "fallback_debug": fallback_debug,
                "timings": timings,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Cloth extraction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

async def handle_wardrobe_flow(request, ctx):
    globals().update(ctx)
    overall_start = time.perf_counter()
    try:
        include_part_base64 = bool(request.include_base64) or request.multipart_mode in {"base64", "both"}
        include_part_url = bool(request.include_upload_url) or request.multipart_mode in {"url", "both"}

        detect_request = DetectGarmentsRequest(
            image_url=request.image_url,
            max_items=max(1, min(5, int(request.max_items))),
            crop_padding_px=max(0, int(request.crop_padding_px)),
            include_base64=include_part_base64,
            include_upload_url=True,
            isolate_garment=False,
        )
        detect_result = await detect_garments(detect_request)
        detect_status = int(detect_result.get("status", 400))
        if detect_status != 200:
            return _build_error_payload(
                title="Detection Failed",
                description="Failed to detect garment sections from the uploaded image.",
                reason_codes=["DETECTION_FAILED"],
                status_code=400,
                result="REJECTED",
            )

        detect_data = detect_result.get("data", {}) if isinstance(detect_result, dict) else {}
        detected_items = list(detect_data.get("items", []))
        if not detected_items:
            return _build_error_payload(
                title="No Garment Found",
                description="We couldn't find a garment in this image. Please upload a clearer photo.",
                reason_codes=["NO_CLOTHING"],
                status_code=400,
                result="REJECTED",
            )

        item_breakdown: List[Dict[str, object]] = []
        internal_image_url_by_component_id: Dict[int, str] = {}
        internal_image_url_by_rank: Dict[int, str] = {}
        for item in detected_items:
            garment_type = _normalize_item_garment_type(str(item.get("garment_type", "all")))
            style_name = str(item.get("style", ""))
            category_meta = _wardrobe_category_from_garment_type(garment_type, style=style_name)
            confidence = float(item.get("combined_score", item.get("confidence", 0.0)) or 0.0)
            rank = int(item.get("rank", 0))
            component_id = int(item.get("component_id", 0))
            internal_image_url = str(item.get("image_url") or "").strip()
            if internal_image_url:
                internal_image_url_by_component_id[component_id] = internal_image_url
                internal_image_url_by_rank[rank] = internal_image_url
            item_payload: Dict[str, object] = {
                "rank": rank,
                "component_id": component_id,
                "garment_type": garment_type,
                "type": garment_type,
                "style": category_meta["style"],
                "primary_category_key": category_meta["primary_category_key"],
                "category_key": category_meta["category_key"],
                "confidence": round(confidence, 4),
                "detector_conf": float(item.get("detector_conf", 0.0) or 0.0),
                "clip_confidence": float(item.get("clip_confidence", 0.0) or 0.0),
                "combined_score": round(confidence, 4),
                "occlusion_ratio": float(item.get("occlusion_ratio", 0.0) or 0.0),
                "is_safe": bool(item.get("is_safe", True)),
                "source": item.get("source"),
                "bbox_original": item.get("bbox_original"),
                "bbox_crop": item.get("bbox_crop"),
                "crop_size": item.get("crop_size"),
            }
            if include_part_url and internal_image_url:
                item_payload["image_url"] = internal_image_url
            if include_part_base64 and "image_base64_png" in item:
                item_payload["image_base64_png"] = item.get("image_base64_png")
            item_breakdown.append(item_payload)

        selected_item: Optional[Dict[str, object]] = None
        invalid_selection = False
        selected_type = _normalize_selection_type(getattr(request, "selected_type", None))

        if selected_type:
            selected_item = _select_best_item_by_type(item_breakdown, selected_type)
            if selected_item is None:
                invalid_selection = True
        elif request.selected_component_id is not None:
            for item in item_breakdown:
                if int(item.get("component_id", -1)) == int(request.selected_component_id):
                    selected_item = item
                    break
            if selected_item is None:
                invalid_selection = True
        elif request.selected_item_rank is not None:
            for item in item_breakdown:
                if int(item.get("rank", -1)) == int(request.selected_item_rank):
                    selected_item = item
                    break
            if selected_item is None:
                invalid_selection = True
        elif len(item_breakdown) == 1:
            selected_item = item_breakdown[0]

        if invalid_selection:
            return _build_error_payload(
                title="Invalid Selection Type",
                description="No detected garment matched the selected type. Please choose top, bottom, dress, or outer.",
                reason_codes=["INVALID_SELECTION_TYPE"],
                status_code=400,
                result="REJECTED",
            )

        if selected_item is None and len(item_breakdown) > 1:
            total_s = round(time.perf_counter() - overall_start, 4)
            data = {
                "result": "REJECTED",
                "title": "Multiple Items Found",
                "description": "Multiple garments were detected. Please select a type (top, bottom, dress, outer) to continue extraction.",
                "reason_codes": ["MULTI_ITEM_SELECTION_REQUIRED"],
                "selection_required": True,
                "total_garments_found": len(item_breakdown),
                "selection_hint": {
                    "expected_field": "type",
                    "allowed_types": ["top", "bottom", "dress", "outer"],
                },
                "item_breakdown": item_breakdown,
                "multipart_data": _build_multipart_parts(
                    items=item_breakdown,
                    include_upload_url=include_part_url,
                    include_base64=include_part_base64,
                ),
                "latencies": {
                    "detect": detect_data.get("timings", {}),
                    "total": total_s,
                },
                "processing_time_ms": int(total_s * 1000),
            }
            return _json_response(_build_success_payload(data=data, status_code=400, message=""))

        if selected_item is None:
            selected_item = item_breakdown[0]

        selected_component_id = int(selected_item.get("component_id", -1))
        selected_rank = int(selected_item.get("rank", -1))
        selected_image_url = internal_image_url_by_component_id.get(selected_component_id, "")
        if not selected_image_url:
            selected_image_url = internal_image_url_by_rank.get(selected_rank, "")
        if not selected_image_url:
            selected_image_url = str(selected_item.get("image_url") or "").strip()
        if not selected_image_url:
            return _build_error_payload(
                title="Item Image Missing",
                description="Selected garment crop image URL is missing. Retry with upload URL enabled.",
                reason_codes=["ITEM_IMAGE_URL_MISSING"],
                status_code=400,
                result="REJECTED",
            )

        selected_type = _normalize_item_garment_type(str(selected_item.get("garment_type", "all")))
        requested_type = request.garment_type.lower().strip()
        extract_result, extract_type_used, last_extract_error = await _extract_cloth_with_fallback_types(
            selected_image_url=selected_image_url,
            selected_type=selected_type,
            preferred_type=requested_type if requested_type not in {"", "auto"} else selected_type,
            crop_padding_px=max(0, int(request.crop_padding_px)),
            use_occlusion_prefilter=request.use_occlusion_prefilter,
            occlusion_proxy_threshold=request.occlusion_proxy_threshold,
            enable_vton_fallback=request.enable_vton_fallback,
            force_vton_fallback=request.force_vton_fallback,
            force_vton_only=bool(selected_type in {"top", "bottom"}),
        )

        if extract_result is None:
            detail = "Garment extraction failed. Please retry with a clearer image."
            status_code = 400
            if last_extract_error is not None:
                detail = str(last_extract_error.detail)
                status_code = int(last_extract_error.status_code)
            return _build_error_payload(
                title="Extraction Failed",
                description=detail,
                reason_codes=["EXTRACTION_FAILED"],
                status_code=status_code,
                result="REJECTED",
            )

        extract_metrics = extract_result.get("metrics", {})
        extraction_path = str(extract_metrics.get("extraction_path", "parser"))
        reason_codes = ["SINGLE_ITEM"]
        if extraction_path == "vton_fallback":
            fallback_debug = extract_metrics.get("fallback_debug", {}) if isinstance(extract_metrics, dict) else {}
            trigger_reason = str(fallback_debug.get("trigger_reason", "")).strip().lower()
            if trigger_reason == "low_parser_coverage":
                reason_codes.append("LOW_PARSER_COVERAGE_FALLBACK")
            elif trigger_reason == "force_vton_fallback":
                reason_codes.append("FORCED_VTON_FALLBACK")
            else:
                reason_codes.append("HIGH_OCCLUSION_FALLBACK")

        # Use granular mapping for final response
        selected_category = _wardrobe_category_from_garment_type(
            str(selected_item.get("garment_type", "all")),
            style=str(selected_item.get("style", ""))
        )
        resolved_output_url = str(extract_result.get("cloth_url") or "")
        output_source = "vton" if extraction_path == "vton_fallback" else "parser"
        parser_cloth_url = None if extraction_path == "vton_fallback" else resolved_output_url
        vton_output_url = resolved_output_url if extraction_path == "vton_fallback" else None
        total_s = round(time.perf_counter() - overall_start, 4)

        combined_score = float(selected_item.get("combined_score", selected_item.get("confidence", 0.1)) or 0.1)

        data = {
            "result": "ACCEPTED",
            "title": "Added To Wardrobe",
            "description": "Garment extracted successfully and ready for wardrobe save.",
            "reason_codes": reason_codes,
            "selection_required": False,
            "clothing_type": selected_category["style"],
            "category_key": selected_category["category_key"],
            "primary_category_key": selected_category["primary_category_key"],
            "style": selected_category["style"],
            "confidence": round(combined_score, 4),
            "quality_score": round(combined_score, 4),
            "total_garments_found": len(item_breakdown),
            "selected_type": selected_type or str(selected_item.get("garment_type", "")),
            "selected_item": selected_item,
            "item_breakdown": item_breakdown,
            "cloth_url": parser_cloth_url,
            "vton_output_url": vton_output_url,
            "occlusion_mask_url": extract_result.get("occlusion_mask_url"),
            "output_image_url": resolved_output_url,
            "output_image_source": output_source,
            "extract_garment_type_used": extract_type_used,
            "multipart_data": _build_multipart_parts(
                items=item_breakdown,
                include_upload_url=include_part_url,
                include_base64=include_part_base64,
                cloth_url=resolved_output_url or None,
                occlusion_mask_url=str(extract_result.get("occlusion_mask_url") or "") or None,
            ),
            "extraction_path": extraction_path,
            "item_coverage": extract_metrics.get("coverage", {}),
            "latencies": {
                "detect": detect_data.get("timings", {}),
                "extract": extract_metrics.get("timings", {}),
                "total": total_s,
            },
            "processing_time_ms": int(total_s * 1000),
        }
        return _build_success_payload(data=data, status_code=200, message="")
    except HTTPException as exc:
        return _build_error_payload(
            title="Request Failed",
            description=str(exc.detail),
            reason_codes=["REQUEST_FAILED"],
            status_code=exc.status_code,
            result="REJECTED",
        )
    except Exception as exc:
        logger.exception("wardrobe_flow failed")
        return _build_error_payload(
            title="Server Error",
            description="Unexpected server error while running wardrobe flow.",
            reason_codes=["SERVER_ERROR"],
            status_code=500,
            result="REJECTED",
            message=str(exc),
        )

async def handle_analyze_multipart(file, authorization, selected_type, ctx):
    globals().update(ctx)
    overall_start = time.perf_counter()

    try:
        auth_payload = _verify_bearer_token(authorization)
    except PermissionError:
        payload = _build_error_payload(
            title="Session Expired",
            description="Please log in again and try uploading your item.",
            reason_codes=["UNAUTHORIZED"],
            status_code=401,
            result="REJECTED",
        )
        return _multipart_form_response(payload)

    try:
        image_bytes = await file.read()
        source_image, normalized_content_type, image_format = _validate_image_upload(file, image_bytes)
    except ValueError as exc:
        code = str(exc)
        if code == "FILE_TOO_LARGE":
            payload = _build_error_payload(
                title="File Too Large",
                description="Please upload an image under 3MB.",
                reason_codes=["FILE_TOO_LARGE"],
                status_code=400,
                result="REJECTED",
            )
            return _multipart_form_response(payload)
        if code == "INVALID_FILE_TYPE":
            payload = _build_error_payload(
                title="Unsupported File",
                description="Please upload a JPG, PNG, or WEBP image.",
                reason_codes=["INVALID_FILE_TYPE"],
                status_code=400,
                result="REJECTED",
            )
            return _multipart_form_response(payload)
        payload = _build_error_payload(
            title="Invalid Image",
            description="Invalid image data in request. Please try again.",
            reason_codes=["INVALID_IMAGE"],
            status_code=400,
            result="REJECTED",
        )
        return _multipart_form_response(payload)

    timings: Dict[str, float] = {}
    timings["read_validate_s"] = round(time.perf_counter() - overall_start, 4)

    step_start = time.perf_counter()
    blur_score = _check_blur_score(source_image)
    timings["blur_check_s"] = round(time.perf_counter() - step_start, 4)
    if blur_score < ANALYZE_BLUR_MIN:
        total_s = round(time.perf_counter() - overall_start, 4)
        payload = _build_error_payload(
            title="Image Too Blurry",
            description="Image is blurry. Please upload a clearer picture to save in wardrobe.",
            reason_codes=["IMAGE_TOO_BLURRY"],
            status_code=400,
            result="REJECTED",
        )
        payload_data = payload["data"]
        if isinstance(payload_data, dict):
            payload_data.update(
                {
                    "total_garments_found": 0,
                    "item_breakdown": [],
                    "multipart_data": {"format": "multipart-data", "parts": []},
                    "latencies": {**timings, "total": total_s},
                    "processing_time_ms": int(total_s * 1000),
                    "debug": {
                        "blur_score": round(float(blur_score), 4),
                        "blur_min": ANALYZE_BLUR_MIN,
                    },
                }
            )
        return _multipart_form_response(payload)

    step_start = time.perf_counter()
    yolo_result = _build_yolo_item_breakdown_from_image(
        source_image=source_image,
        max_items=max(1, min(5, ANALYZE_YOLO_MAX_ITEMS)),
        crop_padding_px=max(0, ANALYZE_CROP_PADDING_PX),
        min_item_pixels=max(1, ANALYZE_MIN_ITEM_PIXELS),
        include_upload_url=ANALYZE_INCLUDE_UPLOAD_URL,
        include_base64=ANALYZE_INCLUDE_BASE64,
    )
    timings["yolo_detect_s"] = round(time.perf_counter() - step_start, 4)

    item_breakdown = list(yolo_result.get("item_breakdown", []))
    yolo_binary_parts = list(yolo_result.get("binary_parts", []))
    raw_instances_count = int(yolo_result.get("raw_instances_count", 0))
    candidate_instances_count = int(yolo_result.get("candidate_instances_count", 0))
    valid_instances_count = int(yolo_result.get("valid_instances_count", 0))
    internal_image_url_by_component_id = dict(yolo_result.get("internal_image_url_by_component_id", {}))
    internal_image_url_by_rank = dict(yolo_result.get("internal_image_url_by_rank", {}))
    internal_isolated_image_url_by_component_id = dict(
        yolo_result.get("internal_isolated_image_url_by_component_id", {})
    )
    internal_isolated_image_url_by_rank = dict(
        yolo_result.get("internal_isolated_image_url_by_rank", {})
    )
    detection_source = str(yolo_result.get("detection_source", "unknown"))
    requested_selected_type = _normalize_selection_type(selected_type)

    if candidate_instances_count == 0 and valid_instances_count == 0:
        total_s = round(time.perf_counter() - overall_start, 4)
        payload = _build_error_payload(
            title="No Clothing Found",
            description="We couldn't find a garment in this photo. Please upload a clear garment image.",
            reason_codes=["NO_CLOTHING"],
            status_code=400,
            result="REJECTED",
        )
        payload_data = payload["data"]
        if isinstance(payload_data, dict):
            payload_data.update(
                {
                    "total_garments_found": 0,
                    "item_breakdown": [],
                    "multipart_data": {"format": "multipart-data", "parts": []},
                    "latencies": {**timings, "total": total_s},
                    "processing_time_ms": int(total_s * 1000),
                    "debug": {
                        "blur_score": round(float(blur_score), 4),
                        "blur_min": ANALYZE_BLUR_MIN,
                        "detection_source": detection_source,
                        "detector_debug": yolo_result.get("detector_debug", {}),
                        "parser_category_debug": yolo_result.get("parser_category_debug", {}),
                        "size_filters": yolo_result.get("size_filters", {}),
                    },
                }
            )
        return _multipart_form_response(payload)

    if valid_instances_count == 0 or not item_breakdown:
        total_s = round(time.perf_counter() - overall_start, 4)
        payload = _build_error_payload(
            title="No Valid Garment Found",
            description="No clear garment area was detected. Please upload an image where the clothing item is clearly visible.",
            reason_codes=["NO_VALID_GARMENT"],
            status_code=400,
            result="REJECTED",
        )
        payload_data = payload["data"]
        if isinstance(payload_data, dict):
            payload_data.update(
                {
                    "total_garments_found": 0,
                    "item_breakdown": [],
                    "multipart_data": {"format": "multipart-data", "parts": []},
                    "latencies": {**timings, "total": total_s},
                    "processing_time_ms": int(total_s * 1000),
                    "debug": {
                        "blur_score": round(float(blur_score), 4),
                        "blur_min": ANALYZE_BLUR_MIN,
                        "detection_source": detection_source,
                        "min_item_pixels": ANALYZE_MIN_ITEM_PIXELS,
                        "raw_instances_count": raw_instances_count,
                        "candidate_instances_count": candidate_instances_count,
                        "valid_instances_count": valid_instances_count,
                        "detector_debug": yolo_result.get("detector_debug", {}),
                        "parser_category_debug": yolo_result.get("parser_category_debug", {}),
                        "size_filters": yolo_result.get("size_filters", {}),
                    },
                }
            )
        return _multipart_form_response(payload)

    if len(item_breakdown) > 1 and not requested_selected_type:
        total_s = round(time.perf_counter() - overall_start, 4)
        data = {
            "result": "REJECTED",
            "title": "Multiple Items Found",
            "description": "Multiple garments were detected. Please select a type (top, bottom, dress, outer) and re-upload to continue.",
            "reason_codes": ["MULTI_ITEM_SELECTION_REQUIRED"],
            "selection_required": True,
            "total_garments_found": len(item_breakdown),
            "selection_hint": {
                "expected_field": "type",
                "allowed_types": ["top", "bottom", "dress", "outer"],
            },
            "item_breakdown": item_breakdown,
            "multipart_data": _build_multipart_parts(
                items=item_breakdown,
                include_upload_url=ANALYZE_INCLUDE_UPLOAD_URL,
                include_base64=ANALYZE_INCLUDE_BASE64,
            ),
            "latencies": {**timings, "total": total_s},
            "processing_time_ms": int(total_s * 1000),
            "debug": {
                "blur_score": round(float(blur_score), 4),
                "blur_min": ANALYZE_BLUR_MIN,
                "detection_source": detection_source,
                "raw_instances_count": raw_instances_count,
                "candidate_instances_count": candidate_instances_count,
                "valid_instances_count": valid_instances_count,
                "detector_debug": yolo_result.get("detector_debug", {}),
                "parser_category_debug": yolo_result.get("parser_category_debug", {}),
            },
        }
        payload = _build_success_payload(data=data, status_code=400, message="")
        return _multipart_form_response(payload, binary_parts=yolo_binary_parts)

    selected_item = _select_best_item_by_type(item_breakdown, requested_selected_type) if requested_selected_type else None
    single_item_type_fallback_enabled = bool(globals().get("ANALYZE_SINGLE_ITEM_TYPE_FALLBACK", True))
    if requested_selected_type and selected_item is None:
        # Cross-type satisfaction: If the user explicitly requested a type (e.g., 'top')
        # but the analyzer labeled the best candidate as something else (e.g., 'dress'),
        # we allow the top-ranked item to satisfy the request to avoid 400 rejections,
        # especially for zoomed-in photos where labeling becomes ambiguous.
        if single_item_type_fallback_enabled and len(item_breakdown) > 0:
            selected_item = item_breakdown[0]
        else:
            total_s = round(time.perf_counter() - overall_start, 4)
            payload = _build_error_payload(
                title="Invalid Selection Type",
                description="No detected garment matched the selected type. Please choose top, bottom, dress, or outer.",
                reason_codes=["INVALID_SELECTION_TYPE"],
                status_code=400,
                result="REJECTED",
            )
            payload_data = payload["data"]
            if isinstance(payload_data, dict):
                payload_data.update(
                    {
                        "total_garments_found": len(item_breakdown),
                        "requested_type": requested_selected_type,
                        "item_breakdown": item_breakdown,
                        "multipart_data": _build_multipart_parts(
                            items=item_breakdown,
                            include_upload_url=ANALYZE_INCLUDE_UPLOAD_URL,
                            include_base64=ANALYZE_INCLUDE_BASE64,
                        ),
                        "latencies": {**timings, "total": total_s},
                        "processing_time_ms": int(total_s * 1000),
                    }
                )
            return _multipart_form_response(payload, binary_parts=yolo_binary_parts)

    if selected_item is None:
        selected_item = item_breakdown[0]

    user_forced_vton_type = requested_selected_type in {"top", "bottom"}

    # === SAFETY & CONFIDENCE CHECKS (Matches cloth-analysis) ===
    # Enforce minimum confidence
    combined_score = float(selected_item.get("combined_score", 0.0))
    clip_conf = float(selected_item.get("clip_confidence", 0.0))

    # We use the same thresholds as cloth-analysis
    # Note: These could be made configurable via ENV
    MIN_COMBINED_SCORE = 0.35
    MIN_CLIP_CONF = 0.18

    if (combined_score < MIN_COMBINED_SCORE or clip_conf < MIN_CLIP_CONF) and not user_forced_vton_type:
        # If a single broad person/dress candidate is low-confidence in no-type mode,
        # split into top/bottom previews so user can explicitly choose the garment part.
        low_conf_split_enabled = bool(globals().get("ANALYZE_LOW_CONFIDENCE_SPLIT_TO_MULTI", True))
        selected_component_id = int(selected_item.get("component_id", -1))
        selected_rank = int(selected_item.get("rank", -1))
        selected_item_type = str(selected_item.get("garment_type", "")).strip().lower()
        selected_class_id = int(selected_item.get("class_id", -1))
        selected_image_url_for_split = str(internal_image_url_by_rank.get(selected_rank, "")).strip()
        if not selected_image_url_for_split:
            selected_image_url_for_split = str(internal_image_url_by_component_id.get(selected_component_id, "")).strip()
        if not selected_image_url_for_split:
            selected_image_url_for_split = str(selected_item.get("image_url") or "").strip()

        should_split_to_multi = (
            low_conf_split_enabled
            and not requested_selected_type
            and len(item_breakdown) == 1
            and selected_class_id in {0, -1}
            and selected_item_type in {"dress", "all", ""}
            and bool(selected_image_url_for_split)
        )
        selected_bbox = selected_item.get("bbox_original")
        selected_bbox_w = 0
        selected_bbox_h = 0
        if isinstance(selected_bbox, dict):
            selected_bbox_w = max(0, int(selected_bbox.get("x1", 0)) - int(selected_bbox.get("x0", 0)))
            selected_bbox_h = max(0, int(selected_bbox.get("y1", 0)) - int(selected_bbox.get("y0", 0)))
        image_w = max(1, int(getattr(source_image, "width", 1)))
        image_h = max(1, int(getattr(source_image, "height", 1)))
        selected_w_ratio = float(selected_bbox_w / image_w)
        selected_h_ratio = float(selected_bbox_h / image_h)
        selected_source = str(selected_item.get("source", "")).strip().lower()
        # If no-type mode found only one low-confidence top/bottom candidate that
        # covers just part of the person frame, treat it as likely multi-garment
        # and return top/bottom split choices instead of LOW_CONFIDENCE.
        likely_two_piece_single_candidate = (
            low_conf_split_enabled
            and not requested_selected_type
            and len(item_breakdown) == 1
            and selected_item_type in {"top", "bottom"}
            and bool(selected_image_url_for_split)
            and selected_h_ratio <= 0.72
            and selected_w_ratio <= 0.92
        )
        if likely_two_piece_single_candidate:
            should_split_to_multi = True

        likely_single_full_piece = (
            selected_item_type in {"dress", "all", ""}
            and selected_h_ratio >= 0.50
            and selected_w_ratio >= 0.28
            and selected_source in {
                "foreground_fallback",
                "single_piece_merge",
                "yolo",
                "yolo_person",
                "yolo_person_split",
                "waist_split",
                "waist_split_top",
                "waist_split_bottom",
                "parser_category",
                "parser_category_overlap_fallback",
            }
        )
        if selected_item_type == "dress" and selected_h_ratio >= 0.40:
            # Single-dress inputs frequently score lower in no-type mode.
            # Keep them in direct VTON flow instead of forcing a split prompt.
            likely_single_full_piece = True
        if selected_item_type == "dress":
            # In no-type mode, once the detector already resolved to a single dress
            # candidate, do not branch into low-confidence top/bottom split.
            likely_single_full_piece = True
        if likely_single_full_piece:
            should_split_to_multi = False

        if should_split_to_multi:
            try:
                split_source_image, _ = _download_image(selected_image_url_for_split)
                split_items: List[Dict[str, object]] = []
                split_binary_parts: List[Dict[str, object]] = []
                for rank_idx, forced_kind in enumerate(["top", "bottom"], start=1):
                    crop_image = _split_crop_for_forced_type(split_source_image, forced_kind)
                    out_buf = io.BytesIO()
                    crop_image.save(out_buf, format="PNG")
                    crop_bytes = out_buf.getvalue()
                    local_path = _save_local_analyze_image(
                        crop_bytes,
                        prefix=f"analyze_lowconf_{forced_kind}",
                        ext="png",
                    )
                    style_name, clip_score = _run_clip_classification(crop_image, garment_type=forced_kind)
                    category_meta = _wardrobe_category_from_garment_type(forced_kind, style=style_name)
                    split_item = {
                        "rank": rank_idx,
                        "component_id": rank_idx,
                        "garment_type": forced_kind,
                        "type": forced_kind,
                        "style": category_meta["style"],
                        "primary_category_key": category_meta["primary_category_key"],
                        "category_key": category_meta["category_key"],
                        "confidence": round(float(clip_score), 4),
                        "detector_conf": round(float(selected_item.get("detector_conf", 0.0) or 0.0), 4),
                        "clip_confidence": round(float(clip_score), 4),
                        "combined_score": round(float(clip_score), 4),
                        "occlusion_ratio": round(float(selected_item.get("occlusion_ratio", 0.0) or 0.0), 4),
                        "is_safe": True,
                        "source": "low_confidence_auto_split",
                        "crop_size": {"width": int(crop_image.width), "height": int(crop_image.height)},
                        "class_id": selected_class_id,
                        "class_name": str(selected_item.get("class_name", "")),
                    }
                    if local_path:
                        split_item["image_local_path"] = local_path
                        split_item["image_local_file_url"] = f"file://{local_path}"
                        split_item["image_url"] = f"file://{local_path}"
                    split_items.append(split_item)
                    split_binary_parts.append(
                        {
                            "name": f"item_{rank_idx}",
                            "filename": f"item_{rank_idx}.png",
                            "content_type": "image/png",
                            "bytes": crop_bytes,
                        }
                    )

                total_s = round(time.perf_counter() - overall_start, 4)
                data = {
                    "result": "REJECTED",
                    "title": "Multiple Items Found",
                    "description": "Multiple garments were detected. Please select a type (top, bottom, dress, outer) and re-upload to continue.",
                    "reason_codes": ["MULTI_ITEM_SELECTION_REQUIRED", "LOW_CONFIDENCE_SPLIT_REQUIRED"],
                    "selection_required": True,
                    "total_garments_found": len(split_items),
                    "selection_hint": {
                        "expected_field": "type",
                        "allowed_types": ["top", "bottom", "dress", "outer"],
                    },
                    "item_breakdown": split_items,
                    "multipart_data": _build_multipart_parts(
                        items=split_items,
                        include_upload_url=ANALYZE_INCLUDE_UPLOAD_URL,
                        include_base64=ANALYZE_INCLUDE_BASE64,
                    ),
                    "latencies": {**timings, "total": total_s},
                    "processing_time_ms": int(total_s * 1000),
                    "debug": {
                        "trigger": "low_confidence_split_to_multi",
                        "combined_score": round(combined_score, 4),
                        "clip_confidence": round(clip_conf, 4),
                        "thresholds": {"combined": MIN_COMBINED_SCORE, "clip": MIN_CLIP_CONF},
                    },
                }
                payload = _build_success_payload(data=data, status_code=400, message="")
                return _multipart_form_response(payload, binary_parts=split_binary_parts)
            except Exception as split_exc:
                logger.warning("Low-confidence split fallback failed: %s", split_exc)

        if not likely_single_full_piece:
            total_s = round(time.perf_counter() - overall_start, 4)
            payload = _build_error_payload(
                title="Unclear Garment",
                description="The garment type isn't clear. Please upload a sharper, centered photo.",
                reason_codes=["LOW_CONFIDENCE"],
                status_code=400,
                result="REJECTED",
            )
            payload_data = payload["data"]
            if isinstance(payload_data, dict):
                payload_data.update({
                    "latencies": {**timings, "total": total_s},
                    "processing_time_ms": int(total_s * 1000),
                    "item_breakdown": item_breakdown,
                    "debug": {
                        "combined_score": round(combined_score, 4),
                        "clip_confidence": round(clip_conf, 4),
                        "thresholds": {"combined": MIN_COMBINED_SCORE, "clip": MIN_CLIP_CONF}
                    }
                })
            return _multipart_form_response(payload, binary_parts=yolo_binary_parts)

    if not bool(selected_item.get("is_safe", True)):
        total_s = round(time.perf_counter() - overall_start, 4)
        payload = _build_error_payload(
            title="Item Not Supported",
            description="This item can't be added. Please upload a different garment.",
            reason_codes=["BANNED_ITEM"],
            status_code=400,
            result="REJECTED",
        )
        payload_data = payload["data"]
        if isinstance(payload_data, dict):
            payload_data.update({
                "latencies": {**timings, "total": total_s},
                "processing_time_ms": int(total_s * 1000),
                "item_breakdown": item_breakdown,
            })
        return _multipart_form_response(payload, binary_parts=yolo_binary_parts)

    selected_component_id = int(selected_item.get("component_id", -1))
    selected_rank = int(selected_item.get("rank", -1))
    use_isolated_top_crop = bool(globals().get("ANALYZE_USE_ISOLATED_TOP_CROP", False))
    use_isolated_bottom_crop = bool(globals().get("ANALYZE_USE_ISOLATED_BOTTOM_CROP", False))
    forced_type_uses_isolated = (
        (requested_selected_type == "top" and use_isolated_top_crop)
        or (requested_selected_type == "bottom" and use_isolated_bottom_crop)
    )

    # For `top`, isolated previews can help remove bottom spillover.
    # For `bottom`, keep raw crop priority so waistband/shape context is preserved.
    # Rank-level previews are type-specific and must win when a single detector component
    # is split into top/bottom parser categories that share the same component_id.
    selected_image_url = ""
    if forced_type_uses_isolated:
        selected_image_url = str(
            internal_isolated_image_url_by_rank.get(selected_rank, "")
        ).strip()
        if not selected_image_url:
            selected_image_url = str(
                internal_isolated_image_url_by_component_id.get(selected_component_id, "")
            ).strip()
    if not selected_image_url:
        selected_image_url = str(internal_image_url_by_rank.get(selected_rank, "")).strip()
    if not selected_image_url:
        same_component_items = [
            it
            for it in item_breakdown
            if int(it.get("component_id", -1)) == selected_component_id
        ]
        if len(same_component_items) == 1:
            selected_image_url = str(
                internal_image_url_by_component_id.get(selected_component_id, "")
            ).strip()
    if not selected_image_url:
        selected_image_url = str(selected_item.get("image_url") or "").strip()
    if not selected_image_url and requested_selected_type == "bottom":
        selected_image_url = str(
            internal_isolated_image_url_by_rank.get(selected_rank, "")
        ).strip()
        if not selected_image_url:
            selected_image_url = str(
                internal_isolated_image_url_by_component_id.get(selected_component_id, "")
            ).strip()
    if not selected_image_url:
        payload = _build_error_payload(
            title="Item Image Missing",
            description="Detected garment crop image URL is missing. Please retry upload.",
            reason_codes=["ITEM_IMAGE_URL_MISSING"],
            status_code=400,
            result="REJECTED",
        )
        return _multipart_form_response(payload)

    step_start = time.perf_counter()
    selected_type = _normalize_item_garment_type(str(selected_item.get("garment_type", "all")))
    selected_source = str(selected_item.get("source", "")).lower().strip()
    selected_occlusion_ratio = float(selected_item.get("occlusion_ratio", 0.0) or 0.0)
    dress_force_vton = (
        ANALYZE_FORCE_VTON_FOR_SINGLE_PIECE_DRESS
        and selected_type == "dress"
        and selected_source == "single_piece_merge"
        and selected_occlusion_ratio >= max(0.0, min(1.0, ANALYZE_DRESS_VTON_OCCLUSION_THRESHOLD))
    )
    preferred_extract_type = requested_selected_type if user_forced_vton_type else selected_type

    forced_type_split_crop_applied = False
    forced_type_style_override: Optional[str] = None
    forced_type_clip_override: Optional[float] = None
    force_split_crop_input = bool(globals().get("ANALYZE_FORCE_SPLIT_CROP_INPUT", False))
    if (
        user_forced_vton_type
        and requested_selected_type in {"top", "bottom"}
        and len(item_breakdown) == 1
        and selected_type not in {"top", "bottom"}
    ):
        try:
            forced_source_image, _ = _download_image(selected_image_url)
            forced_crop = _split_crop_for_forced_type(forced_source_image, requested_selected_type)
            try:
                forced_style_name, forced_style_conf = _run_clip_classification(
                    forced_crop,
                    garment_type=requested_selected_type,
                )
                forced_type_style_override = str(forced_style_name or "").strip() or None
                forced_type_clip_override = float(forced_style_conf)
            except Exception:
                forced_type_style_override = None
                forced_type_clip_override = None
            out_buf = io.BytesIO()
            forced_crop.save(out_buf, format="PNG")
            forced_bytes = out_buf.getvalue()
            forced_local_path = _save_local_analyze_image(
                forced_bytes,
                prefix=f"analyze_forced_{requested_selected_type}",
                ext="png",
            )
            if forced_local_path and (requested_selected_type == "top" or force_split_crop_input):
                selected_image_url = f"file://{forced_local_path}"
                forced_type_split_crop_applied = True
        except Exception as exc:
            logger.warning("Forced type split crop failed for analyze flow: %s", exc)

    force_vton_path = bool(dress_force_vton or ANALYZE_ALWAYS_USE_VTON)
    extract_result, extract_type_used, last_extract_error = await _extract_cloth_with_fallback_types(
        selected_image_url=selected_image_url,
        selected_type=selected_type,
        preferred_type=preferred_extract_type,
        crop_padding_px=max(0, ANALYZE_CROP_PADDING_PX),
        use_occlusion_prefilter=True,
        occlusion_proxy_threshold=ANALYZE_OCCLUSION_THRESHOLD,
        enable_vton_fallback=True,
        force_vton_fallback=bool(force_vton_path or user_forced_vton_type),
        force_vton_only=bool(user_forced_vton_type),
    )
    timings["extract_s"] = round(time.perf_counter() - step_start, 4)
    if extract_result is None:
        detail = "Garment extraction failed. Please retry with a clearer image."
        status_code = 400
        if last_extract_error is not None:
            detail = str(last_extract_error.detail)
            status_code = int(last_extract_error.status_code)
        payload = _build_error_payload(
            title="Extraction Failed",
            description=detail,
            reason_codes=["EXTRACTION_FAILED"],
            status_code=status_code,
            result="REJECTED",
        )
        return _multipart_form_response(payload, binary_parts=yolo_binary_parts)

    extract_metrics = extract_result.get("metrics", {}) if isinstance(extract_result, dict) else {}
    extraction_path = str(extract_metrics.get("extraction_path", "parser"))
    reason_codes = ["SINGLE_ITEM"]
    if extraction_path == "vton_fallback":
        fallback_debug = extract_metrics.get("fallback_debug", {}) if isinstance(extract_metrics, dict) else {}
        trigger_reason = str(fallback_debug.get("trigger_reason", "")).strip().lower()
        if trigger_reason == "low_parser_coverage":
            reason_codes.append("LOW_PARSER_COVERAGE_FALLBACK")
        elif trigger_reason == "force_vton_fallback":
            if ANALYZE_ALWAYS_USE_VTON:
                reason_codes.append("VTON_ONLY_PIPELINE")
            else:
                reason_codes.append("FORCED_VTON_FALLBACK")
        else:
            reason_codes.append("HIGH_OCCLUSION_FALLBACK")
    if user_forced_vton_type:
        reason_codes.append("TYPE_FORCED_VTON")
    if (
        requested_selected_type
        and selected_item is not None
        and str(selected_item.get("garment_type", "")).strip().lower() != requested_selected_type
    ):
        reason_codes.append("TYPE_FALLBACK_SINGLE_ITEM")
    if forced_type_split_crop_applied:
        reason_codes.append("TYPE_FORCED_SPLIT_CROP")

    user_id = str(auth_payload.get("userId"))
    progress_id = str(uuid.uuid4())
    input_ext = "png" if image_format == "PNG" else ("webp" if image_format == "WEBP" else "jpg")
    input_blob_name = f"{user_id}/{progress_id}.{input_ext}"

    try:
        step_start = time.perf_counter()
        input_url = _upload_image_to_azure_container(
            image_bytes,
            content_type=normalized_content_type,
            container_name=AZURE_STORAGE_INPUT_CONTAINER,
            blob_name=input_blob_name,
        )
        timings["upload_input_s"] = round(time.perf_counter() - step_start, 4)
    except Exception as exc:
        logger.error("Input upload failed: %s", exc)
        payload = _build_error_payload(
            title="Upload Failed",
            description="Failed to upload input image for wardrobe sync.",
            reason_codes=["UPLOAD_FAILED"],
            status_code=500,
            result="REJECTED",
        )
        return _multipart_form_response(payload, binary_parts=yolo_binary_parts)

    # Use granular mapping for final response
    selected_category = _wardrobe_category_from_garment_type(
        str(selected_item.get("garment_type", "all")),
        style=str(selected_item.get("style", "")),
    )
    if requested_selected_type in {"top", "bottom"}:
        selected_style = (forced_type_style_override or "").strip()
        if requested_selected_type == "bottom":
            if not selected_style:
                selected_style = str(selected_item.get("style", "") or "").strip()
            if not selected_style or selected_style.lower() in {"top", "bottom", "dress", "outerwear", "outer"}:
                inferred_style = _infer_style_from_detector_class_name(
                    str(selected_item.get("class_name", "")),
                    requested_selected_type,
                )
                if inferred_style:
                    selected_style = inferred_style
        selected_category = _wardrobe_category_from_garment_type(
            requested_selected_type,
            style=selected_style,
        )
        if forced_type_clip_override is not None:
            detector_conf = float(selected_item.get("detector_conf", selected_item.get("confidence", 0.0)) or 0.0)
            combined_score = max(combined_score, (detector_conf * 0.3) + (float(forced_type_clip_override) * 0.7))
    resolved_output_url = str(extract_result.get("cloth_url") or "")
    if not resolved_output_url:
        payload = _build_error_payload(
            title="Extraction Failed",
            description="Extractor returned no output URL.",
            reason_codes=["EXTRACTION_FAILED"],
            status_code=500,
            result="REJECTED",
        )
        return _multipart_form_response(payload, binary_parts=yolo_binary_parts)

    data = {
        "result": "ACCEPTED",
        "title": "Added To Wardrobe",
        "description": "Garment extracted successfully and ready for wardrobe save.",
        "reason_codes": reason_codes,
        "selected_type": requested_selected_type or selected_type,
        "clothing_type": selected_category["style"],
        "category_key": selected_category["category_key"],
        "primary_category_key": selected_category["primary_category_key"],
        "style": selected_category["style"],
        "confidence": round(combined_score, 4),
        "quality_score": round(combined_score, 4),
        "output_image_url": resolved_output_url,
        "wardrobe_progress_id": progress_id,
        "item_breakdown": item_breakdown,
    }

    sync_metadata: Dict[str, object] = {
        "response_data": dict(data),
        "analyze_debug": {
            "blur_score": round(float(blur_score), 4),
            "blur_min": ANALYZE_BLUR_MIN,
            "detection_source": detection_source,
            "raw_instances_count": raw_instances_count,
            "candidate_instances_count": candidate_instances_count,
            "valid_instances_count": valid_instances_count,
            "detector_debug": yolo_result.get("detector_debug", {}),
            "parser_category_debug": yolo_result.get("parser_category_debug", {}),
            "size_filters": yolo_result.get("size_filters", {}),
            "timings": dict(timings),
        },
        "extract_debug": {
            "selected_type": selected_type,
            "extract_type_used": extract_type_used,
            "extraction_path": extraction_path,
            "reason_codes": list(reason_codes),
            "extract_metrics": extract_metrics,
        },
        "request_debug": {
            "filename": file.filename,
            "content_type": normalized_content_type,
            "image_format": image_format,
            "input_size_bytes": len(image_bytes),
            "requested_type": requested_selected_type,
            "selected_image_url": selected_image_url,
        },
    }

    response_binary_parts: List[Dict[str, object]] = list(yolo_binary_parts) if ANALYZE_INCLUDE_ITEM_PREVIEW_ON_SUCCESS else []
    cloth_bytes = _download_binary_from_url(resolved_output_url)
    cloth_local_path = None
    if cloth_bytes:
        cloth_local_path = _save_local_analyze_image(cloth_bytes, prefix="analyze_extracted_cloth", ext="png")
        extracted_filename = f"extracted_cloth_{progress_id}.png"
        response_binary_parts.append(
            {
                "name": "extracted_cloth",
                "filename": extracted_filename,
                "content_type": "image/png",
                "bytes": cloth_bytes,
            }
        )
    if cloth_local_path:
        sync_metadata["extract_debug"] = {
            **dict(sync_metadata.get("extract_debug", {})),
            "extracted_cloth_local_path": cloth_local_path,
        }

    payload = _build_success_payload(data=data, status_code=200, message="")
    background_task: Optional[BackgroundTask] = None
    if ENABLE_WARDROBE_PROGRESS_SYNC and WARDROBE_PROGRESS_API_BASE_URL:
        background_task = BackgroundTask(
            _trigger_background_wardrobe_sync,
            api_base_url=WARDROBE_PROGRESS_API_BASE_URL,
            bearer_token=authorization or "",
            progress_id=progress_id,
            input_url=input_url,
            output_url=resolved_output_url,
            metadata=sync_metadata,
        )
    else:
        logger.info(
            "Skipping wardrobe progress sync (enabled=%s, base_url=%s)",
            ENABLE_WARDROBE_PROGRESS_SYNC,
            bool(WARDROBE_PROGRESS_API_BASE_URL),
        )

    return _multipart_form_response(
        payload,
        binary_parts=response_binary_parts,
        background=background_task,
    )

async def handle_process_image(request, ctx):
    globals().update(ctx)
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA GPU is required")
    try:
        _ensure_tryon_model_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model load failed: {exc}") from exc

    try:
        timings = {}
        overall_start = time.perf_counter()
        requested_steps = request.inference_steps if request.inference_steps is not None else None
        inference_steps = requested_steps if requested_steps is not None else INFERENCE_STEPS
        max_image_size = request.max_image_size if request.max_image_size is not None else MAX_IMAGE_SIZE
        guidance_scale = request.guidance_scale if request.guidance_scale is not None else GUIDANCE_SCALE
        auto_prompt_enabled = request.auto_prompt if request.auto_prompt is not None else AUTO_PROMPT_ENABLED
        adaptive_steps_enabled = request.adaptive_steps if request.adaptive_steps is not None else ADAPTIVE_STEPS_ENABLED
        region_upscale_enabled = request.enable_region_upscale if request.enable_region_upscale is not None else ENABLE_REGION_UPSCALE
        design_match_mode = request.design_match_mode if request.design_match_mode is not None else DESIGN_MATCH_MODE
        strict_design_match = request.strict_design_match if request.strict_design_match is not None else STRICT_DESIGN_MATCH
        condition_use_garment_isolation = (
            request.condition_use_garment_isolation
            if request.condition_use_garment_isolation is not None
            else CONDITION_USE_GARMENT_ISOLATION
        )
        auto_prompt_forced = False
        isolation_forced = False
        design_match_forced = False
        strict_reference_forced = False
        manual_description_ignored = bool((request.person_description or "").strip())

        if FORCE_AUTO_PROMPT and not auto_prompt_enabled:
            auto_prompt_enabled = True
            auto_prompt_forced = True

        if FORCE_GARMENT_ISOLATION and not condition_use_garment_isolation:
            condition_use_garment_isolation = True
            isolation_forced = True

        if FORCE_IMAGE_REFERENCE_PROMPT:
            if not design_match_mode:
                design_match_mode = True
                design_match_forced = True
            if not strict_design_match:
                strict_design_match = True
                strict_reference_forced = True

        if design_match_mode and strict_design_match:
            if not condition_use_garment_isolation:
                condition_use_garment_isolation = True
                isolation_forced = True
            if not auto_prompt_enabled:
                auto_prompt_enabled = True
                auto_prompt_forced = True

        if max_image_size < 256 or max_image_size > 2048:
            raise HTTPException(status_code=400, detail="max_image_size must be between 256 and 2048")

        step_start = time.perf_counter()
        source_image, download_bytes = _download_image(request.image_url)
        timings["download_s"] = round(time.perf_counter() - step_start, 4)
        timings["download_mb"] = round(download_bytes / (1024 * 1024), 4)

        original_w, original_h = source_image.size

        step_start = time.perf_counter()
        person_image = _resize_for_inference(source_image, max_image_size=max_image_size)
        timings["preprocess_resize_s"] = round(time.perf_counter() - step_start, 4)
        new_w, new_h = person_image.size

        step_start = time.perf_counter()
        if condition_use_garment_isolation:
            condition_image, conditioning_debug, conditioning_mask, conditioning_garment_type = _build_garment_conditioning_image(
                person_image,
                request.person_description,
                enable_isolation=condition_use_garment_isolation,
            )
        else:
            condition_image = person_image
            conditioning_debug = {"enabled": False, "reason": "disabled"}
            conditioning_mask = None
            conditioning_garment_type = "all" if FORCE_IMAGE_REFERENCE_PROMPT else _detect_garment_type_from_text(request.person_description)
        timings["conditioning_image_s"] = round(time.perf_counter() - step_start, 4)

        # Two-panel canvas for virtual try-off: [target cloth slot | source person]
        step_start = time.perf_counter()
        canvas = Image.new("RGB", (new_w * 2, new_h), (0, 0, 0))
        canvas.paste(condition_image, (new_w, 0))

        mask = np.zeros((new_h, new_w * 2, 3), dtype=np.uint8)
        mask[:, :new_w] = 255
        mask_image = Image.fromarray(mask)
        timings["canvas_mask_s"] = round(time.perf_counter() - step_start, 4)

        step_start = time.perf_counter()
        if not auto_prompt_enabled and FORCE_AUTO_PROMPT:
            auto_prompt_enabled = True
            auto_prompt_forced = True

        if auto_prompt_enabled:
            prompt, prompt_debug = _build_auto_prompt(
                person_image,
                request.person_description,
                strict_reference=(design_match_mode and strict_design_match),
            )
        else:
            # Kept for explicit legacy mode only (FORCE_AUTO_PROMPT=false).
            garment_description = _normalize_garment_description(request.person_description)
            prompt = (
                f"<MODEL> {garment_description}. <TARGET> exact same garment laid flat product photo, "
                "identical pattern and color, full item visible, no human."
            )
            prompt_debug = {
                "enabled": False,
                "complexity": _estimate_visual_complexity(person_image),
                "garment_type": _detect_garment_type_from_text(request.person_description),
                "dominant_colors": [],
                "manual_description_used": True,
            }
        if design_match_mode and not strict_design_match:
            prompt = (
                prompt
                + " Keep exact print placement and motif scale, preserve original hue and saturation; no extra patterns, no hands, no face."
            )
        timings["auto_prompt_s"] = round(time.perf_counter() - step_start, 4)

        step_start = time.perf_counter()
        if requested_steps is not None:
            inference_steps = requested_steps
            adaptive_debug = {"enabled": False, "reason": "request_override"}
        elif adaptive_steps_enabled:
            complexity = prompt_debug.get("complexity") or _estimate_visual_complexity(person_image)
            inference_steps, router_meta = _select_adaptive_steps(complexity)
            adaptive_debug = {"enabled": True, **router_meta, "complexity": complexity}
        else:
            inference_steps = INFERENCE_STEPS
            adaptive_debug = {"enabled": False, "reason": "global_default"}
        timings["adaptive_steps_s"] = round(time.perf_counter() - step_start, 4)

        if inference_steps < 1 or inference_steps > 100:
            raise HTTPException(status_code=400, detail="inference_steps must be between 1 and 100")

        full_width = new_w * 2

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        if ENABLE_EMPTY_CACHE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        output = pipe(
            prompt=prompt,
            image=canvas,
            mask_image=mask_image,
            strength=1.0,
            height=new_h,
            width=full_width,
            target_width=new_w,
            edit=False,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            output_type="latent",
            generator=torch.Generator(device=DEVICE).manual_seed(42),
        ).images
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["inference_s"] = round(time.perf_counter() - step_start, 4)

        step_start = time.perf_counter()
        latents = pipe._unpack_latents(output, new_h, full_width, pipe.vae_scale_factor)
        latents = latents[:, :, :, : new_w // pipe.vae_scale_factor]
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        timings["latent_unpack_s"] = round(time.perf_counter() - step_start, 4)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        pipe.vae.to(dtype=torch.float32)
        try:
            decoded = pipe.vae.decode(latents.to(dtype=torch.float32), return_dict=False)[0].detach()
        finally:
            pipe.vae.to(dtype=TORCH_DTYPE)
            if ENABLE_EMPTY_CACHE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["vae_decode_s"] = round(time.perf_counter() - step_start, 4)

        step_start = time.perf_counter()
        result_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        if ENABLE_EMPTY_CACHE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        timings["postprocess_s"] = round(time.perf_counter() - step_start, 4)

        if ENABLE_POSTPROCESS:
            step_start = time.perf_counter()
            result_image = _apply_postprocess(result_image, person_image)
            timings["postprocess_enhance_s"] = round(time.perf_counter() - step_start, 4)

        if region_upscale_enabled:
            step_start = time.perf_counter()
            upscale_garment_type = prompt_debug.get("garment_type", conditioning_garment_type)
            result_image, region_upscale_debug = _enhance_region_detail(
                result_image,
                garment_type_hint=upscale_garment_type,
                region_mask=conditioning_mask,
            )
            timings["region_upscale_s"] = round(time.perf_counter() - step_start, 4)
        else:
            region_upscale_debug = {"enabled": False, "reason": "disabled"}

        step_start = time.perf_counter()
        image_buffer = io.BytesIO()
        save_kwargs = {}
        if OUTPUT_IMAGE_FORMAT == "JPEG":
            save_kwargs["quality"] = OUTPUT_IMAGE_QUALITY
            save_kwargs["optimize"] = True
        elif OUTPUT_IMAGE_FORMAT == "WEBP":
            save_kwargs["quality"] = OUTPUT_IMAGE_QUALITY
            save_kwargs["method"] = 6
        result_image.save(image_buffer, format=OUTPUT_IMAGE_FORMAT, **save_kwargs)
        output_bytes = image_buffer.getvalue()
        ext_map = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
        content_type_map = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
        output_url = upload_to_azure(
            output_bytes,
            extension=ext_map.get(OUTPUT_IMAGE_FORMAT, "png"),
            content_type=content_type_map.get(OUTPUT_IMAGE_FORMAT, "image/png"),
        )
        timings["upload_s"] = round(time.perf_counter() - step_start, 4)
        timings["upload_mb"] = round(len(output_bytes) / (1024 * 1024), 4)

        timings["total_s"] = round(time.perf_counter() - overall_start, 4)

        return {
            "status": "success",
            "output_url": output_url,
            "metrics": {
                "request_settings": {
                    "inference_steps": inference_steps,
                    "inference_steps_requested": requested_steps,
                    "max_image_size": max_image_size,
                    "guidance_scale": guidance_scale,
                    "auto_prompt_enabled": auto_prompt_enabled,
                    "adaptive_steps_enabled": adaptive_steps_enabled,
                    "design_match_mode": design_match_mode,
                    "strict_design_match": strict_design_match,
                    "condition_use_garment_isolation": condition_use_garment_isolation,
                    "region_upscale_enabled": region_upscale_enabled,
                    "realesrgan_enabled": ENABLE_REALESRGAN,
                    "force_auto_prompt": FORCE_AUTO_PROMPT,
                    "force_image_reference_prompt": FORCE_IMAGE_REFERENCE_PROMPT,
                    "force_garment_isolation": FORCE_GARMENT_ISOLATION,
                    "design_match_enforcements": {
                        "forced_auto_prompt": auto_prompt_forced,
                        "forced_condition_use_garment_isolation": isolation_forced,
                        "forced_design_match_mode": design_match_forced,
                        "forced_strict_design_match": strict_reference_forced,
                        "manual_description_ignored": manual_description_ignored,
                    },
                },
                "image_size": {
                    "source": {"width": original_w, "height": original_h},
                    "inference": {"width": new_w, "height": new_h},
                },
                "auto_prompt": {
                    "prompt_preview": prompt[:280],
                    "debug": prompt_debug,
                },
                "conditioning": conditioning_debug,
                "adaptive_steps": adaptive_debug,
                "region_upscale": region_upscale_debug,
                "timings": timings,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Try-off API error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
