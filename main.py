import io
import logging
import os
import time
import inspect
import colorsys
import re
import uuid
import json
import base64
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from starlette.background import BackgroundTask

from azure_utils import upload_to_azure
from src.api_features.category_mapping import (
    BANNED_CLOTHING,
    CLOTHING_STYLES,
    STYLE_TO_CATEGORY_KEYS,
    allowed_primary_keys_for_garment_type as _allowed_primary_keys_for_garment_type,
    is_style_compatible_with_garment_type as _is_style_compatible_with_garment_type,
    normalize_item_garment_type as _normalize_item_garment_type,
    style_category_keys as _style_category_keys,
    wardrobe_category_from_garment_type as _wardrobe_category_from_garment_type,
)
from src.api_features.image_io import (
    download_image as _download_image_impl,
    save_local_analyze_image as _save_local_analyze_image_impl,
)
from src.api_features.response_payloads import (
    build_error_payload as _build_error_payload,
    build_multipart_parts as _build_multipart_parts,
    build_success_payload as _build_success_payload,
    json_response as _json_response,
    multipart_form_response as _multipart_form_response,
)
from src.api_features.schemas import (
    DetectGarmentsRequest,
    ExtractClothRequest,
    TryOffRequest,
    WardrobeFlowRequest,
)
from src.api_features.security import verify_bearer_token as _verify_bearer_token_impl
from src.api_features.storage_sync import (
    download_binary_from_url as _download_binary_from_url,
    trigger_background_wardrobe_sync as _trigger_background_wardrobe_sync_impl,
    upload_image_to_azure_container as _upload_image_to_azure_container_impl,
)
from src.api_features.upload_validation import (
    check_blur_score as _check_blur_score_impl,
    validate_image_upload as _validate_image_upload_impl,
)
from src.api_features.flow_handlers import (
    handle_analyze_multipart,
    handle_detect_garments,
    handle_extract_cloth,
    handle_process_image,
    handle_wardrobe_flow,
)
from src.api_features.vton_fallback import (
    VtonFallbackConfig,
    VtonFallbackDeps,
    map_extract_type_to_vton_category as _map_extract_type_to_vton_category_impl,
    maybe_execute_vton_fallback as _maybe_execute_vton_fallback_impl,
    run_occlusion_prefilter as _run_occlusion_prefilter_impl,
    run_vton_cloth_only_fallback as _run_vton_cloth_only_fallback_impl,
)
from src.api_features.yolo_cropper import (
    YoloCropperConfig,
    YoloCropperDeps,
    build_component_preview_asset as _build_component_preview_asset_impl,
    build_isolated_preview_image as _build_isolated_preview_image_impl,
    build_yolo_item_breakdown_from_image as _build_yolo_item_breakdown_from_image_impl,
)
from src.garment_detector import detect_garment_instances
from src.multi_garment_section import crop_with_padding, extract_component_sections, image_to_base64_png

try:
    import jwt
except ImportError:  # pragma: no cover
    jwt = None

try:
    import cv2
except ImportError:
    cv2 = None

# Note: src.pipeline_tryon must be present in the project root
IMPORT_ERROR = None
try:
    from diffusers import AutoencoderKL, FluxTransformer2DModel
    from src.pipeline_tryon import FluxTryonPipeline
except ImportError as exc:
    FluxTryonPipeline = None
    FluxTransformer2DModel = None
    AutoencoderKL = None
    IMPORT_ERROR = exc

try:
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
except ImportError:
    AutoModelForSemanticSegmentation = None
    SegformerImageProcessor = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None

try:
    # torchvision>=0.19 moved rgb_to_grayscale off functional_tensor; basicsr still imports old path.
    import sys
    import types
    from torchvision.transforms.functional import rgb_to_grayscale as _rgb_to_grayscale

    _tv_ft_module = types.ModuleType("torchvision.transforms.functional_tensor")
    _tv_ft_module.rgb_to_grayscale = _rgb_to_grayscale
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _tv_ft_module)
except Exception:
    pass

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    RealESRGANer = None
    RRDBNet = None

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("any2any-tryoff")

app = FastAPI(title="Any2Any TryOff API")

# Torch 2.4 does not accept `enable_gqa`; newer stacks may pass it.
try:
    _sdpa_params = inspect.signature(F.scaled_dot_product_attention).parameters
except (TypeError, ValueError):
    _sdpa_params = {}

if "enable_gqa" not in _sdpa_params:
    _orig_sdpa = F.scaled_dot_product_attention

    def _sdpa_compat(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _sdpa_compat

pipe = None
pipe_lock = Lock()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16

MODEL_PATH = os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev")
ANY2ANY_LORA_REPO = os.getenv("ANY2ANY_LORA_REPO", "loooooong/Any2anyTryon")
ANY2ANY_LORA_WEIGHT = os.getenv("ANY2ANY_LORA_WEIGHT", "dev_lora_any2any_multi.safetensors")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
INFERENCE_STEPS = int(os.getenv("INFERENCE_STEPS", "4"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "0.0"))
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
ENABLE_ATTENTION_SLICING = os.getenv("ENABLE_ATTENTION_SLICING", "true").lower() == "true"
ENABLE_VAE_SLICING = os.getenv("ENABLE_VAE_SLICING", "true").lower() == "true"
ENABLE_VAE_TILING = os.getenv("ENABLE_VAE_TILING", "true").lower() == "true"
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"
ENABLE_EMPTY_CACHE = os.getenv("ENABLE_EMPTY_CACHE", "true").lower() == "true"
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
OUTPUT_IMAGE_FORMAT = os.getenv("OUTPUT_IMAGE_FORMAT", "PNG").upper()
OUTPUT_IMAGE_QUALITY = int(os.getenv("OUTPUT_IMAGE_QUALITY", "95"))
ENABLE_POSTPROCESS = os.getenv("ENABLE_POSTPROCESS", "false").lower() == "true"
POSTPROCESS_COLOR_MATCH = float(os.getenv("POSTPROCESS_COLOR_MATCH", "0.15"))
POSTPROCESS_CONTRAST = float(os.getenv("POSTPROCESS_CONTRAST", "1.02"))
POSTPROCESS_SHARPNESS = float(os.getenv("POSTPROCESS_SHARPNESS", "1.05"))
POSTPROCESS_UNSHARP_RADIUS = float(os.getenv("POSTPROCESS_UNSHARP_RADIUS", "1.0"))
POSTPROCESS_UNSHARP_PERCENT = int(os.getenv("POSTPROCESS_UNSHARP_PERCENT", "110"))
POSTPROCESS_UNSHARP_THRESHOLD = int(os.getenv("POSTPROCESS_UNSHARP_THRESHOLD", "2"))
PARSING_MODEL_PATH = os.getenv("PARSING_MODEL_PATH", "mattmdjaga/segformer_b2_clothes")
PARSING_DEVICE = os.getenv("PARSING_DEVICE", "cpu").lower()
PARSING_TORCH_DTYPE = os.getenv("PARSING_TORCH_DTYPE", "auto").lower()
ENABLE_MASK_REFINEMENT = os.getenv("ENABLE_MASK_REFINEMENT", "true").lower() == "true"
MASK_KEEP_COMPONENTS = int(os.getenv("MASK_KEEP_COMPONENTS", "2"))
MASK_MIN_COMPONENT_AREA_RATIO = float(os.getenv("MASK_MIN_COMPONENT_AREA_RATIO", "0.001"))
MASK_EDGE_FEATHER_PX = int(os.getenv("MASK_EDGE_FEATHER_PX", "2"))
MASK_EDGE_MIN_ALPHA = int(os.getenv("MASK_EDGE_MIN_ALPHA", "10"))
DRESS_SECOND_COMPONENT_ENABLE = os.getenv("DRESS_SECOND_COMPONENT_ENABLE", "true").lower() == "true"
DRESS_SECOND_COMPONENT_MIN_MAIN_AREA_RATIO = float(os.getenv("DRESS_SECOND_COMPONENT_MIN_MAIN_AREA_RATIO", "0.01"))
DRESS_SECOND_COMPONENT_MAX_GAP_PX = int(os.getenv("DRESS_SECOND_COMPONENT_MAX_GAP_PX", "256"))
DRESS_SECOND_COMPONENT_MIN_X_OVERLAP = float(os.getenv("DRESS_SECOND_COMPONENT_MIN_X_OVERLAP", "0.15"))
LOAD_TRYON_MODEL_ON_STARTUP = os.getenv("LOAD_TRYON_MODEL_ON_STARTUP", "false").lower() == "true"
DETECT_USE_YOLO = os.getenv("DETECT_USE_YOLO", "true").lower() == "true"
DETECT_YOLO_MIN_CONF = float(os.getenv("DETECT_YOLO_MIN_CONF", "0.25"))
DETECT_YOLO_IOU = float(os.getenv("DETECT_YOLO_IOU", "0.5"))
DETECT_YOLO_MIN_AREA_RATIO = float(os.getenv("DETECT_YOLO_MIN_AREA_RATIO", "0.002"))
DETECT_MIN_ITEM_PIXELS = int(os.getenv("DETECT_MIN_ITEM_PIXELS", "0"))
DETECT_MIN_ITEM_WIDTH_PX = int(os.getenv("DETECT_MIN_ITEM_WIDTH_PX", "0"))
DETECT_MIN_ITEM_HEIGHT_PX = int(os.getenv("DETECT_MIN_ITEM_HEIGHT_PX", "0"))
DETECT_MIN_GARMENT_OVERLAP_RATIO = float(os.getenv("DETECT_MIN_GARMENT_OVERLAP_RATIO", "0.25"))
DETECT_ENABLE_WAIST_SPLIT = os.getenv("DETECT_ENABLE_WAIST_SPLIT", "true").lower() == "true"
DETECT_FRAGMENT_SUPPRESS_ENABLED = os.getenv("DETECT_FRAGMENT_SUPPRESS_ENABLED", "true").lower() == "true"
DETECT_FRAGMENT_SECONDARY_MAX_RATIO = float(os.getenv("DETECT_FRAGMENT_SECONDARY_MAX_RATIO", "0.20"))
DETECT_FRAGMENT_DOMINANCE_MIN_RATIO = float(os.getenv("DETECT_FRAGMENT_DOMINANCE_MIN_RATIO", "0.82"))
DETECT_SINGLE_PIECE_MERGE_ENABLED = os.getenv("DETECT_SINGLE_PIECE_MERGE_ENABLED", "true").lower() == "true"
DETECT_SINGLE_PIECE_DRESS_HINT_MIN_RATIO = float(os.getenv("DETECT_SINGLE_PIECE_DRESS_HINT_MIN_RATIO", "0.90"))
DETECT_SINGLE_PIECE_MIN_X_OVERLAP = float(os.getenv("DETECT_SINGLE_PIECE_MIN_X_OVERLAP", "0.45"))
DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_PX = int(os.getenv("DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_PX", "64"))
DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_RATIO = float(os.getenv("DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_RATIO", "0.08"))
DETECT_PARSING_TYPE_OVERRIDE_MIN_SCORE = float(os.getenv("DETECT_PARSING_TYPE_OVERRIDE_MIN_SCORE", "0.35"))
EXTRACT_OCCLUSION_PREFILTER_ENABLED = os.getenv("EXTRACT_OCCLUSION_PREFILTER_ENABLED", "true").lower() == "true"
EXTRACT_OCCLUSION_PROXY_THRESHOLD = float(os.getenv("EXTRACT_OCCLUSION_PROXY_THRESHOLD", "0.24"))
EXTRACT_PREFILTER_YOLO_MIN_CONF = float(os.getenv("EXTRACT_PREFILTER_YOLO_MIN_CONF", "0.2"))
EXTRACT_PREFILTER_YOLO_IOU = float(os.getenv("EXTRACT_PREFILTER_YOLO_IOU", "0.5"))
EXTRACT_PREFILTER_YOLO_MIN_AREA_RATIO = float(os.getenv("EXTRACT_PREFILTER_YOLO_MIN_AREA_RATIO", "0.002"))
EXTRACT_PREFILTER_MAX_ITEMS = int(os.getenv("EXTRACT_PREFILTER_MAX_ITEMS", "8"))
EXTRACT_VTON_FALLBACK_ENABLED = os.getenv("EXTRACT_VTON_FALLBACK_ENABLED", "true").lower() == "true"
EXTRACT_VTON_CLOTH_ONLY_ENDPOINT = os.getenv("EXTRACT_VTON_CLOTH_ONLY_ENDPOINT", "http://127.0.0.1:8004/vton-v15/cloth-only")
EXTRACT_VTON_TIMEOUT_S = int(os.getenv("EXTRACT_VTON_TIMEOUT_S", "90"))
EXTRACT_VTON_QUALITY_PRESET = os.getenv("EXTRACT_VTON_QUALITY_PRESET", "balanced")
EXTRACT_VTON_TIMESTEPS = int(os.getenv("EXTRACT_VTON_TIMESTEPS", "20"))
EXTRACT_VTON_GUIDANCE_SCALE = float(os.getenv("EXTRACT_VTON_GUIDANCE_SCALE", "2.2"))
EXTRACT_VTON_SEGMENTATION_FREE = os.getenv("EXTRACT_VTON_SEGMENTATION_FREE", "false").lower() == "true"
EXTRACT_VTON_CUTOUT_FEATHER_PX = int(os.getenv("EXTRACT_VTON_CUTOUT_FEATHER_PX", "1"))
EXTRACT_VTON_ZOOM_PADDING_RATIO = float(os.getenv("EXTRACT_VTON_ZOOM_PADDING_RATIO", "0.12"))
EXTRACT_VTON_UPSCALE_ENABLED = os.getenv("EXTRACT_VTON_UPSCALE_ENABLED", "false").lower() == "true"
EXTRACT_VTON_UPSCALE_FACTOR = int(os.getenv("EXTRACT_VTON_UPSCALE_FACTOR", "2"))
EXTRACT_VTON_FAST_MODE = os.getenv("EXTRACT_VTON_FAST_MODE", "true").lower() == "true"
EXTRACT_VTON_FAST_TIMESTEPS = int(os.getenv("EXTRACT_VTON_FAST_TIMESTEPS", "20"))
EXTRACT_VTON_FAST_GUIDANCE_SCALE = float(os.getenv("EXTRACT_VTON_FAST_GUIDANCE_SCALE", "2.2"))
EXTRACT_VTON_FAST_DISABLE_UPSCALE = os.getenv("EXTRACT_VTON_FAST_DISABLE_UPSCALE", "true").lower() == "true"
EXTRACT_VTON_USE_SHOWROOM_PERSON = os.getenv("EXTRACT_VTON_USE_SHOWROOM_PERSON", "true").lower() == "true"
EXTRACT_VTON_SHOWROOM_PERSON_IMAGE_URL = os.getenv(
    "EXTRACT_VTON_SHOWROOM_PERSON_IMAGE_URL",
    "file:///workspace/Any2anyTryon/inputs/fashn_showroom_model.png",
).strip()
EXTRACT_PARSER_MIN_COVERAGE_RATIO = float(os.getenv("EXTRACT_PARSER_MIN_COVERAGE_RATIO", "0.08"))
EXTRACT_DISABLE_FINAL_CROP = os.getenv("EXTRACT_DISABLE_FINAL_CROP", "true").lower() == "true"
AUTO_PROMPT_ENABLED = os.getenv("AUTO_PROMPT_ENABLED", "true").lower() == "true"
AUTO_PROMPT_USE_PARSING = os.getenv("AUTO_PROMPT_USE_PARSING", "true").lower() == "true"
AUTO_PROMPT_MAX_COLORS = int(os.getenv("AUTO_PROMPT_MAX_COLORS", "3"))
FORCE_AUTO_PROMPT = os.getenv("FORCE_AUTO_PROMPT", "true").lower() == "true"
FORCE_IMAGE_REFERENCE_PROMPT = os.getenv("FORCE_IMAGE_REFERENCE_PROMPT", "true").lower() == "true"
ADAPTIVE_STEPS_ENABLED = os.getenv("ADAPTIVE_STEPS_ENABLED", "true").lower() == "true"
ADAPTIVE_STEPS_LOW = int(os.getenv("ADAPTIVE_STEPS_LOW", "24"))
ADAPTIVE_STEPS_HIGH = int(os.getenv("ADAPTIVE_STEPS_HIGH", "30"))
ADAPTIVE_COMPLEXITY_THRESHOLD = float(os.getenv("ADAPTIVE_COMPLEXITY_THRESHOLD", "0.42"))
DESIGN_MATCH_MODE = os.getenv("DESIGN_MATCH_MODE", "false").lower() == "true"
STRICT_DESIGN_MATCH = os.getenv("STRICT_DESIGN_MATCH", "true").lower() == "true"
CONDITION_USE_GARMENT_ISOLATION = os.getenv("CONDITION_USE_GARMENT_ISOLATION", "false").lower() == "true"
FORCE_GARMENT_ISOLATION = os.getenv("FORCE_GARMENT_ISOLATION", "true").lower() == "true"
CONDITION_MASK_CLOSE_KERNEL = int(os.getenv("CONDITION_MASK_CLOSE_KERNEL", "13"))
CONDITION_MASK_MIN_AREA_RATIO = float(os.getenv("CONDITION_MASK_MIN_AREA_RATIO", "0.01"))
ENABLE_REGION_UPSCALE = os.getenv("ENABLE_REGION_UPSCALE", "true").lower() == "true"
REGION_UPSCALE_FACTOR = float(os.getenv("REGION_UPSCALE_FACTOR", "2.0"))
REGION_UPSCALE_BG_THRESHOLD = int(os.getenv("REGION_UPSCALE_BG_THRESHOLD", "12"))
REGION_UPSCALE_MIN_PIXELS = int(os.getenv("REGION_UPSCALE_MIN_PIXELS", "1200"))
REGION_UPSCALE_SHARPNESS = float(os.getenv("REGION_UPSCALE_SHARPNESS", "1.08"))
REGION_UPSCALE_PARSE_MASK = os.getenv("REGION_UPSCALE_PARSE_MASK", "false").lower() == "true"
ENABLE_REALESRGAN = os.getenv("ENABLE_REALESRGAN", "true").lower() == "true"
REALESRGAN_MODEL_NAME = os.getenv("REALESRGAN_MODEL_NAME", "RealESRGAN_x2plus.pth")
REALESRGAN_MODEL_DIR = os.getenv("REALESRGAN_MODEL_DIR", "./models/upscalers")
REALESRGAN_MODEL_URL = os.getenv(
    "REALESRGAN_MODEL_URL",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
)
REALESRGAN_TILE = int(os.getenv("REALESRGAN_TILE", "0"))
REALESRGAN_TILE_PAD = int(os.getenv("REALESRGAN_TILE_PAD", "10"))
REALESRGAN_PRE_PAD = int(os.getenv("REALESRGAN_PRE_PAD", "0"))
REALESRGAN_OUTSCALE = float(os.getenv("REALESRGAN_OUTSCALE", "2.0"))
REALESRGAN_PRESERVE_SIZE = os.getenv("REALESRGAN_PRESERVE_SIZE", "true").lower() == "true"

PARSING_LABELS = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "glove": 3,
    "sunglasses": 4,
    "upper": 5,
    "dress": 6,
    "coat": 7,
    "socks": 8,
    "pants": 9,
    "jumpsuit": 10,
    "scarf": 11,
    "skirt": 12,
    "face": 13,
    "left_arm": 14,
    "right_arm": 15,
    "left_leg": 16,
    "right_leg": 17,
    "left_shoe": 18,
    "right_shoe": 19,
}
GARMENT_LABEL_MAP = {
    "dress": {PARSING_LABELS["dress"], PARSING_LABELS["jumpsuit"]},
    "top": {PARSING_LABELS["upper"], PARSING_LABELS["coat"], PARSING_LABELS["scarf"]},
    "bottom": {PARSING_LABELS["pants"], PARSING_LABELS["skirt"]},
    "outer": {PARSING_LABELS["coat"], PARSING_LABELS["scarf"]},
    "all": {
        PARSING_LABELS["upper"],
        PARSING_LABELS["dress"],
        PARSING_LABELS["coat"],
        PARSING_LABELS["pants"],
        PARSING_LABELS["jumpsuit"],
        PARSING_LABELS["scarf"],
        PARSING_LABELS["skirt"],
    },
}
OCCLUDER_LABELS = {
    PARSING_LABELS["left_arm"],
    PARSING_LABELS["right_arm"],
    PARSING_LABELS["glove"],
    PARSING_LABELS["hair"],
    PARSING_LABELS["face"],
    PARSING_LABELS["sunglasses"],
    PARSING_LABELS["hat"],
}

GARMENT_LABEL_KEYWORDS = {
    "dress": ("dress", "gown", "jumpsuit", "romper"),
    "top": (
        "upper",
        "top",
        "shirt",
        "blouse",
        "tee",
        "tshirt",
        "t shirt",
        "sweater",
        "hoodie",
        "coat",
        "jacket",
        "cardigan",
        "scarf",
        "vest",
        "outerwear",
        "outwear",
    ),
    "bottom": ("pant", "trouser", "jean", "legging", "short", "skirt", "bottom"),
    "outer": ("coat", "jacket", "cardigan", "blazer", "outerwear", "outwear", "scarf", "shawl", "cloak"),
}
OCCLUDER_LABEL_KEYWORDS = (
    "arm",
    "hand",
    "glove",
    "hair",
    "face",
    "head",
    "hat",
    "sunglass",
    "sunglasses",
    "bag",
    "purse",
    "backpack",
    "belt",
    "watch",
    "bracelet",
    "earring",
    "necklace",
    "jewelry",
)

parsing_processor = None
parsing_model = None
parsing_lock = Lock()
realesrgan_upsampler = None
realesrgan_lock = Lock()
clip_model = None
clip_processor = None
clip_lock = Lock()
clip_text_inputs = None

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Referer": "https://www.google.com/",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(3 * 1024 * 1024)))
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
ANALYZE_BLUR_MIN = float(os.getenv("ANALYZE_BLUR_MIN", "25.0"))
ANALYZE_YOLO_MAX_ITEMS = int(os.getenv("ANALYZE_YOLO_MAX_ITEMS", "5"))
ANALYZE_YOLO_MIN_CONF = float(os.getenv("ANALYZE_YOLO_MIN_CONF", str(DETECT_YOLO_MIN_CONF)))
ANALYZE_YOLO_IOU = float(os.getenv("ANALYZE_YOLO_IOU", str(DETECT_YOLO_IOU)))
ANALYZE_YOLO_MIN_AREA_RATIO = float(os.getenv("ANALYZE_YOLO_MIN_AREA_RATIO", str(DETECT_YOLO_MIN_AREA_RATIO)))
ANALYZE_USE_HUMAN_PARSER = os.getenv("ANALYZE_USE_HUMAN_PARSER", "false").lower() == "true"
ANALYZE_MIN_COMPONENT_AREA_RATIO = float(os.getenv("ANALYZE_MIN_COMPONENT_AREA_RATIO", "0.005"))
ANALYZE_MIN_ITEM_PIXELS = int(os.getenv("ANALYZE_MIN_ITEM_PIXELS", str(max(1200, DETECT_MIN_ITEM_PIXELS))))
ANALYZE_CROP_PADDING_PX = int(os.getenv("ANALYZE_CROP_PADDING_PX", "12"))
ANALYZE_OCCLUSION_THRESHOLD = float(os.getenv("ANALYZE_OCCLUSION_THRESHOLD", "0.10"))
ANALYZE_FORCE_VTON_FOR_SINGLE_PIECE_DRESS = os.getenv(
    "ANALYZE_FORCE_VTON_FOR_SINGLE_PIECE_DRESS",
    "true",
).lower() == "true"
ANALYZE_ALWAYS_USE_VTON = os.getenv("ANALYZE_ALWAYS_USE_VTON", "true").lower() == "true"
ANALYZE_DRESS_VTON_OCCLUSION_THRESHOLD = float(os.getenv("ANALYZE_DRESS_VTON_OCCLUSION_THRESHOLD", "0.02"))
ANALYZE_SINGLE_ITEM_TYPE_FALLBACK = os.getenv("ANALYZE_SINGLE_ITEM_TYPE_FALLBACK", "true").lower() == "true"
ANALYZE_FORCE_SPLIT_CROP_INPUT = os.getenv("ANALYZE_FORCE_SPLIT_CROP_INPUT", "false").lower() == "true"
ANALYZE_LOW_CONFIDENCE_SPLIT_TO_MULTI = os.getenv("ANALYZE_LOW_CONFIDENCE_SPLIT_TO_MULTI", "true").lower() == "true"
ANALYZE_USE_ISOLATED_TOP_CROP = os.getenv("ANALYZE_USE_ISOLATED_TOP_CROP", "false").lower() == "true"
ANALYZE_INCLUDE_BASE64 = os.getenv("ANALYZE_INCLUDE_BASE64", "false").lower() == "true"
ANALYZE_INCLUDE_UPLOAD_URL = os.getenv("ANALYZE_INCLUDE_UPLOAD_URL", "false").lower() == "true"
ANALYZE_INCLUDE_ITEM_PREVIEW_ON_SUCCESS = os.getenv("ANALYZE_INCLUDE_ITEM_PREVIEW_ON_SUCCESS", "false").lower() == "true"
ANALYZE_SAVE_LOCAL_ITEMS = os.getenv("ANALYZE_SAVE_LOCAL_ITEMS", "true").lower() == "true"
ANALYZE_MULTI_ITEM_ISOLATE_PREVIEW = os.getenv("ANALYZE_MULTI_ITEM_ISOLATE_PREVIEW", "true").lower() == "true"
ANALYZE_MULTI_ITEM_PREVIEW_KEEP_OCCLUDERS = os.getenv("ANALYZE_MULTI_ITEM_PREVIEW_KEEP_OCCLUDERS", "true").lower() == "true"
ANALYZE_MULTI_ITEM_ISOLATE_MIN_MASK_RATIO = float(os.getenv("ANALYZE_MULTI_ITEM_ISOLATE_MIN_MASK_RATIO", "0.35"))
ANALYZE_MULTI_ITEM_ISOLATE_TRIM_PADDING_PX = int(os.getenv("ANALYZE_MULTI_ITEM_ISOLATE_TRIM_PADDING_PX", "2"))
ANALYZE_LOCAL_ITEMS_DIR = os.getenv("ANALYZE_LOCAL_ITEMS_DIR", str(Path(__file__).resolve().parent / "outputs" / "analyze_items"))
WARDROBE_PROGRESS_API_BASE_URL = (
    os.getenv("WARDROBE_PROGRESS_API_BASE_URL", "").strip()
    or os.getenv("API_BASE_URL", "").strip()
    or os.getenv("WARDROBE_API_BASE_URL", "").strip()
)
ENABLE_WARDROBE_PROGRESS_SYNC = os.getenv("ENABLE_WARDROBE_PROGRESS_SYNC", "true").lower() == "true"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_INPUT_CONTAINER = os.getenv("AZURE_STORAGE_INPUT_CONTAINER", "wardrobe-inputs")
AZURE_STORAGE_OUTPUT_CONTAINER = os.getenv("AZURE_STORAGE_OUTPUT_CONTAINER", "wardrobe-outputs")
JWT_ACCESS_SECRET = os.getenv("JWT_ACCESS_SECRET", "")

_background_executor = ThreadPoolExecutor(max_workers=max(2, int(os.getenv("BACKGROUND_WORKERS", "4"))))


def _verify_bearer_token(authorization: Optional[str]) -> Dict[str, object]:
    return _verify_bearer_token_impl(
        authorization,
        jwt_access_secret=JWT_ACCESS_SECRET,
        jwt_module=jwt,
        logger=logger,
    )


def _validate_image_upload(file: UploadFile, image_bytes: bytes) -> Tuple[Image.Image, str, str]:
    return _validate_image_upload_impl(
        file,
        image_bytes,
        max_upload_bytes=MAX_UPLOAD_BYTES,
        allowed_image_mime_types=ALLOWED_IMAGE_MIME_TYPES,
        allowed_image_formats=ALLOWED_IMAGE_FORMATS,
    )


def _check_blur_score(img: Image.Image) -> float:
    return _check_blur_score_impl(img, cv2)


def _upload_image_to_azure_container(
    image_bytes: bytes,
    *,
    content_type: str,
    container_name: str,
    blob_name: str,
) -> str:
    return _upload_image_to_azure_container_impl(
        image_bytes,
        content_type=content_type,
        container_name=container_name,
        blob_name=blob_name,
        connection_string=AZURE_STORAGE_CONNECTION_STRING,
    )


def _trigger_background_wardrobe_sync(**kwargs) -> None:
    _trigger_background_wardrobe_sync_impl(
        background_executor=_background_executor,
        logger=logger,
        **kwargs,
    )


def _download_image(url: str):
    return _download_image_impl(url, REQUEST_HEADERS, timeout_s=45)


def _save_local_analyze_image(image_bytes: bytes, *, prefix: str, ext: str = "png") -> Optional[str]:
    return _save_local_analyze_image_impl(
        image_bytes,
        prefix=prefix,
        ext=ext,
        enabled=ANALYZE_SAVE_LOCAL_ITEMS,
        output_dir=ANALYZE_LOCAL_ITEMS_DIR,
        logger=logger,
    )


def _yolo_cropper_config() -> YoloCropperConfig:
    return YoloCropperConfig(
        analyze_use_human_parser=ANALYZE_USE_HUMAN_PARSER,
        analyze_min_component_area_ratio=ANALYZE_MIN_COMPONENT_AREA_RATIO,
        detect_min_item_width_px=DETECT_MIN_ITEM_WIDTH_PX,
        detect_min_item_height_px=DETECT_MIN_ITEM_HEIGHT_PX,
        detect_min_garment_overlap_ratio=DETECT_MIN_GARMENT_OVERLAP_RATIO,
        detect_use_yolo=DETECT_USE_YOLO,
        analyze_yolo_min_conf=ANALYZE_YOLO_MIN_CONF,
        analyze_yolo_min_area_ratio=ANALYZE_YOLO_MIN_AREA_RATIO,
        analyze_yolo_iou=ANALYZE_YOLO_IOU,
        detect_enable_waist_split=DETECT_ENABLE_WAIST_SPLIT,
        detect_parsing_type_override_min_score=DETECT_PARSING_TYPE_OVERRIDE_MIN_SCORE,
        analyze_multi_item_isolate_preview=ANALYZE_MULTI_ITEM_ISOLATE_PREVIEW,
        analyze_multi_item_preview_keep_occluders=ANALYZE_MULTI_ITEM_PREVIEW_KEEP_OCCLUDERS,
        analyze_multi_item_isolate_min_mask_ratio=ANALYZE_MULTI_ITEM_ISOLATE_MIN_MASK_RATIO,
        analyze_multi_item_isolate_trim_padding_px=ANALYZE_MULTI_ITEM_ISOLATE_TRIM_PADDING_PX,
    )


def _yolo_cropper_deps() -> YoloCropperDeps:
    return YoloCropperDeps(
        cv2_module=cv2,
        binary_open_fn=_binary_open,
        binary_close_fn=_binary_close,
        build_soft_alpha_fn=_build_soft_alpha,
        save_local_analyze_image_fn=_save_local_analyze_image,
        detect_garment_instances_fn=detect_garment_instances,
        run_human_parsing_fn=_run_human_parsing,
        resolve_label_ids_from_model_fn=_resolve_label_ids_from_model,
        refine_garment_mask_fn=_refine_garment_mask,
        build_parser_category_components_fn=_build_parser_category_components,
        build_parsing_category_masks_fn=_build_parsing_category_masks,
        max_component_overlap_ratio_fn=_max_component_overlap_ratio,
        attempt_waist_split_components_fn=_attempt_waist_split_components,
        extract_component_sections_fn=extract_component_sections,
        infer_component_garment_type_from_parsing_fn=_infer_component_garment_type_from_parsing,
        postprocess_single_piece_candidates_fn=_postprocess_single_piece_candidates,
        crop_with_padding_fn=crop_with_padding,
        estimate_occlusion_from_parsing_masks_fn=_estimate_occlusion_from_parsing_masks,
        run_clip_classification_fn=_run_clip_classification,
        normalize_item_garment_type_fn=_normalize_item_garment_type,
        wardrobe_category_from_garment_type_fn=_wardrobe_category_from_garment_type,
        upload_to_azure_fn=upload_to_azure,
    )


def _build_isolated_preview_image(
    crop_image: Image.Image,
    crop_mask: np.ndarray,
) -> Tuple[Image.Image, Dict[str, object]]:
    return _build_isolated_preview_image_impl(
        crop_image,
        crop_mask,
        config=_yolo_cropper_config(),
        deps=_yolo_cropper_deps(),
    )


def _build_component_preview_asset(
    *,
    crop_image: Image.Image,
    crop_mask: np.ndarray,
    rank: int,
) -> Dict[str, object]:
    return _build_component_preview_asset_impl(
        crop_image=crop_image,
        crop_mask=crop_mask,
        rank=rank,
        config=_yolo_cropper_config(),
        deps=_yolo_cropper_deps(),
        logger=logger,
    )


def _build_yolo_item_breakdown_from_image(
    *,
    source_image: Image.Image,
    max_items: int,
    crop_padding_px: int,
    min_item_pixels: int,
    include_upload_url: bool,
    include_base64: bool,
) -> Dict[str, object]:
    return _build_yolo_item_breakdown_from_image_impl(
        source_image=source_image,
        max_items=max_items,
        crop_padding_px=crop_padding_px,
        min_item_pixels=min_item_pixels,
        include_upload_url=include_upload_url,
        include_base64=include_base64,
        banned_clothing=BANNED_CLOTHING,
        config=_yolo_cropper_config(),
        deps=_yolo_cropper_deps(),
        logger=logger,
    )


async def _extract_cloth_with_fallback_types(
    *,
    selected_image_url: str,
    selected_type: str,
    preferred_type: str,
    crop_padding_px: int,
    use_occlusion_prefilter: Optional[bool],
    occlusion_proxy_threshold: Optional[float],
    enable_vton_fallback: Optional[bool],
    force_vton_fallback: bool,
    force_vton_only: bool = False,
) -> Tuple[Optional[Dict[str, object]], str, Optional[HTTPException]]:
    preferred = (preferred_type or "").lower().strip()
    selected = _normalize_item_garment_type(selected_type)
    extract_type = preferred if preferred in {"dress", "top", "bottom", "outer", "all"} else selected
    if extract_type == "all":
        extract_type = "dress" if selected == "all" else selected

    attempt_types: List[str] = [extract_type]
    if not force_vton_only:
        if selected and selected not in attempt_types:
            attempt_types.append(selected)
        if "all" not in attempt_types:
            attempt_types.append("all")
        if "dress" not in attempt_types:
            attempt_types.append("dress")

    last_error: Optional[HTTPException] = None
    for attempt_type in attempt_types:
        try:
            request = ExtractClothRequest(
                image_url=selected_image_url,
                garment_type=attempt_type,  # type: ignore[arg-type]
                crop_to_garment=True,
                include_occlusion_mask=False,
                crop_padding_px=max(0, int(crop_padding_px)),
                use_occlusion_prefilter=use_occlusion_prefilter,
                occlusion_proxy_threshold=occlusion_proxy_threshold,
                enable_vton_fallback=True if force_vton_only else enable_vton_fallback,
                force_vton_fallback=True if force_vton_only else force_vton_fallback,
            )
            result = await extract_cloth(request)
            if isinstance(result, dict) and result.get("status") == "success":
                if force_vton_only:
                    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                    extraction_path = str(metrics.get("extraction_path", "")).strip().lower()
                    if extraction_path != "vton_fallback":
                        last_error = HTTPException(
                            status_code=502,
                            detail="VTON extraction failed for requested type. Please retry.",
                        )
                        continue
                return result, attempt_type, None
        except HTTPException as exc:
            last_error = exc
            continue
    return None, extract_type, last_error


def _resize_for_inference(image: Image.Image, max_image_size: int) -> Image.Image:
    width, height = image.size
    scale = max_image_size / max(width, height)
    new_w = max(16, (int(width * scale) // 16) * 16)
    new_h = max(16, (int(height * scale) // 16) * 16)
    return image.resize((new_w, new_h))


def _garment_type_matches_requested(request_type: str, candidate_type: str) -> bool:
    req = (request_type or "all").lower()
    cand = (candidate_type or "").lower()
    if req == "all":
        return cand in {"top", "bottom", "dress", "outer"}
    if req == "top":
        return cand in {"top", "outer"}
    if req == "outer":
        return cand in {"outer", "top"}
    return cand == req


def _vton_fallback_config() -> VtonFallbackConfig:
    return VtonFallbackConfig(
        extract_prefilter_max_items=EXTRACT_PREFILTER_MAX_ITEMS,
        extract_prefilter_yolo_min_conf=EXTRACT_PREFILTER_YOLO_MIN_CONF,
        extract_prefilter_yolo_min_area_ratio=EXTRACT_PREFILTER_YOLO_MIN_AREA_RATIO,
        extract_prefilter_yolo_iou=EXTRACT_PREFILTER_YOLO_IOU,
        extract_vton_cloth_only_endpoint=EXTRACT_VTON_CLOTH_ONLY_ENDPOINT,
        extract_vton_timeout_s=EXTRACT_VTON_TIMEOUT_S,
        extract_vton_quality_preset=EXTRACT_VTON_QUALITY_PRESET,
        extract_vton_timesteps=EXTRACT_VTON_TIMESTEPS,
        extract_vton_guidance_scale=EXTRACT_VTON_GUIDANCE_SCALE,
        extract_vton_segmentation_free=EXTRACT_VTON_SEGMENTATION_FREE,
        extract_vton_cutout_feather_px=EXTRACT_VTON_CUTOUT_FEATHER_PX,
        extract_vton_zoom_padding_ratio=EXTRACT_VTON_ZOOM_PADDING_RATIO,
        extract_vton_upscale_enabled=EXTRACT_VTON_UPSCALE_ENABLED,
        extract_vton_upscale_factor=EXTRACT_VTON_UPSCALE_FACTOR,
        extract_vton_fast_mode=EXTRACT_VTON_FAST_MODE,
        extract_vton_fast_timesteps=EXTRACT_VTON_FAST_TIMESTEPS,
        extract_vton_fast_guidance_scale=EXTRACT_VTON_FAST_GUIDANCE_SCALE,
        extract_vton_fast_disable_upscale=EXTRACT_VTON_FAST_DISABLE_UPSCALE,
        extract_vton_use_showroom_person=EXTRACT_VTON_USE_SHOWROOM_PERSON,
        extract_vton_showroom_person_image_url=EXTRACT_VTON_SHOWROOM_PERSON_IMAGE_URL,
    )


def _vton_fallback_deps() -> VtonFallbackDeps:
    return VtonFallbackDeps(
        detect_garment_instances_fn=detect_garment_instances,
        garment_type_matches_requested_fn=_garment_type_matches_requested,
    )


def _map_extract_type_to_vton_category(garment_type: str) -> str:
    return _map_extract_type_to_vton_category_impl(garment_type)


def _run_occlusion_prefilter(
    source_image: Image.Image,
    garment_type: str,
    min_area_pixels: int,
) -> tuple:
    return _run_occlusion_prefilter_impl(
        source_image,
        garment_type,
        min_area_pixels,
        config=_vton_fallback_config(),
        deps=_vton_fallback_deps(),
    )


def _run_vton_cloth_only_fallback(
    image_url: str,
    garment_type: str,
) -> Dict[str, object]:
    return _run_vton_cloth_only_fallback_impl(
        image_url,
        garment_type,
        config=_vton_fallback_config(),
        logger=logger,
    )


def _maybe_execute_vton_fallback(
    *,
    request_image_url: str,
    request_garment_type: str,
    source_image: Image.Image,
    prefilter_debug: Dict[str, object],
    fallback_debug: Dict[str, object],
    timings: Dict[str, float],
    overall_start: float,
) -> Optional[Dict[str, object]]:
    return _maybe_execute_vton_fallback_impl(
        request_image_url=request_image_url,
        request_garment_type=request_garment_type,
        source_image=source_image,
        prefilter_debug=prefilter_debug,
        fallback_debug=fallback_debug,
        timings=timings,
        overall_start=overall_start,
        config=_vton_fallback_config(),
        logger=logger,
    )


def _build_parsing_category_masks(parsing: np.ndarray) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for category in ("top", "bottom", "dress", "outer"):
        labels, _, _ = _resolve_label_ids_from_model(category)
        masks[category] = np.isin(parsing, list(labels))
    return masks


def _infer_component_garment_type_from_parsing(
    component_mask: np.ndarray,
    category_masks: Dict[str, np.ndarray],
) -> tuple:
    mask = component_mask.astype(bool)
    area = int(mask.sum())
    if area <= 0:
        return "all", 0.0, {}

    overlap_by_type: Dict[str, float] = {}
    for category, category_mask in category_masks.items():
        overlap = int((mask & category_mask.astype(bool)).sum())
        overlap_by_type[category] = float(overlap / max(1, area))

    sorted_types = sorted(overlap_by_type.items(), key=lambda item: item[1], reverse=True)
    if not sorted_types or sorted_types[0][1] <= 0.05:
        return "all", 0.0, overlap_by_type
    best_type, best_score = sorted_types[0]
    return best_type, float(best_score), overlap_by_type


def _ensure_parsing_model_loaded():
    global parsing_model, parsing_processor
    if parsing_model is not None and parsing_processor is not None:
        return

    if AutoModelForSemanticSegmentation is None or SegformerImageProcessor is None:
        raise RuntimeError("Parsing dependencies are not available")

    with parsing_lock:
        if parsing_model is not None and parsing_processor is not None:
            return

        device = PARSING_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        parsing_dtype = torch.float32
        if device == "cuda":
            if PARSING_TORCH_DTYPE in ("auto", "float16", "fp16", "half"):
                parsing_dtype = torch.float16
            elif PARSING_TORCH_DTYPE in ("bfloat16", "bf16"):
                parsing_dtype = torch.bfloat16
            elif PARSING_TORCH_DTYPE in ("float32", "fp32"):
                parsing_dtype = torch.float32
            else:
                parsing_dtype = torch.float16

        logger.info("Loading parsing model: %s (device=%s)", PARSING_MODEL_PATH, device)
        token = os.getenv("HUGGING_FACE_KEY")
        parsing_processor = SegformerImageProcessor.from_pretrained(PARSING_MODEL_PATH, token=token)
        parsing_model = AutoModelForSemanticSegmentation.from_pretrained(
            PARSING_MODEL_PATH,
            token=token,
            torch_dtype=parsing_dtype,
        )
        parsing_model.to(device)
        parsing_model.eval()


def _binary_dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        out = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return out.astype(bool)
    t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
    d = F.max_pool2d(t, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (d[0, 0].cpu().numpy() > 0.5)


def _binary_erode(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        out = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return out.astype(bool)
    t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
    inv = 1.0 - t
    e = 1.0 - F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (e[0, 0].cpu().numpy() > 0.5)


def _binary_open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask, kernel_size), kernel_size)


def _binary_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, kernel_size), kernel_size)


def _run_human_parsing(image: Image.Image) -> np.ndarray:
    _ensure_parsing_model_loaded()
    device = next(parsing_model.parameters()).device
    dtype = next(parsing_model.parameters()).dtype
    inputs = parsing_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.inference_mode():
        logits = parsing_model(pixel_values).logits
    logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)
    return pred


def _get_clip_model():
    global clip_model, clip_processor, clip_text_inputs
    if clip_model is not None:
        return clip_model, clip_processor

    if CLIPModel is None or CLIPProcessor is None:
        return None, None

    with clip_lock:
        if clip_model is not None:
            return clip_model, clip_processor

        model_id = "openai/clip-vit-large-patch14"
        logger.info(f"📦 Loading CLIP identifier ({model_id})...")
        try:
            # We use half precision if on CUDA to save memory (total ~1.1GB vs 2.1GB)
            clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE)
            if DEVICE == "cuda":
                clip_model = clip_model.half()

            clip_processor = CLIPProcessor.from_pretrained(model_id)

            # Pre-tokenize labels to speed up inference
            clip_text_inputs = clip_processor(
                text=CLOTHING_STYLES,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            
            # If we used half() on model, ensure text inputs are also handled (though CLIP text is usually float32)
            clip_model.eval()
            logger.info("✅ CLIP loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ CLIP load failed: {e}")
            clip_model = False  # Mark as failed to avoid retries
            clip_processor = None
            clip_text_inputs = None

        return clip_model, clip_processor


def _run_clip_classification(image: Image.Image, garment_type: str = "all") -> Tuple[str, float]:
    model, processor = _get_clip_model()
    if not model or model is False:
        return "Unknown Style", 0.0

    try:
        rgb_image = image.convert("RGB")
        image_inputs = processor(images=rgb_image, return_tensors="pt").to(DEVICE)
        
        # Match model dtype (fp16 vs fp32)
        model_dtype = next(model.parameters()).dtype
        image_inputs["pixel_values"] = image_inputs["pixel_values"].to(dtype=model_dtype)

        with torch.inference_mode():
            outputs = model(
                input_ids=clip_text_inputs["input_ids"],
                attention_mask=clip_text_inputs["attention_mask"],
                pixel_values=image_inputs["pixel_values"]
            )
            probs = outputs.logits_per_image.softmax(dim=1)

        allowed_primary = _allowed_primary_keys_for_garment_type(garment_type)
        best_idx = probs.argmax(dim=1).item()
        best_conf = float(probs[0, best_idx].item())
        style_name = CLOTHING_STYLES[best_idx]
        if allowed_primary:
            sorted_idx = probs[0].argsort(descending=True).tolist()
            for idx in sorted_idx:
                candidate_style = CLOTHING_STYLES[int(idx)]
                keys = _style_category_keys(candidate_style)
                if not keys:
                    continue
                primary_key, _ = keys
                if primary_key in allowed_primary:
                    style_name = candidate_style
                    best_conf = float(probs[0, int(idx)].item())
                    break

        return style_name.title(), best_conf
    except Exception as e:
        logger.warning(f"⚠️ CLIP classification failed: {e}")
        return "Unknown Style", 0.0


def _normalize_label(label: str) -> str:
    normalized = name.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in normalized)
    return " ".join(normalized.split())


def _normalize_label_name(name: str) -> str:
    normalized = name.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in normalized)
    return " ".join(normalized.split())


def _label_matches_keyword(label: str, keyword: str) -> bool:
    keyword = _normalize_label_name(keyword)
    if not keyword:
        return False
    if " " in keyword:
        return keyword in label
    for token in label.split():
        if token == keyword or token.startswith(keyword):
            return True
    return False


def _resolve_label_ids_from_model(garment_type: str):
    raw_id2label = getattr(getattr(parsing_model, "config", None), "id2label", {}) or {}
    model_labels = {}
    for raw_idx, raw_name in raw_id2label.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        model_labels[idx] = _normalize_label_name(str(raw_name))

    if not model_labels:
        return GARMENT_LABEL_MAP[garment_type], OCCLUDER_LABELS, {"used_fallback": True, "reason": "missing_id2label"}

    def collect_ids(keywords):
        matched = set()
        for idx, label in model_labels.items():
            if any(_label_matches_keyword(label, kw) for kw in keywords):
                matched.add(idx)
        return matched

    per_type = {
        "dress": collect_ids(GARMENT_LABEL_KEYWORDS["dress"]),
        "top": collect_ids(GARMENT_LABEL_KEYWORDS["top"]),
        "bottom": collect_ids(GARMENT_LABEL_KEYWORDS["bottom"]),
        "outer": collect_ids(GARMENT_LABEL_KEYWORDS["outer"]),
    }
    per_type["all"] = set().union(*per_type.values())

    garment_ids = per_type.get(garment_type, set())
    occluder_ids = collect_ids(OCCLUDER_LABEL_KEYWORDS)

    used_fallback = False
    if not garment_ids:
        garment_ids = GARMENT_LABEL_MAP[garment_type]
        used_fallback = True
    if not occluder_ids:
        occluder_ids = OCCLUDER_LABELS
        used_fallback = True

    matched_labels = sorted(
        [{"id": idx, "name": model_labels.get(idx, "")} for idx in garment_ids],
        key=lambda item: item["id"],
    )
    matched_occluders = sorted(
        [{"id": idx, "name": model_labels.get(idx, "")} for idx in occluder_ids],
        key=lambda item: item["id"],
    )
    return garment_ids, occluder_ids, {
        "used_fallback": used_fallback,
        "model_labels_count": len(model_labels),
        "garment_labels": matched_labels,
        "occluder_labels": matched_occluders,
    }


def _crop_image_and_masks(
    image: Image.Image,
    primary_mask: np.ndarray,
    secondary_mask: np.ndarray,
    padding_px: int,
):
    ys, xs = np.where(primary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return image, primary_mask, secondary_mask

    x0 = max(int(xs.min()) - padding_px, 0)
    x1 = min(int(xs.max()) + padding_px + 1, image.width)
    y0 = max(int(ys.min()) - padding_px, 0)
    y1 = min(int(ys.max()) + padding_px + 1, image.height)

    cropped_image = image.crop((x0, y0, x1, y1))
    cropped_primary = primary_mask[y0:y1, x0:x1]
    cropped_secondary = secondary_mask[y0:y1, x0:x1]
    return cropped_image, cropped_primary, cropped_secondary


def _refine_garment_mask(mask: np.ndarray, garment_type: str):
    if not ENABLE_MASK_REFINEMENT:
        return mask.astype(bool), {"enabled": False}

    refined = mask.astype(bool)
    meta = {"enabled": True, "cv2_used": cv2 is not None}
    before_pixels = int(refined.sum())

    if cv2 is not None and before_pixels > 0:
        binary = refined.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            min_area = int(refined.size * max(MASK_MIN_COMPONENT_AREA_RATIO, 0.0))
            component_ids = list(range(1, num_labels))
            component_ids.sort(key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)

            kept = np.zeros_like(refined, dtype=bool)
            keep_budget = max(1, MASK_KEEP_COMPONENTS)
            kept_ids = []

            # Primary component (largest garment region)
            main_id = None
            main_area = 0
            main_bbox = None
            for idx in component_ids:
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                main_id = idx
                main_area = area
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                main_bbox = (x, y, x + w - 1, y + h - 1)
                kept |= (labels == idx)
                kept_ids.append(idx)
                break

            # Dress-specific optimization: allow one nearby lower component
            # to preserve split hems/skirts while still filtering background noise.
            if (
                main_id is not None
                and garment_type == "dress"
                and DRESS_SECOND_COMPONENT_ENABLE
                and keep_budget == 1
                and main_bbox is not None
            ):
                mx0, my0, mx1, my1 = main_bbox
                main_w = max(1, mx1 - mx0 + 1)
                for idx in component_ids:
                    if idx == main_id:
                        continue
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    if area < int(main_area * max(0.0, DRESS_SECOND_COMPONENT_MIN_MAIN_AREA_RATIO)):
                        continue

                    x = int(stats[idx, cv2.CC_STAT_LEFT])
                    y = int(stats[idx, cv2.CC_STAT_TOP])
                    w = int(stats[idx, cv2.CC_STAT_WIDTH])
                    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                    cx0, cy0, cx1, cy1 = x, y, x + w - 1, y + h - 1

                    x_overlap = max(0, min(mx1, cx1) - max(mx0, cx0) + 1)
                    x_overlap_ratio = x_overlap / main_w

                    if cy0 > my1:
                        v_gap = cy0 - my1
                    elif my0 > cy1:
                        v_gap = my0 - cy1
                    else:
                        v_gap = 0

                    if (
                        x_overlap_ratio >= max(0.0, DRESS_SECOND_COMPONENT_MIN_X_OVERLAP)
                        and v_gap <= max(0, DRESS_SECOND_COMPONENT_MAX_GAP_PX)
                    ):
                        kept |= (labels == idx)
                        kept_ids.append(idx)
                        meta["dress_secondary_component_added"] = True
                        meta["dress_secondary_component_area"] = area
                        break

            # Standard component budget (when >1) keeps top-N valid components.
            if len(kept_ids) < keep_budget:
                for idx in component_ids:
                    if idx in kept_ids:
                        continue
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    kept |= (labels == idx)
                    kept_ids.append(idx)
                    if len(kept_ids) >= keep_budget:
                        break

            if kept.any():
                refined = kept
                meta["kept_components"] = len(kept_ids)
                meta["total_components"] = num_labels - 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_u8 = refined.astype(np.uint8)
        refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        refined = refined_u8.astype(bool)
    else:
        refined = _binary_open(refined, 3)
        refined = _binary_close(refined, 3)

    after_pixels = int(refined.sum())
    meta["pixels_before"] = before_pixels
    meta["pixels_after"] = after_pixels
    meta["removed_pixels"] = max(before_pixels - after_pixels, 0)
    return refined, meta


def _build_soft_alpha(mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.uint8) * 255)
    feather_px = max(0, MASK_EDGE_FEATHER_PX)
    if feather_px <= 0:
        return alpha

    if cv2 is not None:
        kernel = feather_px * 2 + 1
        blurred = cv2.GaussianBlur(alpha, (kernel, kernel), sigmaX=max(0.8, feather_px * 0.8))
    else:
        blurred = np.asarray(Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=feather_px)))

    erode_kernel = max(1, feather_px * 2 + 1)
    inner = (_binary_erode(mask, erode_kernel).astype(np.uint8) * 255)
    soft_alpha = np.maximum(blurred, inner).astype(np.uint8)
    min_alpha = max(0, min(255, MASK_EDGE_MIN_ALPHA))
    soft_alpha[soft_alpha < min_alpha] = 0
    return soft_alpha


def _estimate_occlusion_from_parsing_masks(
    garment_mask: np.ndarray,
    occluder_mask: np.ndarray,
) -> Dict[str, object]:
    garment = np.asarray(garment_mask).astype(bool)
    occluder = np.asarray(occluder_mask).astype(bool)
    empty_mask = np.zeros_like(garment, dtype=bool)
    garment_pixels = int(garment.sum())
    if garment_pixels <= 0:
        return {
            "occlusion_mask": empty_mask,
            "garment_pixels": 0,
            "occluded_pixels": 0,
            "direct_ratio": 0.0,
            "structural_ratio": 0.0,
            "hole_ratio": 0.0,
            "missing_ratio": 0.0,
            "core_missing_ratio": 0.0,
            "proxy_ratio": 0.0,
            "kernel_size": 0,
        }

    # Direct occluder-on-garment overlap (classic metric).
    direct_support = _binary_dilate(garment, 15)
    direct_mask = occluder & direct_support
    direct_ratio = float(direct_mask.sum() / max(garment_pixels, 1))

    # Structural estimate: close gaps in garment silhouette and measure occluders inside that expected region.
    ys, xs = np.where(garment)
    h, w = garment.shape
    pad = max(4, int(round(min(h, w) * 0.01)))
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    bbox_region = np.zeros_like(garment, dtype=bool)
    bbox_region[y0:y1, x0:x1] = True

    kernel_size = max(9, int(round(min(h, w) * 0.035)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 121)

    expected_region = _binary_close(garment, kernel_size) & bbox_region
    if int(expected_region.sum()) < garment_pixels:
        expected_region = garment.copy()

    structural_mask = occluder & expected_region
    structural_ratio = float(structural_mask.sum() / max(int(expected_region.sum()), 1))

    missing_region = expected_region & (~garment)
    hole_occ_mask = occluder & missing_region
    hole_ratio = float(hole_occ_mask.sum() / max(int(expected_region.sum()), 1))
    missing_ratio = float(missing_region.sum() / max(int(expected_region.sum()), 1))

    # Emphasize center/mid garment zones where hand/hair occlusion destroys identity most.
    h_span = max(1, y1 - y0)
    w_span = max(1, x1 - x0)
    mid_y0 = y0 + int(0.12 * h_span)
    mid_y1 = y0 + int(0.72 * h_span)
    core_y0 = y0 + int(0.15 * h_span)
    core_y1 = y0 + int(0.75 * h_span)
    core_x0 = x0 + int(0.20 * w_span)
    core_x1 = x0 + int(0.80 * w_span)

    mid_region = np.zeros_like(garment, dtype=bool)
    mid_region[max(0, mid_y0):min(h, mid_y1), x0:x1] = True
    mid_mask = garment & mid_region
    mid_occ = direct_mask & mid_region
    mid_ratio = float(mid_occ.sum() / max(int(mid_mask.sum()), 1)) if int(mid_mask.sum()) > 0 else 0.0

    core_region = np.zeros_like(garment, dtype=bool)
    core_region[max(0, core_y0):min(h, core_y1), max(0, core_x0):min(w, core_x1)] = True
    core_mask = garment & core_region
    core_occ = direct_mask & core_region
    core_ratio = float(core_occ.sum() / max(int(core_mask.sum()), 1)) if int(core_mask.sum()) > 0 else 0.0
    core_expected = expected_region & core_region
    core_missing = missing_region & core_region
    core_missing_ratio = (
        float(core_missing.sum() / max(int(core_expected.sum()), 1)) if int(core_expected.sum()) > 0 else 0.0
    )

    # Favor hole coverage because it better reflects hidden middle sections of the garment.
    weighted_hole_ratio = min(1.0, hole_ratio * 1.8)
    weighted_mid_ratio = min(1.0, mid_ratio * 2.0)
    weighted_core_ratio = min(1.0, core_ratio * 2.2)
    # Count parser "missing garment" directly, even when occluder labels miss accessories/hair edges.
    weighted_missing_ratio = min(1.0, missing_ratio * 1.25)
    weighted_core_missing_ratio = min(1.0, core_missing_ratio * 1.7)
    proxy_ratio = float(
        max(
            direct_ratio,
            structural_ratio,
            weighted_hole_ratio,
            weighted_mid_ratio,
            weighted_core_ratio,
            weighted_missing_ratio,
            weighted_core_missing_ratio,
        )
    )
    combined_mask = direct_mask | hole_occ_mask

    return {
        "occlusion_mask": combined_mask,
        "garment_pixels": garment_pixels,
        "occluded_pixels": int(combined_mask.sum()),
        "direct_ratio": round(direct_ratio, 4),
        "structural_ratio": round(structural_ratio, 4),
        "hole_ratio": round(hole_ratio, 4),
        "missing_ratio": round(missing_ratio, 4),
        "mid_ratio": round(mid_ratio, 4),
        "core_ratio": round(core_ratio, 4),
        "core_missing_ratio": round(core_missing_ratio, 4),
        "weighted_hole_ratio": round(weighted_hole_ratio, 4),
        "weighted_mid_ratio": round(weighted_mid_ratio, 4),
        "weighted_core_ratio": round(weighted_core_ratio, 4),
        "weighted_missing_ratio": round(weighted_missing_ratio, 4),
        "weighted_core_missing_ratio": round(weighted_core_missing_ratio, 4),
        "proxy_ratio": round(proxy_ratio, 4),
        "kernel_size": int(kernel_size),
    }


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    a_area = max(1, (ax1 - ax0) * (ay1 - ay0))
    b_area = max(1, (bx1 - bx0) * (by1 - by0))
    union = a_area + b_area - inter_area
    return float(inter_area / max(1, union))


def _max_component_overlap_ratio(components: List[Dict[str, object]], garment_mask: np.ndarray) -> float:
    best = 0.0
    if not components:
        return best
    for component in components:
        raw_mask = component.get("mask")
        if raw_mask is None:
            continue
        component_mask = np.asarray(raw_mask).astype(bool)
        area = int(component_mask.sum())
        if area <= 0:
            continue
        overlap_pixels = int((component_mask & garment_mask).sum())
        overlap_ratio = float(overlap_pixels / max(area, 1))
        if overlap_ratio > best:
            best = overlap_ratio
    return float(best)


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


def _postprocess_single_piece_candidates(
    components: List[Dict[str, object]],
    *,
    image_height: int,
    min_item_pixels: int,
    parsing: Optional[np.ndarray] = None,
    source_image: Optional[Image.Image] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not components:
        return components, {"applied": False, "reason": "empty"}

    sorted_components = sorted(
        components,
        key=lambda item: (int(item.get("area", 0)), float(item.get("detector_conf", 0.0))),
        reverse=True,
    )
    debug: Dict[str, object] = {
        "applied": False,
        "input_count": len(sorted_components),
    }

    if len(sorted_components) < 2:
        debug["reason"] = "single_candidate"
        return sorted_components, debug

    primary = sorted_components[0]
    secondary = sorted_components[1]
    primary_area = int(primary.get("area", 0))
    secondary_area = int(secondary.get("area", 0))
    total_area = max(1, sum(max(0, int(c.get("area", 0))) for c in sorted_components))
    secondary_ratio = float(secondary_area / max(primary_area, 1))
    primary_dominance = float(primary_area / total_area)

    if (
        DETECT_FRAGMENT_SUPPRESS_ENABLED
        and secondary_ratio <= max(0.0, min(1.0, DETECT_FRAGMENT_SECONDARY_MAX_RATIO))
        and primary_dominance >= max(0.0, min(1.0, DETECT_FRAGMENT_DOMINANCE_MIN_RATIO))
    ):
        debug.update(
            {
                "applied": True,
                "mode": "fragment_suppression",
                "secondary_ratio": round(secondary_ratio, 4),
                "primary_dominance": round(primary_dominance, 4),
                "kept_component_id": int(primary.get("component_id", 0)),
            }
        )
        return [primary], debug

    if not DETECT_SINGLE_PIECE_MERGE_ENABLED:
        debug["reason"] = "single_piece_merge_disabled"
        return sorted_components, debug

    top_candidate = next((c for c in sorted_components if str(c.get("garment_type", "")).lower() in {"top", "outer"}), None)
    bottom_candidate = next((c for c in sorted_components if str(c.get("garment_type", "")).lower() == "bottom"), None)
    if top_candidate is None or bottom_candidate is None:
        debug["reason"] = "no_top_bottom_pair"
        return sorted_components, debug

    # If upstream split logic explicitly marked this pair as multi-piece,
    # do not collapse back into a single dress candidate.
    merge_blocked = bool(
        top_candidate.get("prevent_single_piece_merge", False)
        or bottom_candidate.get("prevent_single_piece_merge", False)
    )
    if merge_blocked:
        debug.update(
            {
                "reason": "split_signal_prefers_multi",
                "top_source": str(top_candidate.get("source", "")),
                "bottom_source": str(bottom_candidate.get("source", "")),
            }
        )
        return sorted_components, debug

    dress_hint_ratio = 0.0
    parserless_single_piece_candidate = False
    if parsing is not None:
        garment_pixels = int(np.isin(parsing, list(GARMENT_LABEL_MAP["all"])).sum())
        dress_pixels = int(np.isin(parsing, list(GARMENT_LABEL_MAP["dress"])).sum())
        if garment_pixels > 0:
            dress_hint_ratio = float(dress_pixels / garment_pixels)
    else:
        top_source = str(top_candidate.get("source", "")).lower().strip()
        bottom_source = str(bottom_candidate.get("source", "")).lower().strip()
        top_class_id = int(top_candidate.get("class_id", -1))
        bottom_class_id = int(bottom_candidate.get("class_id", -1))
        split_sources = {
            "waist_split_top",
            "waist_split_bottom",
            "person_top_bottom_split",
            "yolo_person_split",
        }
        both_from_split = top_source in split_sources and bottom_source in split_sources
        both_generic_person_like = top_class_id in {0, -1} and bottom_class_id in {0, -1}
        parserless_single_piece_candidate = bool(both_from_split or both_generic_person_like)
        if parserless_single_piece_candidate:
            dress_hint_ratio = 1.0

    top_bbox = tuple(top_candidate.get("bbox", (0, 0, 0, 0)))
    bottom_bbox = tuple(bottom_candidate.get("bbox", (0, 0, 0, 0)))
    x_overlap = _bbox_x_overlap_ratio(top_bbox, bottom_bbox)
    vertical_gap = _bbox_vertical_gap(top_bbox, bottom_bbox)
    max_gap_px = max(
        max(0, int(DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_PX)),
        max(0, int(max(0.0, DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_RATIO) * max(1, int(image_height)))),
    )

    # If YOLO natively output a Top and Bottom but they completely align,
    # evaluate if they form a perfect color/texture match (solid dress logic)
    if not parserless_single_piece_candidate and source_image is not None:
        if x_overlap >= max(0.0, DETECT_SINGLE_PIECE_MIN_X_OVERLAP) and vertical_gap <= max_gap_px:
            try:
                top_mask = np.asarray(top_candidate.get("mask", np.zeros((1, 1), dtype=bool))).astype(bool)
                bottom_mask = np.asarray(bottom_candidate.get("mask", np.zeros((1, 1), dtype=bool))).astype(bool)
                
                rgb = np.asarray(source_image.convert("RGB"), dtype=np.float32)
                t_ys, t_xs = np.where(top_mask)
                b_ys, b_xs = np.where(bottom_mask)
                
                if len(t_ys) > 0 and len(b_ys) > 0:
                    split_y = int((np.max(t_ys) + np.min(b_ys)) / 2)
                    band = max(10, min(48, int(image_height * 0.10)))
                    
                    t_band_mask = top_mask & (np.arange(top_mask.shape[0])[:, None] >= split_y - band)
                    b_band_mask = bottom_mask & (np.arange(bottom_mask.shape[0])[:, None] <= split_y + band)
                    
                    if t_band_mask.sum() > 64 and b_band_mask.sum() > 64:
                        top_pixels = rgb[t_band_mask]
                        bot_pixels = rgb[b_band_mask]
                        
                        top_med = np.median(top_pixels, axis=0)
                        bot_med = np.median(bot_pixels, axis=0)
                        top_std = float(np.std(top_pixels) / 255.0)
                        bot_std = float(np.std(bot_pixels) / 255.0)
                        
                        color_dist = float(np.linalg.norm(top_med - bot_med) / 255.0)
                        texture_delta = abs(top_std - bot_std)
                        
                        # Increased thresholds slightly to be more permissive for patterned/textured dresses
                        if color_dist < 0.20 and texture_delta < 0.08:
                            parserless_single_piece_candidate = True
                            dress_hint_ratio = 1.0
                            debug["native_yolo_perfect_color_match"] = True
                            debug["color_dist"] = round(color_dist, 4)
                            debug["texture_delta"] = round(texture_delta, 4)
            except Exception:
                pass

    if dress_hint_ratio < max(0.0, DETECT_SINGLE_PIECE_DRESS_HINT_MIN_RATIO):
        debug.update(
            {
                "reason": "dress_hint_too_low",
                "dress_hint_ratio": round(dress_hint_ratio, 4),
                "x_overlap": round(x_overlap, 4),
                "vertical_gap_px": int(vertical_gap),
                "max_gap_px": int(max_gap_px),
                "parserless_single_piece_candidate": bool(parserless_single_piece_candidate),
            }
        )
        return sorted_components, debug
    if x_overlap < max(0.0, DETECT_SINGLE_PIECE_MIN_X_OVERLAP):
        debug.update(
            {
                "reason": "x_overlap_too_low",
                "dress_hint_ratio": round(dress_hint_ratio, 4),
                "x_overlap": round(x_overlap, 4),
                "vertical_gap_px": int(vertical_gap),
                "max_gap_px": int(max_gap_px),
            }
        )
        return sorted_components, debug
    if vertical_gap > max_gap_px:
        debug.update(
            {
                "reason": "vertical_gap_too_large",
                "dress_hint_ratio": round(dress_hint_ratio, 4),
                "x_overlap": round(x_overlap, 4),
                "vertical_gap_px": int(vertical_gap),
                "max_gap_px": int(max_gap_px),
            }
        )
        return sorted_components, debug

    top_mask = np.asarray(top_candidate.get("mask", np.zeros((1, 1), dtype=bool))).astype(bool)
    bottom_mask = np.asarray(bottom_candidate.get("mask", np.zeros((1, 1), dtype=bool))).astype(bool)
    merged_mask = top_mask | bottom_mask
    merged_component = _mask_to_component(merged_mask, component_id=int(top_candidate.get("component_id", 1)))
    if merged_component is None:
        debug["reason"] = "merged_component_empty"
        return sorted_components, debug

    merged_area = int(merged_component.get("area", 0))
    if merged_area < max(1, int(min_item_pixels)):
        debug["reason"] = "merged_component_too_small"
        return sorted_components, debug

    merged_component["source"] = "single_piece_merge"
    merged_component["garment_type"] = "dress"
    merged_component["class_id"] = 3
    merged_component["class_name"] = "dress"
    merged_component["detector_conf"] = round(
        max(float(top_candidate.get("detector_conf", 0.0)), float(bottom_candidate.get("detector_conf", 0.0))),
        4,
    )
    merged_component["garment_type_score"] = round(
        max(float(top_candidate.get("garment_type_score", 0.0)), float(bottom_candidate.get("garment_type_score", 0.0)), dress_hint_ratio),
        4,
    )
    merged_component["merge_members"] = [
        int(top_candidate.get("component_id", 0)),
        int(bottom_candidate.get("component_id", 0)),
    ]

    merged_member_ids = set(merged_component["merge_members"])
    remaining = [c for c in sorted_components if int(c.get("component_id", 0)) not in merged_member_ids]
    if remaining:
        keep_threshold = max(min_item_pixels, int(merged_area * 0.25))
        remaining = [c for c in remaining if int(c.get("area", 0)) >= keep_threshold]

    merged_output = [merged_component] + remaining
    merged_output.sort(
        key=lambda item: (int(item.get("area", 0)), float(item.get("detector_conf", 0.0))),
        reverse=True,
    )
    debug.update(
        {
            "applied": True,
            "mode": "single_piece_merge",
            "dress_hint_ratio": round(dress_hint_ratio, 4),
            "x_overlap": round(x_overlap, 4),
            "vertical_gap_px": int(vertical_gap),
            "max_gap_px": int(max_gap_px),
            "output_count": len(merged_output),
            "merged_area": int(merged_area),
        }
    )
    return merged_output, debug


def _build_parser_category_components(
    parsing: np.ndarray,
    min_area_pixels: int,
    max_items: int,
) -> tuple:
    categories = ("top", "bottom", "dress", "outer")
    candidates: List[Dict[str, object]] = []
    category_debug: Dict[str, object] = {}
    category_class_id = {"top": 1, "bottom": 2, "dress": 3, "outer": 4}

    for category in categories:
        labels, _, label_debug = _resolve_label_ids_from_model(category)
        raw_mask = np.isin(parsing, list(labels))
        if cv2 is not None:
            # Fast path for section split: light denoise without full refinement.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_u8 = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
            refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            split_mask = refined_u8.astype(bool)
        else:
            split_mask = raw_mask

        if not split_mask.any():
            category_debug[category] = {
                "kept": 0,
                "label_debug": label_debug,
                "refinement_debug": {"enabled": False, "reason": "empty_after_fast_filter"},
            }
            continue

        parts = extract_component_sections(
            garment_mask=split_mask,
            max_items=2 if category == "dress" else 1,
            min_area_pixels=min_area_pixels,
        )
        kept_for_category = 0
        for part in parts:
            area = int(part.get("area", 0))
            if area < min_area_pixels:
                continue
            comp = dict(part)
            comp["source"] = f"parser_{category}"
            comp["garment_type"] = category
            comp["class_id"] = category_class_id[category]
            comp["detector_conf"] = round(min(0.99, 0.55 + min(area / max(1, parsing.size), 0.44)), 4)
            candidates.append(comp)
            kept_for_category += 1

        category_debug[category] = {
            "kept": kept_for_category,
            "label_debug": label_debug,
            "refinement_debug": {
                "enabled": True,
                "mode": "fast_split",
                "pixels_before": int(raw_mask.sum()),
                "pixels_after": int(split_mask.sum()),
            },
        }

    # De-duplicate highly overlapping parser categories; keep larger area.
    candidates.sort(key=lambda item: int(item.get("area", 0)), reverse=True)
    deduped: List[Dict[str, object]] = []
    for candidate in candidates:
        bbox = tuple(candidate.get("bbox", (0, 0, 0, 0)))
        if any(_bbox_iou(bbox, tuple(kept.get("bbox", (0, 0, 0, 0)))) > 0.82 for kept in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= max(1, max_items):
            break

    return deduped, category_debug


def _attempt_waist_split_components(
    mask: np.ndarray,
    min_area_pixels: int,
) -> List[Dict[str, object]]:
    if cv2 is None:
        return []
    base = mask.astype(bool)
    if not base.any():
        return []
    ys, xs = np.where(base)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h = y1 - y0
    if h < 120:
        return []

    crop = base[y0:y1, x0:x1]
    row_counts = crop.sum(axis=1).astype(np.float32)
    smooth = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, 21), sigmaX=0, sigmaY=0).reshape(-1)
    s0 = max(5, int(0.30 * h))
    s1 = min(h - 6, int(0.75 * h))
    if s1 <= s0:
        return []

    search = smooth[s0:s1]
    split_local = int(np.argmin(search))
    split_y = s0 + split_local

    top_peak = float(np.max(smooth[: max(split_y, 1)]))
    bottom_peak = float(np.max(smooth[min(split_y + 1, h - 1) :]))
    valley = float(smooth[split_y])
    if min(top_peak, bottom_peak) <= 0:
        return []

    # Split only when a very clear waist valley exists.
    # This avoids false top/bottom splits for single-piece dresses.
    if valley > min(top_peak, bottom_peak) * 0.50:
        return []

    overlap = max(8, int(0.10 * h))
    upper = np.zeros_like(base, dtype=bool)
    lower = np.zeros_like(base, dtype=bool)
    upper_end = min(h, split_y + overlap)
    lower_start = max(0, split_y - overlap)
    upper[y0 : y0 + upper_end, x0:x1] = crop[:upper_end, :]
    lower[y0 + lower_start : y1, x0:x1] = crop[lower_start:, :]

    top_parts = extract_component_sections(upper, max_items=1, min_area_pixels=min_area_pixels)
    bottom_parts = extract_component_sections(lower, max_items=1, min_area_pixels=min_area_pixels)
    if not top_parts or not bottom_parts:
        return []

    top = dict(top_parts[0])
    top["source"] = "waist_split_top"
    top["garment_type"] = "top"
    top["class_id"] = 1
    top["detector_conf"] = 0.6

    bottom = dict(bottom_parts[0])
    bottom["source"] = "waist_split_bottom"
    bottom["garment_type"] = "bottom"
    bottom["class_id"] = 2
    bottom["detector_conf"] = 0.6

    return [bottom, top] if int(bottom.get("area", 0)) >= int(top.get("area", 0)) else [top, bottom]


def _mask_to_component(mask: np.ndarray, component_id: int) -> Optional[Dict[str, object]]:
    m = mask.astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return {
        "mask": m,
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        "area": int(m.sum()),
        "component_id": component_id,
    }


def _force_split_single_component(
    mask: np.ndarray,
    min_area_pixels: int,
) -> List[Dict[str, object]]:
    base = mask.astype(bool)
    if not base.any():
        return []
    ys, xs = np.where(base)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h = y1 - y0
    if h < 80:
        return []

    split_y = y0 + (h // 2)
    overlap = max(10, int(0.12 * h))
    upper = np.zeros_like(base, dtype=bool)
    lower = np.zeros_like(base, dtype=bool)
    upper[y0 : min(y1, split_y + overlap), x0:x1] = base[y0 : min(y1, split_y + overlap), x0:x1]
    lower[max(y0, split_y - overlap) : y1, x0:x1] = base[max(y0, split_y - overlap) : y1, x0:x1]

    top_comp = _mask_to_component(upper, component_id=1)
    bottom_comp = _mask_to_component(lower, component_id=2)
    if not top_comp or not bottom_comp:
        return []
    if int(top_comp["area"]) < max(64, min_area_pixels // 2) or int(bottom_comp["area"]) < max(64, min_area_pixels // 2):
        return []

    top = dict(top_comp)
    top["source"] = "forced_split_top"
    top["garment_type"] = "top"
    top["class_id"] = 1
    top["detector_conf"] = 0.5

    bottom = dict(bottom_comp)
    bottom["source"] = "forced_split_bottom"
    bottom["garment_type"] = "bottom"
    bottom["class_id"] = 2
    bottom["detector_conf"] = 0.5
    return [bottom, top] if int(bottom.get("area", 0)) >= int(top.get("area", 0)) else [top, bottom]


def _apply_postprocess(result_image: Image.Image, source_reference: Image.Image) -> Image.Image:
    image = result_image.convert("RGB")
    reference = source_reference.convert("RGB").resize(image.size)

    # Gently align channel statistics with source to preserve palette character.
    if POSTPROCESS_COLOR_MATCH > 0:
        out_arr = np.asarray(image, dtype=np.float32)
        ref_arr = np.asarray(reference, dtype=np.float32)
        matched = out_arr.copy()
        for channel in range(3):
            out_mean = out_arr[..., channel].mean()
            out_std = out_arr[..., channel].std() + 1e-6
            ref_mean = ref_arr[..., channel].mean()
            ref_std = ref_arr[..., channel].std() + 1e-6
            matched[..., channel] = ((out_arr[..., channel] - out_mean) / out_std) * ref_std + ref_mean
        alpha = min(max(POSTPROCESS_COLOR_MATCH, 0.0), 1.0)
        mixed = np.clip((1.0 - alpha) * out_arr + alpha * matched, 0, 255).astype(np.uint8)
        image = Image.fromarray(mixed, mode="RGB")

    if abs(POSTPROCESS_CONTRAST - 1.0) > 1e-4:
        image = ImageEnhance.Contrast(image).enhance(POSTPROCESS_CONTRAST)

    if abs(POSTPROCESS_SHARPNESS - 1.0) > 1e-4:
        image = ImageEnhance.Sharpness(image).enhance(POSTPROCESS_SHARPNESS)

    if POSTPROCESS_UNSHARP_PERCENT > 0:
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=max(0.0, POSTPROCESS_UNSHARP_RADIUS),
                percent=max(0, POSTPROCESS_UNSHARP_PERCENT),
                threshold=max(0, POSTPROCESS_UNSHARP_THRESHOLD),
            )
        )

    return image


def _download_file(url: str, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download Real-ESRGAN model: {response.status_code}")
    with target_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _resolve_realesrgan_model_path() -> Path:
    model_dir = Path(REALESRGAN_MODEL_DIR)
    model_path = model_dir / REALESRGAN_MODEL_NAME
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    logger.info("Downloading Real-ESRGAN model to %s", str(model_path))
    _download_file(REALESRGAN_MODEL_URL, model_path)
    return model_path


def _ensure_realesrgan_loaded():
    global realesrgan_upsampler
    if realesrgan_upsampler is not None:
        return
    if not ENABLE_REALESRGAN:
        return
    if RealESRGANer is None or RRDBNet is None:
        raise RuntimeError("Real-ESRGAN dependencies are not available")

    with realesrgan_lock:
        if realesrgan_upsampler is not None:
            return
        model_path = _resolve_realesrgan_model_path()
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        half = bool(torch.cuda.is_available())
        gpu_id = 0 if torch.cuda.is_available() else None
        realesrgan_upsampler = RealESRGANer(
            scale=2,
            model_path=str(model_path),
            model=model,
            tile=max(0, REALESRGAN_TILE),
            tile_pad=max(0, REALESRGAN_TILE_PAD),
            pre_pad=max(0, REALESRGAN_PRE_PAD),
            half=half,
            gpu_id=gpu_id,
        )
        logger.info(
            "Real-ESRGAN loaded: model=%s tile=%s tile_pad=%s half=%s preserve_size=%s",
            str(model_path),
            max(0, REALESRGAN_TILE),
            max(0, REALESRGAN_TILE_PAD),
            half,
            REALESRGAN_PRESERVE_SIZE,
        )


def _detect_garment_type_from_text(text: str) -> str:
    normalized = _normalize_label_name(text)
    if any(keyword in normalized for keyword in ("dress", "gown", "jumpsuit", "romper")):
        return "dress"
    if any(keyword in normalized for keyword in ("pant", "trouser", "jean", "legging", "skirt", "short")):
        return "bottom"
    if any(keyword in normalized for keyword in ("coat", "jacket", "cardigan", "blazer", "outerwear", "scarf")):
        return "outer"
    if any(keyword in normalized for keyword in ("top", "shirt", "blouse", "hoodie", "sweater", "tee", "tshirt")):
        return "top"
    return "all"


def _rgb_to_color_name(rgb) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if v < 0.15:
        return "black"
    if s < 0.2:
        if v > 0.85:
            return "white"
        if v > 0.6:
            return "light gray"
        return "gray"
    deg = h * 360.0
    if deg < 15 or deg >= 345:
        return "red"
    if deg < 40:
        return "orange"
    if deg < 65:
        return "yellow"
    if deg < 165:
        return "green"
    if deg < 200:
        return "teal"
    if deg < 250:
        return "blue"
    if deg < 290:
        return "purple"
    if deg < 330:
        return "pink"
    return "red"


def _estimate_visual_complexity(image: Image.Image, region_mask: Optional[np.ndarray] = None):
    rgb = np.asarray(image.convert("RGB"))
    gray = np.asarray(image.convert("L"))

    if cv2 is not None:
        edges = cv2.Canny(gray, 80, 170) > 0
    else:
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
        gy = np.abs(np.diff(gray.astype(np.float32), axis=0))
        mag = np.zeros_like(gray, dtype=np.float32)
        mag[:, 1:] += gx
        mag[1:, :] += gy
        edges = mag > 36.0

    if region_mask is not None and region_mask.shape == gray.shape and region_mask.any():
        active = region_mask.astype(bool)
    else:
        active = np.ones_like(gray, dtype=bool)

    active_pixels = max(int(active.sum()), 1)
    edge_density = float((edges & active).sum()) / float(active_pixels)
    color_std = float(rgb[active].std() / 255.0) if active.any() else float(rgb.std() / 255.0)
    complexity = min(1.0, (edge_density * 2.7) + (color_std * 0.8))

    return {
        "edge_density": round(edge_density, 4),
        "color_std": round(color_std, 4),
        "complexity_score": round(complexity, 4),
    }


def _extract_dominant_color_names(image: Image.Image, region_mask: Optional[np.ndarray], max_colors: int):
    rgb = np.asarray(image.convert("RGB"))
    pixels = rgb.reshape(-1, 3)
    if region_mask is not None and region_mask.shape[:2] == rgb.shape[:2] and region_mask.any():
        flat_mask = region_mask.reshape(-1)
        pixels = pixels[flat_mask]
    if pixels.size == 0:
        return []

    sample_limit = 50000
    if len(pixels) > sample_limit:
        stride = max(1, len(pixels) // sample_limit)
        pixels = pixels[::stride]

    reduced = (pixels // 24) * 24
    unique, counts = np.unique(reduced, axis=0, return_counts=True)
    if len(unique) == 0:
        return []

    order = np.argsort(counts)[::-1]
    names = []
    seen = set()
    for idx in order:
        color_name = _rgb_to_color_name(unique[idx])
        if color_name in seen:
            continue
        seen.add(color_name)
        names.append(color_name)
        if len(names) >= max(1, max_colors):
            break
    return names


def _infer_primary_mask_for_prompt(image: Image.Image, person_description: str = ""):
    mask = None
    text_hint = _detect_garment_type_from_text(person_description or "")
    garment_type = text_hint
    debug = {"garment_type_from_text": text_hint, "mask_source": "none"}

    if not AUTO_PROMPT_USE_PARSING:
        return mask, garment_type, debug

    try:
        parsing = _run_human_parsing(image)
        type_counts = {}
        best_type = "dress"
        best_count = 0
        for candidate in ("dress", "top", "bottom", "outer"):
            ids, _, _ = _resolve_label_ids_from_model(candidate)
            count = int(np.isin(parsing, list(ids)).sum())
            type_counts[candidate] = count
            if count > best_count:
                best_count = count
                best_type = candidate
        selected_type = best_type if best_count > 0 else "all"

        if not FORCE_IMAGE_REFERENCE_PROMPT and text_hint in GARMENT_LABEL_MAP and text_hint != "all":
            hinted = int(type_counts.get(text_hint, 0))
            if hinted > 0 and hinted >= int(best_count * 0.6):
                selected_type = text_hint
                debug["type_selection"] = "text_hint"
            else:
                debug["type_selection"] = "parsing_best"
        else:
            debug["type_selection"] = "parsing_best"

        debug["garment_type_from_parsing"] = selected_type
        debug["garment_pixel_counts"] = type_counts

        label_ids, _, _ = _resolve_label_ids_from_model(selected_type if selected_type in GARMENT_LABEL_MAP else "all")
        candidate_mask = np.isin(parsing, list(label_ids))
        candidate_mask = _binary_open(candidate_mask, 3)
        candidate_mask = _binary_close(candidate_mask, 5)
        candidate_mask, _ = _refine_garment_mask(candidate_mask, selected_type if selected_type in GARMENT_LABEL_MAP else "all")
        if candidate_mask.any():
            mask = candidate_mask
            garment_type = selected_type
            debug["mask_source"] = "parsing"
    except Exception:
        logger.exception("Auto prompt parsing path failed; using full image fallback")

    return mask, garment_type, debug


def _extract_garment_mask_from_parsing(parsing: np.ndarray, garment_type_hint: str):
    selected_type = garment_type_hint if garment_type_hint in GARMENT_LABEL_MAP else "all"
    if selected_type == "all":
        best_type = "dress"
        best_count = 0
        for candidate in ("dress", "top", "bottom", "outer"):
            ids, _, _ = _resolve_label_ids_from_model(candidate)
            count = int(np.isin(parsing, list(ids)).sum())
            if count > best_count:
                best_count = count
                best_type = candidate
        selected_type = best_type if best_count > 0 else "all"

    label_ids, _, _ = _resolve_label_ids_from_model(selected_type)
    mask = np.isin(parsing, list(label_ids))
    mask = _binary_open(mask, 3)
    close_kernel = max(3, CONDITION_MASK_CLOSE_KERNEL)
    if close_kernel % 2 == 0:
        close_kernel += 1
    mask = _binary_close(mask, close_kernel)
    mask, _ = _refine_garment_mask(mask, selected_type)
    return mask.astype(bool), selected_type


def _build_garment_conditioning_image(
    image: Image.Image,
    person_description: str,
    enable_isolation: bool,
):
    if not enable_isolation:
        return image, {"enabled": False, "reason": "disabled"}, None, _detect_garment_type_from_text(person_description)

    hint_type = "all" if FORCE_IMAGE_REFERENCE_PROMPT else _detect_garment_type_from_text(person_description)
    try:
        parsing = _run_human_parsing(image)
        mask, selected_type = _extract_garment_mask_from_parsing(parsing, hint_type)
        min_pixels = int(image.width * image.height * max(0.0, CONDITION_MASK_MIN_AREA_RATIO))
        mask_pixels = int(mask.sum())
        if mask_pixels < max(128, min_pixels):
            return image, {
                "enabled": False,
                "reason": "mask_too_small",
                "mask_pixels": mask_pixels,
                "min_pixels": max(128, min_pixels),
            }, None, selected_type

        rgb = np.asarray(image.convert("RGB"))
        alpha = _build_soft_alpha(mask).astype(np.float32)[:, :, None] / 255.0
        neutral = np.full_like(rgb, 245, dtype=np.uint8)
        isolated = np.clip((rgb.astype(np.float32) * alpha) + (neutral.astype(np.float32) * (1.0 - alpha)), 0, 255).astype(np.uint8)

        condition_image = Image.fromarray(isolated, mode="RGB")
        return condition_image, {
            "enabled": True,
            "mask_pixels": mask_pixels,
            "garment_type": selected_type,
            "source": "human_parsing",
        }, mask, selected_type
    except Exception:
        logger.exception("Garment conditioning image build failed; using person image")
        return image, {"enabled": False, "reason": "exception"}, None, hint_type


def _normalize_garment_description(text: str) -> str:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return "a garment"

    # Remove person/model framing so prompt stays garment-centric.
    patterns = (
        r"^\s*(a|an|the)?\s*(person|woman|man|female model|male model|model)\s+wearing\s+",
        r"^\s*(a|an|the)?\s*(person|woman|man|female model|male model|model)\s+in\s+",
    )
    for pat in patterns:
        normalized = re.sub(pat, "", normalized, flags=re.IGNORECASE).strip()

    if not normalized:
        return "a garment"
    if normalized.lower().startswith(("a ", "an ", "the ")):
        return normalized
    return f"a {normalized}"


def _build_auto_prompt(image: Image.Image, person_description: str, strict_reference: bool = False):
    primary_mask, garment_type, mask_debug = _infer_primary_mask_for_prompt(image, person_description)
    complexity = _estimate_visual_complexity(image, primary_mask)
    colors = _extract_dominant_color_names(image, primary_mask, AUTO_PROMPT_MAX_COLORS)
    color_phrase = ", ".join(colors) if colors else "original color palette"

    if complexity["edge_density"] >= 0.14:
        texture_phrase = "preserve intricate print motifs and fine texture details"
    elif complexity["edge_density"] >= 0.09:
        texture_phrase = "preserve printed pattern details and texture"
    else:
        texture_phrase = "preserve smooth fabric texture and subtle seams"

    garment_noun = {
        "dress": "dress",
        "top": "top",
        "bottom": "bottom garment",
        "outer": "outerwear piece",
        "all": "garment",
    }.get(garment_type, "garment")

    effective_strict = strict_reference or FORCE_IMAGE_REFERENCE_PROMPT
    garment_description = f"a {garment_noun} from the reference image"
    prompt = (
        f"<MODEL> {garment_description}. <TARGET> exact same {garment_noun} laid flat product photo, "
        f"full item visible, preserve silhouette, colors ({color_phrase}), and {texture_phrase}, no human. "
        "Use the right-side reference image as the single source of truth; do not invent or replace garment type, "
        "print, embroidery, trims, panel layout, buttons, seams, or accessories. "
        "Preserve any existing logo, text, letters, typography, and their placement exactly as seen in reference."
    )

    debug = {
        "enabled": True,
        "strict_reference": effective_strict,
        "garment_type": garment_type,
        "dominant_colors": colors,
        "complexity": complexity,
        "manual_description_used": False,
        "mask_debug": mask_debug,
    }
    return prompt, debug


def _select_adaptive_steps(complexity: dict):
    low = max(1, min(100, ADAPTIVE_STEPS_LOW))
    high = max(1, min(100, ADAPTIVE_STEPS_HIGH))
    if high < low:
        low, high = high, low
    threshold = max(0.0, min(1.0, ADAPTIVE_COMPLEXITY_THRESHOLD))

    selected = high if complexity["complexity_score"] >= threshold else low
    reason = "high_complexity" if selected == high else "low_complexity"
    return selected, {"low": low, "high": high, "threshold": threshold, "reason": reason}


def _enhance_region_detail(
    image: Image.Image,
    garment_type_hint: str = "all",
    region_mask: Optional[np.ndarray] = None,
):
    if REGION_UPSCALE_FACTOR <= 1.0:
        return image, {"enabled": False, "reason": "factor<=1"}

    rgb = np.asarray(image.convert("RGB"))
    fg_mask = None
    if region_mask is not None and region_mask.shape == rgb.shape[:2] and region_mask.any():
        fg_mask = region_mask.astype(bool)
    elif REGION_UPSCALE_PARSE_MASK:
        try:
            parsing = _run_human_parsing(image)
            fg_mask, _ = _extract_garment_mask_from_parsing(parsing, garment_type_hint)
        except Exception:
            logger.exception("Region upscale parsing path failed; using luminance mask fallback")

    if fg_mask is None:
        fg_mask = (rgb.max(axis=2) > max(0, min(255, REGION_UPSCALE_BG_THRESHOLD)))
        fg_mask = _binary_open(fg_mask, 3)
        fg_mask = _binary_close(fg_mask, 5)

    if cv2 is not None and fg_mask.any():
        labels = fg_mask.astype(np.uint8)
        num_labels, cc_map, stats, _ = cv2.connectedComponentsWithStats(labels, connectivity=8)
        if num_labels > 1:
            best_idx = 1
            best_area = int(stats[1, cv2.CC_STAT_AREA])
            for idx in range(2, num_labels):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area > best_area:
                    best_area = area
                    best_idx = idx
            fg_mask = (cc_map == best_idx)

    fg_pixels = int(fg_mask.sum())
    if fg_pixels < max(1, REGION_UPSCALE_MIN_PIXELS):
        return image, {"enabled": False, "reason": "insufficient_foreground", "foreground_pixels": fg_pixels}

    ys, xs = np.where(fg_mask)
    if len(xs) == 0 or len(ys) == 0:
        return image, {"enabled": False, "reason": "empty_bbox", "foreground_pixels": fg_pixels}

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = image.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    if cw < 16 or ch < 16:
        return image, {"enabled": False, "reason": "small_crop", "crop_size": {"width": cw, "height": ch}}

    resample = getattr(Image, "Resampling", Image).LANCZOS
    upscale_engine = "classic"
    upscaled_size = {"width": cw, "height": ch}
    try:
        if ENABLE_REALESRGAN:
            _ensure_realesrgan_loaded()
        if ENABLE_REALESRGAN and realesrgan_upsampler is not None:
            crop_bgr = np.asarray(crop.convert("RGB"))[:, :, ::-1]
            sr_bgr, _ = realesrgan_upsampler.enhance(crop_bgr, outscale=max(1.0, REALESRGAN_OUTSCALE))
            sr_rgb = sr_bgr[:, :, ::-1]
            upscaled = Image.fromarray(sr_rgb.astype(np.uint8), mode="RGB")
            upscaled_size = {"width": upscaled.width, "height": upscaled.height}
            upscale_engine = "realesrgan"
            if REALESRGAN_PRESERVE_SIZE:
                refined = upscaled.resize((cw, ch), resample=resample)
            else:
                refined = upscaled
        else:
            up_w = max(cw, int(round(cw * REGION_UPSCALE_FACTOR)))
            up_h = max(ch, int(round(ch * REGION_UPSCALE_FACTOR)))
            upscaled = crop.resize((up_w, up_h), resample=resample)
            if abs(REGION_UPSCALE_SHARPNESS - 1.0) > 1e-4:
                upscaled = ImageEnhance.Sharpness(upscaled).enhance(REGION_UPSCALE_SHARPNESS)
            refined = upscaled.resize((cw, ch), resample=resample)
            upscaled_size = {"width": up_w, "height": up_h}
    except Exception:
        logger.exception("Region Real-ESRGAN upscale failed; falling back to classic refine")
        up_w = max(cw, int(round(cw * REGION_UPSCALE_FACTOR)))
        up_h = max(ch, int(round(ch * REGION_UPSCALE_FACTOR)))
        upscaled = crop.resize((up_w, up_h), resample=resample)
        if abs(REGION_UPSCALE_SHARPNESS - 1.0) > 1e-4:
            upscaled = ImageEnhance.Sharpness(upscaled).enhance(REGION_UPSCALE_SHARPNESS)
        refined = upscaled.resize((cw, ch), resample=resample)
        upscaled_size = {"width": up_w, "height": up_h}
        upscale_engine = "classic_fallback"

    if refined.size != (cw, ch):
        refined = refined.resize((cw, ch), resample=resample)

    output = image.copy()
    output.paste(refined, (x0, y0, x1, y1))
    return output, {
        "enabled": True,
        "foreground_pixels": fg_pixels,
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "upscale_factor": REGION_UPSCALE_FACTOR,
        "mask_source": "provided" if region_mask is not None else ("parsing" if REGION_UPSCALE_PARSE_MASK else "luminance"),
        "upscale_engine": upscale_engine,
        "upscaled_size": upscaled_size,
        "realesrgan_enabled": ENABLE_REALESRGAN,
    }


def _ensure_tryon_model_loaded():
    global pipe
    if pipe is not None:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Failed to import Any2Any dependencies: {IMPORT_ERROR}") from IMPORT_ERROR

    with pipe_lock:
        if pipe is not None:
            return

        token = os.getenv("HUGGING_FACE_KEY")
        logger.info("Loading model: %s", MODEL_PATH)
        local_pipe = None
        try:
            transformer = FluxTransformer2DModel.from_pretrained(
                MODEL_PATH,
                subfolder="transformer",
                torch_dtype=TORCH_DTYPE,
                token=token,
            )
            vae = AutoencoderKL.from_pretrained(
                MODEL_PATH,
                subfolder="vae",
                torch_dtype=TORCH_DTYPE,
                token=token,
            )
            local_pipe = FluxTryonPipeline.from_pretrained(
                MODEL_PATH,
                transformer=transformer,
                vae=vae,
                torch_dtype=TORCH_DTYPE,
                token=token,
            )

            logger.info("Loading Any2Any LoRA: %s / %s", ANY2ANY_LORA_REPO, ANY2ANY_LORA_WEIGHT)
            local_pipe.load_lora_weights(
                ANY2ANY_LORA_REPO,
                weight_name=ANY2ANY_LORA_WEIGHT,
                adapter_name="any2any",
            )
            local_pipe.set_adapters(["any2any"], adapter_weights=[1.0])

            if ENABLE_CPU_OFFLOAD:
                local_pipe.enable_model_cpu_offload()
            else:
                local_pipe.to(DEVICE)

            if ENABLE_ATTENTION_SLICING:
                local_pipe.enable_attention_slicing()
            if ENABLE_VAE_SLICING:
                local_pipe.vae.enable_slicing()
            if ENABLE_VAE_TILING:
                local_pipe.vae.enable_tiling()

            try:
                import xformers  # noqa: F401

                local_pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers enabled")
            except ImportError:
                logger.info("xformers not found, continuing without it")

            if ENABLE_TORCH_COMPILE:
                try:
                    local_pipe.transformer = torch.compile(local_pipe.transformer, mode="reduce-overhead", fullgraph=False)
                    logger.info("torch.compile enabled for transformer")
                except Exception:
                    logger.exception("torch.compile failed; continuing without compile")

            pipe = local_pipe
            logger.info(
                (
                "runtime_flags: cpu_offload=%s attention_slicing=%s vae_slicing=%s "
                "vae_tiling=%s torch_compile=%s empty_cache=%s max_sequence_length=%s "
                "output_format=%s output_quality=%s postprocess=%s auto_prompt=%s "
                "adaptive_steps=%s design_match=%s strict_design_match=%s condition_isolation=%s "
                "region_upscale=%s realesrgan=%s force_auto_prompt=%s force_ref_prompt=%s force_isolation=%s"
            ),
                ENABLE_CPU_OFFLOAD,
                ENABLE_ATTENTION_SLICING,
                ENABLE_VAE_SLICING,
                ENABLE_VAE_TILING,
                ENABLE_TORCH_COMPILE,
                ENABLE_EMPTY_CACHE,
                MAX_SEQUENCE_LENGTH,
                OUTPUT_IMAGE_FORMAT,
                OUTPUT_IMAGE_QUALITY,
                ENABLE_POSTPROCESS,
                AUTO_PROMPT_ENABLED,
                ADAPTIVE_STEPS_ENABLED,
                DESIGN_MATCH_MODE,
                STRICT_DESIGN_MATCH,
                CONDITION_USE_GARMENT_ISOLATION,
                ENABLE_REGION_UPSCALE,
                ENABLE_REALESRGAN,
                FORCE_AUTO_PROMPT,
                FORCE_IMAGE_REFERENCE_PROMPT,
                FORCE_GARMENT_ISOLATION,
            )
            logger.info("Any2Any try-off model loaded successfully")
        except Exception:
            pipe = None
            local_pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.exception("Any2Any model load failed; pipeline state reset")
            raise


@app.on_event("startup")
async def load_model():
    if LOAD_TRYON_MODEL_ON_STARTUP:
        _ensure_tryon_model_loaded()
    else:
        logger.info("Skipping try-off model preload (LOAD_TRYON_MODEL_ON_STARTUP=false)")


@app.get("/")
def root_ready():
    return {"status": "ok", "service": "any2any-tryoff", "ready": True}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "model_path": MODEL_PATH,
        "load_tryon_on_startup": LOAD_TRYON_MODEL_ON_STARTUP,
        "gpu_available": torch.cuda.is_available(),
        "parsing_loaded": parsing_model is not None,
        "parsing_model_path": PARSING_MODEL_PATH,
        "auto_prompt_enabled": AUTO_PROMPT_ENABLED,
        "force_auto_prompt": FORCE_AUTO_PROMPT,
        "force_image_reference_prompt": FORCE_IMAGE_REFERENCE_PROMPT,
        "adaptive_steps_enabled": ADAPTIVE_STEPS_ENABLED,
        "design_match_mode": DESIGN_MATCH_MODE,
        "strict_design_match": STRICT_DESIGN_MATCH,
        "condition_use_garment_isolation": CONDITION_USE_GARMENT_ISOLATION,
        "force_garment_isolation": FORCE_GARMENT_ISOLATION,
        "region_upscale_enabled": ENABLE_REGION_UPSCALE,
        "realesrgan_enabled": ENABLE_REALESRGAN,
        "realesrgan_loaded": realesrgan_upsampler is not None,
        "detect_use_yolo": DETECT_USE_YOLO,
        "detect_min_item_pixels": DETECT_MIN_ITEM_PIXELS,
        "detect_min_item_width_px": DETECT_MIN_ITEM_WIDTH_PX,
        "detect_min_item_height_px": DETECT_MIN_ITEM_HEIGHT_PX,
        "detect_min_garment_overlap_ratio": DETECT_MIN_GARMENT_OVERLAP_RATIO,
        "detect_enable_waist_split": DETECT_ENABLE_WAIST_SPLIT,
        "detect_fragment_suppress_enabled": DETECT_FRAGMENT_SUPPRESS_ENABLED,
        "detect_fragment_secondary_max_ratio": DETECT_FRAGMENT_SECONDARY_MAX_RATIO,
        "detect_fragment_dominance_min_ratio": DETECT_FRAGMENT_DOMINANCE_MIN_RATIO,
        "detect_single_piece_merge_enabled": DETECT_SINGLE_PIECE_MERGE_ENABLED,
        "detect_single_piece_dress_hint_min_ratio": DETECT_SINGLE_PIECE_DRESS_HINT_MIN_RATIO,
        "detect_single_piece_min_x_overlap": DETECT_SINGLE_PIECE_MIN_X_OVERLAP,
        "detect_single_piece_max_vertical_gap_px": DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_PX,
        "detect_single_piece_max_vertical_gap_ratio": DETECT_SINGLE_PIECE_MAX_VERTICAL_GAP_RATIO,
        "detect_parsing_type_override_min_score": DETECT_PARSING_TYPE_OVERRIDE_MIN_SCORE,
        "extract_occlusion_prefilter_enabled": EXTRACT_OCCLUSION_PREFILTER_ENABLED,
        "extract_occlusion_proxy_threshold": EXTRACT_OCCLUSION_PROXY_THRESHOLD,
        "extract_vton_fallback_enabled": EXTRACT_VTON_FALLBACK_ENABLED,
        "extract_vton_cloth_only_endpoint": EXTRACT_VTON_CLOTH_ONLY_ENDPOINT,
        "extract_vton_quality_preset": EXTRACT_VTON_QUALITY_PRESET,
        "extract_vton_timesteps": EXTRACT_VTON_TIMESTEPS,
        "extract_vton_guidance_scale": EXTRACT_VTON_GUIDANCE_SCALE,
        "extract_vton_fast_mode": EXTRACT_VTON_FAST_MODE,
        "extract_vton_fast_timesteps": EXTRACT_VTON_FAST_TIMESTEPS,
        "extract_vton_fast_guidance_scale": EXTRACT_VTON_FAST_GUIDANCE_SCALE,
        "extract_vton_fast_disable_upscale": EXTRACT_VTON_FAST_DISABLE_UPSCALE,
        "extract_vton_upscale_enabled": EXTRACT_VTON_UPSCALE_ENABLED,
        "extract_vton_upscale_factor": EXTRACT_VTON_UPSCALE_FACTOR,
        "extract_vton_use_showroom_person": EXTRACT_VTON_USE_SHOWROOM_PERSON,
        "extract_vton_showroom_person_image_url": EXTRACT_VTON_SHOWROOM_PERSON_IMAGE_URL,
        "extract_parser_min_coverage_ratio": EXTRACT_PARSER_MIN_COVERAGE_RATIO,
        "extract_disable_final_crop": EXTRACT_DISABLE_FINAL_CROP,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "analyze_blur_min": ANALYZE_BLUR_MIN,
        "analyze_use_human_parser": ANALYZE_USE_HUMAN_PARSER,
        "analyze_min_item_pixels": ANALYZE_MIN_ITEM_PIXELS,
        "analyze_occlusion_threshold": ANALYZE_OCCLUSION_THRESHOLD,
        "analyze_force_vton_for_single_piece_dress": ANALYZE_FORCE_VTON_FOR_SINGLE_PIECE_DRESS,
        "analyze_always_use_vton": ANALYZE_ALWAYS_USE_VTON,
        "analyze_dress_vton_occlusion_threshold": ANALYZE_DRESS_VTON_OCCLUSION_THRESHOLD,
        "analyze_single_item_type_fallback": ANALYZE_SINGLE_ITEM_TYPE_FALLBACK,
        "analyze_force_split_crop_input": ANALYZE_FORCE_SPLIT_CROP_INPUT,
        "analyze_low_confidence_split_to_multi": ANALYZE_LOW_CONFIDENCE_SPLIT_TO_MULTI,
        "analyze_use_isolated_top_crop": ANALYZE_USE_ISOLATED_TOP_CROP,
        "analyze_include_upload_url": ANALYZE_INCLUDE_UPLOAD_URL,
        "analyze_include_item_preview_on_success": ANALYZE_INCLUDE_ITEM_PREVIEW_ON_SUCCESS,
        "analyze_save_local_items": ANALYZE_SAVE_LOCAL_ITEMS,
        "analyze_multi_item_isolate_preview": ANALYZE_MULTI_ITEM_ISOLATE_PREVIEW,
        "analyze_multi_item_preview_keep_occluders": ANALYZE_MULTI_ITEM_PREVIEW_KEEP_OCCLUDERS,
        "analyze_multi_item_isolate_min_mask_ratio": ANALYZE_MULTI_ITEM_ISOLATE_MIN_MASK_RATIO,
        "analyze_multi_item_isolate_trim_padding_px": ANALYZE_MULTI_ITEM_ISOLATE_TRIM_PADDING_PX,
        "analyze_min_component_area_ratio": ANALYZE_MIN_COMPONENT_AREA_RATIO,
        "analyze_local_items_dir": ANALYZE_LOCAL_ITEMS_DIR,
        "wardrobe_progress_sync_enabled": ENABLE_WARDROBE_PROGRESS_SYNC,
        "wardrobe_progress_api_configured": bool(WARDROBE_PROGRESS_API_BASE_URL),
    }


@app.post("/detect_garments", include_in_schema=False)
async def detect_garments(request: DetectGarmentsRequest):
    return await handle_detect_garments(request, ctx=globals())



@app.post("/extract_cloth", include_in_schema=False)
async def extract_cloth(request: ExtractClothRequest):
    return await handle_extract_cloth(request, ctx=globals())



@app.post("/wardrobe_flow", include_in_schema=False)
async def wardrobe_flow(request: WardrobeFlowRequest):
    return await handle_wardrobe_flow(request, ctx=globals())



@app.post("/analyze")
async def analyze_multipart(
    file: UploadFile = File(...),
    selected_type: Optional[str] = Form(default=None, alias="type"),
    authorization: str = Header(default=None, alias="Authorization"),
):
    return await handle_analyze_multipart(file, authorization, selected_type, ctx=globals())



@app.post("/process")
async def process_image(request: TryOffRequest):
    return await handle_process_image(request, ctx=globals())



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
