"""Image-frame coordinate conversion and offline evidence gates for localization."""
from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Any

from app.models.document_localization_v1 import ImageFrameRecord


class LocalizationFrameError(ValueError):
    """Frame registry or coordinate conversion failed."""


def registry_digest(registry: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        registry,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_image_registry(registry: list[dict[str, Any]], page_size: tuple[int, int]) -> None:
    width, height = page_size
    if not registry:
        raise LocalizationFrameError("IMAGE_REGISTRY_EMPTY")
    seen: set[str] = set()
    for row in registry:
        record = ImageFrameRecord.model_validate(row)
        if record.image_id in seen:
            raise LocalizationFrameError("DUPLICATE_IMAGE_FRAME_ID")
        seen.add(record.image_id)
        x0, y0, x1, y1 = record.window_px
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise LocalizationFrameError("IMAGE_FRAME_OUTSIDE_PAGE")
    if "img0" not in seen:
        raise LocalizationFrameError("IMAGE_REGISTRY_REQUIRES_FULL_PAGE_IMG0")


def local_box_to_full_page_box(
    box_2d: list[float],
    frame: dict[str, Any],
) -> list[float]:
    if len(box_2d) != 4:
        raise LocalizationFrameError("INVALID_LOCAL_BOX")
    if not all(type(v) in {int, float} and math.isfinite(v) for v in box_2d):
        raise LocalizationFrameError("INVALID_LOCAL_BOX")
    y0, x0, y1, x1 = box_2d
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        raise LocalizationFrameError("INVALID_LOCAL_BOX")
    record = ImageFrameRecord.model_validate(frame)
    wx0, wy0, wx1, wy1 = record.window_px
    window_width = wx1 - wx0
    window_height = wy1 - wy0
    return [
        wy0 + y0 / 1000 * window_height,
        wx0 + x0 / 1000 * window_width,
        wy0 + y1 / 1000 * window_height,
        wx0 + x1 / 1000 * window_width,
    ]


def full_page_box_to_normalized_region(
    box: list[float],
    page_size: tuple[int, int],
) -> dict[str, float]:
    width, height = page_size
    y0, x0, y1, x1 = box
    return {
        "x": x0 / width,
        "y": y0 / height,
        "width": (x1 - x0) / width,
        "height": (y1 - y0) / height,
    }


def full_page_box_to_pixel_bounds(
    box: list[float],
    page_size: tuple[int, int],
) -> list[int]:
    width, height = page_size
    y0, x0, y1, x1 = box
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise LocalizationFrameError("FULL_PAGE_BOX_OUTSIDE_PAGE")
    return [
        math.floor(x0),
        math.floor(y0),
        math.ceil(x1),
        math.ceil(y1),
    ]


def image_visible_content_gate(data: bytes) -> dict[str, Any]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        converted = image.convert("RGBA")
        extrema = converted.getextrema()
    uniform = all(channel[0] == channel[1] for channel in extrema)
    return {
        "status": "BLOCKED_NO_VISIBLE_CONTENT" if uniform else "UNVERIFIED",
        "reason": "UNIFORM_VISIBLE_PIXELS" if uniform else None,
        "visual_association_validated": False,
    }


def edge_contact_warnings(
    bounds: list[int],
    frame_size: tuple[int, int],
) -> list[str]:
    x0, y0, x1, y1 = bounds
    width, height = frame_size
    warnings = []
    if x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height:
        warnings.append("REGION_TOUCHES_SOURCE_IMAGE_BORDER")
    return warnings
