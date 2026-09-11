"""Offline, read-only source preparation for PDF and single JPEG/PNG images.

No application settings, providers, prompts or network are imported. PDFium calls
in this module are serialized. Do not run other PDFium consumers concurrently in
this process; integrate into isolated workers before exposing untrusted uploads.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import threading
import unicodedata
import warnings
from contextlib import closing
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from app.models.source_preparation_v1 import (
    PreparedPage,
    PreparedSource,
    SourcePreparationOptions,
)

_PREPARATION_LOCK = threading.Lock()
_SCHEMA_VERSION = 1


class SourcePreparationError(ValueError):
    """An input/preparation failure, never a successful semantic extraction."""


def _load_engines():
    # Lazy loading keeps the existing extractor independent of the new dependency.
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SourcePreparationError(
            'Falta una dependencia: ejecutar uv add "pypdfium2>=5.8,<6".'
        ) from exc
    return pdfium, raw, Image, ImageOps


def document_identity(logical_name: str, content_sha256: str) -> str:
    """Order-independent identity; equal bytes in different paths remain distinct."""
    name = unicodedata.normalize("NFC", logical_name.replace("\\", "/"))
    digest = hashlib.sha256((name + "\0" + content_sha256).encode("utf-8")).hexdigest()
    return "doc-" + digest[:24]


def _media_type(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    raise SourcePreparationError("UNSUPPORTED_SOURCE: se admite PDF, JPEG y PNG.")


def _json_write(path: Path, value: Any) -> None:
    # Exclusive creation; never replace source documents or earlier run artifacts.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def _scale_for_size(width: float, height: float, options: SourcePreparationOptions,
                    requested: float) -> float:
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise SourcePreparationError("INVALID_PAGE_SIZE")
    scale = min(
        requested,
        options.max_render_edge / width,
        options.max_render_edge / height,
        math.sqrt(options.max_render_pixels / (width * height)),
    )
    # PDFium rounds pixel dimensions upwards. Enforce the budget after rounding too.
    for _ in range(20):
        w, h = math.ceil(width * scale), math.ceil(height * scale)
        if max(w, h) <= options.max_render_edge and w * h <= options.max_render_pixels:
            return scale
        scale *= 0.999
    raise SourcePreparationError("RENDER_BUDGET_CANNOT_BE_SATISFIED")


def _box_in_preview(page, box, width_px: int, height_px: int, raw) -> dict[str, Any]:
    """Use the engine's transform, including CropBox and intrinsic PDF rotation.

    Returned un-clipped bounds preserve the original result; normalized visible
    bounds are explicitly an intersection, not a 'repaired' original coordinate.
    """
    left, bottom, right, top = box
    if not all(math.isfinite(v) for v in box):
        raise SourcePreparationError("NONFINITE_CHARACTER_BOX")
    points = []
    for x, y in ((left, bottom), (left, top), (right, bottom), (right, top)):
        dx, dy = ctypes.c_int(), ctypes.c_int()
        ok = raw.FPDF_PageToDevice(
            page, 0, 0, width_px, height_px, 0, x, y,
            ctypes.byref(dx), ctypes.byref(dy),
        )
        if not ok:
            raise SourcePreparationError("PAGE_TO_PREVIEW_TRANSFORM_FAILED")
        points.append((dx.value, dy.value))
    x0, y0 = min(p[0] for p in points), min(p[1] for p in points)
    x1, y1 = max(p[0] for p in points), max(p[1] for p in points)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width_px, x1), min(height_px, y1)
    region = None
    if cx1 > cx0 and cy1 > cy0:
        region = {
            "x": cx0 / width_px, "y": cy0 / height_px,
            "width": (cx1 - cx0) / width_px, "height": (cy1 - cy0) / height_px,
        }
    return {
        "unclipped_bbox_px": [x0, y0, x1, y1],
        "visible_region": region,
        "clipped_to_visible_page": [x0, y0, x1, y1] != [cx0, cy0, cx1, cy1],
    }


def _read_native_characters(page, width_px: int, height_px: int, options, raw):
    chars: list[dict[str, Any]] = []
    notes: list[str] = []
    textpage = page.get_textpage()
    try:
        count = textpage.count_chars()
        if count < 0:
            raise SourcePreparationError("NATIVE_TEXT_COUNT_FAILED")
        if count > options.max_chars_per_page:
            # No silent truncation: the page will be marked partial by the caller.
            raise SourcePreparationError("NATIVE_TEXT_CHARACTER_LIMIT_EXCEEDED")
        unlocated = 0
        unicode_errors = 0
        for index in range(count):
            codepoint = int(raw.FPDFText_GetUnicode(textpage, index))
            decoded = chr(codepoint) if 0 < codepoint <= 0x10FFFF else "\ufffd"
            if 0xD800 <= codepoint <= 0xDFFF:
                decoded = "\ufffd"
            mapping_error = bool(raw.FPDFText_HasUnicodeMapError(textpage, index))
            generated = bool(raw.FPDFText_IsGenerated(textpage, index))
            unicode_errors += int(mapping_error or decoded == "\ufffd")
            item: dict[str, Any] = {
                "pdf_char_index": index,
                "text": decoded,
                "unicode_codepoint_reported": codepoint,
                "generated_by_text_engine": generated,
                "unicode_mapping_error": mapping_error,
                "bbox_pdf_canvas": None,
                "angle_radians_pdf_canvas": None,
                "unclipped_bbox_px": None,
                "visible_region": None,
                "clipped_to_visible_page": False,
            }
            # Engine-inserted spaces/line breaks are not glyphs to be localized.
            if not decoded.isspace() and not generated and codepoint > 0:
                try:
                    box = textpage.get_charbox(index)
                    item["bbox_pdf_canvas"] = list(box)
                    angle = float(raw.FPDFText_GetCharAngle(textpage, index))
                    item["angle_radians_pdf_canvas"] = angle if math.isfinite(angle) and angle >= 0 else None
                    item.update(_box_in_preview(page, box, width_px, height_px, raw))
                    if item["visible_region"] is None:
                        unlocated += 1
                except (ValueError, RuntimeError, OverflowError):
                    item["location_error"] = "NATIVE_CHARACTER_LOCATION_UNAVAILABLE"
                    unlocated += 1
            chars.append(item)
        if unlocated:
            notes.append(f"UNLOCATED_OR_OUTSIDE_NATIVE_CHARACTERS:{unlocated}")
        if unicode_errors:
            notes.append(f"NATIVE_UNICODE_MAPPING_ERRORS:{unicode_errors}")
        # This is the engine's index order; it is NOT validated document reading order.
        text = "".join(char["text"] for char in chars)
        return chars, text, notes
    finally:
        textpage.close()


def _prepare_pdf_page(pdf, page_index, source_id, output_dir, options, pdfium, raw):
    stem = f"page-{page_index + 1:04d}"
    preview_file, observations_file = stem + ".png", stem + ".json"
    with closing(pdf[page_index]) as page:
        width, height = page.get_size()
        scale = _scale_for_size(width, height, options, options.render_scale)
        bitmap = page.render(scale=scale, rotation=0, draw_annots=True)
        try:
            with bitmap.to_pil() as preview:
                width_px, height_px = preview.size
                with (output_dir / preview_file).open("xb") as out:
                    preview.save(out, format="PNG")
        finally:
            bitmap.close()
        notes = []
        if scale < options.render_scale - 1e-9:
            notes.append("PREVIEW_DOWNSCALED_TO_RESOURCE_LIMIT")
        try:
            chars, text, text_warnings = _read_native_characters(
                page, width_px, height_px, options, raw,
            )
            notes.extend(text_warnings)
            text_state = "NATIVE_TEXT_PRESENT_UNVERIFIED" if text.strip() else "NO_NATIVE_TEXT"
            status = "PREPARED"
            error = None
        except Exception as exc:
            chars, text = [], ""
            text_state, status = "TEXT_EXTRACTION_FAILED", "PARTIAL"
            error = type(exc).__name__
            notes.append(str(exc) if isinstance(exc, SourcePreparationError)
                         else "NATIVE_TEXT_EXTRACTION_FAILED")
        if text_state == "NO_NATIVE_TEXT":
            notes.append("VISUAL_READING_REQUIRED_NO_OCR_RUN")
        data = {
            "schema_version": _SCHEMA_VERSION,
            "document_id": source_id,
            "page_number": page_index + 1,
            "pdf_page_label": pdf.get_page_label(page_index) or None,
            "sheet_number_from_drawing": None,
            "source_type": "PDF",
            "status": status,
            "frame": {
                "origin_preview": "TOP_LEFT", "x_direction": "RIGHT", "y_direction": "DOWN",
                "preview_width_px": width_px, "preview_height_px": height_px,
                "pdf_size_canvas_units": [width, height],
                "pdf_visible_bbox_lbrt": list(page.get_bbox()),
                "pdf_intrinsic_rotation_clockwise": page.get_rotation(),
                "additional_rotation_clockwise": 0,
                "render_scale_requested": options.render_scale,
                "render_scale_used": scale,
                "mapping": "FPDF_PageToDevice using full preview dimensions and rotation=0",
                "length_unit_of_drawing": None,
            },
            "preview_file": preview_file,
            "text_state": text_state,
            "raw_text_engine_order": text,
            "characters": chars,
            "warnings": notes,
            "limitations": [
                "Native text is decoded, not proof of visual completeness or correctness.",
                "Invisible/OCR layers in the source may disagree with visible content.",
                "No table, dimension endpoints, element identity or semantic meaning resolved.",
                "PDF canvas units are not construction units; UserUnit is not inferred.",
                "Small text may need a higher resolution local render from the original PDF.",
            ],
        }
        _json_write(output_dir / observations_file, data)
        return PreparedPage(
            view_index=page_index + 1, page_number=page_index + 1,
            pdf_page_label=data["pdf_page_label"], status=status,
            preview_file=preview_file, observations_file=observations_file,
            width_px=width_px, height_px=height_px, native_char_count=len(chars),
            located_char_count=sum(c["visible_region"] is not None for c in chars),
            text_state=text_state, warnings=notes, error=error,
        )


def _prepare_pdf(data, source, output_dir, options, pdfium, raw):
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise SourcePreparationError(
            f"PDF_OPEN_FAILED:{type(exc).__name__}:code={getattr(exc, 'err_code', None)}"
        ) from exc
    with pdf:
        source.page_count = len(pdf)
        if source.page_count > options.max_pages:
            raise SourcePreparationError("PDF_PAGE_LIMIT_EXCEEDED")
        if source.page_count < 1:
            raise SourcePreparationError("PDF_HAS_NO_PAGES")
        # Render widgets, when present, as well as ordinary page content/annotations.
        pdf.init_forms()
        for index in range(source.page_count):
            try:
                prepared = _prepare_pdf_page(
                    pdf, index, source.document_id, output_dir, options, pdfium, raw,
                )
            except Exception as exc:
                prepared = PreparedPage(
                    view_index=index + 1, page_number=index + 1, pdf_page_label=None,
                    status="FAILED", preview_file=None, observations_file=None,
                    error=str(exc) if isinstance(exc, SourcePreparationError)
                    else f"PAGE_PREPARATION_FAILED:{type(exc).__name__}",
                )
            source.pages.append(prepared)


def _prepare_image(data, source, output_dir, options, Image, ImageOps):
    import io
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as original:
            width, height = original.size
            if width * height > options.max_image_pixels:
                raise SourcePreparationError("SOURCE_IMAGE_PIXEL_LIMIT_EXCEEDED")
            if getattr(original, "n_frames", 1) != 1:
                raise SourcePreparationError("ANIMATED_OR_MULTIFRAME_IMAGE_UNSUPPORTED")
            orientation = original.getexif().get(274)
            if orientation is not None and orientation not in range(1, 9):
                raise SourcePreparationError("INVALID_EXIF_ORIENTATION")
            oriented = ImageOps.exif_transpose(original)
            try:
                scale = _scale_for_size(*oriented.size, options, requested=1.0)
                size = (max(1, math.floor(oriented.width * scale)),
                        max(1, math.floor(oriented.height * scale)))
                # Composite transparency explicitly; never mutate the original source.
                rgba = oriented.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                try:
                    white.alpha_composite(rgba)
                    rgb = white.convert("RGB")
                    try:
                        with rgb.resize(size, Image.Resampling.LANCZOS) as preview:
                            with (output_dir / "image.png").open("xb") as out:
                                preview.save(out, format="PNG")
                    finally:
                        rgb.close()
                finally:
                    rgba.close()
                    white.close()
                frame = {
                    "original_size_px": [width, height],
                    "original_exif_orientation": orientation,
                    "exif_transform_applied": orientation in range(2, 9)
                    if orientation is not None else False,
                    "oriented_size_px": list(oriented.size),
                    "preview_width_px": size[0], "preview_height_px": size[1],
                    "origin_preview": "TOP_LEFT",
                    "scale_x_from_oriented": size[0] / oriented.width,
                    "scale_y_from_oriented": size[1] / oriented.height,
                    "visual_orientation_verified": False,
                    "length_unit_of_drawing": None,
                }
            finally:
                oriented.close()
    notes = ["VISUAL_READING_REQUIRED_NO_OCR_RUN", "VISUAL_ORIENTATION_NOT_INFERRED"]
    if scale < 1:
        notes.append("PREVIEW_DOWNSCALED_TO_RESOURCE_LIMIT")
    _json_write(output_dir / "image.json", {
        "schema_version": _SCHEMA_VERSION,
        "document_id": source.document_id, "page_number": None,
        "source_type": "IMAGE", "status": "PREPARED", "preview_file": "image.png",
        "frame": frame, "text_state": "IMAGE_NO_NATIVE_TEXT",
        "raw_text_engine_order": "", "characters": [], "warnings": notes,
    })
    source.page_count = None
    source.pages.append(PreparedPage(
        view_index=1, page_number=None, pdf_page_label=None, status="PREPARED",
        preview_file="image.png", observations_file="image.json",
        width_px=size[0], height_px=size[1], text_state="IMAGE_NO_NATIVE_TEXT", warnings=notes,
    ))


def prepare_source(
    path: Path,
    output_root: Path,
    *,
    logical_name: str | None = None,
    input_index: int = 1,
    expected_sha256: str | None = None,
    options: SourcePreparationOptions | None = None,
) -> PreparedSource:
    """Prepare one source. Original bytes are only read; artifacts are written separately.

    PDF/image failures are recorded in source.json where possible. Invalid paths,
    changed inputs and occupied output destinations raise SourcePreparationError.
    PREPARED means artifacts produced, never 'AI extraction approved'.
    """
    options = options or SourcePreparationOptions()
    path, output_root = Path(path), Path(output_root)
    if path.is_symlink() or not path.is_file():
        raise SourcePreparationError("INPUT_MUST_BE_A_REGULAR_FILE")
    if type(input_index) is not int or input_index < 1:
        raise SourcePreparationError("INPUT_INDEX_MUST_BE_POSITIVE")
    if path.stat().st_size > options.max_file_bytes:
        raise SourcePreparationError("SOURCE_FILE_SIZE_LIMIT_EXCEEDED")
    with path.open("rb") as stream:
        data = stream.read(options.max_file_bytes + 1)
    if len(data) > options.max_file_bytes:
        raise SourcePreparationError("SOURCE_FILE_SIZE_LIMIT_EXCEEDED")
    if not data:
        raise SourcePreparationError("EMPTY_SOURCE")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise SourcePreparationError("SOURCE_HASH_CHANGED_OR_MISMATCHED")
    media_type = _media_type(data)
    label = logical_name if logical_name is not None else path.name
    identity = document_identity(label, digest)
    destination = output_root / identity
    if destination.is_symlink() or destination.exists():
        raise SourcePreparationError("OUTPUT_ALREADY_EXISTS_USE_NEW_RUN_DIRECTORY")
    pdfium, raw, Image, ImageOps = _load_engines()
    destination.mkdir(parents=True, exist_ok=False)
    source = PreparedSource(
        schema_version=_SCHEMA_VERSION, document_id=identity, content_sha256=digest,
        logical_name=label, input_index=input_index, size_bytes=len(data),
        media_type=media_type, status="PREPARING", page_count=None,
        engine_versions={"pypdfium2": str(pdfium.PYPDFIUM_INFO),
                         "pdfium": str(pdfium.PDFIUM_INFO), "pillow": version("pillow")},
        provenance={
            "input_bytes_modified": False,
            "semantic_extraction_run": False, "ocr_run": False, "network_calls": 0,
            "upstream_source_id": None,
            "identity_note": "Private source identity; not a replacement for API source-N IDs.",
            "options": asdict(options),
        },
    )
    with _PREPARATION_LOCK:
        try:
            if media_type == "application/pdf":
                _prepare_pdf(data, source, destination, options, pdfium, raw)
            else:
                _prepare_image(data, source, destination, options, Image, ImageOps)
        except Exception as exc:
            source.warnings.append(str(exc) if isinstance(exc, SourcePreparationError)
                                   else f"SOURCE_PREPARATION_FAILED:{type(exc).__name__}")
            source.status = "FAILED"
        else:
            source.status = "PREPARED" if all(p.status == "PREPARED" for p in source.pages) \
                else ("PARTIAL" if any(p.status != "FAILED" for p in source.pages) else "FAILED")
    _json_write(destination / "source.json", asdict(source))
    return source
