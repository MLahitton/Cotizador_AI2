"""Prepare and validate page-localization jobs without calling any AI provider.

Generic over prepared PDF/JPEG/PNG sources. Reads no gold answers or project rules.
All model boxes/links remain proposals; valid coordinates are not semantic validation.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from app.models.document_localization_v1 import PageLocalizationProposal
from app.services.localization_prompt_v1 import (
    PROMPT_VERSION,
    build_localization_prompt,
    prompt_digest,
)

MAX_JSON_BYTES = 80 * 1024 * 1024
MAX_PREVIEW_BYTES = 20 * 1024 * 1024
MAX_PREVIEW_PIXELS = 32_000_000
MAX_VIEWS = 500
MAX_NATIVE_TOKENS = 500
MAX_NATIVE_CHARS = 8000


class LocalizationError(ValueError):
    """Invalid source/plan/response; never a successful interpretation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, limit: int = MAX_JSON_BYTES) -> Any:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise LocalizationError("JSON_MISSING_OR_TOO_LARGE")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LocalizationError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise LocalizationError("NONFINITE_JSON_VALUE")

    try:
        return json.loads(path.read_text(encoding="utf-8-sig"),
                          object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LocalizationError("INVALID_JSON") from exc


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def safe_child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise LocalizationError("INVALID_ARTIFACT_PATH")
    if "\0" in relative or ":" in relative:
        raise LocalizationError("INVALID_ARTIFACT_PATH")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in {"..", ".", ""} for part in relative.split("/")):
        raise LocalizationError("INVALID_ARTIFACT_PATH")
    root = Path(root).resolve()
    child = root.joinpath(*rel.parts)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise LocalizationError("SYMLINK_ARTIFACT_NOT_ALLOWED")
    if not child.resolve().is_relative_to(root) or not child.is_file():
        raise LocalizationError("ARTIFACT_MISSING_OR_OUTSIDE_RUN")
    return child


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise LocalizationError(f"INVALID_{name}")
    return value


def _image_bytes_and_size(path: Path) -> tuple[bytes, tuple[int, int]]:
    from PIL import Image

    if path.stat().st_size > MAX_PREVIEW_BYTES:
        raise LocalizationError("PREVIEW_BYTE_LIMIT")
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise LocalizationError("EXPECTED_SINGLE_PREPARED_PNG")
        size = image.size
        if min(size) < 1 or size[0] * size[1] > MAX_PREVIEW_PIXELS:
            raise LocalizationError("PREVIEW_PIXEL_LIMIT")
        image.verify()
    return data, size


def _valid_region(region: Any) -> bool:
    if not isinstance(region, dict):
        return False
    values = [region.get(key) for key in ("x", "y", "width", "height")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        return False
    x, y, width, height = values
    return (x >= 0 and y >= 0 and width > 0 and height > 0
            and x + width <= 1 + 1e-9 and y + height <= 1 + 1e-9)


def _union(regions: list[dict]) -> dict[str, float]:
    x = min(r["x"] for r in regions)
    y = min(r["y"] for r in regions)
    x1 = max(r["x"] + r["width"] for r in regions)
    y1 = max(r["y"] + r["height"] for r in regions)
    return {"x": x, "y": y, "width": x1 - x, "height": y1 - y}


def native_tokens(observations: dict) -> dict[str, Any]:
    """Short spatially adjacent runs, never rebuilt rows or inferred table cells.

    PDF engine index order is kept. Spaces, missing/error glyphs and large geometric
    gaps terminate a run. The original character observations remain untouched.
    """
    tokens = []
    run = []
    characters = observations.get("characters", [])
    if not isinstance(characters, list) or len(characters) > 100_000:
        raise LocalizationError("INVALID_CHARACTER_OBSERVATIONS")

    def flush():
        if run:
            tokens.append({
                "text": "".join(c["text"] for c in run),
                "char_indices": [c["pdf_char_index"] for c in run],
                "region": _union([c["visible_region"] for c in run]),
            })
            run.clear()

    for char in characters:
        text = char.get("text", "")
        region = char.get("visible_region")
        usable = (isinstance(text, str) and text and not text.isspace()
                  and text.isprintable() and "\ufffd" not in text
                  and not char.get("unicode_mapping_error")
                  and not char.get("generated_by_text_engine") and _valid_region(region)
                  and type(char.get("pdf_char_index")) is int)
        if not usable:
            flush()
            continue
        if run:
            prev = run[-1]
            pr = prev["visible_region"]
            dx = abs((region["x"] + region["width"] / 2) - (pr["x"] + pr["width"] / 2))
            dy = abs((region["y"] + region["height"] / 2) - (pr["y"] + pr["height"] / 2))
            horizontal = dy < 0.6 * max(region["height"], pr["height"]) and dx < 2.5 * max(
                region["width"], pr["width"])
            vertical = dx < 0.6 * max(region["width"], pr["width"]) and dy < 2.5 * max(
                region["height"], pr["height"])
            if (char["pdf_char_index"] != prev["pdf_char_index"] + 1
                    or len(run) >= 64 or not (horizontal or vertical)):
                flush()
        run.append(char)
    flush()
    selected = []
    size = 0
    for token in tokens:
        if len(selected) >= MAX_NATIVE_TOKENS or size + len(token["text"]) > MAX_NATIVE_CHARS:
            break
        selected.append(token)
        size += len(token["text"])
    return {
        "origin": "PDF_NATIVE_DECODED_UNVERIFIED",
        "reading_order": "ENGINE_INDEX_ORDER_NOT_VALIDATED",
        "tokens": selected, "total_available": len(tokens),
        "included": len(selected), "truncated": len(selected) != len(tokens),
        "warning": "No token or box establishes a field meaning or an element association.",
    }


def tile_windows(width: int, height: int, mode: str) -> list[list[int]]:
    if mode not in {"auto", "none"}:
        raise LocalizationError("INVALID_TILE_MODE")
    if mode == "none" or max(width, height) <= 2400:
        return []
    # Preview crops only: they do not add detail lost when Phase 2 downsized the source.
    mx, my = width // 2, height // 2
    ox, oy = max(1, width // 40), max(1, height // 40)
    return [
        [0, 0, min(width, mx + ox), min(height, my + oy)],
        [max(0, mx - ox), 0, width, min(height, my + oy)],
        [0, max(0, my - oy), min(width, mx + ox), height],
        [max(0, mx - ox), max(0, my - oy), width, height],
    ]


def build_plan(report_path: Path, *, tiles: str = "auto") -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    report = read_json(report_path, 5 * 1024 * 1024)
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("status") != "PREPARATION_COMPLETED"
            or report.get("source_archive_unchanged") is not True):
        raise LocalizationError("EXPECTED_COMPLETED_PHASE2_REPORT")
    docs = report.get("documents", [])
    if not isinstance(docs, list) or not docs:
        raise LocalizationError("NO_PREPARED_DOCUMENTS")
    jobs = []
    identities = set()
    for doc in docs:
        if doc.get("status") != "PREPARED":
            raise LocalizationError("SOURCE_NOT_PREPARED")
        document_id = doc.get("document_id", "")
        if not re.fullmatch(r"doc-[a-f0-9]{24}", document_id) or document_id in identities:
            raise LocalizationError("INVALID_OR_DUPLICATE_DOCUMENT_ID")
        identities.add(document_id)
        project_id = doc.get("project_id")
        if not isinstance(project_id, str) or not project_id:
            raise LocalizationError("MISSING_PROJECT_ID")
        media = doc.get("media_type")
        if media not in {"application/pdf", "image/jpeg", "image/png"}:
            raise LocalizationError("UNSUPPORTED_PREPARED_MEDIA")
        pages = doc.get("pages", [])
        if not pages:
            raise LocalizationError("NO_PREPARED_VIEWS")
        if media == "application/pdf" and len(pages) != doc.get("page_count"):
            raise LocalizationError("INCOMPLETE_PDF_VIEW_LIST")
        if media != "application/pdf" and len(pages) != 1:
            raise LocalizationError("EXPECTED_SINGLE_IMAGE_VIEW")
        for index, page in enumerate(pages, start=1):
            if page.get("status") != "PREPARED" or page.get("view_index") != index:
                raise LocalizationError("INVALID_OR_UNPREPARED_VIEW")
            pn = index if media == "application/pdf" else None
            if page.get("page_number") != pn:
                raise LocalizationError("INVALID_PHYSICAL_PAGE_NUMBER")
            directory = doc.get("artifact_directory", "")
            preview_rel = directory + "/" + str(page.get("preview_file", ""))
            observations_rel = directory + "/" + str(page.get("observations_file", ""))
            preview = safe_child(report_path.parent, preview_rel)
            observation_path = safe_child(report_path.parent, observations_rel)
            _data, (width, height) = _image_bytes_and_size(preview)
            width_expected = _positive_int(page.get("width_px"), "WIDTH")
            height_expected = _positive_int(page.get("height_px"), "HEIGHT")
            obs = read_json(observation_path)
            frame = obs.get("frame", {})
            if ((width, height) != (width_expected, height_expected)
                    or obs.get("document_id") != document_id or obs.get("page_number") != pn
                    or frame.get("preview_width_px") != width or frame.get("preview_height_px") != height
                    or frame.get("origin_preview") != "TOP_LEFT"
                    or obs.get("preview_file") != page.get("preview_file")
                    or obs.get("schema_version") != 1 or obs.get("status") != "PREPARED"):
                raise LocalizationError("PREVIEW_OBSERVATION_IDENTITY_OR_FRAME_MISMATCH")
            tokens = native_tokens(obs)
            windows = tile_windows(width, height, tiles)
            jobs.append({
                "job_id": f"{document_id}-v{index:04d}",
                "project_id": project_id, "document_id": document_id,
                "corpus_document_id": doc.get("corpus_document_id"),
                "logical_name": doc.get("logical_name"),
                "content_sha256": doc.get("content_sha256"),
                "page_number": pn, "view_index": index, "media_type": media,
                "preview_relative_path": preview_rel,
                "observations_relative_path": observations_rel,
                "preview_sha256": file_sha256(preview),
                "observations_sha256": file_sha256(observation_path),
                "width_px": width, "height_px": height,
                "tile_windows_px": windows, "images_per_request": 1 + len(windows),
                "native_tokens_included": tokens["included"],
                "native_tokens_available": tokens["total_available"],
                "native_tokens_truncated": tokens["truncated"],
                "text_state": page.get("text_state"),
                "preparation_warnings": list(page.get("warnings", [])),
            })
            if len(jobs) > MAX_VIEWS:
                raise LocalizationError("VIEW_LIMIT_EXCEEDED")
    # Ordering does not define identity; do not reuse API source-N ids across runs.
    jobs.sort(key=lambda j: (j["project_id"], j["document_id"], j["view_index"]))
    return {
        "schema_version": 1, "status": "LOCALIZATION_PLAN_READY",
        "preparation_report_path": str(report_path),
        "preparation_report_sha256": file_sha256(report_path),
        "preparation_run_id": report.get("run_id"),
        "source_archive_sha256": report.get("source_archive_sha256"),
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_digest(),
        "response_schema_sha256": response_schema_digest(),
        "tile_mode": tiles, "projects": len({j["project_id"] for j in jobs}),
        "documents": len(docs), "views": len(jobs),
        "planned_requests": len(jobs),
        "planned_images": sum(j["images_per_request"] for j in jobs),
        "views_without_native_text": sum(j["native_tokens_available"] == 0 for j in jobs),
        "network_calls": 0, "localization_model_run": False,
        "canonical_extraction_run": False, "corpus_approved": False,
        "jobs": jobs,
        "limitations": [
            "Readiness is not localization accuracy or semantic extraction.",
            "Asset hashes freeze current prepared artifacts, not proof of visual completeness.",
            "Tile crops do not recover detail beyond Phase 2 preview resolution.",
            "Rotated handwritten images may need an additional verified orientation step.",
        ],
    }


def response_schema_digest() -> str:
    raw = json.dumps(PageLocalizationProposal.model_json_schema(), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def select_jobs(plan: dict, project: str | None = None,
                document: str | None = None) -> list[dict]:
    jobs = plan["jobs"]
    selected = [j for j in jobs if (project is None or j["project_id"] == project)
                and (document is None or j["document_id"] == document
                     or j["corpus_document_id"] == document)]
    if not selected:
        raise LocalizationError("SELECTION_HAS_NO_JOBS")
    return selected


def request_for_job(plan: dict, job: dict) -> tuple[str, list[tuple[str, str, bytes]], dict]:
    """Recheck immutable assets immediately before a request or offline replay."""
    if plan["prompt_sha256"] != prompt_digest() or plan["response_schema_sha256"] != response_schema_digest():
        raise LocalizationError("CODE_CHANGED_REBUILD_LOCALIZATION_PLAN")
    report_path = Path(plan["preparation_report_path"])
    if file_sha256(report_path) != plan["preparation_report_sha256"]:
        raise LocalizationError("PREPARATION_REPORT_CHANGED")
    preview = safe_child(report_path.parent, job["preview_relative_path"])
    obs_path = safe_child(report_path.parent, job["observations_relative_path"])
    if (file_sha256(preview) != job["preview_sha256"]
            or file_sha256(obs_path) != job["observations_sha256"]):
        raise LocalizationError("PREPARED_ASSET_CHANGED")
    data, size = _image_bytes_and_size(preview)
    if list(size) != [job["width_px"], job["height_px"]]:
        raise LocalizationError("PREVIEW_SIZE_CHANGED")
    obs = read_json(obs_path)
    width, height = size
    from PIL import Image

    photo = job["media_type"].startswith("image/")

    def encode(image):
        # Photo PNGs are much larger than their original JPEG. Use a recorded
        # bounded JPEG derivative for the MODEL only; archival preview stays intact.
        stream = io.BytesIO()
        if photo:
            with image.convert("RGB") as rgb:
                rgb.save(stream, format="JPEG", quality=95, subsampling=0)
            return "image/jpeg", stream.getvalue()
        image.save(stream, format="PNG", compress_level=1)
        return "image/png", stream.getvalue()

    with Image.open(io.BytesIO(data)) as original:
        mime, full_data = encode(original) if photo else ("image/png", data)
    images = [("Pagina/imagen completa. Marco de coordenadas para TODAS las cajas.", mime, full_data)]

    tile_context = []
    with Image.open(io.BytesIO(data)) as image:
        for index, window in enumerate(job["tile_windows_px"], start=1):
            x0, y0, x1, y1 = window
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise LocalizationError("INVALID_TILE_WINDOW")
            page_box = [y0 * 1000 / height, x0 * 1000 / width,
                        y1 * 1000 / height, x1 * 1000 / width]
            with image.crop(window) as crop:
                mime, encoded = encode(crop)
                images.append((f"Ampliacion {index} de la MISMA pagina. Su box_2d global: {page_box}",
                               mime, encoded))
            tile_context.append({"tile_index": index, "full_page_box_2d": page_box})
    context = {
        "job_id": job["job_id"], "physical_page_number": job["page_number"],
        "view_index": job["view_index"], "source_media_type": job["media_type"],
        "frame": {"origin": "TOP_LEFT", "width_px": width, "height_px": height,
                  "box_2d_order": "y0,x0,y1,x1", "box_scale": 1000},
        "tiles": tile_context,
        "model_image_derivative": "JPEG_QUALITY_95_FOR_PHOTO" if photo else "PREPARED_PNG",
        "native": native_tokens(obs),
        "preparation_warnings": job["preparation_warnings"],
    }
    prompt = build_localization_prompt(context)
    # Inline base64 has overhead; conservative limit below the usual inline request size.
    if sum(len(b) for _, _, b in images) * 4 / 3 + len(prompt.encode()) > 18 * 1024 * 1024:
        raise LocalizationError("INLINE_REQUEST_BUDGET_EXCEEDED")
    return prompt, images, obs


def parse_proposal(text: str, job_id: str) -> PageLocalizationProposal:
    if not isinstance(text, str) or len(text.encode()) > 2 * 1024 * 1024:
        raise LocalizationError("MODEL_RESPONSE_EMPTY_OR_TOO_LARGE")
    # Strict JSON only; never repair numbers, boxes, ids or truncated responses.
    try:
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError("Duplicate key")
                value[key] = item
            return value
        data = json.loads(text, object_pairs_hook=pairs)
        proposal = PageLocalizationProposal.model_validate(data)
    except (ValueError, TypeError) as exc:
        raise LocalizationError("INVALID_LOCALIZATION_RESPONSE") from exc
    if proposal.job_id != job_id:
        raise LocalizationError("MODEL_JOB_ID_MISMATCH")
    return proposal


def _native_region_slice(obs: dict, box: list[float]) -> dict:
    y0, x0, y1, x1 = [v / 1000 for v in box]
    inside, touching = [], []
    for char in obs.get("characters", []):
        r = char.get("visible_region")
        if not _valid_region(r):
            continue
        rx0, ry0, rx1, ry1 = r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]
        if not (rx1 > x0 and rx0 < x1 and ry1 > y0 and ry0 < y1):
            continue
        target = inside if rx0 >= x0 and rx1 <= x1 and ry0 >= y0 and ry1 <= y1 else touching
        target.append(char["pdf_char_index"])
    return {"contained_char_indices": inside, "boundary_char_indices": touching,
            "association": "GEOMETRIC_OVERLAP_ONLY_NOT_SEMANTIC_PROOF"}


def save_proposal_artifacts(plan: dict, job: dict, text: str, output_dir: Path,
                            *, origin: str = "MODEL_RESPONSE") -> dict:
    """Write exact region crops and links for inspection. Never edit source or extraction."""
    _prompt, images, obs = request_for_job(plan, job)
    proposal = parse_proposal(text, job["job_id"])
    width, height = job["width_px"], job["height_px"]
    crop_pixels = sum(
        math.ceil((r.box_2d[3] - r.box_2d[1]) * width / 1000)
        * math.ceil((r.box_2d[2] - r.box_2d[0]) * height / 1000)
        for r in proposal.regions
    )
    if crop_pixels > 128_000_000:
        raise LocalizationError("OUTPUT_CROP_PIXEL_BUDGET_EXCEEDED")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise LocalizationError("OUTPUT_ALREADY_EXISTS")
    output_dir.mkdir(parents=True, exist_ok=False)
    from PIL import Image, ImageDraw

    region_rows = []
    preview_path = safe_child(Path(plan["preparation_report_path"]).parent,
                              job["preview_relative_path"])
    with Image.open(preview_path) as image:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        width, height = image.size
        try:
            for region in proposal.regions:
                y0, x0, y1, x1 = region.box_2d
                bounds = [math.floor(x0 * width / 1000), math.floor(y0 * height / 1000),
                          math.ceil(x1 * width / 1000), math.ceil(y1 * height / 1000)]
                with image.crop(bounds) as crop:
                    crop.save(output_dir / f"{region.region_id}.png")
                draw.rectangle(bounds, outline="red", width=max(1, width // 1200))
                draw.text((bounds[0] + 2, bounds[1] + 2), region.region_id, fill="red")
                row = region.model_dump(mode="json")
                row.update({"crop_file": f"{region.region_id}.png", "crop_bbox_px": bounds,
                            "region_normalized": {"x": x0 / 1000, "y": y0 / 1000,
                                                  "width": (x1 - x0) / 1000,
                                                  "height": (y1 - y0) / 1000},
                            "text_origin": "MODEL_TRANSCRIPTION_UNVERIFIED",
                            "native_slice": _native_region_slice(obs, region.box_2d)})
                region_rows.append(row)
            annotated.save(output_dir / "regions_overview.png")
        finally:
            annotated.close()
    linked = {rid for element in proposal.elements for rid in element.linked_region_ids()}
    result = {
        "schema_version": 1, "status": "PROPOSALS_RECORDED_UNVERIFIED", "origin": origin,
        "job_id": job["job_id"], "document_id": job["document_id"],
        "content_sha256": job["content_sha256"], "project_id": job["project_id"],
        "physical_page_number": job["page_number"], "view_index": job["view_index"],
        "preview_sha256": job["preview_sha256"], "observations_sha256": job["observations_sha256"],
        "page_roles": proposal.page_roles, "coverage_claim": proposal.coverage,
        "suggested_rotation_clockwise": proposal.suggested_rotation_clockwise,
        "orientation_applied": False,
        "elements": [element.model_dump(mode="json") for element in proposal.elements],
        "regions": region_rows,
        "unassigned_region_ids": [r.region_id for r in proposal.regions if r.region_id not in linked],
        "issues": proposal.issues,
        "structural_validation": "PASSED", "visual_association_validated": False,
        "canonical_extraction_run": False, "corpus_approved": False,
        "limitations": ["Boxes and links are model proposals, not verified ownership.",
                        "Crops use prepared preview pixels and no resolution enhancement.",
                        "Repeated references are preserved; no cross-page identity merge.",
                        "No quantity, measurement, specification or scope is resolved here."],
    }
    write_json(output_dir / "localization.json", result)
    return result
