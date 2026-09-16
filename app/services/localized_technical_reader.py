"""Experimental technical reading from saved localization artifacts."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from pydantic import ValidationError

from app.models.localized_technical_reading import (
    LocalizedCandidateTrace,
    LocalizedTechnicalReadEnvelope,
    LocalizedTechnicalReadingResult,
)
from app.providers.gemini_localization_v1 import _compact_api_schema
from app.services.document_localization_v1 import (
    file_sha256,
    read_json,
    safe_child,
    write_json,
)
from app.services.localized_technical_prompt import (
    LOCALIZED_TECHNICAL_PROMPT_VERSION,
    build_localized_technical_prompt,
    localized_technical_prompt_digest,
)

MAX_CONTEXT_IMAGES_PER_CANDIDATE = 16
MAX_CONTEXT_PIXELS_PER_CANDIDATE = 24_000_000


class LocalizedTechnicalReaderError(ValueError):
    """Invalid saved localization or technical model response."""


class LocalizedTechnicalProvider(Protocol):
    model: str

    def generate_localized_technical(
        self,
        prompt: str,
        images: list[tuple[str, str, bytes]],
    ) -> dict[str, Any]:
        """Return a response envelope with at least text and finish_reason."""

    def close(self) -> None:
        """Release provider resources."""


def localized_technical_api_schema() -> dict[str, Any]:
    schema = LocalizedTechnicalReadEnvelope.model_json_schema()
    return _compact_api_schema(schema, schema.get("$defs", {}))


def candidate_identity(report: dict[str, Any], candidate: dict[str, Any]) -> str:
    return "|".join((
        str(report["document_id"]),
        str(report["view_index"]),
        str(candidate["candidate_id"]),
    ))


def build_localized_technical_plan(
    localization_report_path: Path,
    *,
    preparation_report_path: Path | None = None,
) -> dict[str, Any]:
    report = read_json(localization_report_path, 10 * 1024 * 1024)
    if not isinstance(report, dict) or report.get("status") != "PROPOSALS_RECORDED_UNVERIFIED":
        raise LocalizedTechnicalReaderError("EXPECTED_LOCALIZATION_PROPOSALS_REPORT")
    if report.get("visual_association_validated") is not False:
        raise LocalizedTechnicalReaderError("LOCALIZATION_MUST_REMAIN_UNVERIFIED")
    prepared = preparation_report_path or report.get("preparation_report_path")
    if prepared is None:
        raise LocalizedTechnicalReaderError("PREPARATION_REPORT_REQUIRED")
    prepared_path = Path(prepared).resolve()
    if not prepared_path.is_file():
        raise LocalizedTechnicalReaderError("PREPARATION_REPORT_MISSING")
    candidates = []
    for candidate in report.get("elements", []):
        candidates.append({
            "technical_candidate_id": candidate_identity(report, candidate),
            "candidate_id": candidate["candidate_id"],
            "reference_raw": candidate.get("reference_raw"),
            "linked_region_ids": _linked_region_ids(candidate),
        })
    return {
        "schema_version": 1,
        "status": "LOCALIZED_TECHNICAL_READING_PLAN_READY",
        "localization_report_path": str(Path(localization_report_path).resolve()),
        "localization_report_sha256": file_sha256(Path(localization_report_path)),
        "preparation_report_path": str(prepared_path),
        "preparation_report_sha256": file_sha256(prepared_path),
        "prompt_version": LOCALIZED_TECHNICAL_PROMPT_VERSION,
        "prompt_sha256": localized_technical_prompt_digest(),
        "response_schema_sha256": _schema_digest(localized_technical_api_schema()),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "network_calls": 0,
        "experimental": True,
        "ready_for_pricing": False,
    }


def select_candidate_jobs(plan: dict[str, Any], candidate_id: str | None = None) -> list[dict]:
    candidates = plan.get("candidates", [])
    selected = [
        candidate for candidate in candidates
        if candidate_id is None
        or candidate["candidate_id"] == candidate_id
        or candidate["technical_candidate_id"] == candidate_id
    ]
    if not selected:
        raise LocalizedTechnicalReaderError("SELECTION_HAS_NO_CANDIDATES")
    return selected


def request_for_candidate(
    plan: dict[str, Any],
    candidate_job: dict[str, Any],
    *,
    margin_px: int = 24,
) -> tuple[str, list[tuple[str, str, bytes]], dict[str, Any]]:
    if plan.get("prompt_sha256") != localized_technical_prompt_digest():
        raise LocalizedTechnicalReaderError("LOCALIZED_TECHNICAL_PROMPT_CHANGED")
    if plan.get("response_schema_sha256") != _schema_digest(localized_technical_api_schema()):
        raise LocalizedTechnicalReaderError("LOCALIZED_TECHNICAL_SCHEMA_CHANGED")
    report_path = Path(plan["localization_report_path"])
    if file_sha256(report_path) != plan["localization_report_sha256"]:
        raise LocalizedTechnicalReaderError("LOCALIZATION_REPORT_CHANGED")
    report = read_json(report_path, 10 * 1024 * 1024)
    if file_sha256(Path(plan["preparation_report_path"])) != plan["preparation_report_sha256"]:
        raise LocalizedTechnicalReaderError("PREPARATION_REPORT_CHANGED")
    candidate = _candidate_by_id(report, candidate_job["candidate_id"])
    root = Path(plan["preparation_report_path"]).resolve().parent
    preview = safe_child(root, _preview_relative_path(report))
    with Image.open(preview) as image:
        width, height = image.size
        images = [_encode_image("page_context", image)]
        package_regions = []
        blocked = set(candidate.get("evidence_blocked_region_ids", []))
        for region in _linked_regions(report, candidate):
            row = dict(region)
            row["usable_as_support"] = _region_usable_as_support(row, blocked)
            context = _context_bounds(row["crop_bbox_px"], (width, height), margin_px)
            row["context_bbox_px"] = context
            row["proposed_bbox_px"] = row["crop_bbox_px"]
            package_regions.append(row)
            if row["usable_as_support"] and len(images) < MAX_CONTEXT_IMAGES_PER_CANDIDATE:
                with image.crop(tuple(context)) as crop:
                    images.append(_encode_image(f"context_{row['region_id']}", crop))
    if sum(_image_pixels(data) for _, _, data in images) > MAX_CONTEXT_PIXELS_PER_CANDIDATE:
        raise LocalizedTechnicalReaderError("CANDIDATE_CONTEXT_PIXEL_BUDGET_EXCEEDED")
    source_id = f"localized:{report['document_id']}:{report['view_index']}"
    context = {
        "technical_candidate_id": candidate_job["technical_candidate_id"],
        "temporary_id": candidate_job["technical_candidate_id"],
        "candidate_id": candidate["candidate_id"],
        "reference_raw": candidate.get("reference_raw"),
        "document_id": report["document_id"],
        "view_index": report["view_index"],
        "page_number": report.get("physical_page_number"),
        "source_id": source_id,
        "input_image_ids": [image_id for image_id, _, _ in images],
        "regions": package_regions,
        "evidence_gate": report.get("evidence_gate"),
        "warnings": _candidate_warnings(report, candidate["candidate_id"]),
        "visual_association_validated": False,
        "ready_for_pricing": False,
    }
    return build_localized_technical_prompt(context), images, context


def parse_localized_technical_response(
    text: str,
    context: dict[str, Any],
) -> LocalizedTechnicalReadEnvelope:
    if not isinstance(text, str) or not text:
        raise LocalizedTechnicalReaderError("MODEL_RESPONSE_EMPTY")
    try:
        envelope = LocalizedTechnicalReadEnvelope.model_validate_json(text)
    except ValidationError as exc:
        raise LocalizedTechnicalReaderError(
            "INVALID_LOCALIZED_TECHNICAL_RESPONSE:"
            + json.dumps(
                exc.errors(include_input=False, include_context=False, include_url=False),
                separators=(",", ":"),
            )
        ) from exc
    if len(envelope.enrichment.elements) != 1:
        raise LocalizedTechnicalReaderError("EXPECTED_ONE_ENRICHED_ELEMENT")
    if envelope.enrichment.elements[0].temporary_id != context["temporary_id"]:
        raise LocalizedTechnicalReaderError("TECHNICAL_CANDIDATE_ID_MISMATCH")
    allowed_images = set(context["input_image_ids"])
    allowed_source = context["source_id"]
    for evidence in envelope.field_evidence:
        if evidence.input_image_id not in allowed_images:
            raise LocalizedTechnicalReaderError("UNKNOWN_INPUT_IMAGE_ID")
        if evidence.source_id != allowed_source:
            raise LocalizedTechnicalReaderError("UNKNOWN_SOURCE_ID")
    return envelope


def record_localized_technical_response(
    plan: dict[str, Any],
    candidate_job: dict[str, Any],
    response: dict[str, Any],
    output_dir: Path,
) -> LocalizedTechnicalReadingResult:
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise LocalizedTechnicalReaderError("OUTPUT_ALREADY_EXISTS")
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt, images, context = request_for_candidate(plan, candidate_job)
    (output_dir / "request_prompt.txt").write_text(prompt, encoding="utf-8")
    write_json(output_dir / "candidate_package.json", context)
    write_json(output_dir / "response_envelope.json", response)
    if response.get("finish_reason") != "STOP":
        raise LocalizedTechnicalReaderError("RESPONSE_NOT_COMPLETE_STOP")
    text = response.get("text")
    envelope = parse_localized_technical_response(text, context)
    write_json(output_dir / "raw_parsed_reading.json", envelope.model_dump(mode="json"))
    trace = LocalizedCandidateTrace(
        technical_candidate_id=context["technical_candidate_id"],
        document_id=context["document_id"],
        view_index=context["view_index"],
        localization_candidate_id=context["candidate_id"],
        reference_raw=context["reference_raw"],
        source_id=context["source_id"],
        linked_region_ids=[row["region_id"] for row in context["regions"]],
        excluded_blocked_region_ids=[
            row["region_id"] for row in context["regions"] if not row["usable_as_support"]
        ],
        field_evidence=envelope.field_evidence,
        pending=envelope.pending,
        conflicts=envelope.conflicts,
        warnings=context["warnings"],
    )
    result = LocalizedTechnicalReadingResult(
        status="LOCALIZED_TECHNICAL_READING_RECORDED_UNVERIFIED",
        enrichment=envelope.enrichment,
        candidate_traces=[trace],
        warnings=list(envelope.enrichment.warnings),
        source_map={context["source_id"]: context["technical_candidate_id"]},
    )
    write_json(output_dir / "localized_technical_reading.json", result.model_dump(mode="json"))
    return result


def run_localized_technical_plan(
    plan: dict[str, Any],
    output_dir: Path,
    provider: LocalizedTechnicalProvider,
    *,
    max_calls: int,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    jobs = select_candidate_jobs(plan, candidate_id)
    if type(max_calls) is not int or not 1 <= max_calls <= 50:
        raise LocalizedTechnicalReaderError("SET_EXPLICIT_MAX_CALLS_BETWEEN_1_AND_50")
    if len(jobs) > max_calls:
        raise LocalizedTechnicalReaderError("REQUEST_BUDGET_EXCEEDED")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise LocalizedTechnicalReaderError("OUTPUT_ALREADY_EXISTS")
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCALIZED_TECHNICAL_READING_RECORDED_UNVERIFIED",
        "requested_model": provider.model,
        "requested_candidates": len(jobs),
        "network_calls_attempted": 0,
        "responses_received": 0,
        "experimental": True,
        "ready_for_pricing": False,
        "jobs": [],
    }
    try:
        for job in jobs:
            prompt, images, _context = request_for_candidate(plan, job)
            report["network_calls_attempted"] += 1
            response = provider.generate_localized_technical(prompt, images)
            report["responses_received"] += 1
            result = record_localized_technical_response(
                plan,
                job,
                response,
                output_dir / job["technical_candidate_id"].replace("|", "__"),
            )
            report["jobs"].append({
                "technical_candidate_id": job["technical_candidate_id"],
                "candidate_id": job["candidate_id"],
                "status": result.status,
                "elements": len(result.enrichment.elements),
            })
    finally:
        provider.close()
    write_json(output_dir / "localized_technical_reading_report.json", report)
    return report


def replay_localized_technical_response(
    plan: dict[str, Any],
    candidate_id: str,
    response_path: Path,
    output_dir: Path,
) -> LocalizedTechnicalReadingResult:
    jobs = select_candidate_jobs(plan, candidate_id)
    if len(jobs) != 1:
        raise LocalizedTechnicalReaderError("REPLAY_REQUIRES_ONE_CANDIDATE")
    response = read_json(response_path, 2 * 1024 * 1024)
    return record_localized_technical_response(plan, jobs[0], response, output_dir)


def _schema_digest(schema: dict[str, Any]) -> str:
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _linked_region_ids(candidate: dict[str, Any]) -> list[str]:
    linked = []
    for field in (
        "reference_region_ids",
        "drawing_region_ids",
        "table_region_ids",
        "dimension_region_ids",
        "note_region_ids",
    ):
        linked.extend(candidate.get(field, []))
    return list(dict.fromkeys(linked))


def _candidate_by_id(report: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        item for item in report.get("elements", [])
        if item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise LocalizedTechnicalReaderError("UNKNOWN_OR_DUPLICATE_CANDIDATE")
    return matches[0]


def _linked_regions(report: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    region_map = {region["region_id"]: region for region in report.get("regions", [])}
    regions = []
    for region_id in _linked_region_ids(candidate):
        if region_id not in region_map:
            raise LocalizedTechnicalReaderError("CANDIDATE_REGION_LINK_MISSING")
        regions.append(region_map[region_id])
    return regions


def _region_usable_as_support(region: dict[str, Any], blocked_ids: set[str]) -> bool:
    evidence_status = region.get("evidence_status")
    return (
        region["region_id"] not in blocked_ids
        and not (
            isinstance(evidence_status, str)
            and evidence_status.startswith("BLOCKED_")
        )
    )


def _candidate_warnings(report: dict[str, Any], candidate_id: str) -> list[str]:
    warnings = []
    gate = report.get("evidence_gate", {})
    for row in gate.get("candidate_warnings", []):
        if row.get("candidate_id") == candidate_id:
            warnings.extend(row.get("warnings", []))
    for row in gate.get("candidates_with_blocked_region_links", []):
        if row.get("candidate_id") == candidate_id:
            warnings.append("CANDIDATE_HAS_BLOCKED_REGION_LINKS")
    return list(dict.fromkeys(warnings))


def _preview_relative_path(localization_report: dict[str, Any]) -> str:
    raw = localization_report.get("preview_relative_path")
    if isinstance(raw, str) and raw:
        return raw
    document_id = localization_report.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        raise LocalizedTechnicalReaderError("LOCALIZATION_REPORT_MISSING_DOCUMENT_ID")
    if localization_report.get("physical_page_number") is None:
        return f"documents/{document_id}/image.png"
    return f"documents/{document_id}/page-{int(localization_report['view_index']):04d}.png"


def _context_bounds(bounds: list[int], size: tuple[int, int], margin_px: int) -> list[int]:
    x0, y0, x1, y1 = bounds
    width, height = size
    margin = max(0, int(margin_px))
    return [
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    ]


def _encode_image(image_id: str, image: Image.Image) -> tuple[str, str, bytes]:
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=1)
    return image_id, "image/png", stream.getvalue()


def _image_pixels(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
    return width * height
