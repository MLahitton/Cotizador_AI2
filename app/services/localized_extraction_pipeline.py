"""Opt-in localized extraction pipeline for the real requirements endpoint."""
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from google.genai import types

from app.models.evidence import Source
from app.models.gemini_discovery import GeminiDiscoveryResult
from app.models.gemini_enrichment import GeminiElementEnrichment, GeminiEnrichmentResult
from app.models.requirement import TokenUsage, Warning
from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini_localization_v1 import GeminiLocalizationClient
from app.services.document_localization_v1 import (
    IMAGE_FRAME_CONTRACT_VERSION,
    IMAGE_FRAME_LOCAL_VALIDATION_POLICY,
    build_plan,
    image_frame_request_for_job,
    image_frame_response_schema_digest,
    save_image_frame_proposal_artifacts,
    write_json,
)
from app.services.gemini_enrichment_pipeline import reconcile_and_build_gemini_extraction
from app.services.gemini_extraction_mapper import map_gemini_extraction_to_requirement_extraction
from app.services.localization_prompt_v1 import image_frame_prompt_digest
from app.services.localized_technical_prompt import LOCALIZED_TECHNICAL_SYSTEM_INSTRUCTION
from app.services.localized_technical_reader import (
    build_localized_technical_plan,
    localized_technical_api_schema,
    record_localized_technical_response,
    request_for_candidate,
    select_candidate_jobs,
)
from app.services.source_preparation_v1 import PreparedSource, prepare_source


class LocalizedExtractionPipelineError(RuntimeError):
    """The opt-in localized_v1 pipeline could not produce useful extraction data."""


class LocalizedTechnicalClient(Protocol):
    model: str

    def generate_localized_technical(
        self,
        prompt: str,
        images: list[tuple[str, str, bytes]],
    ) -> dict[str, Any]:
        """Generate technical enrichment from localized candidate context."""

    def close(self) -> None:
        """Release provider resources."""


class GeminiLocalizedTechnicalClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        from google import genai

        if not api_key or not model:
            raise ValueError("Missing API key or configured model")
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate_localized_technical(
        self,
        prompt: str,
        images: list[tuple[str, str, bytes]],
    ) -> dict[str, Any]:
        parts = []
        for label, mime_type, data in images:
            parts.append(types.Part.from_text(text=label))
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        parts.append(types.Part.from_text(text=prompt))
        response = self._client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=LOCALIZED_TECHNICAL_SYSTEM_INSTRUCTION,
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=localized_technical_api_schema(),
            ),
        )
        return {
            "text": getattr(response, "text", None),
            "finish_reason": _finish_reason(response),
            "requested_model": self.model,
            "usage": _usage(response),
        }

    def close(self) -> None:
        self._client.close()


def extract_requirement_with_localized_v1(
    files: list[Path],
    *,
    project_id: str | None,
    requirement_id: str | None,
    api_key: str,
    model: str,
    localization_client: GeminiLocalizationClient | None = None,
    technical_client: LocalizedTechnicalClient | None = None,
    debug_dir: Path | None = None,
) -> RequirementExtraction:
    """Run source preparation, localization and localized reading for uploaded files."""
    if not files:
        raise ValueError("Debes proporcionar al menos un archivo para extraer.")

    with tempfile.TemporaryDirectory(prefix="ai2-localized-v1-") as temp_dir:
        workspace = Path(temp_dir)
        prepared_sources = _prepare_uploaded_sources(
            [Path(file) for file in files],
            workspace,
            project_id=project_id,
        )
        preparation_report_path = workspace / "source_preparation_report.json"
        preparation_report = _write_preparation_report(
            preparation_report_path,
            prepared_sources,
            project_id=project_id,
        )
        _debug_write(debug_dir, "01-source-preparation.json", preparation_report)

        localization_plan = _image_frame_plan(preparation_report_path)
        localization_result = _run_localization(
            localization_plan,
            workspace / "localization",
            api_key=api_key,
            model=model,
            client=localization_client,
        )
        _debug_write(debug_dir, "02-localization-summary.json", localization_result.summary)

        technical_result = _run_localized_technical(
            localization_result.localization_report_paths,
            preparation_report_path,
            workspace / "technical",
            api_key=api_key,
            model=model,
            client=technical_client,
        )
        _debug_write(debug_dir, "03-localized-technical.json", technical_result.summary)

        warnings = localization_result.warnings + technical_result.warnings
        if not technical_result.elements:
            raise LocalizedExtractionPipelineError(
                "LOCALIZED_V1_NO_USEFUL_EXTRACTION: "
                + "; ".join(warnings[:10])
            )

        enrichment = GeminiEnrichmentResult(
            elements=technical_result.elements,
            warnings=warnings,
            usage=_sum_usage(localization_result.usage + technical_result.usage),
        )
        gemini_extraction, _reconciled, _decisions = reconcile_and_build_gemini_extraction(
            GeminiDiscoveryResult(notes=warnings),
            enrichment,
        )
        _debug_write(debug_dir, "04-pre-mapper.json", gemini_extraction)
        extraction = map_gemini_extraction_to_requirement_extraction(
            gemini_extraction,
            model_provider="google",
            model=model,
            default_source_id="source-1" if len(files) == 1 else None,
            allowed_source_ids=_localized_allowed_source_ids(prepared_sources),
        )
        extraction.requirement.project_id = project_id
        extraction.requirement.requirement_id = requirement_id
        extraction.sources = _sources_from_prepared(prepared_sources)
        extraction.warnings.extend(_warnings_from_messages(warnings))
        extraction.extraction_metadata.source_count = len(prepared_sources)
        extraction.extraction_metadata.element_count = len(extraction.elements)
        extraction.extraction_metadata.model_provider = "google"
        extraction.extraction_metadata.model = model
        extraction.extraction_metadata.pipeline_version = "localized-v1"
        extraction.extraction_metadata.token_usage = enrichment.usage
        _debug_write(debug_dir, "05-final-extraction.json", extraction)
        return extraction


class _LocalizationRunResult:
    def __init__(
        self,
        *,
        localization_report_paths: list[Path],
        warnings: list[str],
        usage: list[TokenUsage | None],
        summary: dict[str, Any],
    ) -> None:
        self.localization_report_paths = localization_report_paths
        self.warnings = warnings
        self.usage = usage
        self.summary = summary


class _TechnicalRunResult:
    def __init__(
        self,
        *,
        elements: list[GeminiElementEnrichment],
        warnings: list[str],
        usage: list[TokenUsage | None],
        summary: dict[str, Any],
    ) -> None:
        self.elements = elements
        self.warnings = warnings
        self.usage = usage
        self.summary = summary


def _prepare_uploaded_sources(
    files: list[Path],
    workspace: Path,
    *,
    project_id: str | None,
) -> list[PreparedSource]:
    prepared_root = workspace / "documents"
    prepared_sources = []
    for index, path in enumerate(files, start=1):
        prepared = prepare_source(
            path,
            prepared_root,
            logical_name=path.name,
            input_index=index,
        )
        if prepared.status == "PREPARED":
            prepared_sources.append(prepared)
        else:
            project = project_id or "uploaded"
            raise LocalizedExtractionPipelineError(
                f"LOCALIZED_V1_SOURCE_PREPARATION_FAILED:{project}:{path.name}"
            )
    return prepared_sources


def _write_preparation_report(
    path: Path,
    prepared_sources: list[PreparedSource],
    *,
    project_id: str | None,
) -> dict[str, Any]:
    project = project_id or "uploaded"
    documents = []
    for source in prepared_sources:
        row = asdict(source)
        row["project_id"] = project
        row["corpus_document_id"] = f"source-{source.input_index}"
        row["archive_path"] = source.logical_name
        row["archive_input_index"] = source.input_index
        row["artifact_directory"] = f"documents/{source.document_id}"
        documents.append(row)
    report = {
        "schema_version": 1,
        "status": "PREPARATION_COMPLETED",
        "run_id": uuid.uuid4().hex,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_archive_sha256": None,
        "source_archive_unchanged": True,
        "projects_prepared": 1 if documents else 0,
        "documents_processed": len(documents),
        "documents_prepared": len(documents),
        "documents_failed": 0,
        "documents": documents,
        "network_calls": 0,
        "corpus_approved": False,
    }
    write_json(path, report)
    return report


def _image_frame_plan(preparation_report_path: Path) -> dict[str, Any]:
    plan = build_plan(preparation_report_path, tiles="auto")
    plan["coordinate_contract"] = IMAGE_FRAME_CONTRACT_VERSION
    plan["local_validation_policy"] = IMAGE_FRAME_LOCAL_VALIDATION_POLICY
    plan["prompt_version"] = IMAGE_FRAME_CONTRACT_VERSION
    plan["prompt_sha256"] = image_frame_prompt_digest()
    plan["response_schema_sha256"] = image_frame_response_schema_digest()
    return plan


def _run_localization(
    plan: dict[str, Any],
    output_root: Path,
    *,
    api_key: str,
    model: str,
    client: GeminiLocalizationClient | None,
) -> _LocalizationRunResult:
    output_root.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    active_client = client or GeminiLocalizationClient(api_key=api_key, model=model)
    paths: list[Path] = []
    warnings: list[str] = []
    usage: list[TokenUsage | None] = []
    jobs = plan.get("jobs", [])
    try:
        for job in jobs:
            job_id = str(job["job_id"])
            try:
                prompt, images, _debug = image_frame_request_for_job(plan, job)
                response = active_client.generate_image_frames(prompt, images)
                usage.append(_usage_from_dict(response.get("usage")))
                result = save_image_frame_proposal_artifacts(
                    plan,
                    job,
                    str(response.get("text") or ""),
                    output_root / job_id,
                )
                paths.append(output_root / job_id / "localization.json")
                if result.get("status") != "PROPOSALS_RECORDED_UNVERIFIED":
                    warnings.append(f"localized_v1 localization nonterminal for {job_id}")
            except Exception as exc:
                warnings.append(
                    f"localized_v1 localization failed for {job_id}: {type(exc).__name__}"
                )
    finally:
        if owns_client:
            active_client.close()
    return _LocalizationRunResult(
        localization_report_paths=paths,
        warnings=warnings,
        usage=usage,
        summary={
            "jobs": len(jobs),
            "succeeded": len(paths),
            "failed": len(jobs) - len(paths),
            "warnings": warnings,
        },
    )


def _run_localized_technical(
    localization_report_paths: list[Path],
    preparation_report_path: Path,
    output_root: Path,
    *,
    api_key: str,
    model: str,
    client: LocalizedTechnicalClient | None,
) -> _TechnicalRunResult:
    output_root.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    active_client = client or GeminiLocalizedTechnicalClient(api_key=api_key, model=model)
    elements: list[GeminiElementEnrichment] = []
    warnings: list[str] = []
    usage: list[TokenUsage | None] = []
    candidates = 0
    succeeded = 0
    try:
        for report_path in localization_report_paths:
            try:
                plan = build_localized_technical_plan(
                    report_path,
                    preparation_report_path=preparation_report_path,
                )
                candidate_jobs = select_candidate_jobs(plan)
            except Exception as exc:
                warnings.append(
                    "localized_v1 technical plan failed for "
                    f"{report_path.name}: {type(exc).__name__}"
                )
                continue
            candidates += len(candidate_jobs)
            for job in candidate_jobs:
                candidate_id = str(job["technical_candidate_id"]).replace("|", "__")
                try:
                    prompt, images, _context = request_for_candidate(plan, job)
                    response = active_client.generate_localized_technical(prompt, images)
                    usage.append(_usage_from_dict(response.get("usage")))
                    result = record_localized_technical_response(
                        plan,
                        job,
                        response,
                        output_root / candidate_id,
                    )
                    elements.extend(result.enrichment.elements)
                    warnings.extend(result.warnings)
                    succeeded += 1
                except Exception as exc:
                    warnings.append(
                        "localized_v1 technical read failed for "
                        f"{candidate_id}: {type(exc).__name__}"
                    )
    finally:
        if owns_client:
            active_client.close()
    return _TechnicalRunResult(
        elements=elements,
        warnings=warnings,
        usage=usage,
        summary={
            "localization_reports": len(localization_report_paths),
            "candidates": candidates,
            "succeeded": succeeded,
            "failed": candidates - succeeded,
            "warnings": warnings,
        },
    )


def _sources_from_prepared(prepared_sources: list[PreparedSource]) -> list[Source]:
    return [
        Source(
            id=f"source-{index}",
            file_name=source.logical_name,
            media_type=source.media_type,
            source_type="document" if source.media_type == "application/pdf" else "image",
            page_count=source.page_count,
        )
        for index, source in enumerate(prepared_sources, start=1)
    ]


def _localized_allowed_source_ids(prepared_sources: list[PreparedSource]) -> list[str]:
    source_ids = [f"source-{index}" for index, _source in enumerate(prepared_sources, start=1)]
    for source in prepared_sources:
        for page in source.pages:
            source_ids.append(f"localized:{source.document_id}:{page.view_index}")
    return source_ids


def _warnings_from_messages(messages: list[str]) -> list[Warning]:
    return [
        Warning(
            id=f"localized-v1-warning-{index}",
            code="LOCALIZED_V1_WARNING",
            severity="warning",
            message=message,
        )
        for index, message in enumerate(dict.fromkeys(messages), start=1)
    ]


def _usage_from_dict(data: object) -> TokenUsage | None:
    if not isinstance(data, dict):
        return None
    values = {
        "input_tokens": data.get("prompt_token_count"),
        "output_tokens": data.get("candidates_token_count"),
        "total_tokens": data.get("total_token_count"),
    }
    if all(value is None for value in values.values()):
        return None
    return TokenUsage(**values)


def _sum_usage(items: list[TokenUsage | None]) -> TokenUsage | None:
    input_tokens = _sum_optional(item.input_tokens for item in items if item is not None)
    output_tokens = _sum_optional(item.output_tokens for item in items if item is not None)
    total_tokens = _sum_optional(item.total_tokens for item in items if item is not None)
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _sum_optional(values) -> int | None:
    materialized = [value for value in values if value is not None]
    return sum(materialized) if materialized else None


def _finish_reason(response: object) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    finish = getattr(finish, "value", finish)
    return str(finish) if finish is not None else None


def _usage(response: object) -> dict[str, Any] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    return {
        field: getattr(usage, field, None)
        for field in ("prompt_token_count", "candidates_token_count", "total_token_count")
    }


def _debug_write(path: Path | None, filename: str, value: object) -> None:
    if path is None:
        return
    serialized = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    target = path / filename
    target.write_text(
        json.dumps(serialized, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
