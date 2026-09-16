"""Corpus V1 runner for the new localization + localized technical reading flow.

This is manual tooling. Expected checks are used only by offline evaluation and
comparison, never while planning requests or running model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models.document_localization_v1 import (  # noqa: E402
    ImageFrameLocalizationProposal,
    PageLocalizationProposal,
)
from app.services.document_localization_v1 import (  # noqa: E402
    IMAGE_FRAME_CONTRACT_VERSION,
    LocalizationError,
    file_sha256,
    image_frame_request_for_job,
    read_json,
    request_for_job,
    save_image_frame_proposal_artifacts,
    save_proposal_artifacts,
    select_jobs,
    write_json,
)
from app.services.localization_prompt_v1 import (  # noqa: E402
    IMAGE_FRAME_LOCALIZATION_RULES,
    SYSTEM_INSTRUCTION,
)
from app.services.localized_technical_reader import (  # noqa: E402
    LocalizedTechnicalReaderError,
    build_localized_technical_plan,
    record_localized_technical_response,
    request_for_candidate,
    select_candidate_jobs,
)
from manual_tests.corpus_v1 import localize_sources, runner  # noqa: E402
from manual_tests.corpus_v1.read_localized_elements import (  # noqa: E402
    live_client as live_technical_client,
)

PIPELINE_STATUS = "CORPUS_PIPELINE_PLAN_READY"


class CorpusPipelineError(RuntimeError):
    """Manual corpus pipeline error."""


def _new_output(out: Path | None, kind: str) -> Path:
    output = out or BASE / "results" / "pipeline_v1" / (kind + "-" + uuid.uuid4().hex)
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise CorpusPipelineError("OUTPUT_ALREADY_EXISTS")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _request_hash(
    stage: str,
    prompt: str,
    images: list[tuple[str, str, bytes]],
    *,
    system_instruction: str,
) -> str:
    payload = {
        "stage": stage,
        "system_instruction": system_instruction,
        "prompt_sha256": _hash_bytes(prompt.encode("utf-8")),
        "images": [
            {"label": label, "mime_type": mime_type, "sha256": _hash_bytes(data)}
            for label, mime_type, data in images
        ],
    }
    return _hash_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _load_cached_response(
    request_hash: str,
    roots: list[Path],
) -> tuple[dict[str, Any], Path] | None:
    for root in roots:
        candidate = root / "responses" / f"{request_hash}.json"
        if candidate.is_file():
            response = read_json(candidate, 4 * 1024 * 1024)
            if isinstance(response, dict) and response.get("finish_reason") == "STOP":
                return response, candidate
    return None


def _store_response(response_dir: Path, request_hash: str, response: dict[str, Any]) -> Path:
    response_dir.mkdir(parents=True, exist_ok=True)
    path = response_dir / f"{request_hash}.json"
    if not path.exists():
        write_json(path, response)
    return path


def _selected_jobs(plan: dict[str, Any], project: str | None, document: str | None) -> list[dict]:
    return select_jobs(plan, project, document)


def build_corpus_pipeline_plan(
    prepared: Path,
    out: Path | None = None,
    *,
    image_frames: bool = True,
    project: str | None = None,
    document: str | None = None,
) -> tuple[dict[str, Any], Path]:
    output = _new_output(out, "plan")
    localization_plan_dir = output / "localization"
    localization_report, _localization_output = localize_sources.make_plan(
        prepared,
        localization_plan_dir,
        "auto",
        image_frames=image_frames,
    )
    localization_plan_path = localization_plan_dir / "localization_plan.json"
    localization_plan = read_json(localization_plan_path, 20 * 1024 * 1024)
    jobs = _selected_jobs(localization_plan, project, document)
    plan = {
        "schema_version": 1,
        "status": PIPELINE_STATUS,
        "created_utc": datetime.now(UTC).isoformat(),
        "preparation_report_path": str(Path(prepared).resolve()),
        "preparation_report_sha256": file_sha256(prepared),
        "localization_plan_file": str(localization_plan_path.resolve()),
        "localization_plan_sha256": file_sha256(localization_plan_path),
        "coordinate_contract": localization_plan.get("coordinate_contract"),
        "projects": localization_report["projects"],
        "documents": localization_report["documents"],
        "views": localization_report["views"],
        "localization_jobs": len(jobs),
        "planned_localization_job_ids": [job["job_id"] for job in jobs],
        "planned_technical_candidates": None,
        "max_localization_calls_required": len(jobs),
        "max_technical_calls_required": None,
        "filters": {"project": project, "document": document},
        "network_calls": 0,
        "expected_checks_loaded": False,
        "corpus_approved": False,
    }
    write_json(output / "corpus_plan.json", plan)
    return plan, output


def _live_localization_client():
    return localize_sources._live_client()


def _localization_request(
    localization_plan: dict[str, Any],
    job: dict[str, Any],
) -> tuple[str, list[tuple[str, str, bytes]], str]:
    image_frames = localization_plan.get("coordinate_contract") == IMAGE_FRAME_CONTRACT_VERSION
    if image_frames:
        prompt, images, _debug = image_frame_request_for_job(localization_plan, job)
        return prompt, images, IMAGE_FRAME_LOCALIZATION_RULES
    prompt, images, _obs = request_for_job(localization_plan, job)
    return prompt, images, SYSTEM_INSTRUCTION


def _save_localization_result(
    localization_plan: dict[str, Any],
    job: dict[str, Any],
    response: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if response.get("finish_reason") != "STOP":
        raise LocalizationError("RESPONSE_NOT_COMPLETE_STOP")
    image_frames = localization_plan.get("coordinate_contract") == IMAGE_FRAME_CONTRACT_VERSION
    if image_frames:
        return save_image_frame_proposal_artifacts(
            localization_plan,
            job,
            response.get("text"),
            output_dir,
        )
    return save_proposal_artifacts(localization_plan, job, response.get("text"), output_dir)


def _validate_localization_response_shape(
    response: dict[str, Any],
    *,
    image_frames: bool,
) -> None:
    text = response.get("text")
    if not isinstance(text, str) or not text.strip():
        raise CorpusPipelineError("EMPTY_LOCALIZATION_RESPONSE_TEXT")
    model = ImageFrameLocalizationProposal if image_frames else PageLocalizationProposal
    model.model_validate_json(text)


def _classify_validation_errors(exc: ValidationError) -> list[str]:
    categories = set()
    for error in exc.errors(include_url=False):
        message = str(error.get("msg", ""))
        location = ".".join(str(part) for part in error.get("loc", ()))
        error_type = str(error.get("type", ""))
        folded = f"{location} {message} {error_type}".casefold()
        if "region kind incompatible with link" in folded:
            categories.add("region kind/link incompatibility")
        elif "box_2d" in folded or "coordinate" in folded:
            categories.add("invalid coordinates")
        elif (
            "field required" in folded
            or "require notes" in folded
            or "require dimension links" in folded
            or "requires its visible label/table region" in folded
        ):
            categories.add("missing/empty conditional field")
        elif "source_image_id" in folded:
            categories.add("unknown image id")
        elif "contract_version" in folded or "literal_error" in folded:
            categories.add("same contract mal interpreted")
        elif (
            "extra_forbidden" in folded
            or "forbidden" in folded
            or "string_pattern_mismatch" in folded
        ):
            categories.add("path/schema mismatch")
        else:
            categories.add("otro")
    return sorted(categories)


def diagnose_failed_localization_responses(
    run_dir: Path,
    out: Path | None = None,
) -> dict[str, Any]:
    execution = read_json(Path(run_dir) / "execution_report.json", 20 * 1024 * 1024)
    image_frames = True
    rows = []
    counts: Counter[str] = Counter()
    for row in execution.get("jobs", []):
        failed = row.get("stage_status") == "LOCALIZATION_FAILED" or row.get("status") == "FAILED"
        if row.get("stage") != "localization" or not failed:
            continue
        diagnostic = {
            "job_id": row.get("job_id"),
            "project_id": row.get("project_id"),
            "corpus_document_id": row.get("corpus_document_id"),
            "document_id": row.get("document_id"),
            "view_index": row.get("view_index"),
            "error_type": row.get("error_type"),
            "reason": row.get("reason"),
            "response_file": row.get("response_file"),
            "recovery_status": "NOT_REVALIDATED",
            "categories": [],
        }
        if row.get("error_type") in {"ReadTimeout", "TimeoutError"}:
            diagnostic["recovery_status"] = "RETRY_REQUIRED_TIMEOUT"
            diagnostic["categories"] = ["timeout"]
            counts["timeout"] += 1
        elif row.get("reason") == "INVALID_LOCALIZATION_RESPONSE" and row.get("response_file"):
            try:
                response = read_json(Path(str(row["response_file"])), 4 * 1024 * 1024)
                _validate_localization_response_shape(response, image_frames=image_frames)
            except ValidationError as exc:
                categories = _classify_validation_errors(exc)
                diagnostic["recovery_status"] = "STILL_INVALID"
                diagnostic["categories"] = categories
                for category in categories:
                    counts[category] += 1
            except Exception as exc:
                diagnostic["recovery_status"] = "STILL_INVALID"
                diagnostic["categories"] = ["otro"]
                diagnostic["error_type"] = type(exc).__name__
                counts["otro"] += 1
            else:
                diagnostic["recovery_status"] = "RECOVERED_VALID_WITH_CURRENT_VALIDATOR"
                counts["RECOVERED_VALID_WITH_CURRENT_VALIDATOR"] += 1
        rows.append(diagnostic)
    report = {
        "schema_version": 1,
        "status": "FAILED_LOCALIZATION_RESPONSE_DIAGNOSTICS_RECORDED",
        "run_dir": str(Path(run_dir).resolve()),
        "failed_localization_count": len(rows),
        "counts": dict(counts),
        "rows": rows,
        "network_calls": 0,
        "validators_relaxed": False,
    }
    target = out or Path(run_dir) / "failed_localization_response_diagnostics.json"
    if target.exists() and out is None:
        target = target.with_name(
            f"{target.stem}-{uuid.uuid4().hex}{target.suffix}"
        )
    write_json(target, report)
    report["report_file"] = str(target.resolve())
    return report


def _call_or_reuse(
    *,
    request_hash: str,
    cache_roots: list[Path],
    response_dir: Path,
    call,
) -> tuple[dict[str, Any], str, Path | None]:
    cached = _load_cached_response(request_hash, cache_roots)
    if cached is not None:
        response, path = cached
        return response, "REUSED", path
    response = call()
    path = _store_response(response_dir, request_hash, response)
    return response, "CALLED", path


def _prior_localization_failures(reuse_roots: list[Path]) -> dict[str, dict[str, Any]]:
    failures = {}
    for root in reuse_roots:
        report_path = root / "execution_report.json"
        if not report_path.is_file():
            continue
        report = read_json(report_path, 20 * 1024 * 1024)
        for row in report.get("jobs", []):
            failed = (
                row.get("stage_status") == "LOCALIZATION_FAILED"
                or row.get("status") == "FAILED"
            )
            if row.get("stage") == "localization" and failed and row.get("job_id"):
                failures[str(row["job_id"])] = row
    return failures


def _is_transient_localization_failure(row: dict[str, Any] | None) -> bool:
    if row is None:
        return True
    error_type = str(row.get("error_type", ""))
    reason = str(row.get("reason", ""))
    return any(
        marker in error_type or marker in reason
        for marker in ("Timeout", "ReadTimeout", "Connection", "Network")
    )


def run_corpus_pipeline(
    plan_path: Path,
    out: Path | None = None,
    *,
    allow_paid_calls: bool,
    max_localization_calls: int,
    max_technical_calls: int,
    reuse_from: list[Path] | None = None,
    localization_client_factory=None,
    technical_client_factory=None,
) -> tuple[dict[str, Any], Path]:
    if not allow_paid_calls:
        raise CorpusPipelineError("PAID_CALLS_NOT_AUTHORIZED")
    output = _new_output(out, "run")
    plan = read_json(plan_path, 20 * 1024 * 1024)
    if not isinstance(plan, dict) or plan.get("status") != PIPELINE_STATUS:
        raise CorpusPipelineError("INVALID_CORPUS_PIPELINE_PLAN")
    if file_sha256(Path(plan["localization_plan_file"])) != plan["localization_plan_sha256"]:
        raise CorpusPipelineError("LOCALIZATION_PLAN_CHANGED")
    localization_plan = localize_sources.load_verified_plan(Path(plan["localization_plan_file"]))
    jobs = [
        job for job in localization_plan["jobs"]
        if job["job_id"] in set(plan["planned_localization_job_ids"])
    ]

    response_dir = output / "responses"
    cache_roots = [Path(path).resolve() for path in reuse_from or []]
    cache_roots.append(output)
    prior_failures = _prior_localization_failures(cache_roots)
    execution: dict[str, Any] = {
        "schema_version": 1,
        "status": "CORPUS_PIPELINE_RECORDED_UNVERIFIED",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": file_sha256(plan_path),
        "localization_jobs": len(jobs),
        "technical_candidates": 0,
        "network_calls_attempted": 0,
        "localization_network_calls": 0,
        "technical_network_calls": 0,
        "responses_reused": 0,
        "localization_errors": 0,
        "technical_errors": 0,
        "valid_reuse": 0,
        "terminal_invalid_carried": 0,
        "network_retry_eligible": 0,
        "required_localization_network_calls": 0,
        "technical_stage_status": (
            "TECHNICAL_SKIPPED" if max_technical_calls == 0 else "PENDING"
        ),
        "technical_skipped_localizations": 0,
        "jobs": [],
        "corpus_approved": False,
    }
    localization_client = None
    technical_client = None
    try:
        localization_work: list[dict[str, Any]] = []
        for job in jobs:
            try:
                prompt, images, system_instruction = _localization_request(localization_plan, job)
                req_hash = _request_hash(
                    "localization",
                    prompt,
                    images,
                    system_instruction=system_instruction,
                )
            except Exception as exc:
                localization_work.append({
                    "job": job,
                    "request_hash": None,
                    "action": "REQUEST_CONSTRUCTION_FAILED",
                    "error_type": type(exc).__name__,
                    "reason": str(exc) if isinstance(exc, LocalizationError) else "FAILED",
                })
                continue
            cached = _load_cached_response(req_hash, cache_roots)
            prior_failure = prior_failures.get(job["job_id"])
            if cached is not None:
                response, cached_path = cached
                try:
                    _validate_localization_response_shape(
                        response,
                        image_frames=localization_plan.get("coordinate_contract")
                        == IMAGE_FRAME_CONTRACT_VERSION,
                    )
                except Exception:
                    localization_work.append({
                        "job": job,
                        "prompt": prompt,
                        "images": images,
                        "request_hash": req_hash,
                        "cached_response": response,
                        "cached_path": cached_path,
                        "action": "TERMINAL_INVALID_CARRIED",
                        "prior_failure": prior_failure,
                    })
                    execution["terminal_invalid_carried"] += 1
                    continue
                localization_work.append({
                    "job": job,
                    "prompt": prompt,
                    "images": images,
                    "system_instruction": system_instruction,
                    "request_hash": req_hash,
                    "cached_response": response,
                    "cached_path": cached_path,
                    "action": "REUSED_VALID",
                })
                execution["valid_reuse"] += 1
                continue
            action = (
                "RETRY_ELIGIBLE"
                if _is_transient_localization_failure(prior_failure)
                else "TERMINAL_INVALID_CARRIED"
            )
            localization_work.append({
                "job": job,
                "prompt": prompt,
                "images": images,
                "system_instruction": system_instruction,
                "request_hash": req_hash,
                "action": action,
                "prior_failure": prior_failure,
            })
            if action == "RETRY_ELIGIBLE":
                execution["network_retry_eligible"] += 1
            else:
                execution["terminal_invalid_carried"] += 1
        execution["required_localization_network_calls"] = execution["network_retry_eligible"]
        if execution["required_localization_network_calls"] > max_localization_calls:
            raise CorpusPipelineError("LOCALIZATION_BUDGET_EXCEEDED")

        for item in localization_work:
            job = item["job"]
            row = {
                "stage": "localization",
                "job_id": job["job_id"],
                "project_id": job["project_id"],
                "corpus_document_id": job["corpus_document_id"],
                "document_id": job["document_id"],
                "view_index": job["view_index"],
                "page_number": job["page_number"],
                "status": "PENDING",
                "request_hash": item["request_hash"],
            }
            started = time.perf_counter()
            job_dir = (
                output / "projects" / job["project_id"] / str(job["corpus_document_id"])
                / f"view-{job['view_index']:04d}"
            )
            job_dir.mkdir(parents=True, exist_ok=False)
            try:
                row["stage_status"] = "LOCALIZATION_PENDING"
                if item["action"] == "REQUEST_CONSTRUCTION_FAILED":
                    row["status"] = "FAILED"
                    row["stage_status"] = "LOCALIZATION_FAILED"
                    row["response_source"] = "REQUEST_NOT_BUILT"
                    row["error_type"] = item["error_type"]
                    row["reason"] = item["reason"]
                    execution["localization_errors"] += 1
                    continue
                if item["action"] == "TERMINAL_INVALID_CARRIED":
                    prior_failure = item.get("prior_failure") or {}
                    row["status"] = "FAILED"
                    row["stage_status"] = "LOCALIZATION_FAILED"
                    row["response_source"] = "TERMINAL_INVALID_CARRIED"
                    row["error_type"] = (
                        prior_failure.get("error_type")
                        or "LocalizationResponseValidationError"
                    )
                    row["reason"] = (
                        prior_failure.get("reason")
                        or "INVALID_LOCALIZATION_RESPONSE"
                    )
                    if item.get("cached_path") is not None:
                        row["response_file"] = str(item["cached_path"])
                        row["response_hash"] = file_sha256(item["cached_path"])
                    elif prior_failure.get("response_file"):
                        row["response_file"] = prior_failure["response_file"]
                        if Path(str(row["response_file"])).is_file():
                            row["response_hash"] = file_sha256(Path(str(row["response_file"])))
                    execution["localization_errors"] += 1
                    continue
                prompt = item["prompt"]
                images = item["images"]

                prompt_for_call = prompt
                images_for_call = images
                row_for_call = row

                def make_call(
                    prompt: str = prompt_for_call,
                    images: list[tuple[str, str, bytes]] = images_for_call,
                    row: dict[str, Any] = row_for_call,
                ) -> dict[str, Any]:
                    nonlocal localization_client
                    if localization_client is None:
                        factory = localization_client_factory or _live_localization_client
                        localization_client = factory()
                    execution["network_calls_attempted"] += 1
                    execution["localization_network_calls"] += 1
                    row["response_source"] = "RETRIED_TRANSIENT"
                    if localization_plan.get("coordinate_contract") == IMAGE_FRAME_CONTRACT_VERSION:
                        return localization_client.generate_image_frames(prompt, images)
                    return localization_client.generate(prompt, images)

                if item["action"] == "REUSED_VALID":
                    response = item["cached_response"]
                    source = "REUSED_VALID"
                    cached_path = item["cached_path"]
                else:
                    response, _source, cached_path = _call_or_reuse(
                        request_hash=item["request_hash"],
                        cache_roots=[],
                        response_dir=response_dir,
                        call=make_call,
                    )
                    source = row.get("response_source", "CALLED")
                row["response_source"] = source
                row["response_file"] = str(cached_path) if cached_path else None
                write_json(job_dir / "response_envelope.json", response)
                row["response_hash"] = file_sha256(job_dir / "response_envelope.json")
                result = _save_localization_result(
                    localization_plan,
                    job,
                    response,
                    job_dir / "localization",
                )
                write_json(job_dir / "localization_report.json", result)
                row["status"] = result["status"]
                row["stage_status"] = "LOCALIZATION_SUCCEEDED"
                row["candidate_count"] = len(result.get("elements", []))
                row["region_count"] = len(result.get("regions", []))
                execution["responses_reused"] += int(source == "REUSED_VALID")
            except Exception as exc:
                row["status"] = "FAILED"
                row["stage_status"] = "LOCALIZATION_FAILED"
                if row.get("response_source") == "REUSED_VALID":
                    row["response_source"] = "CACHED_INVALID"
                elif row.get("response_source") == "RETRIED_TRANSIENT":
                    row["response_source"] = "RETRY_FAILED"
                row["error_type"] = type(exc).__name__
                row["reason"] = str(exc) if isinstance(exc, LocalizationError) else "FAILED"
                execution["localization_errors"] += 1
            finally:
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                execution["jobs"].append(row)

        successful_localizations = [
            row for row in execution["jobs"]
            if (
                row["stage"] == "localization"
                and row.get("stage_status") == "LOCALIZATION_SUCCEEDED"
            )
        ]
        if max_technical_calls == 0:
            execution["technical_stage_status"] = "TECHNICAL_SKIPPED"
            execution["technical_skipped_localizations"] = len(successful_localizations)
            return _finish_execution_report(execution, output)

        preparation_report_path = Path(plan["preparation_report_path"])
        if file_sha256(preparation_report_path) != plan["preparation_report_sha256"]:
            raise CorpusPipelineError("PREPARATION_REPORT_CHANGED")
        technical_jobs: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
        for row in successful_localizations:
            loc_report_path = (
                output / "projects" / row["project_id"] / str(row["corpus_document_id"])
                / f"view-{row['view_index']:04d}" / "localization_report.json"
            )
            try:
                technical_plan = build_localized_technical_plan(
                    loc_report_path,
                    preparation_report_path=preparation_report_path,
                )
                for candidate in select_candidate_jobs(technical_plan):
                    technical_jobs.append((technical_plan, candidate, loc_report_path.parent))
            except Exception as exc:
                execution["jobs"].append({
                    "stage": "technical-plan",
                    "project_id": row["project_id"],
                    "corpus_document_id": row["corpus_document_id"],
                    "document_id": row["document_id"],
                    "view_index": row["view_index"],
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "reason": (
                        str(exc)
                        if isinstance(exc, LocalizedTechnicalReaderError)
                        else "FAILED"
                    ),
                })
                execution["technical_errors"] += 1
        execution["technical_candidates"] = len(technical_jobs)
        if len(technical_jobs) > max_technical_calls:
            raise CorpusPipelineError("TECHNICAL_BUDGET_EXCEEDED")
        execution["technical_stage_status"] = "TECHNICAL_RUNNING"

        for technical_plan, candidate, view_dir in technical_jobs:
            row = {
                "stage": "technical",
                "technical_candidate_id": candidate["technical_candidate_id"],
                "candidate_id": candidate["candidate_id"],
                "project_id": view_dir.parents[1].name,
                "corpus_document_id": view_dir.parent.name,
                "status": "PENDING",
                "stage_status": "TECHNICAL_PENDING",
            }
            started = time.perf_counter()
            candidate_dir = (
                view_dir / "technical" / candidate["technical_candidate_id"].replace("|", "__")
            )
            try:
                prompt, images, context = request_for_candidate(technical_plan, candidate)
                row["document_id"] = context["document_id"]
                row["view_index"] = context["view_index"]
                row["reference_raw"] = context["reference_raw"]
                req_hash = _request_hash(
                    "technical",
                    prompt,
                    images,
                    system_instruction="LOCALIZED_TECHNICAL",
                )
                row["request_hash"] = req_hash

                prompt_for_call = prompt
                images_for_call = images

                def make_call(
                    prompt: str = prompt_for_call,
                    images: list[tuple[str, str, bytes]] = images_for_call,
                ) -> dict[str, Any]:
                    nonlocal technical_client
                    if technical_client is None:
                        technical_client = (technical_client_factory or live_technical_client)()
                    execution["network_calls_attempted"] += 1
                    execution["technical_network_calls"] += 1
                    return technical_client.generate_localized_technical(prompt, images)

                response, source, cached_path = _call_or_reuse(
                    request_hash=req_hash,
                    cache_roots=cache_roots,
                    response_dir=response_dir,
                    call=make_call,
                )
                row["response_source"] = source
                row["response_file"] = str(cached_path) if cached_path else None
                result = record_localized_technical_response(
                    technical_plan,
                    candidate,
                    response,
                    candidate_dir,
                )
                row["status"] = result.status
                row["stage_status"] = "TECHNICAL_SUCCEEDED"
                row["elements"] = len(result.enrichment.elements)
                execution["responses_reused"] += int(source == "REUSED")
            except Exception as exc:
                row["status"] = "FAILED"
                row["stage_status"] = "TECHNICAL_FAILED"
                if row.get("response_source") == "REUSED":
                    row["response_source"] = "CACHED_INVALID"
                row["error_type"] = type(exc).__name__
                row["reason"] = (
                    str(exc)
                    if isinstance(exc, LocalizedTechnicalReaderError)
                    else "FAILED"
                )
                execution["technical_errors"] += 1
            finally:
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                execution["jobs"].append(row)
    finally:
        for client in (localization_client, technical_client):
            if client is not None and hasattr(client, "close"):
                client.close()
    if execution["localization_errors"] or execution["technical_errors"]:
        execution["status"] = "CORPUS_PIPELINE_HAS_ERRORS"
    if execution["technical_stage_status"] == "TECHNICAL_RUNNING":
        execution["technical_stage_status"] = (
            "TECHNICAL_FAILED" if execution["technical_errors"] else "TECHNICAL_SUCCEEDED"
        )
    write_json(output / "execution_report.json", execution)
    return execution, output


def _finish_execution_report(
    execution: dict[str, Any],
    output: Path,
) -> tuple[dict[str, Any], Path]:
    if execution["localization_errors"] or execution["technical_errors"]:
        execution["status"] = "CORPUS_PIPELINE_HAS_ERRORS"
    write_json(output / "execution_report.json", execution)
    return execution, output


def _collect_new_elements(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pattern = "projects/*/*/view-*/technical/*/localized_technical_reading.json"
    for path in sorted(Path(run_dir).glob(pattern)):
        result = read_json(path, 4 * 1024 * 1024)
        project_id = path.parts[-6]
        corpus_document_id = path.parts[-5]
        view_part = path.parts[-4]
        view_index = int(view_part.rsplit("-", 1)[1])
        for element in result.get("enrichment", {}).get("elements", []):
            enriched = dict(element)
            enriched["project_id"] = project_id
            enriched["corpus_document_id"] = corpus_document_id
            enriched["view_index"] = view_index
            by_project[project_id].append(enriched)
    return by_project


def _field_name(check: dict[str, Any]) -> str:
    field = str(check.get("field", "unknown"))
    if check.get("kind") == "measurement":
        return field
    if field == "operation":
        return "operation/function"
    if field in {"panel_count", "movable_panel_count", "fixed_panel_count"}:
        return "panel/component information"
    if field.startswith("geometry"):
        return "geometry"
    return field


def evaluate_corpus_run(
    run_dir: Path,
    out: Path | None = None,
    *,
    expected_path: Path = BASE / "expected_checks.json",
) -> dict[str, Any]:
    expected = read_json(expected_path, 2 * 1024 * 1024)
    elements_by_project = _collect_new_elements(run_dir)
    project_reports = []
    field_counts: dict[str, Counter] = defaultdict(Counter)
    for project in expected.get("projects", []):
        project_id = project["project_id"]
        snapshot = {"elements": elements_by_project.get(project_id, [])}
        project_report = runner.evaluate_snapshot(snapshot, expected, project_id, "enrichment")
        project_reports.append(project_report)
        for row in project_report["checks"]:
            field_counts[_field_name(row)][row["state"]] += 1
    counts = Counter()
    for report in project_reports:
        counts.update(report["counts"])
    execution = read_json(Path(run_dir) / "execution_report.json", 20 * 1024 * 1024)
    report = {
        "schema_version": 1,
        "status": "CORPUS_EVALUATION_RECORDED",
        "projects_evaluated": len(project_reports),
        "documents_evaluated": len({
            row.get("corpus_document_id")
            for row in execution.get("jobs", [])
            if row.get("corpus_document_id")
        }),
        "elements_found": sum(len(items) for items in elements_by_project.values()),
        "checks_evaluable": sum(report["checked_values"] for report in project_reports),
        "matches": counts["MATCH"],
        "mismatches": counts["DIFFERENT"],
        "missing": counts["MISSING"],
        "conflicts_preserved": counts["IDENTITY_AMBIGUOUS"],
        "extraction_errors": execution.get("technical_errors", 0),
        "localization_errors": execution.get("localization_errors", 0),
        "by_project": project_reports,
        "by_document": _by_document(execution),
        "by_field": {field: dict(counter) for field, counter in sorted(field_counts.items())},
        "expected_checks_used_only_for_offline_evaluation": True,
        "corpus_approved": False,
    }
    target = out or Path(run_dir) / "evaluation_report.json"
    write_json(target, report)
    return report


def _by_document(execution: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in execution.get("jobs", []):
        key = (str(row.get("project_id")), str(row.get("corpus_document_id")))
        grouped[key][str(row.get("status"))] += 1
    return [
        {
            "project_id": project_id,
            "corpus_document_id": document_id,
            "statuses": dict(counter),
        }
        for (project_id, document_id), counter in sorted(grouped.items())
    ]


def _checks_by_identity(evaluation: dict[str, Any]) -> dict[tuple[str, str, str], str]:
    checks = {}
    for project in evaluation.get("by_project", []):
        for row in project.get("checks", []):
            key = (
                project["project_id"],
                str(row.get("reference", "*")),
                str(row.get("check_id", row.get("field"))),
            )
            checks[key] = row["state"]
    return checks


def _classification(old: str | None, new: str | None) -> str:
    if old is None:
        return "NEWLY_EXTRACTED" if new == "MATCH" else "NOT_COMPARABLE"
    if new is None:
        return "STILL_MISSING" if old == "MISSING" else "NOT_COMPARABLE"
    if old == "MATCH" and new == "MATCH":
        return "UNCHANGED_CORRECT"
    if old == "MATCH" and new != "MATCH":
        return "REGRESSED"
    if old != "MATCH" and new == "MATCH":
        return "FIXED"
    if old == "MISSING" and new == "MISSING":
        return "STILL_MISSING"
    return "STILL_WRONG"


def compare_legacy_vs_new(
    run_dir: Path,
    out: Path | None = None,
    *,
    baseline_path: Path = BASE / "results" / "baseline.json",
) -> dict[str, Any]:
    new_eval_path = Path(run_dir) / "evaluation_report.json"
    if not new_eval_path.is_file():
        evaluate_corpus_run(run_dir)
    new_eval = read_json(new_eval_path, 20 * 1024 * 1024)
    baseline_eval = read_json(baseline_path, 4 * 1024 * 1024) if baseline_path.is_file() else {}
    old_checks = _checks_by_identity({"by_project": [baseline_eval]} if baseline_eval else {})
    new_checks = _checks_by_identity(new_eval)
    rows = []
    for key in sorted(set(old_checks) | set(new_checks)):
        old_state = old_checks.get(key)
        new_state = new_checks.get(key)
        rows.append({
            "project_id": key[0],
            "reference": key[1],
            "check_id": key[2],
            "legacy_state": old_state,
            "new_state": new_state,
            "classification": _classification(old_state, new_state),
        })
    counts = Counter(row["classification"] for row in rows)
    report = {
        "schema_version": 1,
        "status": "LEGACY_VS_NEW_COMPARISON_RECORDED",
        "counts": dict(counts),
        "rows": rows,
        "expected_checks_used_only_for_offline_comparison": True,
        "corpus_approved": False,
    }
    target = out or Path(run_dir) / "comparison_report.json"
    write_json(target, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    plan_parser = subs.add_parser("plan")
    plan_parser.add_argument("--prepared", type=Path, required=True)
    plan_parser.add_argument("--out", type=Path)
    plan_parser.add_argument("--project")
    plan_parser.add_argument("--document")
    plan_parser.add_argument("--legacy-coordinates", action="store_true")
    run_parser = subs.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--out", type=Path)
    run_parser.add_argument("--allow-paid-calls", action="store_true")
    run_parser.add_argument("--max-localization-calls", type=int, required=True)
    run_parser.add_argument("--max-technical-calls", type=int, required=True)
    run_parser.add_argument("--reuse-from", type=Path, action="append", default=[])
    evaluate_parser = subs.add_parser("evaluate")
    evaluate_parser.add_argument("--run", type=Path, required=True)
    evaluate_parser.add_argument("--out", type=Path)
    compare_parser = subs.add_parser("compare")
    compare_parser.add_argument("--run", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path)
    diagnose_parser = subs.add_parser("diagnose-failed-localizations")
    diagnose_parser.add_argument("--run", type=Path, required=True)
    diagnose_parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            report, output = build_corpus_pipeline_plan(
                args.prepared,
                args.out,
                image_frames=not args.legacy_coordinates,
                project=args.project,
                document=args.document,
            )
            print(f"STATUS={report['status']}")
            print(f"PROJECTS={report['projects']}")
            print(f"DOCUMENTS={report['documents']}")
            print(f"VIEWS={report['views']}")
            print(f"LOCALIZATION_JOBS={report['localization_jobs']}")
            print(f"NETWORK_CALLS={report['network_calls']}")
            print(f"PLAN={output / 'corpus_plan.json'}")
            return 0
        if args.command == "run":
            report, output = run_corpus_pipeline(
                args.plan,
                args.out,
                allow_paid_calls=args.allow_paid_calls,
                max_localization_calls=args.max_localization_calls,
                max_technical_calls=args.max_technical_calls,
                reuse_from=args.reuse_from,
            )
            print(f"STATUS={report['status']}")
            print(f"NETWORK_CALLS_ATTEMPTED={report['network_calls_attempted']}")
            print(f"RESPONSES_REUSED={report['responses_reused']}")
            print(f"REPORT={output / 'execution_report.json'}")
            return 1 if report["status"] == "CORPUS_PIPELINE_HAS_ERRORS" else 0
        if args.command == "evaluate":
            report = evaluate_corpus_run(args.run, args.out)
            print(f"STATUS={report['status']}")
            print(f"PROJECTS_EVALUATED={report['projects_evaluated']}")
            print(f"DOCUMENTS_EVALUATED={report['documents_evaluated']}")
            print(f"CHECKS_EVALUABLE={report['checks_evaluable']}")
            print(f"MATCHES={report['matches']}")
            print(f"MISSING={report['missing']}")
            print(f"REPORT={args.out or Path(args.run) / 'evaluation_report.json'}")
            return 0
        if args.command == "diagnose-failed-localizations":
            report = diagnose_failed_localization_responses(args.run, args.out)
            print(f"STATUS={report['status']}")
            print(f"FAILED_LOCALIZATIONS={report['failed_localization_count']}")
            for key, value in sorted(report["counts"].items()):
                print(f"{key.upper().replace(' ', '_').replace('/', '_')}={value}")
            print(f"REPORT={report['report_file']}")
            return 0
        report = compare_legacy_vs_new(args.run, args.out)
        print(f"STATUS={report['status']}")
        for key in (
            "FIXED",
            "REGRESSED",
            "STILL_WRONG",
            "STILL_MISSING",
            "NEWLY_EXTRACTED",
            "UNCHANGED_CORRECT",
            "NOT_COMPARABLE",
        ):
            print(f"{key}={report['counts'].get(key, 0)}")
        print(f"REPORT={args.out or Path(args.run) / 'comparison_report.json'}")
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, CorpusPipelineError) else type(exc).__name__
        print(f"CORPUS_PIPELINE_ERROR={reason}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
