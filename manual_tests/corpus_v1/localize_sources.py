"""Phase 3: page-localization plan, opt-in model run, and offline response replay.

No expected answers are read. The current application extraction flow is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.providers.gemini_localization_v1 import (  # noqa: E402
    GeminiLocalizationAPIError,
    build_localization_api_schema,
    localization_api_schema_metadata,
)
from app.services.document_localization_v1 import (  # noqa: E402
    LocalizationError,
    build_plan,
    file_sha256,
    read_json,
    request_for_job,
    save_proposal_artifacts,
    select_jobs,
    write_json,
)
from app.services.localization_prompt_v1 import SYSTEM_INSTRUCTION  # noqa: E402

NEW_FILES = (
    "app/models/document_localization_v1.py",
    "app/services/document_localization_v1.py",
    "app/services/localization_prompt_v1.py",
    "app/providers/gemini_localization_v1.py",
    "manual_tests/corpus_v1/localize_sources.py",
)


def provenance() -> dict:
    versions = {}
    for dependency in ("pydantic", "pillow", "google-genai"):
        try:
            versions[dependency] = version(dependency)
        except PackageNotFoundError:
            versions[dependency] = "NOT_INSTALLED"
    return {"python": platform.python_version(), "platform": platform.system(),
            "versions": versions,
            "files_sha256": {f: file_sha256(REPO_ROOT / f) for f in NEW_FILES}}


def new_output(out: Path | None, kind: str, preparation_root: Path) -> Path:
    output = out or BASE / "results" / "source_localization" / (kind + "-" + uuid.uuid4().hex)
    output = output.resolve()
    if output.is_relative_to(preparation_root.resolve()):
        raise LocalizationError("DO_NOT_WRITE_INTO_PREPARATION_RUN")
    if output.exists() or output.is_symlink():
        raise LocalizationError("OUTPUT_ALREADY_EXISTS")
    output.mkdir(parents=True, exist_ok=False)
    return output


def make_plan(prepared: Path, out: Path | None = None, tiles: str = "auto") -> tuple[dict, Path]:
    plan = build_plan(prepared, tiles=tiles)
    # build_plan validates every asset/frame and indexes bounded native tokens.
    # Encoding model tiles is deferred to run/replay to avoid repeating costly work.
    # execute_plan validates all selected request sizes before creating the client.
    output = new_output(out, "plan", prepared.parent)
    plan["created_utc"] = datetime.now(UTC).isoformat()
    plan["code_provenance"] = provenance()
    write_json(output / "localization_plan.json", plan)
    report = {key: value for key, value in plan.items() if key != "jobs"}
    report["projects_detail"] = [
        {"project_id": p, "documents": len({j["document_id"] for j in plan["jobs"]
                                            if j["project_id"] == p}),
         "views": sum(j["project_id"] == p for j in plan["jobs"])}
        for p in sorted({j["project_id"] for j in plan["jobs"]})
    ]
    report["plan_file"] = str((output / "localization_plan.json").resolve())
    write_json(output / "localization_plan_report.json", report)
    return report, output


def load_verified_plan(plan_path: Path) -> dict:
    plan = read_json(plan_path, 10 * 1024 * 1024)
    if not isinstance(plan, dict) or plan.get("status") != "LOCALIZATION_PLAN_READY":
        raise LocalizationError("INVALID_LOCALIZATION_PLAN")
    prepared = Path(plan["preparation_report_path"])
    if file_sha256(prepared) != plan["preparation_report_sha256"]:
        raise LocalizationError("PREPARATION_REPORT_CHANGED")
    current = build_plan(prepared, tiles=plan["tile_mode"])
    for key, value in current.items():
        if plan.get(key) != value:
            raise LocalizationError("PLAN_CHANGED_OR_STALE_REBUILD_IT")
    return plan


def _live_client():
    # Load settings only AFTER --allow-paid-calls and the full request budget pass.
    from app.core.settings import get_settings
    from app.providers.gemini_localization_v1 import GeminiLocalizationClient

    settings = get_settings()
    return GeminiLocalizationClient(api_key=settings.gemini_api_key, model=settings.gemini_model)


def execute_plan(plan_path: Path, *, allow_paid_calls: bool, max_calls: int,
                 project: str | None = None, document: str | None = None,
                 out: Path | None = None, client_factory=None) -> tuple[dict, Path]:
    if not allow_paid_calls:
        raise LocalizationError("PAID_CALLS_NOT_AUTHORIZED")
    if type(max_calls) is not int or not 1 <= max_calls <= 100:
        raise LocalizationError("SET_EXPLICIT_MAX_CALLS_BETWEEN_1_AND_100")
    plan = load_verified_plan(plan_path)
    jobs = select_jobs(plan, project, document)
    if len(jobs) > max_calls:
        raise LocalizationError(
            f"REQUEST_BUDGET_EXCEEDED: selected={len(jobs)}, max={max_calls}"
        )
    for job in jobs:
        request_for_job(plan, job)
    output = new_output(out, "run", Path(plan["preparation_report_path"]).parent)
    report: dict[str, Any] = {
        "schema_version": 1, "status": "LOCALIZATION_RUNNING",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": file_sha256(plan_path),
        "preparation_report_sha256": plan["preparation_report_sha256"],
        "source_archive_sha256": plan["source_archive_sha256"],
        "prompt_sha256": plan["prompt_sha256"],
        "schema_sha256": plan["response_schema_sha256"],
        "api_schema": localization_api_schema_metadata(),
        "api_schema_file": "api_response_schema.json",
        "requested_jobs": len(jobs), "request_budget": max_calls,
        "sdk_retry_attempts": 1,
        "network_calls_attempted": 0, "responses_received": 0,
        "pages_with_valid_proposal_shape": 0, "unverified_element_proposals": 0,
        "model_reported_partial_views": 0, "orientation_review_views": 0,
        "unreadable_region_proposals": 0,
        "localization_model_run": False, "canonical_extraction_run": False,
        "visual_association_validated": False, "corpus_approved": False,
        "code_provenance": provenance(), "jobs": [],
    }
    # Create once to prove output is writable before paying for any requests.
    write_json(output / "api_response_schema.json", build_localization_api_schema())
    write_json(output / "run_started.json", report)
    client = None
    failure = False
    try:
        client = (client_factory or _live_client)()
        report["requested_model"] = client.model
        for job in jobs:
            row = {"job_id": job["job_id"], "project_id": job["project_id"],
                   "corpus_document_id": job["corpus_document_id"],
                   "page_number": job["page_number"], "status": "PENDING"}
            if failure:
                row["status"] = "NOT_RUN_AFTER_FAILURE"
                report["jobs"].append(row)
                continue
            started = time.perf_counter()
            job_dir = output / job["job_id"]
            job_dir.mkdir(exist_ok=False)
            try:
                prompt, images, _obs = request_for_job(plan, job)
                with (job_dir / "request_prompt.txt").open("x", encoding="utf-8") as stream:
                    stream.write(SYSTEM_INSTRUCTION + "\n\n" + prompt)
                row["prompt_sha256"] = file_sha256(job_dir / "request_prompt.txt")
                row["image_sha256"] = [hashlib.sha256(b).hexdigest() for _, _, b in images]
                row["image_mime_types"] = [mime for _, mime, _ in images]
                report["network_calls_attempted"] += 1
                report["localization_model_run"] = True
                response = client.generate(prompt, images)
                report["responses_received"] += 1
                write_json(job_dir / "response_envelope.json", response)
                row["response_sha256"] = file_sha256(job_dir / "response_envelope.json")
                row["usage"] = response.get("usage")
                row["response_model_version"] = response.get("response_model_version")
                if response.get("finish_reason") != "STOP":
                    raise LocalizationError("RESPONSE_NOT_COMPLETE_STOP")
                result = save_proposal_artifacts(
                    plan, job, response.get("text"), job_dir / "proposals")
                row["status"] = result["status"]
                row["coverage_claim"] = result["coverage_claim"]
                row["suggested_rotation_clockwise"] = result["suggested_rotation_clockwise"]
                row["issues"] = result["issues"]
                row["element_proposals"] = len(result["elements"])
                row["region_proposals"] = len(result["regions"])
                row["artifact_directory"] = job["job_id"] + "/proposals"
                report["pages_with_valid_proposal_shape"] += 1
                report["unverified_element_proposals"] += len(result["elements"])
                report["model_reported_partial_views"] += int(
                    result["coverage_claim"] != "FULL_SCAN_CLAIMED"
                )
                report["orientation_review_views"] += int(
                    result["suggested_rotation_clockwise"] != 0
                )
                report["unreadable_region_proposals"] += sum(
                    r["legibility"] != "READABLE" for r in result["regions"])
            except Exception as exc:
                # Provider exceptions can embed headers/keys/payloads: do NOT log str(exc).
                row["status"] = "FAILED"
                row["error_type"] = type(exc).__name__
                row["reason"] = (
                    str(exc)
                    if isinstance(exc, LocalizationError)
                    else "REQUEST_OR_ARTIFACT_FAILED"
                )
                if isinstance(exc, GeminiLocalizationAPIError):
                    row["error_type"] = exc.original_error_type
                    row["reason"] = "GEMINI_API_ERROR"
                    row["api_error"] = exc.diagnostic()
                failure = True
            finally:
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                report["jobs"].append(row)
                write_json(job_dir / "job_report.json", row)
                print(
                    f"{row['corpus_document_id']} view={job['view_index']}: "
                    f"{row['status']}",
                    flush=True,
                )
                if "api_error" in row:
                    info = row["api_error"]
                    print(
                        f"API_HTTP_CODE={info['http_code']} | "
                        f"API_STATUS={info['status']}",
                        flush=True,
                    )
                    print(f"API_MESSAGE_REDACTED={info['message_redacted']}", flush=True)
    except Exception as exc:
        report["setup_error_type"] = type(exc).__name__
        done = {r["job_id"] for r in report["jobs"]}
        report["jobs"].extend({"job_id": j["job_id"], "status": "NOT_RUN_SETUP_FAILURE"}
                              for j in jobs if j["job_id"] not in done)
        failure = True
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                report["client_close_warning"] = True
        report["status"] = (
            "LOCALIZATION_HAS_ERRORS" if failure else "LOCALIZATION_RECORDED_UNVERIFIED"
        )
        write_json(output / "localization_report.json", report)
    return report, output


def replay_response(plan_path: Path, job_id: str, response_path: Path,
                    out: Path | None = None) -> tuple[dict, Path]:
    plan = load_verified_plan(plan_path)
    matches = [job for job in plan["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise LocalizationError("UNKNOWN_JOB_ID")
    envelope = read_json(response_path, 2 * 1024 * 1024)
    if envelope.get("finish_reason") != "STOP":
        raise LocalizationError("RESPONSE_NOT_COMPLETE_STOP")
    output = new_output(out, "replay", Path(plan["preparation_report_path"]).parent)
    result = save_proposal_artifacts(
        plan, matches[0], envelope.get("text"), output / "proposals", origin="OFFLINE_REPLAY")
    result["network_calls"] = 0
    result["response_sha256"] = file_sha256(response_path)
    write_json(output / "replay_report.json", result)
    return result, output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    plan_parser = subs.add_parser(
        "plan",
        help="Sin red ni claves: comprueba entradas y lista solicitudes",
    )
    plan_parser.add_argument("--prepared", type=Path, required=True)
    plan_parser.add_argument("--tiles", choices=["auto", "none"], default="auto")
    plan_parser.add_argument("--out", type=Path)
    live = subs.add_parser(
        "run",
        help="OPT-IN: envia las vistas seleccionadas a Gemini (puede tener costo)",
    )
    live.add_argument("--plan", type=Path, required=True)
    live.add_argument("--project")
    live.add_argument("--document")
    live.add_argument("--max-calls", type=int, required=True)
    live.add_argument("--allow-paid-calls", action="store_true")
    live.add_argument("--out", type=Path)
    replay = subs.add_parser(
        "replay",
        help="Valida otra vez una respuesta guardada, sin llamada al modelo",
    )
    replay.add_argument("--plan", type=Path, required=True)
    replay.add_argument("--job", required=True)
    replay.add_argument("--response", type=Path, required=True)
    replay.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            report, output = make_plan(args.prepared, args.out, args.tiles)
            for field in ("status", "projects", "documents", "views", "planned_requests",
                          "planned_images", "views_without_native_text", "network_calls"):
                print(f"{field.upper()}={report[field]}")
            print("LOCALIZATION_MODEL_RUN=NO | CANONICAL_EXTRACTION_RUN=NO | CORPUS_APPROVED=NO")
            print(f"PLAN={output / 'localization_plan.json'}")
            print(f"REPORT={output / 'localization_plan_report.json'}")
            return 0
        if args.command == "run":
            report, output = execute_plan(
                args.plan, allow_paid_calls=args.allow_paid_calls, max_calls=args.max_calls,
                project=args.project, document=args.document, out=args.out)
            for field in (
                "status",
                "requested_jobs",
                "network_calls_attempted",
                "responses_received",
                "pages_with_valid_proposal_shape",
                "unverified_element_proposals",
            ):
                print(f"{field.upper()}={report[field]}")
            print(
                "VISUAL_ASSOCIATION_VALIDATED=NO | CANONICAL_EXTRACTION_RUN=NO | "
                "CORPUS_APPROVED=NO"
            )
            print(f"REPORT={output / 'localization_report.json'}")
            return 1 if report["status"] == "LOCALIZATION_HAS_ERRORS" else 0
        report, output = replay_response(args.plan, args.job, args.response, args.out)
        print(f"STATUS={report['status']} | NETWORK_CALLS=0 | CORPUS_APPROVED=NO")
        print(f"REPORT={output / 'replay_report.json'}")
        return 0
    except Exception as exc:
        # No credentials or response text on console, including settings validation errors.
        reason = str(exc) if isinstance(exc, LocalizationError) else type(exc).__name__
        print(f"LOCALIZATION_ERROR={reason}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
