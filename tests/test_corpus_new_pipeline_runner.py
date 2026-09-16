"""Tests for the manual Corpus V1 new pipeline runner; no network calls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.document_localization_v1 import IMAGE_FRAME_CONTRACT_VERSION, file_sha256

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "manual_tests" / "corpus_v1" / "run_new_pipeline.py"
SPEC = importlib.util.spec_from_file_location("corpus_v1_new_pipeline_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def job(job_id: str, project_id: str, document_id: str, view_index: int = 1) -> dict:
    return {
        "job_id": job_id,
        "project_id": project_id,
        "corpus_document_id": document_id,
        "document_id": f"doc-{job_id}",
        "view_index": view_index,
        "page_number": view_index,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def localization_plan(jobs: list[dict]) -> dict:
    return {
        "status": "LOCALIZATION_PLAN_READY",
        "coordinate_contract": IMAGE_FRAME_CONTRACT_VERSION,
        "jobs": jobs,
    }


def corpus_plan(tmp_path: Path, jobs: list[dict]) -> Path:
    loc_path = tmp_path / "localization_plan.json"
    write_json(loc_path, localization_plan(jobs))
    prepared = tmp_path / "source_preparation_report.json"
    prepared.write_text("{}", encoding="utf-8")
    plan = {
        "status": pipeline.PIPELINE_STATUS,
        "preparation_report_path": str(prepared),
        "preparation_report_sha256": file_sha256(prepared),
        "localization_plan_file": str(loc_path),
        "localization_plan_sha256": file_sha256(loc_path),
        "planned_localization_job_ids": [item["job_id"] for item in jobs],
    }
    plan_path = tmp_path / "corpus_plan.json"
    write_json(plan_path, plan)
    return plan_path


def install_run_fakes(monkeypatch, jobs: list[dict], *, fail_first=False) -> None:
    monkeypatch.setattr(
        pipeline.localize_sources,
        "load_verified_plan",
        lambda _path: localization_plan(jobs),
    )

    def fake_localization_request(_plan, item):
        if fail_first and item["job_id"] == jobs[0]["job_id"]:
            raise pipeline.LocalizationError("SYNTHETIC_LOCALIZATION_FAILURE")
        return f"loc-{item['job_id']}", [("page", "image/png", item["job_id"].encode())], "sys"

    monkeypatch.setattr(pipeline, "_localization_request", fake_localization_request)
    monkeypatch.setattr(
        pipeline,
        "_save_localization_result",
        lambda _plan, item, _response, _out: {
            "status": "PROPOSALS_RECORDED_UNVERIFIED",
            "elements": [{"candidate_id": f"c-{item['job_id']}", "reference_raw": "V-01"}],
            "regions": [],
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_localization_response_shape",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline,
        "build_localized_technical_plan",
        lambda _path, *, preparation_report_path=None: {
            "preparation_report_path": str(preparation_report_path),
            "candidates": [{"candidate_id": "c1", "technical_candidate_id": "t1"}],
        },
    )
    monkeypatch.setattr(pipeline, "select_candidate_jobs", lambda plan: plan["candidates"])
    monkeypatch.setattr(
        pipeline,
        "request_for_candidate",
        lambda _plan, candidate: (
            f"tech-{candidate['candidate_id']}",
            [("page_context", "image/png", b"img")],
            {
                "document_id": "doc",
                "view_index": 1,
                "reference_raw": "V-01",
            },
        ),
    )

    def fake_record(_plan, _candidate, _response, output_dir):
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            output_dir / "localized_technical_reading.json",
            {
                "enrichment": {
                    "elements": [{
                        "temporary_id": "t1",
                        "reference": "V-01",
                        "measurements": [],
                    }],
                    "warnings": [],
                }
            },
        )
        return SimpleNamespace(
            status="LOCALIZED_TECHNICAL_READING_RECORDED_UNVERIFIED",
            enrichment=SimpleNamespace(elements=[{"reference": "V-01"}]),
        )

    monkeypatch.setattr(pipeline, "record_localized_technical_response", fake_record)


class FakeLocalizationClient:
    model = "fake-localizer"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def generate_image_frames(self, _prompt, _images):
        self.calls += 1
        return {"finish_reason": "STOP", "text": "{}"}

    def close(self):
        self.closed = True


class FakeTechnicalClient:
    model = "fake-reader"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def generate_localized_technical(self, _prompt, _images):
        self.calls += 1
        return {"finish_reason": "STOP", "text": "{}"}

    def close(self):
        self.closed = True


def test_plan_covers_multiple_projects_and_documents(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}", encoding="utf-8")
    loc_jobs = [job("j1", "p1", "D1"), job("j2", "p2", "D2")]

    def fake_make_plan(_prepared, out, _tiles, *, image_frames):
        assert image_frames is True
        loc_plan = localization_plan(loc_jobs)
        write_json(out / "localization_plan.json", loc_plan)
        return {
            "projects": 2,
            "documents": 2,
            "views": 2,
        }, out

    monkeypatch.setattr(pipeline.localize_sources, "make_plan", fake_make_plan)
    report, output = pipeline.build_corpus_pipeline_plan(prepared, tmp_path / "plan")
    assert report["status"] == pipeline.PIPELINE_STATUS
    assert report["projects"] == 2
    assert report["documents"] == 2
    assert report["views"] == 2
    assert report["localization_jobs"] == 2
    assert report["expected_checks_loaded"] is False
    assert (output / "corpus_plan.json").is_file()


def test_failed_localization_job_does_not_abort_remaining_jobs(tmp_path, monkeypatch):
    loc_jobs = [job("j1", "p1", "D1"), job("j2", "p1", "D2")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs, fail_first=True)
    report, _output = pipeline.run_corpus_pipeline(
        plan_path,
        tmp_path / "run",
        allow_paid_calls=True,
        max_localization_calls=2,
        max_technical_calls=2,
        localization_client_factory=FakeLocalizationClient,
        technical_client_factory=FakeTechnicalClient,
    )
    localization_statuses = [
        row["stage_status"] for row in report["jobs"] if row["stage"] == "localization"
    ]
    assert localization_statuses == ["LOCALIZATION_FAILED", "LOCALIZATION_SUCCEEDED"]
    assert report["status"] == "CORPUS_PIPELINE_HAS_ERRORS"
    assert report["technical_candidates"] == 1
    assert report["localization_network_calls"] == 1
    assert report["jobs"][0]["response_source"] == "REQUEST_NOT_BUILT"
    assert report["jobs"][1]["candidate_count"] == 1


def test_localization_api_failure_counts_attempt_and_continues(tmp_path, monkeypatch):
    loc_jobs = [job("j1", "p1", "D1"), job("j2", "p1", "D2")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs)

    class FailingOnceClient(FakeLocalizationClient):
        def generate_image_frames(self, _prompt, _images):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("synthetic timeout")
            return {"finish_reason": "STOP", "text": "{}"}

    report, _output = pipeline.run_corpus_pipeline(
        plan_path,
        tmp_path / "run",
        allow_paid_calls=True,
        max_localization_calls=2,
        max_technical_calls=0,
        localization_client_factory=FailingOnceClient,
    )
    localization_rows = [row for row in report["jobs"] if row["stage"] == "localization"]
    assert [row["stage_status"] for row in localization_rows] == [
        "LOCALIZATION_FAILED",
        "LOCALIZATION_SUCCEEDED",
    ]
    assert report["localization_network_calls"] == 2
    assert localization_rows[0]["response_source"] == "RETRY_FAILED"
    assert localization_rows[1]["response_source"] == "RETRIED_TRANSIENT"


def test_global_plan_integrity_error_still_aborts_before_jobs(tmp_path, monkeypatch):
    loc_jobs = [job("j1", "p1", "D1")]
    plan_path = corpus_plan(tmp_path, loc_jobs)

    def invalid_plan(_path):
        raise pipeline.LocalizationError("PLAN_CHANGED_OR_STALE_REBUILD_IT")

    monkeypatch.setattr(pipeline.localize_sources, "load_verified_plan", invalid_plan)
    with pytest.raises(pipeline.LocalizationError, match="PLAN_CHANGED_OR_STALE"):
        pipeline.run_corpus_pipeline(
            plan_path,
            tmp_path / "run",
            allow_paid_calls=True,
            max_localization_calls=1,
            max_technical_calls=0,
            localization_client_factory=FakeLocalizationClient,
        )


def test_zero_technical_budget_skips_technical_stage(tmp_path, monkeypatch):
    loc_jobs = [job("j1", "p1", "D1")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs)

    def fail_technical_plan(*_args, **_kwargs):
        raise AssertionError("technical stage must not be planned")

    monkeypatch.setattr(pipeline, "build_localized_technical_plan", fail_technical_plan)
    report, _output = pipeline.run_corpus_pipeline(
        plan_path,
        tmp_path / "run",
        allow_paid_calls=True,
        max_localization_calls=1,
        max_technical_calls=0,
        localization_client_factory=FakeLocalizationClient,
        technical_client_factory=FakeTechnicalClient,
    )
    assert report["technical_stage_status"] == "TECHNICAL_SKIPPED"
    assert report["technical_skipped_localizations"] == 1
    assert report["technical_errors"] == 0
    assert report["technical_network_calls"] == 0
    assert [row["stage"] for row in report["jobs"]] == ["localization"]


def test_preparation_report_is_propagated_to_technical_plan(tmp_path, monkeypatch):
    loc_jobs = [job("j1", "p1", "D1")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    seen = {}
    install_run_fakes(monkeypatch, loc_jobs)

    def capture_preparation(_path, *, preparation_report_path=None):
        seen["preparation_report_path"] = preparation_report_path
        return {"candidates": [{"candidate_id": "c1", "technical_candidate_id": "t1"}]}

    monkeypatch.setattr(pipeline, "build_localized_technical_plan", capture_preparation)
    pipeline.run_corpus_pipeline(
        plan_path,
        tmp_path / "run",
        allow_paid_calls=True,
        max_localization_calls=1,
        max_technical_calls=1,
        localization_client_factory=FakeLocalizationClient,
        technical_client_factory=FakeTechnicalClient,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert seen["preparation_report_path"] == Path(plan["preparation_report_path"])


def test_cached_valid_response_is_reused_without_network(tmp_path, monkeypatch):
    loc_jobs = [job("j1", "p1", "D1")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs)
    loc_hash = pipeline._request_hash(
        "localization",
        "loc-j1",
        [("page", "image/png", b"j1")],
        system_instruction="sys",
    )
    tech_hash = pipeline._request_hash(
        "technical",
        "tech-c1",
        [("page_context", "image/png", b"img")],
        system_instruction="LOCALIZED_TECHNICAL",
    )
    reuse = tmp_path / "reuse"
    write_json(reuse / "responses" / f"{loc_hash}.json", {"finish_reason": "STOP", "text": "{}"})
    write_json(reuse / "responses" / f"{tech_hash}.json", {"finish_reason": "STOP", "text": "{}"})
    report, _output = pipeline.run_corpus_pipeline(
        plan_path,
        tmp_path / "run",
        allow_paid_calls=True,
        max_localization_calls=1,
        max_technical_calls=1,
        reuse_from=[reuse],
        localization_client_factory=FakeLocalizationClient,
        technical_client_factory=FakeTechnicalClient,
    )
    assert report["network_calls_attempted"] == 0
    assert report["responses_reused"] == 2


def test_cached_invalid_response_is_revalidated_and_not_counted_as_reused(
    tmp_path,
    monkeypatch,
):
    loc_jobs = [job("j1", "p1", "D1")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs)
    loc_hash = pipeline._request_hash(
        "localization",
        "loc-j1",
        [("page", "image/png", b"j1")],
        system_instruction="sys",
    )
    reuse = tmp_path / "reuse"
    write_json(reuse / "responses" / f"{loc_hash}.json", {"finish_reason": "STOP", "text": "{}"})

    def reject_cached(_plan, _item, _response, _out):
        raise pipeline.LocalizationError("INVALID_LOCALIZATION_RESPONSE")

    monkeypatch.setattr(pipeline, "_save_localization_result", reject_cached)
    report, _output = pipeline.run_corpus_pipeline(
        plan_path,
        tmp_path / "run",
        allow_paid_calls=True,
        max_localization_calls=1,
        max_technical_calls=0,
        reuse_from=[reuse],
        localization_client_factory=FakeLocalizationClient,
        technical_client_factory=FakeTechnicalClient,
    )
    assert report["network_calls_attempted"] == 0
    assert report["responses_reused"] == 0
    assert report["jobs"][0]["response_source"] == "CACHED_INVALID"


def test_resumable_budget_counts_only_transient_retry_eligible_jobs(tmp_path, monkeypatch):
    valid_jobs = [job(f"valid{i}", "p", f"D{i}") for i in range(37)]
    invalid_jobs = [job(f"invalid{i}", "p", f"X{i}") for i in range(5)]
    timeout_job = job("timeout", "p", "T1")
    loc_jobs = valid_jobs + invalid_jobs + [timeout_job]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs)
    reuse = tmp_path / "reuse"
    previous_rows = []
    for item in valid_jobs:
        request_hash = pipeline._request_hash(
            "localization",
            f"loc-{item['job_id']}",
            [("page", "image/png", item["job_id"].encode())],
            system_instruction="sys",
        )
        response_file = reuse / "responses" / f"{request_hash}.json"
        write_json(response_file, {"finish_reason": "STOP", "text": "{}"})
        previous_rows.append({
            "stage": "localization",
            "job_id": item["job_id"],
            "status": "PROPOSALS_RECORDED_UNVERIFIED",
        })
    for item in invalid_jobs:
        request_hash = pipeline._request_hash(
            "localization",
            f"loc-{item['job_id']}",
            [("page", "image/png", item["job_id"].encode())],
            system_instruction="sys",
        )
        response_file = reuse / "responses" / f"{request_hash}.json"
        write_json(response_file, {"finish_reason": "STOP", "text": "{}"})
        previous_rows.append({
            "stage": "localization",
            "job_id": item["job_id"],
            "status": "FAILED",
            "error_type": "LocalizationResponseValidationError",
            "reason": "INVALID_LOCALIZATION_RESPONSE",
            "response_file": str(response_file),
        })
    previous_rows.append({
        "stage": "localization",
        "job_id": timeout_job["job_id"],
        "status": "FAILED",
        "error_type": "ReadTimeout",
        "reason": "FAILED",
    })
    write_json(reuse / "execution_report.json", {"jobs": previous_rows})

    invalid_hashes = {
        pipeline._request_hash(
            "localization",
            f"loc-{item['job_id']}",
            [("page", "image/png", item["job_id"].encode())],
            system_instruction="sys",
        )
        for item in invalid_jobs
    }

    def validate_by_response(response, **_kwargs):
        if response.get("marker") == "invalid":
            raise ValueError("INVALID_LOCALIZATION_RESPONSE")

    for request_hash in invalid_hashes:
        write_json(reuse / "responses" / f"{request_hash}.json", {
            "finish_reason": "STOP",
            "text": "{}",
            "marker": "invalid",
        })
    monkeypatch.setattr(pipeline, "_validate_localization_response_shape", validate_by_response)
    report, _output = pipeline.run_corpus_pipeline(
        plan_path,
        tmp_path / "run",
        allow_paid_calls=True,
        max_localization_calls=1,
        max_technical_calls=0,
        reuse_from=[reuse],
        localization_client_factory=FakeLocalizationClient,
    )
    assert report["valid_reuse"] == 37
    assert report["terminal_invalid_carried"] == 5
    assert report["network_retry_eligible"] == 1
    assert report["required_localization_network_calls"] == 1
    assert report["localization_network_calls"] == 1
    assert report["responses_reused"] == 37
    sources = [row["response_source"] for row in report["jobs"] if row["stage"] == "localization"]
    assert sources.count("REUSED_VALID") == 37
    assert sources.count("TERMINAL_INVALID_CARRIED") == 5
    assert sources.count("RETRIED_TRANSIENT") == 1


def test_zero_budget_with_transient_pending_exceeds_budget(tmp_path, monkeypatch):
    loc_jobs = [job("timeout", "p", "D1")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs)
    reuse = tmp_path / "reuse"
    write_json(
        reuse / "execution_report.json",
        {
            "jobs": [{
                "stage": "localization",
                "job_id": "timeout",
                "status": "FAILED",
                "error_type": "ReadTimeout",
                "reason": "FAILED",
            }],
        },
    )
    with pytest.raises(pipeline.CorpusPipelineError, match="LOCALIZATION_BUDGET_EXCEEDED"):
        pipeline.run_corpus_pipeline(
            plan_path,
            tmp_path / "run",
            allow_paid_calls=True,
            max_localization_calls=0,
            max_technical_calls=0,
            reuse_from=[reuse],
            localization_client_factory=FakeLocalizationClient,
        )


def test_budgets_are_enforced_before_calls(tmp_path, monkeypatch):
    loc_jobs = [job("j1", "p1", "D1"), job("j2", "p1", "D2")]
    plan_path = corpus_plan(tmp_path, loc_jobs)
    install_run_fakes(monkeypatch, loc_jobs)
    with pytest.raises(pipeline.CorpusPipelineError, match="LOCALIZATION_BUDGET_EXCEEDED"):
        pipeline.run_corpus_pipeline(
            plan_path,
            tmp_path / "run",
            allow_paid_calls=True,
            max_localization_calls=1,
            max_technical_calls=10,
        )


def test_expected_checks_are_only_loaded_by_evaluation(tmp_path, monkeypatch):
    expected = tmp_path / "expected_checks.json"
    write_json(expected, {"projects": [{"project_id": "p1", "elements": []}]})
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "execution_report.json",
        {"jobs": [], "technical_errors": 0, "localization_errors": 0},
    )
    report = pipeline.evaluate_corpus_run(run_dir, expected_path=expected)
    assert report["expected_checks_used_only_for_offline_evaluation"] is True
    assert report["projects_evaluated"] == 1


def test_failed_response_diagnostics_separate_invalid_and_timeout(tmp_path):
    run_dir = tmp_path / "run"
    invalid_response = run_dir / "responses" / "invalid.json"
    write_json(invalid_response, {"finish_reason": "STOP", "text": "{}"})
    write_json(
        run_dir / "execution_report.json",
        {
            "jobs": [
                {
                    "stage": "localization",
                    "stage_status": "LOCALIZATION_FAILED",
                    "job_id": "j1",
                    "reason": "INVALID_LOCALIZATION_RESPONSE",
                    "error_type": "LocalizationResponseValidationError",
                    "response_file": str(invalid_response),
                },
                {
                    "stage": "localization",
                    "stage_status": "LOCALIZATION_FAILED",
                    "job_id": "j2",
                    "reason": "REQUEST_OR_ARTIFACT_FAILED",
                    "error_type": "ReadTimeout",
                },
            ],
        },
    )
    report = pipeline.diagnose_failed_localization_responses(run_dir)
    statuses = {row["job_id"]: row["recovery_status"] for row in report["rows"]}
    assert statuses == {"j1": "STILL_INVALID", "j2": "RETRY_REQUIRED_TIMEOUT"}
    assert report["counts"]["timeout"] == 1


def test_failed_raw_response_can_revalidate_offline(tmp_path):
    run_dir = tmp_path / "run"
    response = {
        "contract_version": "page-localization-image-frames-v1.0",
        "job_id": "j1",
        "image_registry_sha256": "a" * 64,
        "page_roles": ["DETAIL_SHEET"],
        "suggested_rotation_clockwise": 0,
        "coverage": "FULL_SCAN_CLAIMED",
        "regions": [{
            "region_id": "r1",
            "kind": "REFERENCE_LABEL",
            "source_image_id": "img0",
            "box_2d": [0, 0, 100, 100],
            "observed_label": "V-01",
            "transcription": "V-01",
            "legibility": "READABLE",
        }],
        "elements": [{
            "candidate_id": "e1",
            "reference_raw": "V-01",
            "description_of_location": "Synthetic",
            "reference_region_ids": ["r1"],
            "drawing_region_ids": [],
            "table_region_ids": [],
            "dimension_region_ids": [],
            "note_region_ids": [],
            "missing_dimension_area_links": ["not visible"],
            "dimension_area_status": "NOT_VISIBLE",
            "table_status": "NOT_VISIBLE",
            "localization_notes": "Synthetic missing dimensions.",
            "association": "PROPOSED",
            "association_basis": "Visible label.",
        }],
        "issues": [],
    }
    response_path = run_dir / "responses" / "valid.json"
    write_json(response_path, {"finish_reason": "STOP", "text": json.dumps(response)})
    write_json(
        run_dir / "execution_report.json",
        {
            "jobs": [{
                "stage": "localization",
                "status": "FAILED",
                "job_id": "j1",
                "reason": "INVALID_LOCALIZATION_RESPONSE",
                "error_type": "LocalizationResponseValidationError",
                "response_file": str(response_path),
            }],
        },
    )
    report = pipeline.diagnose_failed_localization_responses(run_dir)
    assert report["rows"][0]["recovery_status"] == "RECOVERED_VALID_WITH_CURRENT_VALIDATOR"


@pytest.mark.parametrize(
    ("legacy_state", "new_state", "classification"),
    [
        ("DIFFERENT", "MATCH", "FIXED"),
        ("MATCH", "DIFFERENT", "REGRESSED"),
        ("DIFFERENT", "DIFFERENT", "STILL_WRONG"),
        ("MISSING", "MISSING", "STILL_MISSING"),
        (None, "MATCH", "NEWLY_EXTRACTED"),
        ("MATCH", "MATCH", "UNCHANGED_CORRECT"),
    ],
)
def test_legacy_vs_new_classifications(legacy_state, new_state, classification):
    assert pipeline._classification(legacy_state, new_state) == classification


def test_corpus_names_do_not_change_request_hashes():
    images = [("page", "image/png", b"same")]
    first = pipeline._request_hash("localization", "same prompt", images, system_instruction="sys")
    second = pipeline._request_hash("localization", "same prompt", images, system_instruction="sys")
    assert first == second
