"""Offline tests of localization plumbing, NOT Gemini accuracy on client drawings."""
from __future__ import annotations

import copy
import hashlib
import json
import socket

import pytest
from PIL import Image

from app.services.document_localization_v1 import (
    LocalizationError,
    build_plan,
    file_sha256,
    native_tokens,
    parse_proposal,
    read_json,
    request_for_job,
    safe_child,
    save_proposal_artifacts,
    select_jobs,
    tile_windows,
)
from manual_tests.corpus_v1.localize_sources import (
    execute_plan,
    load_verified_plan,
    main,
    make_plan,
    replay_response,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("No network is allowed in localization tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def character(index=0, text="V", x=0.05, y=0.05, **kwargs):
    return {"pdf_char_index": index, "text": text,
            "visible_region": {"x": x, "y": y, "width": 0.01, "height": 0.02},
            "unicode_mapping_error": False, "generated_by_text_engine": False, **kwargs}


def prepared_fixture(root, *, pdf=True, count=1, size=(200, 100), text_state=None):
    docs = []
    for i in range(count):
        document_id = "doc-" + f"{i + 1:024x}"
        directory = root / "documents" / document_id
        directory.mkdir(parents=True, exist_ok=True)
        png = "page-0001.png" if pdf else "image.png"
        obsname = "page-0001.json" if pdf else "image.json"
        with Image.new("RGB", size, "white") as image:
            image.save(directory / png)
        text_state_resolved = text_state or (
            "NATIVE_TEXT_PRESENT_UNVERIFIED" if pdf else "IMAGE_NO_NATIVE_TEXT"
        )
        chars = [character()] if pdf and text_state_resolved != "NO_NATIVE_TEXT" else []
        dump(directory / obsname, {
            "schema_version": 1, "document_id": document_id,
            "page_number": 1 if pdf else None, "status": "PREPARED",
            "preview_file": png,
            "frame": {"origin_preview": "TOP_LEFT", "preview_width_px": size[0],
                      "preview_height_px": size[1]},
            "characters": chars,
        })
        docs.append({
            "status": "PREPARED", "project_id": "synthetic_project",
            "corpus_document_id": f"S{i}", "document_id": document_id,
            "logical_name": f"synthetic_{i}.pdf" if pdf else f"synthetic_{i}.jpg",
            "content_sha256": hashlib.sha256(f"source-{i}".encode()).hexdigest(),
            "media_type": "application/pdf" if pdf else "image/jpeg",
            "page_count": 1 if pdf else None, "artifact_directory": "documents/" + document_id,
            "pages": [{"status": "PREPARED", "view_index": 1,
                       "page_number": 1 if pdf else None, "preview_file": png,
                       "observations_file": obsname, "width_px": size[0], "height_px": size[1],
                       "text_state": text_state_resolved, "warnings": []}],
        })
    report = root / "source_preparation_report.json"
    dump(report, {"schema_version": 1, "status": "PREPARATION_COMPLETED", "run_id": "fake_run",
                  "source_archive_unchanged": True, "source_archive_sha256": "0" * 64,
                  "documents": docs})
    return report


def proposal(job_id):
    return {
        "job_id": job_id, "page_roles": ["DETAIL_SHEET"],
        "suggested_rotation_clockwise": 0, "coverage": "FULL_SCAN_CLAIMED",
        "regions": [
            {"region_id": "r1", "kind": "REFERENCE_LABEL", "box_2d": [0, 0, 200, 300],
             "observed_label": "Label", "transcription": "V", "legibility": "READABLE"},
            {"region_id": "r2", "kind": "DRAWING", "box_2d": [250, 100, 950, 900],
             "observed_label": "Drawing", "transcription": None, "legibility": "READABLE"},
        ],
        "elements": [{"candidate_id": "e1", "reference_raw": "V",
                      "description_of_location": "Left-hand label and drawing below",
                      "reference_region_ids": ["r1"], "drawing_region_ids": ["r2"],
                      "table_region_ids": [], "dimension_region_ids": [], "note_region_ids": [],
                      "association": "PROPOSED", "association_basis": "Visible caption"}],
        "issues": [],
    }


def test_plan_prepares_no_network_or_model(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", count=2)
    before = {p: file_sha256(p) for p in report.parent.rglob("*") if p.is_file()}
    result, output = make_plan(report, tmp_path / "plan")
    assert result["views"] == 2
    assert result["planned_requests"] == 2
    assert result["network_calls"] == 0
    assert result["localization_model_run"] is False
    assert result["corpus_approved"] is False
    assert (output / "localization_plan_report.json").is_file()
    assert before == {p: file_sha256(p) for p in before}


def test_plan_preserves_image_without_fabricated_page(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", pdf=False)
    plan = build_plan(report)
    assert plan["jobs"][0]["page_number"] is None
    assert plan["views_without_native_text"] == 1


def test_no_native_pdf_still_gets_visual_request(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", text_state="NO_NATIVE_TEXT")
    plan = build_plan(report)
    assert len(plan["jobs"]) == 1
    text, images, obs = request_for_job(plan, plan["jobs"][0])
    assert len(images) == 1
    assert '"included":0' in text
    assert obs["characters"] == []


def test_jobs_do_not_depend_on_report_order(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", count=3)
    jobs = build_plan(report)["jobs"]
    data = read_json(report)
    data["documents"].reverse()
    dump(report, data)
    assert build_plan(report)["jobs"] == jobs


@pytest.mark.parametrize(
    "relative",
    ["../escape", "/etc/passwd", "C:/file", "C:\\file", ".env:secret", "", "a//b", "a/./b"],
)
def test_unsafe_paths_rejected(tmp_path, relative):
    with pytest.raises(LocalizationError):
        safe_child(tmp_path, relative)


def test_symlink_artifact_rejected(tmp_path):
    target = tmp_path / "source.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("OS does not permit test symlinks")
    with pytest.raises(LocalizationError, match="SYMLINK"):
        safe_child(tmp_path, "link.json")


@pytest.mark.parametrize("field,value", [("page_number", 2), ("view_index", 2),
                                         ("width_px", 999), ("status", "PARTIAL")])
def test_report_page_metadata_mismatch_rejected(tmp_path, field, value):
    report = prepared_fixture(tmp_path / "prepared")
    data = read_json(report)
    data["documents"][0]["pages"][0][field] = value
    dump(report, data)
    with pytest.raises(LocalizationError):
        build_plan(report)


def test_missing_assets_not_silently_skipped(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    for image in report.parent.rglob("*.png"):
        image.unlink()
    with pytest.raises(LocalizationError):
        build_plan(report)


def test_duplicate_document_id_is_not_deduped(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    data = read_json(report)
    data["documents"].append(copy.deepcopy(data["documents"][0]))
    dump(report, data)
    with pytest.raises(LocalizationError):
        build_plan(report)


def test_observation_identity_rejected(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    obs = next(report.parent.rglob("page-0001.json"))
    data = read_json(obs)
    data["document_id"] = "wrong"
    dump(obs, data)
    with pytest.raises(LocalizationError):
        build_plan(report)


def test_plan_rechecks_modified_asset_before_request(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    plan = build_plan(report)
    obs = next(report.parent.rglob("page-0001.json"))
    obs.write_text(obs.read_text() + " ")
    with pytest.raises(LocalizationError, match="ASSET_CHANGED"):
        request_for_job(plan, plan["jobs"][0])


def test_saved_plan_tampering_rejected(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    _result, output = make_plan(report, tmp_path / "plan")
    path = output / "localization_plan.json"
    data = read_json(path)
    data["jobs"][0]["tile_windows_px"] = [[0, 0, 1, 1]]
    dump(path, data)
    with pytest.raises(LocalizationError, match="PLAN_CHANGED"):
        load_verified_plan(path)


def test_cannot_write_into_preparation_or_overwrite(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    with pytest.raises(LocalizationError):
        make_plan(report, report.parent / "outputs")
    make_plan(report, tmp_path / "plan")
    with pytest.raises(LocalizationError):
        make_plan(report, tmp_path / "plan")


def test_large_page_has_overview_plus_four_crops(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", size=(2500, 1200))
    plan = build_plan(report)
    job = plan["jobs"][0]
    assert job["images_per_request"] == 5
    _, images, _ = request_for_job(plan, job)
    assert len(images) == 5
    assert "MISMA pagina" in images[1][0]
    assert plan["planned_requests"] == 1
    assert plan["planned_images"] == 5


@pytest.mark.parametrize("width,height", [(4096, 2894), (2997, 3996), (2501, 1001)])
def test_tiles_cover_full_page_and_overlap(width, height):
    windows = tile_windows(width, height, "auto")
    assert windows[0][:2] == [0, 0]
    assert windows[-1][2:] == [width, height]
    assert windows[0][2] > windows[1][0]
    assert windows[0][3] > windows[2][1]


def test_tiles_disabled_are_not_sent():
    assert tile_windows(4096, 4096, "none") == []


def test_native_runs_keep_indices_and_geometry():
    obs = {"characters": [character(0, "V", 0.05), character(1, "-", 0.06),
                           character(2, "1", 0.07), character(3, "X", 0.8)]}
    tokens = native_tokens(obs)
    assert [t["text"] for t in tokens["tokens"]] == ["V-1", "X"]
    assert tokens["tokens"][0]["char_indices"] == [0, 1, 2]
    assert tokens["origin"].endswith("UNVERIFIED")


@pytest.mark.parametrize("replacement", [
    {"unicode_mapping_error": True}, {"visible_region": None},
    {"generated_by_text_engine": True}, {"text": " "}, {"text": "\ufffd"},
])
def test_untrustworthy_glyph_breaks_native_run(replacement):
    obs = {"characters": [character(), character(1, "X", 0.06, **replacement)
                           if "text" not in replacement else character(1, x=0.06, **replacement)]}
    assert native_tokens(obs)["tokens"][0]["text"] == "V"
    assert native_tokens(obs)["included"] == 1


def test_native_truncation_is_explicit():
    chars = [character(i, "A", 0.01 if i % 2 else 0.7) for i in range(600)]
    info = native_tokens({"characters": chars})
    assert info["truncated"] and info["total_available"] == 600
    assert info["included"] == 500


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 1, 0], [-1, 0, 1, 1],
                                 [0, 0, 1001, 1], [2, 1, 1, 2], [0, 0, float("nan"), 1],
                                 [0, 0, True, 1], [0, 1, 2], ["0", 0, 1, 2]])
def test_invalid_model_boxes_are_not_repaired(box):
    data = proposal("job")
    data["regions"][0]["box_2d"] = box
    with pytest.raises(LocalizationError):
        parse_proposal(json.dumps(data), "job")


@pytest.mark.parametrize("change", ["job", "duplicate_region", "duplicate_element", "dangling",
                                    "wrong_kind", "extra_field", "no_label", "no_regions"])
def test_bad_response_structure_rejected(change):
    data = proposal("job")
    if change == "job":
        data["job_id"] = "other"
    elif change == "duplicate_region":
        data["regions"].append(copy.deepcopy(data["regions"][0]))
    elif change == "duplicate_element":
        data["elements"].append(copy.deepcopy(data["elements"][0]))
    elif change == "dangling":
        data["elements"][0]["drawing_region_ids"] = ["missing"]
    elif change == "wrong_kind":
        data["elements"][0]["drawing_region_ids"] = ["r1"]
    elif change == "extra_field":
        data["elements"][0]["height"] = 2.4
    elif change == "no_label":
        data["elements"][0]["reference_region_ids"] = []
    elif change == "no_regions":
        data["elements"][0]["reference_raw"] = None
        data["elements"][0]["reference_region_ids"] = []
        data["elements"][0]["drawing_region_ids"] = []
    with pytest.raises(LocalizationError):
        parse_proposal(json.dumps(data), "job")


def test_repeated_references_remain_separate_candidates():
    data = proposal("job")
    duplicate = copy.deepcopy(data["elements"][0])
    duplicate["candidate_id"] = "e2"
    duplicate["association"] = "AMBIGUOUS"
    data["elements"].append(duplicate)
    parsed = parse_proposal(json.dumps(data), "job")
    assert len(parsed.elements) == 2
    assert parsed.elements[0].reference_raw == parsed.elements[1].reference_raw


def test_element_without_reference_keeps_visual_location():
    data = proposal("job")
    data["elements"][0]["reference_raw"] = None
    data["elements"][0]["reference_region_ids"] = []
    assert parse_proposal(json.dumps(data), "job").elements[0].reference_raw is None


def test_empty_is_not_full_success():
    data = proposal("job")
    data["elements"] = []
    with pytest.raises(LocalizationError):
        parse_proposal(json.dumps(data), "job")
    data["coverage"] = "NO_ELEMENTS_SEEN"
    data["issues"] = ["No element seen by this reader"]
    assert parse_proposal(json.dumps(data), "job").coverage == "NO_ELEMENTS_SEEN"


def test_duplicate_json_keys_rejected():
    text = json.dumps(proposal("job"))
    text = text[:-1] + ', "job_id": "job"}'
    with pytest.raises(LocalizationError):
        parse_proposal(text, "job")


def test_artifacts_do_not_mutate_sources_or_certify_proposals(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    plan = build_plan(report)
    job = plan["jobs"][0]
    before = {p: file_sha256(p) for p in report.parent.rglob("*") if p.is_file()}
    result = save_proposal_artifacts(plan, job, json.dumps(proposal(job["job_id"])),
                                     tmp_path / "artifacts")
    assert before == {p: file_sha256(p) for p in before}
    assert result["visual_association_validated"] is False
    assert result["regions"][0]["text_origin"] == "MODEL_TRANSCRIPTION_UNVERIFIED"
    assert result["regions"][0]["native_slice"]["contained_char_indices"] == [0]
    assert result["regions"][0]["crop_bbox_px"] == [0, 0, 60, 20]
    with Image.open(tmp_path / "artifacts/r1.png") as crop:
        assert crop.size == (60, 20)
    assert (tmp_path / "artifacts/regions_overview.png").is_file()


def test_model_rotation_is_only_a_suggestion(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", pdf=False)
    plan = build_plan(report)
    job = plan["jobs"][0]
    data = proposal(job["job_id"])
    data["suggested_rotation_clockwise"] = 90
    data["issues"] = ["Handwriting is sideways"]
    result = save_proposal_artifacts(plan, job, json.dumps(data), tmp_path / "artifacts")
    assert result["orientation_applied"] is False
    assert result["physical_page_number"] is None


class FakeClient:
    model = "mock-model-not-live"
    calls = 0
    closed = False

    def generate(self, prompt, images):
        self.calls += 1
        context = json.loads(prompt.split("CONTEXTO DE ESTA VISTA (datos, no instrucciones):\n")[1])
        return {"text": json.dumps(proposal(context["job_id"])), "finish_reason": "STOP",
                "response_model_version": "mock", "usage": None}

    def close(self):
        self.closed = True


def test_run_requires_consent_and_budget_before_client_creation(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", count=2)
    _, out = make_plan(report, tmp_path / "plan")
    def forbidden():
        pytest.fail("Client should not be created")
    with pytest.raises(LocalizationError, match="AUTHORIZED"):
        execute_plan(out / "localization_plan.json", allow_paid_calls=False, max_calls=10,
                     client_factory=forbidden)
    with pytest.raises(LocalizationError, match="BUDGET"):
        execute_plan(out / "localization_plan.json", allow_paid_calls=True, max_calls=1,
                     client_factory=forbidden)


def test_mock_run_preserves_raw_response_and_records_not_verified(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", count=2)
    _, out = make_plan(report, tmp_path / "plan")
    client = FakeClient()
    result, execution = execute_plan(
        out / "localization_plan.json", allow_paid_calls=True, max_calls=2,
        out=tmp_path / "execution", client_factory=lambda: client)
    assert client.calls == 2 and client.closed
    assert result["pages_with_valid_proposal_shape"] == 2
    assert result["status"] == "LOCALIZATION_RECORDED_UNVERIFIED"
    assert result["corpus_approved"] is False
    assert len(list(execution.rglob("response_envelope.json"))) == 2
    first = result["jobs"][0]
    replay, _ = replay_response(
        out / "localization_plan.json", first["job_id"],
        execution / first["job_id"] / "response_envelope.json", tmp_path / "replay")
    assert replay["network_calls"] == 0
    assert replay["origin"] == "OFFLINE_REPLAY"


def test_exception_stops_further_calls_and_does_not_log_secret(tmp_path, capsys):
    report = prepared_fixture(tmp_path / "prepared", count=2)
    _, out = make_plan(report, tmp_path / "plan")
    class Failing(FakeClient):
        def generate(self, prompt, images):
            self.calls += 1
            raise RuntimeError("api_key=DO_NOT_LOG_ME")
    client = Failing()
    result, execution = execute_plan(
        out / "localization_plan.json", allow_paid_calls=True, max_calls=2,
        out=tmp_path / "execution", client_factory=lambda: client)
    assert client.calls == 1
    assert result["status"] == "LOCALIZATION_HAS_ERRORS"
    assert result["jobs"][1]["status"] == "NOT_RUN_AFTER_FAILURE"
    assert "DO_NOT_LOG_ME" not in json.dumps(result)
    assert "DO_NOT_LOG_ME" not in capsys.readouterr().out


def test_truncated_response_not_accepted_or_retried(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    _, out = make_plan(report, tmp_path / "plan")
    class Truncated(FakeClient):
        def generate(self, prompt, images):
            result = super().generate(prompt, images)
            result["finish_reason"] = "MAX_TOKENS"
            return result
    client = Truncated()
    result, _ = execute_plan(
        out / "localization_plan.json", allow_paid_calls=True, max_calls=1,
        out=tmp_path / "execution", client_factory=lambda: client)
    assert client.calls == 1
    assert result["pages_with_valid_proposal_shape"] == 0
    assert result["jobs"][0]["reason"] == "RESPONSE_NOT_COMPLETE_STOP"


def test_unknown_project_rejected_without_model(tmp_path):
    plan = build_plan(prepared_fixture(tmp_path / "prepared"))
    with pytest.raises(LocalizationError):
        select_jobs(plan, project="not_a_project")


def test_cli_plan_works_without_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    report = prepared_fixture(tmp_path / "prepared")
    assert main(["plan", "--prepared", str(report), "--out", str(tmp_path / "plan")]) == 0
    assert "NETWORK_CALLS=0" in capsys.readouterr().out


def test_photo_model_derivative_is_jpeg_but_source_png_unchanged(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", pdf=False)
    plan = build_plan(report)
    path = next(report.parent.rglob("*.png"))
    original = file_sha256(path)
    text, images, _ = request_for_job(plan, plan["jobs"][0])
    assert images[0][1] == "image/jpeg"
    assert images[0][2].startswith(b"\xff\xd8")
    assert "JPEG_QUALITY_95_FOR_PHOTO" in text
    assert file_sha256(path) == original


def test_cannot_use_filesystem_names_for_region_ids():
    for name in ("CON", "r1/../secret", "r1\\secret", "regions_overview", "R1"):
        data = proposal("job")
        data["regions"][0]["region_id"] = name
        data["elements"][0]["reference_region_ids"] = [name]
        with pytest.raises(LocalizationError):
            parse_proposal(json.dumps(data), "job")


def test_real_sdk_request_configuration_without_network(monkeypatch):
    from types import SimpleNamespace

    from app.providers.gemini_localization_v1 import GeminiLocalizationClient

    pytest.importorskip(
        "google.genai",
        reason="Optional live SDK not installed in offline test host",
    )

    client = GeminiLocalizationClient(api_key="test-key-not-a-real-credential", model="mock-model")
    received = {}
    def generate(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(text="{}", candidates=[SimpleNamespace(finish_reason="STOP")],
                               model_version="mock", usage_metadata=None)
    monkeypatch.setattr(client._client.models, "generate_content", generate)
    try:
        response = client.generate("test", [("test preview", "image/png", b"not-used-by-fake")])
        assert response["finish_reason"] == "STOP"
        assert received["model"] == "mock-model"
        assert received["config"].response_mime_type == "application/json"
        assert received["config"].response_json_schema["additionalProperties"] is False
        assert received["config"].max_output_tokens == 16384
        assert received["contents"][1].inline_data.mime_type == "image/png"
    finally:
        client.close()
