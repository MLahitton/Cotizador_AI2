"""Synthetic checks for opt-in image-frame localization. No live Gemini calls."""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from app.providers.gemini_localization_v1 import (
    image_frame_localization_api_schema_metadata,
    localization_api_schema_metadata,
)
from app.services.document_localization_v1 import (
    IMAGE_FRAME_CONTRACT_VERSION,
    LocalizationError,
    LocalizationResponseValidationError,
    image_frame_request_for_job,
    parse_image_frame_proposal,
    parse_proposal,
    previous_image_frame_notes_schema_digest,
    read_json,
    response_schema_digest,
    save_image_frame_proposal_artifacts,
    save_proposal_artifacts,
)
from app.services.localization_frame_guard import (
    LocalizationFrameError,
    local_box_to_full_page_box,
    registry_digest,
    validate_image_registry,
)
from app.services.localization_prompt_v1 import (
    image_frame_prompt_digest,
    prompt_digest,
)
from manual_tests.corpus_v1.localize_sources import (
    execute_plan,
    make_plan,
    replay_response,
    write_json,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("No network allowed in image-frame localization tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def prepared_fixture(root, *, size=(3000, 1200), mark=False, text=True):
    document_id = "doc-" + "1" * 24
    directory = root / "documents" / document_id
    directory.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", size, "white") as image:
        if mark:
            image.putpixel((20, 20), (254, 254, 254))
        image.save(directory / "page-0001.png")
    chars = [{
        "pdf_char_index": 0,
        "text": "V",
        "visible_region": {"x": 0.01, "y": 0.01, "width": 0.01, "height": 0.01},
        "unicode_mapping_error": False,
        "generated_by_text_engine": False,
    }] if text else []
    dump(directory / "page-0001.json", {
        "schema_version": 1,
        "document_id": document_id,
        "page_number": 1,
        "status": "PREPARED",
        "preview_file": "page-0001.png",
        "frame": {"origin_preview": "TOP_LEFT", "preview_width_px": size[0],
                  "preview_height_px": size[1]},
        "characters": chars,
    })
    report = root / "source_preparation_report.json"
    dump(report, {
        "schema_version": 1,
        "status": "PREPARATION_COMPLETED",
        "run_id": "synthetic",
        "source_archive_unchanged": True,
        "source_archive_sha256": "0" * 64,
        "documents": [{
            "status": "PREPARED",
            "project_id": "synthetic",
            "corpus_document_id": "SYN",
            "document_id": document_id,
            "logical_name": "synthetic.pdf",
            "content_sha256": hashlib.sha256(b"source").hexdigest(),
            "media_type": "application/pdf",
            "page_count": 1,
            "artifact_directory": "documents/" + document_id,
            "pages": [{
                "status": "PREPARED",
                "view_index": 1,
                "page_number": 1,
                "preview_file": "page-0001.png",
                "observations_file": "page-0001.json",
                "width_px": size[0],
                "height_px": size[1],
                "text_state": "NATIVE_TEXT_PRESENT_UNVERIFIED" if text else "NO_NATIVE_TEXT",
                "warnings": [],
            }],
        }],
    })
    return report


def image_frame_plan(tmp_path, *, mark=False, text=True):
    report = prepared_fixture(tmp_path / "prepared", mark=mark, text=text)
    _summary, output = make_plan(
        report,
        tmp_path / "plan",
        "auto",
        image_frames=True,
    )
    plan = read_json(output / "localization_plan.json")
    job = plan["jobs"][0]
    prompt, images, debug = image_frame_request_for_job(plan, job)
    return plan, job, prompt, images, debug


def registry_sha256(debug):
    return registry_digest(debug["image_registry"])


def proposal(job_id, registry_digest, *, source_image_id="img0", box=None):
    return {
        "contract_version": IMAGE_FRAME_CONTRACT_VERSION,
        "job_id": job_id,
        "image_registry_sha256": registry_digest,
        "page_roles": ["DETAIL_SHEET"],
        "suggested_rotation_clockwise": 0,
        "coverage": "FULL_SCAN_CLAIMED",
        "regions": [
            {"region_id": "r1", "kind": "REFERENCE_LABEL", "source_image_id": source_image_id,
             "box_2d": box or [0, 0, 100, 100],
             "observed_label": "V-1", "transcription": "V-1", "legibility": "READABLE"},
            {"region_id": "r2", "kind": "TABLE", "source_image_id": source_image_id,
             "box_2d": [100, 100, 300, 300],
             "observed_label": "Table", "transcription": "V-1", "legibility": "READABLE"},
        ],
        "elements": [{
            "candidate_id": "e1",
            "reference_raw": "V-1",
            "description_of_location": "Synthetic candidate",
            "reference_region_ids": ["r1"],
            "drawing_region_ids": [],
            "table_region_ids": ["r2"],
            "dimension_region_ids": [],
            "note_region_ids": [],
            "missing_dimension_area_links": ["no dimension area visible"],
            "dimension_area_status": "NOT_VISIBLE",
            "table_status": "LOCATED",
            "localization_notes": "Dimension area is not visible in the synthetic crop.",
            "association": "PROPOSED",
            "association_basis": "Visible label and table row.",
        }],
        "issues": [],
    }


def legacy_proposal(job_id, *, count=1, with_dimension=False):
    regions = [
        {"region_id": "r1", "kind": "REFERENCE_LABEL", "box_2d": [0, 0, 100, 100],
         "observed_label": "V", "transcription": "V", "legibility": "READABLE"},
        {"region_id": "r2", "kind": "DRAWING", "box_2d": [100, 100, 300, 300],
         "observed_label": "Drawing", "transcription": None, "legibility": "READABLE"},
    ]
    if with_dimension:
        regions.append(
            {"region_id": "r3", "kind": "DIMENSION_AREA", "box_2d": [300, 300, 400, 400],
             "observed_label": "Dimension", "transcription": "1.20", "legibility": "READABLE"}
        )
    elements = []
    for index in range(count):
        elements.append({
            "candidate_id": f"e{index + 1}",
            "reference_raw": f"V-{index + 1}",
            "description_of_location": "Synthetic candidate",
            "reference_region_ids": ["r1"],
            "drawing_region_ids": ["r2"],
            "table_region_ids": [],
            "dimension_region_ids": ["r3"] if with_dimension else [],
            "note_region_ids": [],
            "association": "PROPOSED",
            "association_basis": "Visible label and drawing.",
        })
    return {
        "job_id": job_id,
        "page_roles": ["DETAIL_SHEET"],
        "suggested_rotation_clockwise": 0,
        "coverage": "FULL_SCAN_CLAIMED",
        "regions": regions,
        "elements": elements,
        "issues": [],
    }


def test_full_page_identity_frame_is_registered(tmp_path):
    plan, job, prompt, images, debug = image_frame_plan(tmp_path)
    registry = debug["image_registry"]
    assert plan["coordinate_contract"] == IMAGE_FRAME_CONTRACT_VERSION
    assert plan["prompt_sha256"] == image_frame_prompt_digest()
    assert registry[0]["image_id"] == "img0"
    assert registry[0]["window_px"] == [0, 0, job["width_px"], job["height_px"]]
    assert images[0][0].startswith("img0 ")
    assert '"image_registry_sha256"' in prompt


def test_crop_coordinates_convert_with_offset_and_distinct_size(tmp_path):
    _plan, _job, _prompt, _images, debug = image_frame_plan(tmp_path)
    crop = debug["image_registry"][1]
    y0, x0, y1, x1 = local_box_to_full_page_box([250, 100, 750, 900], crop)
    assert x0 == pytest.approx(crop["window_px"][0] + 0.1 * crop["image_width_px"])
    assert y0 == pytest.approx(crop["window_px"][1] + 0.25 * crop["image_height_px"])
    assert x1 == pytest.approx(crop["window_px"][0] + 0.9 * crop["image_width_px"])
    assert y1 == pytest.approx(crop["window_px"][1] + 0.75 * crop["image_height_px"])


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 1001, 1], [0, 0, float("nan"), 1]])
def test_invalid_local_boxes_are_rejected(box):
    frame = {"image_id": "img0", "window_px": [0, 0, 100, 100],
             "image_width_px": 100, "image_height_px": 100,
             "mime_type": "image/png", "sha256": "a" * 64}
    with pytest.raises(LocalizationFrameError):
        local_box_to_full_page_box(box, frame)


def test_invalid_registry_and_unknown_source_id_are_rejected(tmp_path):
    with pytest.raises(LocalizationFrameError):
        validate_image_registry([], (100, 100))
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug), source_image_id="img99")
    with pytest.raises(LocalizationError, match="UNKNOWN_SOURCE_IMAGE_ID"):
        parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])


def test_image_registry_hash_tampering_is_rejected(tmp_path):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], "0" * 64)
    with pytest.raises(LocalizationError, match="IMAGE_REGISTRY_DIGEST_MISMATCH"):
        parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])


def test_legacy_parser_does_not_accept_image_frame_contract(tmp_path):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    with pytest.raises(LocalizationError):
        parse_proposal(json.dumps(data), job["job_id"])


def test_image_frame_parser_rejects_legacy_contract(tmp_path):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = {
        "job_id": job["job_id"], "page_roles": ["DETAIL_SHEET"],
        "suggested_rotation_clockwise": 0, "coverage": "NO_ELEMENTS_SEEN",
        "regions": [], "elements": [], "issues": ["No candidates"],
    }
    with pytest.raises(LocalizationError):
        parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])


def test_image_frame_located_with_empty_notes_passes_without_mutation(tmp_path):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    data["elements"][0]["localization_notes"] = ""
    parsed = parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])
    assert parsed.elements[0].localization_notes == ""


@pytest.mark.parametrize("field", ["dimension_area_status", "table_status"])
@pytest.mark.parametrize("status", ["UNREADABLE", "UNRESOLVED"])
@pytest.mark.parametrize("notes", ["", "   "])
def test_unreadable_or_unresolved_states_require_notes(tmp_path, field, status, notes):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    data["elements"][0][field] = status
    data["elements"][0]["localization_notes"] = notes
    with pytest.raises(LocalizationResponseValidationError) as caught:
        parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])
    diagnostic = caught.value.diagnostic()
    assert diagnostic["stage"] == "LOCAL_RESPONSE_VALIDATION"
    assert diagnostic["contract"] == IMAGE_FRAME_CONTRACT_VERSION
    assert diagnostic["validator"] == "ImageFrameLocalizationProposal"
    assert diagnostic["errors"][0]["type"] == "value_error"
    assert "input" not in json.dumps(diagnostic)


@pytest.mark.parametrize("field", ["dimension_area_status", "table_status"])
@pytest.mark.parametrize("status", ["UNREADABLE", "UNRESOLVED"])
def test_unreadable_or_unresolved_states_accept_explained_notes(tmp_path, field, status):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    data["elements"][0][field] = status
    data["elements"][0]["localization_notes"] = "Visible but not readable."
    parsed = parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])
    assert parsed.elements[0].localization_notes == "Visible but not readable."


@pytest.mark.parametrize("value", [None, 12, "x" * 1501])
def test_localization_notes_null_wrong_type_and_too_long_rejected(tmp_path, value):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    data["elements"][0]["localization_notes"] = value
    with pytest.raises(LocalizationResponseValidationError):
        parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])
    del data["elements"][0]["localization_notes"]
    with pytest.raises(LocalizationResponseValidationError):
        parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])


def test_uniform_white_crop_blocks_evidence_but_keeps_candidate(tmp_path):
    plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    result = save_image_frame_proposal_artifacts(
        plan,
        job,
        json.dumps(data),
        tmp_path / "artifacts",
    )
    gate = read_json(tmp_path / "artifacts" / "evidence_gate.json")
    assert result["elements"][0]["candidate_id"] == "e1"
    assert result["elements"][0]["evidence_blocked_region_ids"]
    assert gate["status"] == "REVIEW_REQUIRED"
    assert "r1" in gate["blocked_region_ids"]
    assert gate["candidates_with_blocked_region_links"] == [{
        "candidate_id": "e1",
        "reference_raw": "V-1",
        "blocked_region_ids": ["r1", "r2"],
    }]


def test_minimal_contrast_and_absent_native_text_do_not_block(tmp_path):
    plan, job, _prompt, _images, debug = image_frame_plan(tmp_path, mark=True, text=False)
    data = proposal(job["job_id"], registry_sha256(debug))
    result = save_image_frame_proposal_artifacts(
        plan,
        job,
        json.dumps(data),
        tmp_path / "artifacts",
    )
    assert result["regions"][0]["evidence_status"] == "UNVERIFIED"
    assert result["regions"][0]["native_slice"]["contained_char_indices"] == []
    assert result["evidence_gate"]["candidates_without_dimension_links_count"] == 1
    assert result["evidence_gate"]["candidate_warnings"] == []


def test_border_contact_is_warning_not_correction(tmp_path):
    plan, job, _prompt, _images, debug = image_frame_plan(tmp_path, mark=True)
    data = proposal(job["job_id"], registry_sha256(debug), box=[0, 0, 100, 100])
    result = save_image_frame_proposal_artifacts(
        plan,
        job,
        json.dumps(data),
        tmp_path / "artifacts",
    )
    assert result["regions"][0]["evidence_warnings"] == ["REGION_TOUCHES_SOURCE_IMAGE_BORDER"]
    assert result["visual_association_validated"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("dimension_area_status", "LOCATED"),
        ("table_status", "NOT_VISIBLE"),
    ],
)
def test_contradictory_status_and_links_are_rejected(tmp_path, field, value):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    data["elements"][0][field] = value
    with pytest.raises(LocalizationError):
        parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])


def test_missing_dimension_links_and_shared_tables_are_preserved(tmp_path):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path)
    data = proposal(job["job_id"], registry_sha256(debug))
    second = dict(data["elements"][0], candidate_id="e2", reference_raw="V-2")
    data["elements"].append(second)
    parsed = parse_image_frame_proposal(json.dumps(data), job["job_id"], debug["image_registry"])
    assert parsed.elements[0].missing_dimension_area_links == ["no dimension area visible"]
    assert parsed.elements[0].table_region_ids == parsed.elements[1].table_region_ids


def test_legacy_gate_reports_candidates_without_dimension_links(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", mark=True)
    _summary, output = make_plan(report, tmp_path / "plan", "none")
    plan = read_json(output / "localization_plan.json")
    job = plan["jobs"][0]
    data = legacy_proposal(job["job_id"], count=7)
    result = save_proposal_artifacts(
        plan,
        job,
        json.dumps(data),
        tmp_path / "artifacts",
    )
    gate = read_json(tmp_path / "artifacts" / "evidence_gate.json")
    assert len(result["elements"]) == 7
    assert gate == result["evidence_gate"]
    assert gate["candidates_without_dimension_links_count"] == 7
    assert len(gate["candidates_without_dimension_links"]) == 7
    assert gate["candidate_warnings"][0]["warnings"] == ["DIMENSION_LINKS_NOT_LOCALIZED"]
    assert data["issues"] == []


def test_legacy_dimension_links_do_not_warn_as_missing(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", mark=True)
    _summary, output = make_plan(report, tmp_path / "plan", "none")
    plan = read_json(output / "localization_plan.json")
    job = plan["jobs"][0]
    result = save_proposal_artifacts(
        plan,
        job,
        json.dumps(legacy_proposal(job["job_id"], with_dimension=True)),
        tmp_path / "artifacts",
    )
    assert result["evidence_gate"]["candidates_without_dimension_links"] == []
    assert result["evidence_gate"]["candidate_warnings"] == []


def test_replay_image_frame_response_without_network(tmp_path):
    plan, job, _prompt, _images, debug = image_frame_plan(tmp_path, mark=True)
    data = proposal(job["job_id"], registry_sha256(debug))
    data["elements"][0]["localization_notes"] = ""
    response = tmp_path / "response.json"
    write_json(response, {
        "finish_reason": "STOP",
        "text": json.dumps(data),
    })
    result, _output = replay_response(
        tmp_path / "plan" / "localization_plan.json",
        job["job_id"],
        response,
        tmp_path / "replay",
    )
    assert result["network_calls"] == 0
    assert result["coordinate_contract"] == IMAGE_FRAME_CONTRACT_VERSION
    gate = read_json(tmp_path / "replay" / "proposals" / "evidence_gate.json")
    assert result["evidence_gate"] == gate


def test_legacy_digests_are_unchanged():
    assert prompt_digest() == (
        "4d8ef44b0f0c4dae7739ef3e83a170a99949494a75cff9a5eb7e1a4b6714dabe"
    )
    assert response_schema_digest() == (
        "0e68ddc2109761d86e6cd737706b111248408d8f949cfc85ee77a09f4630df5a"
    )
    assert localization_api_schema_metadata()["api_schema_sha256"] == (
        "520bec7ba05c6f713135082382cdd7a1606e2adb4df1c4fa6c0a929c7b518d5b"
    )


def test_image_frame_mode_is_opt_in(tmp_path):
    report = prepared_fixture(tmp_path / "prepared")
    _legacy_summary, legacy_out = make_plan(report, tmp_path / "legacy", "none")
    _new_summary, new_out = make_plan(
        report,
        tmp_path / "frames",
        "none",
        image_frames=True,
    )
    legacy = read_json(legacy_out / "localization_plan.json")
    framed = read_json(new_out / "localization_plan.json")
    assert "coordinate_contract" not in legacy
    assert framed["coordinate_contract"] == IMAGE_FRAME_CONTRACT_VERSION


def test_image_frame_plan_execute_preflight_uses_image_frame_contract(tmp_path):
    plan, job, _prompt, _images, debug = image_frame_plan(tmp_path, mark=True)
    data = proposal(job["job_id"], registry_sha256(debug))
    data["elements"][0]["localization_notes"] = ""
    response_text = json.dumps(data)

    class FakeClient:
        model = "mock-model"
        calls = 0
        closed = False

        def generate_image_frames(self, prompt, images):
            self.calls += 1
            return {
                "text": response_text,
                "finish_reason": "STOP",
                "usage": None,
                "response_model_version": "mock",
            }

        def close(self):
            self.closed = True

    client = FakeClient()
    result, _output = execute_plan(
        tmp_path / "plan" / "localization_plan.json",
        allow_paid_calls=True,
        max_calls=1,
        out=tmp_path / "run",
        client_factory=lambda: client,
    )
    assert plan["coordinate_contract"] == IMAGE_FRAME_CONTRACT_VERSION
    assert client.calls == 1 and client.closed
    assert result["status"] == "LOCALIZATION_RECORDED_UNVERIFIED"
    assert result["api_schema"] == image_frame_localization_api_schema_metadata()
    assert result["api_schema"] != localization_api_schema_metadata()


def test_replay_accepts_previous_notes_schema_only_for_authorized_policy(tmp_path):
    plan, job, _prompt, _images, debug = image_frame_plan(tmp_path, mark=True)
    plan_path = tmp_path / "plan" / "localization_plan.json"
    data = read_json(plan_path)
    data["response_schema_sha256"] = previous_image_frame_notes_schema_digest()
    data.pop("local_validation_policy", None)
    dump(plan_path, data)
    response = tmp_path / "response.json"
    body = proposal(job["job_id"], registry_sha256(debug))
    body["elements"][0]["localization_notes"] = ""
    write_json(response, {"finish_reason": "STOP", "text": json.dumps(body)})

    result, _output = replay_response(
        plan_path,
        job["job_id"],
        response,
        tmp_path / "replay",
    )
    assert result["schema_provenance"]["previous_notes_policy_accepted"] is True
    assert result["schema_provenance"]["applied_response_schema_sha256"] != (
        previous_image_frame_notes_schema_digest()
    )


def test_run_does_not_accept_previous_notes_schema_as_current_plan(tmp_path):
    _plan, job, _prompt, _images, debug = image_frame_plan(tmp_path, mark=True)
    plan_path = tmp_path / "plan" / "localization_plan.json"
    data = read_json(plan_path)
    data["response_schema_sha256"] = previous_image_frame_notes_schema_digest()
    data.pop("local_validation_policy", None)
    dump(plan_path, data)

    class FakeClient:
        model = "mock-model"

        def generate_image_frames(self, prompt, images):
            return {
                "text": json.dumps(proposal(job["job_id"], registry_sha256(debug))),
                "finish_reason": "STOP",
            }

    with pytest.raises(LocalizationError, match="PLAN_CHANGED_OR_STALE_REBUILD_IT"):
        execute_plan(
            plan_path,
            allow_paid_calls=True,
            max_calls=1,
            out=tmp_path / "run",
            client_factory=lambda: FakeClient(),
        )


def test_image_frame_cli_plan_roundtrips_in_separate_processes(tmp_path):
    report = prepared_fixture(tmp_path / "prepared", mark=True)
    plan_dir = tmp_path / "plan"
    command = [
        sys.executable,
        "manual_tests/corpus_v1/localize_sources.py",
        "plan",
        "--prepared",
        str(report),
        "--image-frames",
        "--out",
        str(plan_dir),
    ]
    planned = subprocess.run(command, cwd=Path.cwd(), check=False, capture_output=True, text=True)
    assert planned.returncode == 0, planned.stderr

    verifier = (
        "from pathlib import Path\n"
        "from manual_tests.corpus_v1.localize_sources import load_verified_plan\n"
        "from app.services.document_localization_v1 import image_frame_request_for_job\n"
        f"plan = load_verified_plan(Path({str(plan_dir / 'localization_plan.json')!r}))\n"
        "prompt, images, debug = image_frame_request_for_job(plan, plan['jobs'][0])\n"
        "assert prompt and images and debug['image_registry'][0]['image_id'] == 'img0'\n"
    )
    checked = subprocess.run(
        [sys.executable, "-c", verifier],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
