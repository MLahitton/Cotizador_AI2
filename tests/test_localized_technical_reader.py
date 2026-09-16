"""Experimental localized technical reader tests with synthetic fixtures only."""
from __future__ import annotations

import json
import socket

import pytest

from app.services.localized_technical_reader import (
    LocalizedTechnicalReaderError,
    build_localized_technical_plan,
    localized_technical_api_schema,
    parse_localized_technical_response,
    replay_localized_technical_response,
    request_for_candidate,
    run_localized_technical_plan,
)
from manual_tests.corpus_v1.localize_sources import replay_response, write_json
from tests.test_localization_image_frames import (
    image_frame_plan,
    proposal,
    registry_sha256,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("No network allowed in localized technical reader tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def localization_artifact(tmp_path, *, blocked_label=False, repeated=False):
    preparation = tmp_path / "prepared" / "source_preparation_report.json"
    plan, job, _prompt, _images, debug = image_frame_plan(tmp_path, mark=True)
    label_box = [400, 400, 500, 500] if blocked_label else None
    data = proposal(job["job_id"], registry_sha256(debug), box=label_box)
    dimension_box = [0, 0, 100, 100] if blocked_label else [300, 300, 400, 500]
    data["regions"].append({
        "region_id": "r3",
        "kind": "DIMENSION_AREA",
        "source_image_id": "img0",
        "box_2d": dimension_box,
        "observed_label": "Dimension area",
        "transcription": "ANCHO 1200",
        "legibility": "READABLE",
    })
    data["elements"][0]["localization_notes"] = ""
    data["elements"][0]["dimension_region_ids"] = ["r3"]
    data["elements"][0]["dimension_area_status"] = "LOCATED"
    if repeated:
        second = dict(data["elements"][0], candidate_id="e2")
        data["elements"].append(second)
    response = tmp_path / "localized_response.json"
    write_json(response, {"finish_reason": "STOP", "text": json.dumps(data)})
    result, output = replay_response(
        tmp_path / "plan" / "localization_plan.json",
        job["job_id"],
        response,
        tmp_path / "localized",
    )
    return result, output / "replay_report.json", preparation


def technical_response(temporary_id, *, field_path="measurements"):
    return {
        "text": json.dumps({
            "enrichment": {
                "elements": [{
                    "temporary_id": temporary_id,
                    "reference": "V-1",
                    "measurements": [{
                        "type": "width",
                        "value": 1200,
                        "unit": "mm",
                        "status": "explicit",
                    }],
                    "quantity": None,
                    "missing_or_unknown": ["quantity"],
                    "status": "explicit",
                }],
                "warnings": ["experimental localized read"],
            },
            "field_evidence": [{
                "field_path": field_path,
                "input_image_id": "page_context",
                "source_id": f"localized:doc-{'1' * 24}:1",
                "region_id": "r2",
                "region": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.2},
                "observed_text": "ANCHO 1200",
                "visual_description": None,
                "status": "SUPPORTED",
                "notes": None,
            }],
            "pending": ["quantity"],
            "conflicts": [],
        }),
        "finish_reason": "STOP",
    }


class FakeProvider:
    model = "fake-localized-reader"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.closed = False

    def generate_localized_technical(self, prompt, images):
        self.calls += 1
        assert prompt
        assert images[0][0] == "page_context"
        return self.response

    def close(self):
        self.closed = True


def test_plan_builds_candidate_packages_with_context_and_schema(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    prompt, images, context = request_for_candidate(plan, plan["candidates"][0])
    assert plan["candidate_count"] == 1
    assert localized_technical_api_schema()["type"] == "object"
    assert "PAQUETE DEL CANDIDATO" in prompt
    assert images[0][0] == "page_context"
    assert (
        context["regions"][0]["context_bbox_px"][0]
        <= context["regions"][0]["proposed_bbox_px"][0]
    )
    assert context["visual_association_validated"] is False


def test_blocked_label_keeps_candidate_with_alternate_support(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path, blocked_label=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for region in report["regions"]:
        if region["region_id"] == "r2":
            region["evidence_status"] = "BLOCKED_SUSPICIOUS_CONTENT"
        elif region["region_id"] == "r3":
            region["evidence_status"] = "UNVERIFIED"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    _prompt, _images, context = request_for_candidate(plan, plan["candidates"][0])
    blocked_regions = [
        region for region in context["regions"] if not region["usable_as_support"]
    ]
    usable_regions = [
        region for region in context["regions"] if region["usable_as_support"]
    ]
    assert context["candidate_id"] == "e1"
    assert {region["region_id"] for region in blocked_regions} >= {"r1", "r2"}
    assert {region["evidence_status"] for region in blocked_regions} >= {
        "BLOCKED_NO_VISIBLE_CONTENT",
        "BLOCKED_SUSPICIOUS_CONTENT",
    }
    assert {region["region_id"] for region in usable_regions} == {"r3"}
    assert usable_regions[0]["evidence_status"] == "UNVERIFIED"
    assert set(plan["candidates"][0]["linked_region_ids"]) >= {"r1", "r3"}


def test_repeated_references_use_candidate_identity_not_reference(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path, repeated=True)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    identities = [candidate["technical_candidate_id"] for candidate in plan["candidates"]]
    assert len(identities) == 2
    assert len(set(identities)) == 2


def test_invalid_field_path_and_unknown_image_or_source_are_rejected(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    candidate = plan["candidates"][0]
    _prompt, _images, context = request_for_candidate(plan, candidate)
    with pytest.raises(LocalizedTechnicalReaderError, match="INVALID_LOCALIZED"):
        parse_localized_technical_response(
            technical_response(context["temporary_id"], field_path="not_a_field")["text"],
            context,
        )
    response = technical_response(context["temporary_id"])
    payload = json.loads(response["text"])
    payload["field_evidence"][0]["input_image_id"] = "missing"
    with pytest.raises(LocalizedTechnicalReaderError, match="UNKNOWN_INPUT_IMAGE_ID"):
        parse_localized_technical_response(json.dumps(payload), context)
    payload["field_evidence"][0]["input_image_id"] = "page_context"
    payload["field_evidence"][0]["source_id"] = "other"
    with pytest.raises(LocalizedTechnicalReaderError, match="UNKNOWN_SOURCE_ID"):
        parse_localized_technical_response(json.dumps(payload), context)


def test_relative_field_path_is_preserved(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    candidate = plan["candidates"][0]
    _prompt, _images, context = request_for_candidate(plan, candidate)
    envelope = parse_localized_technical_response(
        technical_response(context["temporary_id"], field_path="quantity")["text"],
        context,
    )
    assert envelope.field_evidence[0].field_path == "quantity"


def test_full_field_path_is_normalized_to_element_relative_path(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    candidate = plan["candidates"][0]
    _prompt, _images, context = request_for_candidate(plan, candidate)
    envelope = parse_localized_technical_response(
        technical_response(
            context["temporary_id"],
            field_path="enrichment.elements[0].reference",
        )["text"],
        context,
    )
    assert envelope.field_evidence[0].field_path == "reference"


def test_full_nested_list_field_path_is_normalized(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    candidate = plan["candidates"][0]
    _prompt, _images, context = request_for_candidate(plan, candidate)
    envelope = parse_localized_technical_response(
        technical_response(
            context["temporary_id"],
            field_path="enrichment.elements[0].measurements[0].value",
        )["text"],
        context,
    )
    assert envelope.field_evidence[0].field_path == "measurements[0].value"


@pytest.mark.parametrize(
    "field_path",
    [
        "enrichment.sources[0].reference",
        "enrichment.elements[1].reference",
        "enrichment.elements[0].not_a_field",
        "enrichment.elements[0].measurements.value",
    ],
)
def test_invalid_full_field_paths_are_rejected(tmp_path, field_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    candidate = plan["candidates"][0]
    _prompt, _images, context = request_for_candidate(plan, candidate)
    with pytest.raises(LocalizedTechnicalReaderError, match="INVALID_LOCALIZED"):
        parse_localized_technical_response(
            technical_response(context["temporary_id"], field_path=field_path)["text"],
            context,
        )


def test_run_with_fake_provider_persists_raw_and_final_result(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    candidate = plan["candidates"][0]
    response = technical_response(candidate["technical_candidate_id"])
    provider = FakeProvider(response)
    report = run_localized_technical_plan(
        plan,
        tmp_path / "run",
        provider,
        max_calls=1,
        candidate_id=candidate["candidate_id"],
    )
    assert provider.calls == 1 and provider.closed
    assert report["status"] == "LOCALIZED_TECHNICAL_READING_RECORDED_UNVERIFIED"
    output = tmp_path / "run" / candidate["technical_candidate_id"].replace("|", "__")
    assert (output / "response_envelope.json").is_file()
    assert (output / "raw_parsed_reading.json").is_file()
    assert (output / "localized_technical_reading.json").is_file()


def test_replay_requires_saved_technical_response_without_provider(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    candidate = plan["candidates"][0]
    response = tmp_path / "technical_response.json"
    write_json(response, technical_response(candidate["technical_candidate_id"]))
    result = replay_localized_technical_response(
        plan,
        candidate["candidate_id"],
        response,
        tmp_path / "replay",
    )
    assert result.status == "LOCALIZED_TECHNICAL_READING_RECORDED_UNVERIFIED"
    assert result.enrichment.elements[0].temporary_id == candidate["technical_candidate_id"]


def test_max_calls_insufficient_does_not_create_provider(tmp_path):
    _localized, report_path, preparation = localization_artifact(tmp_path, repeated=True)
    plan = build_localized_technical_plan(report_path, preparation_report_path=preparation)
    provider = FakeProvider(technical_response(plan["candidates"][0]["technical_candidate_id"]))
    with pytest.raises(LocalizedTechnicalReaderError, match="REQUEST_BUDGET"):
        run_localized_technical_plan(plan, tmp_path / "run", provider, max_calls=1)
    assert provider.calls == 0
