"""API-schema compatibility projection and unchanged local validation. No live API calls."""
from __future__ import annotations

import copy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models.document_localization_v1 import PageLocalizationProposal
from app.providers.gemini_localization_v1 import (
    GeminiLocalizationClient,
    _compact_api_schema,
    build_localization_api_schema,
    localization_api_schema_metadata,
)
from app.services.document_localization_v1 import LocalizationError, parse_proposal
from manual_tests.corpus_v1 import localize_sources as runner


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("No network permitted in API schema tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def valid_proposal():
    return {
        "job_id": "synthetic-job", "page_roles": ["DETAIL_SHEET"],
        "suggested_rotation_clockwise": 0, "coverage": "FULL_SCAN_CLAIMED",
        "regions": [
            {"region_id": "r1", "kind": "REFERENCE_LABEL", "box_2d": [10, 20, 30, 60],
             "observed_label": "Label", "transcription": "REF-A", "legibility": "READABLE"},
            {"region_id": "r2", "kind": "DRAWING", "box_2d": [40, 20, 500, 600],
             "observed_label": "Drawing", "transcription": None, "legibility": "READABLE"},
        ],
        "elements": [
            {"candidate_id": "e1", "reference_raw": "REF-A", "description_of_location": "Left",
             "reference_region_ids": ["r1"], "drawing_region_ids": ["r2"],
             "table_region_ids": [], "dimension_region_ids": [], "note_region_ids": [],
             "association": "PROPOSED", "association_basis": "Visible label"},
        ],
        "issues": [],
    }


def walk(node):
    yield node
    for child in node.get("properties", {}).values():
        yield from walk(child)
    for key in ("items",):
        if isinstance(node.get(key), dict):
            yield from walk(node[key])
    for child in node.get("anyOf", []):
        yield from walk(child)


def test_schema_keeps_every_field_type_enum_and_null_branch():
    original = PageLocalizationProposal.model_json_schema()
    generated = build_localization_api_schema()
    def compare(left, right):
        if "$ref" in left:
            left = original["$defs"][left["$ref"].split("/")[-1]]
        for key in ("type", "enum", "required", "additionalProperties"):
            assert left.get(key) == right.get(key)
        assert list(left.get("properties", {})) == list(right.get("properties", {}))
        for name, child in left.get("properties", {}).items():
            compare(child, right["properties"][name])
        if "items" in left:
            compare(left["items"], right["items"])
        assert len(left.get("anyOf", [])) == len(right.get("anyOf", []))
        for a, b in zip(left.get("anyOf", []), right.get("anyOf", []), strict=True):
            compare(a, b)
    compare(original, generated)


def test_wire_schema_contains_only_supported_structure_and_box_arity():
    supported = {"type", "properties", "required", "items", "enum", "anyOf",
                 "additionalProperties", "minItems", "maxItems"}
    nodes = list(walk(build_localization_api_schema()))
    assert all(set(node) <= supported for node in nodes)
    bounded = [node for node in nodes if "maxItems" in node or "minItems" in node]
    assert bounded == [{"items": {"type": "number"}, "type": "array",
                        "minItems": 4, "maxItems": 4}]


def test_no_mutation_or_shared_nested_state():
    original = PageLocalizationProposal.model_json_schema()
    before = copy.deepcopy(original)
    first = _compact_api_schema(original, original["$defs"])
    assert original == before
    first["properties"]["regions"]["items"]["required"].clear()
    second = build_localization_api_schema()
    assert second["properties"]["regions"]["items"]["required"]
    assert original == PageLocalizationProposal.model_json_schema()


def test_keywords_as_property_names_are_not_dropped():
    original = {"type": "object", "properties": {
        "pattern": {"type": "string", "pattern": "^ok$"},
        "maxLength": {"type": "integer"},
    }, "required": ["pattern", "maxLength"]}
    result = _compact_api_schema(original, {})
    assert result["properties"] == {"pattern": {"type": "string"},
                                    "maxLength": {"type": "integer"}}
    assert result["required"] == ["pattern", "maxLength"]


@pytest.mark.parametrize("schema,defs", [
    ({"oneOf": [{"type": "string"}]}, {}),
    ({"$ref": "https://example.invalid/schema"}, {}),
    ({"$ref": "#/$defs/Missing"}, {}),
    ({"$ref": "#/$defs/Cycle"}, {"Cycle": {"$ref": "#/$defs/Cycle"}}),
    ({"$ref": "#/$defs/Name", "type": "object"}, {"Name": {"type": "string"}}),
    ({"type": "string", "new_constraint": True}, {}),
])
def test_unreviewed_schema_features_fail_before_network(schema, defs):
    with pytest.raises(ValueError, match="LOCALIZATION_API_SCHEMA"):
        _compact_api_schema(schema, defs)


def test_local_limits_are_still_present_in_pydantic_schema():
    schema = PageLocalizationProposal.model_json_schema()
    assert schema["properties"]["regions"]["maxItems"] == 300
    assert schema["properties"]["elements"]["maxItems"] == 150
    assert schema["$defs"]["LocalizationRegion"]["properties"]["region_id"]["pattern"]
    assert (
        schema["$defs"]["LocalizationRegion"]["properties"]["transcription"]["anyOf"][0][
            "maxLength"
        ]
        == 4000
    )


@pytest.mark.parametrize("violation", [
    "id_pattern", "long_text", "long_reference", "too_many_regions", "too_many_elements",
    "too_many_links", "dangling_link", "duplicate_id", "wrong_link_kind",
    "out_of_range", "inverted_box", "wrong_arity", "wrong_job", "unknown_field",
    "rotation_not_allowed", "empty_full_scan", "empty_basis",
])
def test_invalid_proposals_are_still_rejected_locally(violation):
    data = valid_proposal()
    if violation == "id_pattern":
        data["regions"][0]["region_id"] = "invalid-id"
    elif violation == "long_text":
        data["regions"][0]["transcription"] = "x" * 4001
    elif violation == "long_reference":
        data["elements"][0]["reference_raw"] = "x" * 201
    elif violation == "too_many_regions":
        data["regions"] = [dict(data["regions"][0], region_id=f"r{i + 1}") for i in range(301)]
    elif violation == "too_many_elements":
        data["elements"] = [dict(data["elements"][0], candidate_id=f"e{i + 1}") for i in range(151)]
    elif violation == "too_many_links":
        data["elements"][0]["reference_region_ids"] = ["r1"] * 51
    elif violation == "dangling_link":
        data["elements"][0]["drawing_region_ids"] = ["r99"]
    elif violation == "duplicate_id":
        data["regions"][1]["region_id"] = "r1"
    elif violation == "wrong_link_kind":
        data["elements"][0]["drawing_region_ids"] = ["r1"]
    elif violation == "out_of_range":
        data["regions"][0]["box_2d"] = [10, 20, 2000, 60]
    elif violation == "inverted_box":
        data["regions"][0]["box_2d"] = [50, 20, 30, 60]
    elif violation == "wrong_arity":
        data["regions"][0]["box_2d"] = [10, 20, 30]
    elif violation == "wrong_job":
        data["job_id"] = "another-job"
    elif violation == "unknown_field":
        data["unexpected"] = True
    elif violation == "rotation_not_allowed":
        data["suggested_rotation_clockwise"] = 45
    elif violation == "empty_full_scan":
        data["elements"] = []
    elif violation == "empty_basis":
        data["elements"][0]["association_basis"] = ""
    with pytest.raises((ValidationError, LocalizationError)):
        parse_proposal(json.dumps(data), "synthetic-job")


def test_valid_proposal_is_not_changed_or_promoted_to_approved():
    data = valid_proposal()
    before = copy.deepcopy(data)
    result = parse_proposal(json.dumps(data), data["job_id"])
    assert result.model_dump() == before
    assert result.elements[0].association == "PROPOSED"
    assert data == before


def test_schema_metadata_identifies_generation_separately_from_validation():
    meta = localization_api_schema_metadata()
    encoded = json.dumps(build_localization_api_schema(), ensure_ascii=False, sort_keys=True,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    assert meta["version"] == "page-localization-api-v1.1"
    assert meta["api_schema_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert meta["api_schema_sha256"] != meta["local_validation_schema_sha256"]
    assert "PageLocalizationProposal" in meta["local_validator"]


def test_schema_is_recorded_before_a_failure_without_an_extra_call(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = {
        "preparation_report_path": str(tmp_path / "prepared" / "report.json"),
        "preparation_report_sha256": "a" * 64, "source_archive_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "response_schema_sha256": "d" * 64,
        "jobs": [{"job_id": "synthetic-job", "project_id": "synthetic",
                  "corpus_document_id": "TEST", "page_number": 1, "view_index": 1}],
    }
    monkeypatch.setattr(runner, "load_verified_plan", lambda _: plan)
    monkeypatch.setattr(runner, "request_for_job", lambda *_: ("prompt", [], {}))
    calls = []
    def fail(*_):
        calls.append(1)
        raise RuntimeError("not written to the report")
    client = SimpleNamespace(model="configured-model", generate=fail, close=lambda: None)
    report, output = runner.execute_plan(plan_path, allow_paid_calls=True, max_calls=1,
                                         out=tmp_path / "run", client_factory=lambda: client)
    assert calls == [1]
    assert report["network_calls_attempted"] == 1
    assert report["api_schema"] == localization_api_schema_metadata()
    assert json.loads(
        (output / "api_response_schema.json").read_text()
    ) == build_localization_api_schema()
    assert report["responses_received"] == 0
    assert report["corpus_approved"] is False


def test_model_prompt_images_and_options_unchanged_except_generation_schema():
    calls = []
    class NoAPIError(Exception):
        pass
    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="{}", candidates=[SimpleNamespace(finish_reason="STOP")],
                               model_version="mock", usage_metadata=None)
    provider = GeminiLocalizationClient.__new__(GeminiLocalizationClient)
    provider.model = "unchanged-model"
    provider._api_key_for_redaction = "not-a-real-key"
    provider._api_error_type = NoAPIError
    provider._types = SimpleNamespace(
        Part=SimpleNamespace(from_text=lambda **kw: kw, from_bytes=lambda **kw: kw),
        GenerateContentConfig=lambda **kw: kw,
        AutomaticFunctionCallingConfig=lambda **kw: kw,
    )
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    images = [("PAGE", "image/png", b"original-bytes")]
    result = provider.generate("original-prompt", images)
    assert len(calls) == 1
    assert calls[0]["contents"] == [
        {"text": "PAGE"},
        {"data": b"original-bytes", "mime_type": "image/png"},
        {"text": "original-prompt"},
    ]
    config = calls[0]["config"]
    assert calls[0]["model"] == "unchanged-model"
    assert config["response_json_schema"] == build_localization_api_schema()
    assert config["response_mime_type"] == "application/json"
    assert config["temperature"] == 0 and config["max_output_tokens"] == 16384
    assert config["automatic_function_calling"] == {"disable": True}
    assert result["api_schema"] == localization_api_schema_metadata()
