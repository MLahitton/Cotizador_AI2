"""Pruebas del evaluador offline; NO pruebas de exactitud de Gemini."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "manual_tests" / "corpus_v1" / "runner.py"
SPEC = importlib.util.spec_from_file_location("corpus_v1_runner_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def snapshot(value=1.5, unit="m", **extra):
    return {"elements": [{"reference": "V-01", "measurements": [
        {"type": "height", "value": value, "unit": unit, "status": "explicit"},
    ], **extra}]}


def specification(kind="measurement", field="height", expected=1.5):
    check = {"check_id": "C1", "kind": kind, "field": field, "expected": expected}
    if kind == "measurement":
        check["unit"] = "m"
    return {"projects": [{"project_id": "p", "elements": [{
        "reference": "V-01", "document_id": "D1", "acceptance_case_id": "A1",
        "checks": [check],
    }]}]}


@pytest.mark.parametrize(("value", "unit"), [(1.5, "m"), (150, "cm"), (1500, "mm"), ("1,50", "m")])
def test_equivalent_units_match(value, unit):
    result = runner.evaluate_snapshot(snapshot(value, unit), specification(), "p")
    assert result["matching_values"] == 1
    assert result["corpus_approved"] is False
    assert result["annotation_complete"] is False


@pytest.mark.parametrize(("value", "unit", "state"), [
    (2.4, "m", "DIFFERENT"), (1.5, None, "UNCOMPARABLE"),
    (1.5, "unknown", "UNCOMPARABLE"), ("ilegible", "m", "UNCOMPARABLE"),
    (True, "m", "UNCOMPARABLE"), (float("nan"), "m", "UNCOMPARABLE"),
    (float("inf"), "m", "UNCOMPARABLE"),
])
def test_invalid_or_different_measurement_is_not_a_match(value, unit, state):
    result = runner.evaluate_snapshot(snapshot(value, unit), specification(), "p")
    assert result["checks"][0]["state"] == state
    assert result["matching_values"] == 0


def test_missing_element_is_not_skipped():
    result = runner.evaluate_snapshot({"elements": []}, specification(), "p")
    assert result["checks"][0]["state"] == "MISSING"


def test_missing_measurement_is_not_skipped():
    data = snapshot()
    data["elements"][0]["measurements"] = []
    result = runner.evaluate_snapshot(data, specification(), "p")
    assert result["checks"][0]["state"] == "MISSING"


def test_duplicate_identity_is_not_silently_deduplicated():
    data = snapshot()
    other = copy.deepcopy(data["elements"][0])
    other["reference"] = "V1"
    data["elements"].append(other)
    result = runner.evaluate_snapshot(data, specification(), "p")
    assert result["checks"][0]["state"] == "IDENTITY_AMBIGUOUS"


def test_conflicting_measurement_is_not_hidden():
    data = snapshot()
    data["elements"][0]["measurements"].append({"type": "height", "value": 2.4, "unit": "m"})
    result = runner.evaluate_snapshot(data, specification(), "p")
    assert result["checks"][0]["state"] == "DIFFERENT"


def test_repeated_equal_measurements_can_match():
    data = snapshot()
    data["elements"][0]["measurements"].append({"type": "height", "value": 1500, "unit": "mm"})
    result = runner.evaluate_snapshot(data, specification(), "p")
    assert result["checks"][0]["state"] == "MATCH"


def test_input_not_mutated():
    data, spec = snapshot(), specification()
    before_data, before_spec = copy.deepcopy(data), copy.deepcopy(spec)
    runner.evaluate_snapshot(data, spec, "p")
    assert data == before_data
    assert spec == before_spec


def test_quantity_does_not_use_panel_count():
    result = runner.evaluate_snapshot(
        snapshot(quantity=None, panel_count=2), specification("number", "quantity", 2), "p",
    )
    assert result["checks"][0]["state"] == "MISSING"


@pytest.mark.parametrize("stage,field", [("enrichment", "operation_raw"), ("pre-mapper", "operation")])
def test_operation_schema_adapter(stage, field):
    result = runner.evaluate_snapshot(
        snapshot(**{field: "BATIENTE"}),
        specification("text_one_of", "operation", ["batiente", "swing"]), "p", stage,
    )
    assert result["checks"][0]["state"] == "MATCH"


def test_reference_normalization_does_not_merge_categories():
    assert runner.reference_key("V01") == runner.reference_key("V-1")
    assert runner.reference_key("PV01") != runner.reference_key("V01")


def test_inventory_detects_extra_missing_and_duplicate_references():
    spec = specification()
    spec["projects"][0]["expected_references"] = ["V-01", "V-02"]
    result = runner.evaluate_snapshot(snapshot(), spec, "p")
    assert result["checks"][0]["state"] == "DIFFERENT"


def test_unannotated_project_is_never_green():
    result = runner.evaluate_snapshot(snapshot(), {"projects": [{"project_id": "p", "elements": []}]}, "p")
    assert result["status"] == "NOT_ANNOTATED"
    assert result["checked_values"] == 0
    assert result["corpus_approved"] is False


@pytest.mark.parametrize("bad", [[], {}, {"elements": [{"reference": "V1"}]}, {
    "elements": [{"reference": {"value": "V1"}, "measurements": []}],
}])
def test_unsupported_snapshot_rejected(bad):
    with pytest.raises(runner.InputError):
        runner.evaluate_snapshot(bad, specification(), "p")


def make_archive(tmp_path, payload=b"source", expected=b"source", extra=False):
    archive_path = tmp_path / "input.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Requerimientos/Test/a.pdf", payload)
        if extra:
            archive.writestr("Requerimientos/Test/b.pdf", b"extra")
    manifest = {"projects": [{"project_id": "p"}], "documents": [{
        "document_id": "D1", "archive_path": "Requerimientos/Test/a.pdf",
        "size_bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest(),
    }]}
    return archive_path, manifest


def test_archive_integrity_verified_without_extracting(tmp_path):
    path, manifest = make_archive(tmp_path)
    before = set(tmp_path.iterdir())
    result = runner.verify_archive(path, manifest)
    assert result["status"] == "INPUTS_MATCH"
    assert set(tmp_path.iterdir()) == before


def test_archive_hash_mismatch(tmp_path):
    path, manifest = make_archive(tmp_path, payload=b"badval")
    assert runner.verify_archive(path, manifest)["files"][0]["state"] == "HASH_MISMATCH"


def test_archive_size_mismatch(tmp_path):
    path, manifest = make_archive(tmp_path, payload=b"bad")
    assert runner.verify_archive(path, manifest)["files"][0]["state"] == "SIZE_MISMATCH"


def test_archive_extra_file_not_ignored(tmp_path):
    path, manifest = make_archive(tmp_path, extra=True)
    assert runner.verify_archive(path, manifest)["status"] == "INPUTS_DIFFER"


def test_unsafe_zip_path_rejected(tmp_path):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(runner.InputError):
        runner.verify_archive(path, {"projects": [], "documents": []})


def test_manifest_matrix_and_seed_consistent():
    base = MODULE_PATH.parent
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    matrix = json.loads((base / "acceptance_matrix.json").read_text(encoding="utf-8"))
    seed = json.loads((base / "expected_checks.json").read_text(encoding="utf-8"))
    assert len(manifest["documents"]) == 17
    assert len(manifest["projects"]) == 9
    assert sum(d.get("pdf_pages") or 0 for d in manifest["documents"]) == 40
    assert len(matrix["cases"]) == 33
    doc_ids = {d["document_id"] for d in manifest["documents"]}
    acceptance_ids = {c["id"] for c in matrix["cases"]}
    covered = set()
    for case in matrix["cases"]:
        ids = set(case["document_id"].split("/"))
        assert ids <= doc_ids
        covered.update(d["project_id"] for d in manifest["documents"] if d["document_id"] in ids)
    assert len(covered) == 9
    for project in seed["projects"]:
        assert project["annotation_complete"] is False
        for element in project["elements"]:
            assert element["document_id"] in doc_ids
            assert element["acceptance_case_id"] in acceptance_ids


def test_cli_strict_returns_nonzero_for_wrong_values(tmp_path, monkeypatch):
    spec = specification()
    spec_path = tmp_path / "expected_checks.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    data_path = tmp_path / "snapshot.json"
    data_path.write_text(json.dumps(snapshot(2.4)), encoding="utf-8")
    monkeypatch.setattr(runner, "BASE", tmp_path)
    assert runner.main([
        "evaluate", "--snapshot", str(data_path), "--project", "p",
        "--strict", "--out", str(tmp_path / "report.json"),
    ]) == 1


def test_cli_does_not_overwrite_input(tmp_path, monkeypatch):
    (tmp_path / "expected_checks.json").write_text(json.dumps(specification()), encoding="utf-8")
    data_path = tmp_path / "snapshot.json"
    payload = json.dumps(snapshot())
    data_path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(runner, "BASE", tmp_path)
    assert runner.main([
        "evaluate", "--snapshot", str(data_path), "--project", "p", "--out", str(data_path),
    ]) == 2
    assert data_path.read_text(encoding="utf-8") == payload
