import json
from types import SimpleNamespace

from app.models.evidence import Evidence
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentMeasurement,
    GeminiEnrichmentResult,
)
from app.providers.gemini_extraction import (
    GeminiEnrichmentParseDiagnostic,
    GeminiEnrichmentParseError,
)
from manual_tests import run_effiliving_regression_trace as runner
from manual_tests.run_effiliving_regression_trace import (
    DEFAULT_REFERENCES,
    EXPECTED_TOTAL_QUANTITY,
    _build_report,
    _final_element_row,
)


def test_effiliving_regression_trace_serializes_final_evidence_defensively() -> None:
    evidence = Evidence(
        id="ev-1",
        source_id="source-1",
        type="table",
        extracted_text="V-10 CANTIDAD: 25",
        page_number=3,
    )
    element = SimpleNamespace(
        id="element-1",
        reference=SimpleNamespace(value="V-10"),
        quantity=None,
        evidence_ids=["ev-1"],
        measurements=[],
        geometry=None,
        functional_type=None,
    )
    extraction = SimpleNamespace(evidence=[evidence])

    row = _final_element_row(
        element,
        extraction,
        {"source-1": "EFFILIVING.pdf"},
    )

    assert row["evidence"] == [
        {
            "id": "ev-1",
            "source_id": "source-1",
            "type": "table",
            "page_number": 3,
            "sheet_name": None,
            "cell_range": None,
            "region": None,
            "extracted_text": "V-10 CANTIDAD: 25",
            "visual_description": None,
            "status": "explicit",
            "confidence": None,
            "notes": None,
            "source_file_name": "EFFILIVING.pdf",
        }
    ]


def test_effiliving_regression_trace_defaults_to_twelve_references() -> None:
    assert DEFAULT_REFERENCES == (
        "V-01",
        "V-02",
        "V-03",
        "V-04",
        "V-05",
        "V-06",
        "V-07",
        "V-08",
        "V-09",
        "V-10",
        "V-11",
        "V-12",
    )


def test_effiliving_regression_trace_reports_raw_grounded_final_and_total() -> None:
    extraction = SimpleNamespace(
        sources=[],
        evidence=[],
        elements=[
            _final_element("V-01", 1),
            _final_element("V-02", 1),
            _final_element("V-03", 1),
            _final_element("V-04", 1),
            _final_element("V-05", 1),
            _final_element("V-06", 1),
            _final_element("V-07", 1),
            _final_element("V-08", 5),
            _final_element("V-09", 5),
            _final_element("V-10", 25),
            _final_element("V-11", 1),
            _final_element("V-12", 5),
        ],
    )
    debug_capture = SimpleNamespace(
        model="gemini-test",
        inventory_trace=None,
        numeric_trace=None,
        reconciliation_decisions=[],
        enrichment=GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="v-10",
                    reference="V-10",
                    quantity=25,
                )
            ]
        ),
        enrichment_debug=SimpleNamespace(
            raw_responses=[
                '{"elements": ['
                '{"temporary_id": "v-10", "reference": "V-10", "quantity": 5}'
                "]}"
            ],
            batch_results=[
                GeminiEnrichmentResult(
                    elements=[
                        GeminiElementEnrichment(
                            temporary_id="v-10",
                            reference="V-10",
                            quantity=25,
                        )
                    ]
                )
            ],
            batch_numeric_traces=[],
            merged_numeric_trace=None,
            quantity_grounding_decisions=[],
        ),
    )

    report = _build_report(
        extraction,
        debug_capture,
        references=DEFAULT_REFERENCES,
        elapsed_seconds=1.25,
    )
    v10 = next(row for row in report["quantity_table"] if row["reference"] == "V-10")

    assert len(report["quantity_table"]) == 12
    assert v10["raw_quantity"] == 5
    assert v10["grounded_quantity"] == 25
    assert v10["final_quantity"] == 25
    assert v10["verdict"] == "OK"
    assert report["total_final_quantity"] == EXPECTED_TOTAL_QUANTITY
    assert report["conclusion"] == "ALL_V01_V12_FINAL_QUANTITIES_MATCH_EXPECTED_TOTAL"
    assert report["stage_table"]["V-10"]["RAW_ENRICHMENT"][0]["quantity"] == 5
    assert report["stage_table"]["V-10"]["POST_GROUNDING"][0]["quantity"] == 25


def test_effiliving_regression_trace_missing_reference_does_not_crash() -> None:
    extraction = SimpleNamespace(sources=[], evidence=[], elements=[])
    debug_capture = SimpleNamespace(
        model="gemini-test",
        inventory_trace=None,
        numeric_trace=None,
        reconciliation_decisions=[],
        enrichment=None,
        enrichment_debug=None,
    )

    report = _build_report(
        extraction,
        debug_capture,
        references=("V-01",),
        elapsed_seconds=1.0,
    )

    assert report["quantity_table"][0]["final_quantity"] is None
    assert report["quantity_table"][0]["verdict"] == "MISMATCH"


def test_effiliving_regression_trace_reports_evidence_fidelity_warning() -> None:
    evidence = Evidence(
        id="ev-1",
        source_id="source-1",
        type="table",
        extracted_text="V-10 5.58 x 5.90",
    )
    extraction = SimpleNamespace(
        sources=[],
        evidence=[evidence],
        elements=[
            SimpleNamespace(
                id="element-1",
                reference=SimpleNamespace(value="V-10"),
                quantity=SimpleNamespace(value=25, status=None, confidence=None),
                evidence_ids=["ev-1"],
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=2.8, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.9, unit="m"),
                ],
                geometry=None,
                functional_type=None,
            )
        ],
    )
    debug_capture = SimpleNamespace(
        model="gemini-test",
        inventory_trace=None,
        numeric_trace=None,
        reconciliation_decisions=[],
        enrichment=None,
        enrichment_debug=None,
    )

    report = _build_report(
        extraction,
        debug_capture,
        references=("V-10",),
        elapsed_seconds=1.0,
    )

    assert report["evidence_fidelity_warnings"] == [
        {
            "code": "EVIDENCE_FIDELITY_WARNING",
            "reference": "V-10",
            "stage": "FINAL_REQUIREMENT_EXTRACTION_DETAIL",
            "structured_dimensions": [
                {"type": "width", "value": 2.8, "unit": "m", "raw_label": None},
                {"type": "height", "value": 2.9, "unit": "m", "raw_label": None},
            ],
            "evidence_dimensions": [{"width": 5.58, "height": 5.9}],
        }
    ]


def test_effiliving_regression_trace_writes_enrichment_parse_error_report(
    tmp_path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "effiliving_enrichment_parse_error.json"
    diagnostic = GeminiEnrichmentParseDiagnostic(
        stage="ENRICHMENT",
        batch_number=3,
        requested_temporary_ids=["elem-v-10"],
        response_python_type="list",
        response_top_level_shape="list[len=2]",
        validation_error_summary=[{"loc": ["elements"], "type": "list_type"}],
        raw_text_length=22,
        raw_text_prefix="[REDACTED]",
        raw_text_suffix="[REDACTED]",
    )

    class FakeInnerProvider:
        model = "gemini-test"

    class FakeProvider:
        _provider = FakeInnerProvider()

        def extract_with_discovery_from_files(self, *args, **kwargs):
            raise GeminiEnrichmentParseError("bad enrichment", diagnostic)

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "ENRICHMENT_PARSE_ERROR_PATH", report_path)
    monkeypatch.setattr(runner, "GeminiExtractionProvider", FakeProvider)
    monkeypatch.setattr(runner.sys, "argv", ["run_effiliving_regression_trace.py"])

    exit_code = runner.main()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["model"] == "gemini-test"
    assert payload["stage"] == "ENRICHMENT"
    assert payload["batch"] == 3
    assert payload["requested_temporary_ids"] == ["elem-v-10"]
    assert payload["shape"] == "list[len=2]"
    assert payload["validation_errors"] == [{"loc": ["elements"], "type": "list_type"}]


def _final_element(reference: str, quantity: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"element-{reference}",
        reference=SimpleNamespace(value=reference),
        quantity=SimpleNamespace(value=quantity, status=None, confidence=None),
        evidence_ids=[],
        measurements=[],
        geometry=None,
        functional_type=None,
    )
