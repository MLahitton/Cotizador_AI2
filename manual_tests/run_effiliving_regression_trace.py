import argparse
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.providers.gemini_extraction import (  # noqa: E402
    GeminiEnrichmentParseError,
    GeminiExtractionProvider,
    GeminiFullPipelineDebugCapture,
)
from app.services.pipeline_trace import build_requirement_pipeline_trace  # noqa: E402

DEFAULT_INPUT_PATH = ROOT_DIR / "manual_tests" / "effiliving.pdf"
RESULTS_DIR = ROOT_DIR / "manual_tests" / "results"
REPORT_PATH = RESULTS_DIR / "effiliving_regression_trace.json"
ENRICHMENT_PARSE_ERROR_PATH = RESULTS_DIR / "effiliving_enrichment_parse_error.json"
DEFAULT_REFERENCES = tuple(f"V-{index:02d}" for index in range(1, 13))
EXPECTED_QUANTITIES = {
    "V-01": 1,
    "V-02": 1,
    "V-03": 1,
    "V-04": 1,
    "V-05": 1,
    "V-06": 1,
    "V-07": 1,
    "V-08": 5,
    "V-09": 5,
    "V-10": 25,
    "V-11": 1,
    "V-12": 5,
}
EXPECTED_TOTAL_QUANTITY = 48
NUMERIC_ROLES = {
    "QUANTITY",
    "LEVEL",
    "LEVEL_RANGE",
    "REPETITION_COUNT",
    "COMPONENT_COUNT",
    "PANEL_COUNT",
    "WIDTH",
    "HEIGHT",
}


def main() -> int:
    args = _parse_args()
    started = perf_counter()
    provider: GeminiExtractionProvider | None = None

    try:
        provider = GeminiExtractionProvider()
        debug_capture = GeminiFullPipelineDebugCapture()
        extraction = provider.extract_with_discovery_from_files(
            [args.input],
            project_id=args.project_id,
            requirement_id=args.requirement_id,
            debug_capture=debug_capture,
        )
        elapsed_seconds = perf_counter() - started
        report = _build_report(
            extraction,
            debug_capture,
            references=tuple(args.reference),
            elapsed_seconds=elapsed_seconds,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _print_summary(report)
        return 0
    except GeminiEnrichmentParseError as exc:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ENRICHMENT_PARSE_ERROR_PATH.write_text(
            json.dumps(
                _enrichment_parse_error_report(exc, provider),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"diagnostic_report: {ENRICHMENT_PARSE_ERROR_PATH}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AI2 full pipeline and capture quantity regression trace."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="PDF/XLSX/image path to process.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=list(DEFAULT_REFERENCES),
        help="Reference to include in the trace. Can be repeated.",
    )
    parser.add_argument("--project-id", default="manual-effiliving")
    parser.add_argument("--requirement-id", default="manual-effiliving-regression-001")
    return parser.parse_args()


def _build_report(
    extraction,
    debug_capture: GeminiFullPipelineDebugCapture,
    *,
    references: tuple[str, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    source_names_by_id = {
        source.id: source.file_name
        for source in extraction.sources
    }
    return {
        "title": "EFFILIVING QUANTITY REGRESSION TRACE",
        "elapsed_seconds": round(elapsed_seconds, 2),
        "model": debug_capture.model,
        "references": list(references),
        "stage_counts": _stage_counts(debug_capture),
        "stage_table": {
            reference: _stage_rows(debug_capture, extraction, reference, source_names_by_id)
            for reference in references
        },
        "quantity_table": _quantity_table(debug_capture, extraction, references),
        "total_final_quantity": _total_final_quantity(extraction, references),
        "expected_total_quantity": EXPECTED_TOTAL_QUANTITY,
        "numeric_trace": _numeric_trace(debug_capture, references),
        "quantity_grounding_decisions": _quantity_grounding_decisions(
            debug_capture,
            references,
        ),
        "semantic_review_decisions": _semantic_review_decisions(
            debug_capture,
            references,
        ),
        "reconciliation_decisions": _reconciliation_decisions(debug_capture, references),
        "evidence_fidelity": _evidence_fidelity(debug_capture, extraction, references),
        "evidence_fidelity_warnings": _evidence_fidelity_warnings(
            debug_capture,
            extraction,
            references,
        ),
        "raw_enrichment_batches": _raw_batches(debug_capture, references),
        "pipeline_trace": build_requirement_pipeline_trace(
            entrypoint="REGRESSION_RUNNER",
            processing_attempt_id="manual-effiliving-regression",
            debug_capture=debug_capture,
            extraction=extraction,
        ),
        "conclusion": _quantity_conclusion(debug_capture, extraction, references),
    }


def _stage_counts(debug_capture: GeminiFullPipelineDebugCapture) -> dict[str, int]:
    if debug_capture.inventory_trace is None:
        return {}
    return {
        stage.stage: stage.count
        for stage in debug_capture.inventory_trace.stages
    }


def _stage_rows(
    debug_capture: GeminiFullPipelineDebugCapture,
    extraction,
    reference: str,
    source_names_by_id: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "RAW_ENRICHMENT": _raw_enrichment_rows(debug_capture, reference),
        "POST_GROUNDING": _post_grounding_rows(
            debug_capture,
            reference,
            source_names_by_id,
        ),
    }
    if debug_capture.inventory_trace is not None:
        for stage in debug_capture.inventory_trace.stages:
            rows[stage.stage] = [
                _inventory_element_row(element, source_names_by_id)
                for element in stage.elements
                if _same_reference(element.reference, reference)
            ]
    rows["FINAL_REQUIREMENT_EXTRACTION_DETAIL"] = [
        _final_element_row(element, extraction, source_names_by_id)
        for element in extraction.elements
        if _same_reference(_final_reference(element), reference)
    ]
    return rows


def _raw_enrichment_rows(
    debug_capture: GeminiFullPipelineDebugCapture,
    reference: str,
) -> list[dict[str, Any]]:
    if debug_capture.enrichment_debug is None:
        return []
    rows = []
    for batch_index, raw_response in enumerate(
        debug_capture.enrichment_debug.raw_responses or [],
        start=1,
    ):
        if not raw_response:
            continue
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            continue
        for element in payload.get("elements", []):
            if not isinstance(element, dict):
                continue
            if not _same_reference(element.get("reference"), reference):
                continue
            row = _safe_dump(element)
            row["batch"] = batch_index
            rows.append(row)
    return rows


def _post_grounding_rows(
    debug_capture: GeminiFullPipelineDebugCapture,
    reference: str,
    source_names_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    if debug_capture.enrichment_debug is None:
        return []
    rows = []
    for batch_index, batch in enumerate(
        debug_capture.enrichment_debug.batch_results or [],
        start=1,
    ):
        for element in batch.elements:
            if not _same_reference(element.reference, reference):
                continue
            row = _enrichment_element_row(element, source_names_by_id)
            row["batch"] = batch_index
            rows.append(row)
    return rows


def _inventory_element_row(element, source_names_by_id: dict[str, str]) -> dict[str, Any]:
    return {
        "temporary_id": element.temporary_id,
        "id": element.id,
        "reference": element.reference,
        "quantity": element.quantity,
        "status": element.status,
        "source_ids": element.source_ids,
        "source_file_names": [
            source_names_by_id.get(source_id)
            for source_id in element.source_ids
        ],
        "dimensions": element.dimensions,
        "description": element.description,
    }


def _enrichment_element_row(element, source_names_by_id: dict[str, str]) -> dict[str, Any]:
    row = _safe_dump(element)
    source_ids = [
        evidence.get("source_id")
        for evidence in row.get("evidence", [])
        if isinstance(evidence, dict) and evidence.get("source_id")
    ]
    row["source_file_names"] = [
        source_names_by_id.get(source_id)
        for source_id in source_ids
        if isinstance(source_id, str)
    ]
    return row


def _final_element_row(element, extraction, source_names_by_id: dict[str, str]) -> dict[str, Any]:
    evidence = [
        item
        for item in extraction.evidence
        if item.id in element.evidence_ids
    ]
    return {
        "id": element.id,
        "reference": _final_reference(element),
        "quantity": _traceable_value(element.quantity),
        "quantity_status": _status_value(element.quantity.status) if element.quantity else None,
        "quantity_confidence": element.quantity.confidence if element.quantity else None,
        "dimensions": [
            {
                "type": measurement.type,
                "value": measurement.value,
                "unit": measurement.unit,
                "raw_label": measurement.raw_label,
            }
            for measurement in element.measurements
        ],
        "geometry": _model_value(element.geometry),
        "functional_type": _model_value(element.functional_type),
        "evidence": [
            _evidence_row(item, source_names_by_id)
            for item in evidence
        ],
    }


def _evidence_row(evidence, source_names_by_id: dict[str, str]) -> dict[str, Any]:
    row = _safe_dump(evidence)
    if not isinstance(row, dict):
        return {"value": row}

    source_id = row.get("source_id")
    row["source_file_name"] = (
        source_names_by_id.get(source_id)
        if isinstance(source_id, str)
        else None
    )
    return row


def _numeric_trace(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "batch_numeric_traces": [
            _filter_numeric_trace(trace, references)
            for trace in (
                debug_capture.enrichment_debug.batch_numeric_traces
                if debug_capture.enrichment_debug is not None
                else []
            )
        ],
        "merged_numeric_trace": _filter_numeric_trace(
            debug_capture.enrichment_debug.merged_numeric_trace
            if debug_capture.enrichment_debug is not None
            else None,
            references,
        ),
        "numeric_trace": _filter_numeric_trace(debug_capture.numeric_trace, references),
    }


def _filter_numeric_trace(trace, references: tuple[str, ...]) -> dict[str, Any] | None:
    if trace is None:
        return None
    payload = trace.model_dump(mode="json")
    payload["elements"] = [
        _filter_numeric_element(element)
        for element in payload["elements"]
        if any(_same_reference(element.get("reference"), reference) for reference in references)
    ]
    return payload


def _filter_numeric_element(element: dict[str, Any]) -> dict[str, Any]:
    element = dict(element)
    element["candidates"] = [
        candidate
        for candidate in element.get("candidates", [])
        if candidate.get("semantic_role") in NUMERIC_ROLES
    ]
    return element


def _quantity_table(
    debug_capture: GeminiFullPipelineDebugCapture,
    extraction,
    references: tuple[str, ...],
) -> list[dict[str, Any]]:
    raw_by_reference = _raw_quantities_by_reference(debug_capture, references)
    grounded_by_reference = _post_grounding_quantities_by_reference(debug_capture, references)
    merged_by_reference = _merged_quantities_by_reference(debug_capture, references)
    final_by_reference = _final_quantities_by_reference(extraction, references)
    grounding_by_reference = {
        _canonical(decision.get("reference")): decision
        for decision in _quantity_grounding_decisions(debug_capture, references)
    }

    rows = []
    for reference in references:
        expected = EXPECTED_QUANTITIES.get(_display_reference(reference))
        final_quantity = _single_or_list(final_by_reference.get(_canonical(reference), []))
        rows.append(
            {
                "reference": reference,
                "raw_quantity": _single_or_list(raw_by_reference.get(_canonical(reference), [])),
                "grounded_quantity": _single_or_list(
                    grounded_by_reference.get(_canonical(reference), [])
                ),
                "merged_quantity": _single_or_list(
                    merged_by_reference.get(_canonical(reference), [])
                ),
                "final_quantity": final_quantity,
                "expected": expected,
                "verdict": _verdict(final_quantity, expected),
                "grounding_decision": grounding_by_reference.get(_canonical(reference)),
            }
        )
    return rows


def _raw_quantities_by_reference(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    if debug_capture.enrichment_debug is None:
        return values
    for raw_response in debug_capture.enrichment_debug.raw_responses or []:
        if not raw_response:
            continue
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            continue
        for element in payload.get("elements", []):
            if not isinstance(element, dict):
                continue
            reference = _canonical(element.get("reference"))
            if not any(reference == _canonical(expected) for expected in references):
                continue
            values.setdefault(reference, []).append(element.get("quantity"))
    return values


def _post_grounding_quantities_by_reference(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    if debug_capture.enrichment_debug is None:
        return values
    for batch in debug_capture.enrichment_debug.batch_results or []:
        for element in batch.elements:
            if not any(_same_reference(element.reference, reference) for reference in references):
                continue
            values.setdefault(_canonical(element.reference), []).append(element.quantity)
    return values


def _merged_quantities_by_reference(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    enrichment = debug_capture.enrichment
    if enrichment is None:
        return values
    for element in enrichment.elements:
        if not any(_same_reference(element.reference, reference) for reference in references):
            continue
        values.setdefault(_canonical(element.reference), []).append(element.quantity)
    return values


def _final_quantities_by_reference(
    extraction,
    references: tuple[str, ...],
) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    for element in extraction.elements:
        reference = _final_reference(element)
        if not any(_same_reference(reference, expected) for expected in references):
            continue
        values.setdefault(_canonical(reference), []).append(_traceable_value(element.quantity))
    return values


def _total_final_quantity(extraction, references: tuple[str, ...]) -> int | float:
    total: int | float = 0
    for quantities in _final_quantities_by_reference(extraction, references).values():
        for quantity in quantities:
            if isinstance(quantity, int | float) and not isinstance(quantity, bool):
                total += quantity
    return total


def _reconciliation_decisions(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> list[dict[str, Any]]:
    decisions = debug_capture.reconciliation_decisions or []
    return [
        _serialize(decision)
        for decision in decisions
        if any(
            _same_reference(decision.reference, reference)
            or _same_reference(decision.normalized_reference, reference)
            for reference in references
        )
    ]


def _quantity_grounding_decisions(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> list[dict[str, Any]]:
    decisions = (
        debug_capture.enrichment_debug.quantity_grounding_decisions
        if debug_capture.enrichment_debug is not None
        else []
    )
    return [
        _serialize(decision)
        for decision in decisions
        if any(_same_reference(decision.reference, reference) for reference in references)
    ]


def _semantic_review_decisions(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> list[dict[str, Any]]:
    decisions = (
        getattr(debug_capture.enrichment_debug, "semantic_review_decisions", [])
        if debug_capture.enrichment_debug is not None
        else []
    )
    return [
        _serialize(decision)
        for decision in decisions
        if any(_same_reference(decision.reference, reference) for reference in references)
    ]


def _evidence_fidelity(
    debug_capture: GeminiFullPipelineDebugCapture,
    extraction,
    references: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    fidelity = {}
    for reference in references:
        raw_batch_hits = _raw_batches(debug_capture, (reference,))
        final_rows = [
            _final_element_row(element, extraction, {})
            for element in extraction.elements
            if _same_reference(_final_reference(element), reference)
        ]
        fidelity[reference] = {
            "raw_batch_hits": raw_batch_hits,
            "final_evidence_texts": [
                _evidence_text(evidence)
                for row in final_rows
                for evidence in row["evidence"]
                if _evidence_text(evidence)
            ],
        }
    return fidelity


def _evidence_fidelity_warnings(
    debug_capture: GeminiFullPipelineDebugCapture,
    extraction,
    references: tuple[str, ...],
) -> list[dict[str, Any]]:
    warnings = []
    for reference in references:
        for stage, rows in _stage_rows(debug_capture, extraction, reference, {}).items():
            for row in rows:
                structured_dimensions = row.get("dimensions") or []
                evidence_dimensions = [
                    dimension
                    for evidence in row.get("evidence", [])
                    for dimension in _dimensions_from_text(_evidence_text(evidence))
                ]
                if not evidence_dimensions:
                    continue
                if not _dimensions_overlap(structured_dimensions, evidence_dimensions):
                    warnings.append(
                        {
                            "code": "EVIDENCE_FIDELITY_WARNING",
                            "reference": reference,
                            "stage": stage,
                            "structured_dimensions": structured_dimensions,
                            "evidence_dimensions": evidence_dimensions,
                        }
                    )
    return warnings


def _raw_batches(
    debug_capture: GeminiFullPipelineDebugCapture,
    references: tuple[str, ...],
) -> list[dict[str, Any]]:
    if debug_capture.enrichment_debug is None:
        return []
    hits = []
    raw_responses = debug_capture.enrichment_debug.raw_responses or []
    for index, raw_response in enumerate(raw_responses, start=1):
        if raw_response is None:
            continue
        for reference in references:
            if reference.casefold() in raw_response.casefold():
                hits.append(
                    {
                        "batch": index,
                        "reference": reference,
                        "raw_response": raw_response,
                    }
                )
    return hits


def _print_summary(report: dict[str, Any]) -> None:
    print("EFFILIVING QUANTITY REGRESSION TRACE")
    print(f"model: {report['model']}")
    print(f"elapsed_seconds: {report['elapsed_seconds']}")
    print("\nSTAGE COUNTS")
    for stage, count in report["stage_counts"].items():
        print(f"{stage}: {count}")
    print("\nREFERENCE FINAL QUANTITIES")
    print("reference | raw_quantity | grounded_quantity | final_quantity | expected | verdict")
    for row in report["quantity_table"]:
        print(
            f"{row['reference']} | {row['raw_quantity']} | "
            f"{row['grounded_quantity']} | {row['final_quantity']} | "
            f"{row['expected']} | {row['verdict']}"
        )
    print(f"total_final_quantity: {report['total_final_quantity']}")
    print(f"expected_total_quantity: {report['expected_total_quantity']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"evidence_fidelity_warnings: {len(report['evidence_fidelity_warnings'])}")
    print(f"\ntrace_report: {REPORT_PATH}")


def _enrichment_parse_error_report(
    exc: GeminiEnrichmentParseError,
    provider: GeminiExtractionProvider | None,
) -> dict[str, Any]:
    diagnostic = exc.diagnostic.to_dict()
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": provider._provider.model if provider is not None else None,
        "stage": diagnostic["stage"],
        "batch": diagnostic["batch_number"],
        "requested_temporary_ids": diagnostic["requested_temporary_ids"],
        "shape": diagnostic["response_top_level_shape"],
        "validation_errors": diagnostic["validation_error_summary"],
        "safe_prefix": diagnostic["raw_text_prefix"],
        "safe_suffix": diagnostic["raw_text_suffix"],
        "raw_text_length": diagnostic["raw_text_length"],
        "response_python_type": diagnostic["response_python_type"],
    }


def _same_reference(value: str | None, expected: str) -> bool:
    return _canonical(value) == _canonical(expected)


def _canonical(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().casefold().replace("_", "-")


def _display_reference(value: str) -> str:
    return value.strip().upper().replace("_", "-")


def _final_reference(element) -> str | None:
    if element.reference is None:
        return None
    return element.reference.value


def _traceable_value(value) -> Any:
    if value is None:
        return None
    return value.value


def _status_value(value) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _model_value(value) -> Any:
    return _safe_dump(value)


def _serialize(value) -> Any:
    return _safe_dump(value)


def _safe_dump(value) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return {
            key: _safe_dump(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_safe_dump(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _safe_dump(item)
            for key, item in value.items()
        }
    return str(value)


def _single_or_list(values: list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _verdict(final_quantity: Any, expected: Any) -> str:
    if expected is None:
        return "NO_EXPECTATION"
    if final_quantity == expected:
        return "OK"
    return "MISMATCH"


def _quantity_conclusion(
    debug_capture: GeminiFullPipelineDebugCapture,
    extraction,
    references: tuple[str, ...],
) -> str:
    table = _quantity_table(debug_capture, extraction, references)
    mismatches = [row["reference"] for row in table if row["verdict"] == "MISMATCH"]
    total = _total_final_quantity(extraction, references)
    if not mismatches and total == EXPECTED_TOTAL_QUANTITY:
        return "ALL_V01_V12_FINAL_QUANTITIES_MATCH_EXPECTED_TOTAL"
    if not mismatches:
        return "V01_V12_MATCH_BUT_TOTAL_DIFFERS"
    corrected = [
        row["reference"]
        for row in table
        if row["raw_quantity"] != row["expected"]
        and row["final_quantity"] == row["expected"]
    ]
    if corrected:
        return "SOME_RAW_WRONG_CORRECTED_BUT_OTHERS_STILL_MISMATCH"
    return "RAW_WRONG_REACHES_FINAL_OR_REFERENCE_MISSING"


def _dimensions_from_text(text: str | None) -> list[dict[str, float]]:
    if not text:
        return []
    dimensions = []
    for match in re.finditer(
        r"\b(\d+(?:[.,]\d+)?)\s*(?:x|×)\s*(\d+(?:[.,]\d+)?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        dimensions.append(
            {
                "width": _parse_float(match.group(1)),
                "height": _parse_float(match.group(2)),
            }
        )
    return dimensions


def _dimensions_overlap(
    structured_dimensions: list[Any],
    evidence_dimensions: list[dict[str, float]],
) -> bool:
    structured_values = [
        item.get("value")
        for item in structured_dimensions
        if isinstance(item, dict)
        and item.get("type") in {"width", "height"}
        and isinstance(item.get("value"), int | float)
    ]
    if len(structured_values) < 2:
        return True
    for evidence in evidence_dimensions:
        evidence_values = [evidence["width"], evidence["height"]]
        if all(
            any(abs(float(source) - float(target)) < 0.01 for target in evidence_values)
            for source in structured_values[:2]
        ):
            return True
    return False


def _parse_float(value: str) -> float:
    return float(value.replace(",", "."))


def _evidence_text(evidence: dict[str, Any]) -> str | None:
    for key in ("extracted_text", "text", "visual_description", "notes"):
        value = evidence.get(key)
        if isinstance(value, str) and value:
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
