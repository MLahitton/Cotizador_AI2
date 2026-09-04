from __future__ import annotations

import json
from typing import Any


def build_requirement_pipeline_trace(
    *,
    entrypoint: str,
    processing_attempt_id: str,
    debug_capture: Any,
    extraction: Any,
) -> list[dict[str, Any]]:
    references = _ordered_references(debug_capture, extraction)
    rows: list[dict[str, Any]] = []
    raw_quantities = _raw_enrichment_quantities(debug_capture)
    post_grounding_quantities = _numeric_trace_quantities(
        getattr(getattr(debug_capture, "enrichment_debug", None), "batch_numeric_traces", [])
    )
    post_review_quantities = _enrichment_debug_quantities(
        getattr(getattr(debug_capture, "enrichment_debug", None), "batch_results", [])
    )
    post_reconciliation_quantities = _reconciliation_quantities(debug_capture)
    final_quantities = _final_extraction_quantities(extraction)
    review_by_reference = _semantic_review_by_reference(debug_capture)

    for reference in references:
        rows.append(
            _row(
                entrypoint,
                processing_attempt_id,
                reference,
                "DISCOVERY",
                None,
            )
        )
        rows.append(
            _row(
                entrypoint,
                processing_attempt_id,
                reference,
                "RAW_ENRICHMENT",
                _single_or_list(raw_quantities.get(_canonical(reference), [])),
            )
        )
        rows.append(
            _row(
                entrypoint,
                processing_attempt_id,
                reference,
                "POST_GROUNDING",
                _single_or_list(post_grounding_quantities.get(_canonical(reference), [])),
            )
        )
        review = review_by_reference.get(_canonical(reference), {})
        rows.append(
            _row(
                entrypoint,
                processing_attempt_id,
                reference,
                "POST_SEMANTIC_REVIEW",
                _single_or_list(post_review_quantities.get(_canonical(reference), [])),
                review,
            )
        )
        rows.append(
            _row(
                entrypoint,
                processing_attempt_id,
                reference,
                "POST_RECONCILIATION",
                _single_or_list(post_reconciliation_quantities.get(_canonical(reference), [])),
            )
        )
        rows.append(
            _row(
                entrypoint,
                processing_attempt_id,
                reference,
                "FINAL_REQUIREMENT_EXTRACTION",
                _single_or_list(final_quantities.get(_canonical(reference), [])),
            )
        )
        rows.append(
            _row(
                entrypoint,
                processing_attempt_id,
                reference,
                "PUBLIC_RESPONSE",
                _single_or_list(final_quantities.get(_canonical(reference), [])),
            )
        )
    return rows


def _row(
    entrypoint: str,
    processing_attempt_id: str,
    reference: str,
    stage: str,
    quantity: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "entrypoint": entrypoint,
        "processing_attempt_id": processing_attempt_id,
        "reference": reference,
        "stage": stage,
        "quantity": quantity,
    }
    if extra:
        row.update(extra)
    return row


def _ordered_references(debug_capture: Any, extraction: Any) -> list[str]:
    values: list[str] = []
    discovery = getattr(debug_capture, "discovery", None)
    for element in getattr(discovery, "elements", []) or []:
        _append_reference(values, getattr(element, "reference", None))
    enrichment_debug = getattr(debug_capture, "enrichment_debug", None)
    for batch in getattr(enrichment_debug, "batch_results", []) or []:
        for element in getattr(batch, "elements", []) or []:
            _append_reference(values, getattr(element, "reference", None))
    for element in getattr(extraction, "elements", []) or []:
        reference = getattr(element, "reference", None)
        _append_reference(values, getattr(reference, "value", reference))
    return values


def _raw_enrichment_quantities(debug_capture: Any) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    enrichment_debug = getattr(debug_capture, "enrichment_debug", None)
    for raw_response in getattr(enrichment_debug, "raw_responses", []) or []:
        if not raw_response:
            continue
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        if not isinstance(payload, dict):
            continue
        for element in payload.get("elements", []):
            if not isinstance(element, dict):
                continue
            reference = _canonical(element.get("reference"))
            values.setdefault(reference, []).append(element.get("quantity"))
    return values


def _numeric_trace_quantities(traces: Any) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    for trace in traces or []:
        for element in getattr(trace, "elements", []) or []:
            reference = _canonical(getattr(element, "reference", None))
            final_quantity = getattr(element, "final_quantity", None)
            values.setdefault(reference, []).append(getattr(final_quantity, "value", None))
    return values


def _enrichment_debug_quantities(results: Any) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    for result in results or []:
        for element in getattr(result, "elements", []) or []:
            values.setdefault(_canonical(getattr(element, "reference", None)), []).append(
                getattr(element, "quantity", None)
            )
    return values


def _reconciliation_quantities(debug_capture: Any) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    for decision in getattr(debug_capture, "reconciliation_decisions", []) or []:
        reference = _canonical(getattr(decision, "reference", None))
        candidates = getattr(decision, "candidates", []) or []
        winner_id = getattr(decision, "winner_temporary_id", None)
        winner = next(
            (
                candidate
                for candidate in candidates
                if getattr(candidate, "temporary_id", None) == winner_id
            ),
            None,
        )
        values.setdefault(reference, []).append(getattr(winner, "quantity", None))
    return values


def _final_extraction_quantities(extraction: Any) -> dict[str | None, list[Any]]:
    values: dict[str | None, list[Any]] = {}
    for element in getattr(extraction, "elements", []) or []:
        reference = getattr(element, "reference", None)
        quantity = getattr(element, "quantity", None)
        values.setdefault(_canonical(getattr(reference, "value", reference)), []).append(
            getattr(quantity, "value", quantity)
        )
    return values


def _semantic_review_by_reference(debug_capture: Any) -> dict[str | None, dict[str, Any]]:
    values: dict[str | None, dict[str, Any]] = {}
    enrichment_debug = getattr(debug_capture, "enrichment_debug", None)
    for trace in getattr(enrichment_debug, "semantic_review_decisions", []) or []:
        values[_canonical(getattr(trace, "reference", None))] = {
            "review_called": getattr(trace, "review_called", None),
            "trigger_reason": getattr(trace, "trigger_reason", None),
            "independent_reread": getattr(trace, "independent_reread", None),
            "first_pass_quantity": getattr(trace, "original_value", None),
            "observed_quantity": getattr(trace, "observed_quantity", None),
            "review_decision": getattr(trace, "decision", None),
            "effective_decision": getattr(trace, "effective_decision", None),
            "final_value": getattr(trace, "final_value", None),
            "locator_strength": getattr(trace, "locator_strength", None),
            "correction_support": getattr(trace, "correction_support", None),
            "confirmation_support": getattr(trace, "confirmation_support", None),
        }
    return values


def _append_reference(values: list[str], reference: Any) -> None:
    if not isinstance(reference, str) or not reference:
        return
    if _canonical(reference) not in {_canonical(value) for value in values}:
        values.append(reference)


def _canonical(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().casefold().replace("_", "-")


def _single_or_list(values: list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values
