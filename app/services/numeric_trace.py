from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus
from app.models.evidence import Region
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentMeasurement,
    GeminiEnrichmentResult,
)

MODEL_EXPLICIT = "MODEL_EXPLICIT"
MODEL_INFERRED = "MODEL_INFERRED"
TABLE = "TABLE"
VISION = "VISION"
UNKNOWN_NUMERIC = "UNKNOWN_NUMERIC"


class NumericCandidateTrace(BaseModel):
    element_temporary_id: str
    field_path: str
    semantic_role: str
    value: str | int | float | None
    source_type: str
    source_id: str | None = None
    source_file_name: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    region: Region | None = None
    evidence_text: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class FinalQuantityTrace(BaseModel):
    element_temporary_id: str
    value: str | int | float | None
    origin_candidate_field_path: str | None = None
    resolution_reason: str = "MODEL_OUTPUT"
    status: ExtractionStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class NumericElementTrace(BaseModel):
    element_temporary_id: str
    reference: str | None = None
    candidates: list[NumericCandidateTrace] = Field(default_factory=list)
    final_quantity: FinalQuantityTrace


class NumericResolutionTrace(BaseModel):
    stage: str
    elements: list[NumericElementTrace] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def build_numeric_resolution_trace(
    enrichment: GeminiEnrichmentResult,
    *,
    stage: str,
    source_file_names_by_id: dict[str, str] | None = None,
) -> NumericResolutionTrace:
    source_file_names_by_id = source_file_names_by_id or {}
    return NumericResolutionTrace(
        stage=stage,
        elements=[
            _element_trace(element, source_file_names_by_id)
            for element in enrichment.elements
        ],
    )


def _element_trace(
    element: GeminiElementEnrichment,
    source_file_names_by_id: dict[str, str],
) -> NumericElementTrace:
    candidates: list[NumericCandidateTrace] = []
    if element.quantity not in (None, ""):
        candidates.append(
            _candidate(
                element,
                field_path="quantity",
                semantic_role="QUANTITY",
                value=element.quantity,
                source_type=_source_type_for_status(element.status),
                status=element.status,
                confidence=element.confidence,
            )
        )

    for index, measurement in enumerate(element.measurements, start=1):
        candidates.extend(_measurement_candidates(element, measurement, index))

    for index, component in enumerate(element.components, start=1):
        if component.quantity not in (None, ""):
            candidates.append(
                _candidate(
                    element,
                    field_path=f"components[{index}].quantity",
                    semantic_role="COMPONENT_COUNT",
                    value=component.quantity,
                    source_type=_source_type_for_status(component.status),
                    status=component.status,
                    confidence=component.confidence,
                    evidence_text=component.notes or component.description,
                )
            )

    if element.panel_count is not None:
        candidates.append(
            _candidate(
                element,
                field_path="panel_count",
                semantic_role="PANEL_COUNT",
                value=element.panel_count,
                source_type=_source_type_for_status(element.status),
                status=element.status,
                confidence=element.confidence,
                evidence_text=element.configuration_raw,
            )
        )

    for index, evidence in enumerate(element.evidence, start=1):
        candidates.extend(
            _evidence_numeric_candidates(
                element,
                evidence,
                index,
                source_file_names_by_id,
            )
        )

    for index, note in enumerate(element.evidence_notes, start=1):
        candidates.extend(
            _text_numeric_candidates(
                element,
                text=note,
                field_path=f"evidence_notes[{index}]",
                source_type=UNKNOWN_NUMERIC,
                source_id=None,
                source_file_name=None,
                page_number=None,
                sheet_name=None,
                cell_range=None,
                region=None,
            )
        )

    return NumericElementTrace(
        element_temporary_id=element.temporary_id,
        reference=element.reference,
        candidates=candidates,
        final_quantity=FinalQuantityTrace(
            element_temporary_id=element.temporary_id,
            value=element.quantity,
            origin_candidate_field_path="quantity"
            if element.quantity not in (None, "")
            else None,
            status=element.status,
            confidence=element.confidence,
        ),
    )


def _measurement_candidates(
    element: GeminiElementEnrichment,
    measurement: GeminiEnrichmentMeasurement,
    index: int,
) -> list[NumericCandidateTrace]:
    if measurement.value is None:
        return []

    semantic_role = _measurement_role(measurement.type or measurement.raw_label)
    return [
        _candidate(
            element,
            field_path=f"measurements[{index}]",
            semantic_role=semantic_role,
            value=measurement.value,
            source_type=_source_type_for_status(measurement.status),
            status=measurement.status,
            confidence=measurement.confidence,
            evidence_text=measurement.text or measurement.raw_label or measurement.notes,
        )
    ]


def _candidate(
    element: GeminiElementEnrichment,
    *,
    field_path: str,
    semantic_role: str,
    value: str | int | float | None,
    source_type: str,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_text: str | None = None,
) -> NumericCandidateTrace:
    return NumericCandidateTrace(
        element_temporary_id=element.temporary_id,
        field_path=field_path,
        semantic_role=semantic_role,
        value=value,
        source_type=source_type,
        evidence_text=evidence_text,
        status=status,
        confidence=confidence,
    )


def _measurement_role(value: str | None) -> str:
    text = _compact(value or "")
    if text in {"width", "ancho"}:
        return "WIDTH"
    if text in {"height", "alto"}:
        return "HEIGHT"
    if text in {"area", "m2", "m²", "ma2"}:
        return "AREA"
    if text in {"thickness", "espesor"}:
        return "GLASS_THICKNESS"
    return UNKNOWN_NUMERIC


def _evidence_numeric_candidates(
    element: GeminiElementEnrichment,
    evidence: GeminiEnrichmentEvidenceNote,
    index: int,
    source_file_names_by_id: dict[str, str],
) -> list[NumericCandidateTrace]:
    text = evidence.text or evidence.visual_description or evidence.notes
    if not text:
        return []

    return _text_numeric_candidates(
        element,
        text=text,
        field_path=f"evidence[{index}]",
        source_type=_evidence_source_type(evidence),
        source_id=evidence.source_id,
        source_file_name=source_file_names_by_id.get(evidence.source_id or ""),
        page_number=evidence.page_number,
        sheet_name=evidence.sheet_name,
        cell_range=evidence.cell_range,
        region=evidence.region,
    )


def _text_numeric_candidates(
    element: GeminiElementEnrichment,
    *,
    text: str,
    field_path: str,
    source_type: str,
    source_id: str | None,
    source_file_name: str | None,
    page_number: int | None,
    sheet_name: str | None,
    cell_range: str | None,
    region: Region | None,
) -> list[NumericCandidateTrace]:
    candidates: list[NumericCandidateTrace] = []
    for semantic_role, pattern in _TEXT_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append(
                NumericCandidateTrace(
                    element_temporary_id=element.temporary_id,
                    field_path=field_path,
                    semantic_role=semantic_role,
                    value=_match_value(semantic_role, match),
                    source_type=source_type,
                    source_id=source_id,
                    source_file_name=source_file_name,
                    page_number=page_number,
                    sheet_name=sheet_name,
                    cell_range=cell_range,
                    region=region,
                    evidence_text=match.group(0),
                    status=_status_for_text_role(semantic_role),
                    confidence=None,
                )
            )
            if semantic_role == "LEVEL_RANGE":
                repetition_count = _inclusive_range_count(match.group(1), match.group(2))
                if repetition_count is not None:
                    candidates.append(
                        NumericCandidateTrace(
                            element_temporary_id=element.temporary_id,
                            field_path=field_path,
                            semantic_role="REPETITION_COUNT",
                            value=repetition_count,
                            source_type=MODEL_INFERRED,
                            source_id=source_id,
                            source_file_name=source_file_name,
                            page_number=page_number,
                            sheet_name=sheet_name,
                            cell_range=cell_range,
                            region=region,
                            evidence_text=match.group(0),
                            status=ExtractionStatus.INFERRED,
                            confidence=None,
                        )
                    )
    return candidates


_NUMBER = r"\d+(?:[.,]\d+)?"
_TEXT_PATTERNS = (
    (
        "QUANTITY",
        re.compile(
            rf"\b(?:cantidad(?:\s+total)?|cant\.?|cnt|qty|unidades|und)\s*:?\s*({_NUMBER})\b",
            flags=re.IGNORECASE,
        ),
    ),
    ("LEVEL", re.compile(rf"\bn\.?\s*p\.?\s*_?\s*({_NUMBER})\b", flags=re.IGNORECASE)),
    ("FLOOR", re.compile(rf"\bpiso\s*:?\s*({_NUMBER})\b", flags=re.IGNORECASE)),
    ("LEVEL", re.compile(rf"\bnivel\s*:?\s*({_NUMBER})\b", flags=re.IGNORECASE)),
    (
        "LEVEL_RANGE",
        re.compile(
            rf"\bniveles?\s+({_NUMBER})\s*(?:al|a|-)\s*({_NUMBER})\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "COMPONENT_COUNT",
        re.compile(rf"\b({_NUMBER})\s*(?:cuerpos?|secciones?|tramos?)\b", flags=re.IGNORECASE),
    ),
    (
        "SECTION_COUNT",
        re.compile(rf"\b({_NUMBER})\s*(?:modulos?|sections?)\b", flags=re.IGNORECASE),
    ),
    (
        "GLASS_THICKNESS",
        re.compile(rf"\b({_NUMBER})\s*mm\b", flags=re.IGNORECASE),
    ),
    (
        "ITEM_NUMBER",
        re.compile(rf"\b(?:item|ítem|itm)\s*:?\s*({_NUMBER})\b", flags=re.IGNORECASE),
    ),
    (
        "WIDTH",
        re.compile(rf"\b({_NUMBER})\s*(?:x|×)\s*{_NUMBER}\b", flags=re.IGNORECASE),
    ),
    (
        "HEIGHT",
        re.compile(rf"\b{_NUMBER}\s*(?:x|×)\s*({_NUMBER})\b", flags=re.IGNORECASE),
    ),
)


def _match_value(semantic_role: str, match: re.Match[str]) -> str | int | float:
    if semantic_role == "LEVEL_RANGE":
        start = _parse_number(match.group(1))
        end = _parse_number(match.group(2))
        return f"{start:g}-{end:g}"

    return _parse_number(match.group(1))


def _parse_number(value: str) -> int | float:
    parsed = float(value.replace(",", "."))
    return int(parsed) if parsed.is_integer() else parsed


def _inclusive_range_count(start: str, end: str) -> int | None:
    parsed_start = _parse_number(start)
    parsed_end = _parse_number(end)
    if not isinstance(parsed_start, int) or not isinstance(parsed_end, int):
        return None
    if parsed_end < parsed_start:
        return None
    return parsed_end - parsed_start + 1


def _status_for_text_role(semantic_role: str) -> ExtractionStatus:
    if semantic_role == "REPETITION_COUNT":
        return ExtractionStatus.INFERRED
    return ExtractionStatus.EXPLICIT


def _source_type_for_status(status: ExtractionStatus | None) -> str:
    if status == ExtractionStatus.INFERRED:
        return MODEL_INFERRED
    return MODEL_EXPLICIT


def _evidence_source_type(evidence: GeminiEnrichmentEvidenceNote) -> str:
    text = _compact(evidence.type or "")
    if text in {"table", "tabla", "sheet", "spreadsheet"}:
        return TABLE
    if text in {"visual", "vision", "drawing", "image", "dibujo"} or evidence.region:
        return VISION
    return UNKNOWN_NUMERIC


def _compact(value: str) -> str:
    return value.casefold().strip().replace("_", "").replace("-", "").replace(" ", "")
