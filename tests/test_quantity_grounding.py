from pathlib import Path

from openpyxl import Workbook

from app.models.common import ExtractionStatus
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentResult,
)
from app.services.numeric_trace import (
    MODEL_EVIDENCE_QUANTITY,
    MODEL_EXPLICIT_QUANTITY,
    MODEL_INFERRED_QUANTITY,
    MODEL_NOTE,
    build_numeric_resolution_trace,
)
from app.services.quantity_grounding import (
    MODEL_EVIDENCE_QUANTITY as GROUNDING_MODEL_EVIDENCE_QUANTITY,
)
from app.services.quantity_grounding import (
    NO_SOURCE_GROUNDING_AVAILABLE,
    QUANTITY_GROUNDING_CONFLICT,
    SOURCE_INDEPENDENT_GROUNDED_WINS,
    SOURCE_INDEPENDENT_QUANTITY,
    SOURCE_MODEL_AGREE,
    SPREADSHEET_CELL,
    QuantityGroundingCandidate,
    build_source_independent_quantity_candidates,
    validate_enrichment_quantities,
)


class _FileSpec:
    def __init__(self, path: Path, mime_type: str) -> None:
        self.path = path
        self.mime_type = mime_type


def test_source_independent_quantity_replaces_floor_contaminated_model_quantity() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="item",
                reference="V-01",
                quantity=2,
                status=ExtractionStatus.EXPLICIT,
                confidence=0.95,
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="V-01 CANTIDAD: 2",
                    )
                ],
            )
        ]
    )

    result, decisions = validate_enrichment_quantities(
        enrichment,
        source_candidates_by_reference={
            "V-01": [_source_candidate("V-01", 1, source_id="source-1")]
        },
    )

    assert result.elements[0].quantity == 1
    assert result.elements[0].status == ExtractionStatus.EXPLICIT
    assert decisions[0].action == "REPLACE"
    assert decisions[0].reason == SOURCE_INDEPENDENT_GROUNDED_WINS
    assert decisions[0].source_candidates[0].value == 1
    assert decisions[0].model_evidence_candidates[0].value == 2
    assert decisions[0].model_evidence_candidates[0].source_type == (
        GROUNDING_MODEL_EVIDENCE_QUANTITY
    )


def test_source_independent_quantity_replaces_repetition_count_model_quantity() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="item",
                reference="V-10",
                quantity=5,
                status=ExtractionStatus.EXPLICIT,
                confidence=0.95,
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="V-10 CANTIDAD: 5 NIVELES 5 AL 9",
                    )
                ],
            )
        ]
    )

    result, decisions = validate_enrichment_quantities(
        enrichment,
        source_candidates_by_reference={
            "V-10": [_source_candidate("V-10", 25, source_id="source-1")]
        },
    )

    assert result.elements[0].quantity == 25
    assert decisions[0].action == "REPLACE"
    assert decisions[0].reason == SOURCE_INDEPENDENT_GROUNDED_WINS


def test_matching_source_independent_quantity_stays_explicit() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="item",
                reference="V-09",
                quantity=5,
                status=ExtractionStatus.EXPLICIT,
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="V-09 CANTIDAD: 5",
                    )
                ],
            )
        ]
    )

    result, decisions = validate_enrichment_quantities(
        enrichment,
        source_candidates_by_reference={
            "V-09": [_source_candidate("V-09", 5, source_id="source-1")]
        },
    )

    assert result.elements[0].quantity == 5
    assert result.elements[0].status == ExtractionStatus.EXPLICIT
    assert decisions[0].action == "KEEP"
    assert decisions[0].reason == SOURCE_MODEL_AGREE
    assert result.warnings == []


def test_model_evidence_quantity_without_independent_source_is_downgraded() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="item",
                reference="V-12",
                quantity=1,
                status=ExtractionStatus.EXPLICIT,
                confidence=0.9,
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="V-12 CANTIDAD: 1",
                    )
                ],
            )
        ]
    )

    result, decisions = validate_enrichment_quantities(enrichment)

    assert result.elements[0].quantity == 1
    assert result.elements[0].status == ExtractionStatus.EXPLICIT
    assert result.elements[0].confidence == 0.9
    assert result.elements[0].quantity_status == ExtractionStatus.INFERRED
    assert result.elements[0].quantity_confidence == 0.5
    assert NO_SOURCE_GROUNDING_AVAILABLE in result.elements[0].missing_or_unknown
    assert decisions[0].action == "DOWNGRADE"
    assert decisions[0].reason == NO_SOURCE_GROUNDING_AVAILABLE
    assert decisions[0].source_candidates == ()
    assert decisions[0].model_evidence_candidates[0].value == 1


def test_source_independent_quantity_conflict_marks_ambiguous() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="item",
                reference="V-04",
                quantity=3,
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result, decisions = validate_enrichment_quantities(
        enrichment,
        source_candidates_by_reference={
            "V-04": [
                _source_candidate("V-04", 2, source_id="source-1"),
                _source_candidate("V-04", 3, source_id="source-1"),
            ]
        },
    )

    assert result.elements[0].quantity == 3
    assert result.elements[0].status == ExtractionStatus.EXPLICIT
    assert result.elements[0].quantity_status == ExtractionStatus.AMBIGUOUS
    assert decisions[0].action == "MARK_AMBIGUOUS"
    assert decisions[0].reason == QUANTITY_GROUNDING_CONFLICT


def test_source_independent_candidates_do_not_cross_references() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="item",
                reference="V-01",
                quantity=5,
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result, decisions = validate_enrichment_quantities(
        enrichment,
        source_candidates_by_reference={
            "V-02": [_source_candidate("V-02", 1, source_id="source-1")]
        },
    )

    assert result.elements[0].quantity == 5
    assert result.elements[0].status == ExtractionStatus.EXPLICIT
    assert result.elements[0].quantity_status == ExtractionStatus.INFERRED
    assert decisions[0].source_candidates == ()


def test_quantity_downgrade_does_not_degrade_explicit_functional_metadata() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="item",
                reference="V-01",
                quantity=1,
                functional_type_raw="PROJECTING",
                status=ExtractionStatus.EXPLICIT,
                confidence=0.95,
            )
        ]
    )

    result, _ = validate_enrichment_quantities(enrichment)

    assert result.elements[0].quantity_status == ExtractionStatus.INFERRED
    assert result.elements[0].quantity_confidence == 0.5
    assert result.elements[0].status == ExtractionStatus.EXPLICIT
    assert result.elements[0].confidence == 0.95
    assert result.elements[0].functional_type_raw == "PROJECTING"


def test_xlsx_source_independent_candidates_use_row_reference_and_quantity_label(
    tmp_path: Path,
) -> None:
    workbook_path = tmp_path / "cuadro.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Cantidades"
    sheet.append(["REF", "CANTIDAD", "N.P"])
    sheet.append(["V-01", 1, 2])
    sheet.append(["V-02", "CANT. 5", "niveles 5 al 9"])
    workbook.save(workbook_path)

    candidates = build_source_independent_quantity_candidates(
        [
            _FileSpec(
                workbook_path,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        ]
    )

    assert candidates["V-01"][0].value == 1
    assert candidates["V-01"][0].origin == SPREADSHEET_CELL
    assert candidates["V-01"][0].source_id == "source-1"
    assert candidates["V-01"][0].source_file_name == "cuadro.xlsx"
    assert candidates["V-01"][0].sheet_name == "Cantidades"
    assert candidates["V-02"][0].value == 5


def test_numeric_trace_marks_model_evidence_and_notes_separately() -> None:
    trace = build_numeric_resolution_trace(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="item",
                    reference="V-12",
                    quantity=5,
                    status=ExtractionStatus.EXPLICIT,
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-1",
                            text="V-12 CANTIDAD: 5",
                        )
                    ],
                    evidence_notes=["Se asigna cantidad 5 por niveles 5 al 9"],
                )
            ]
        ),
        stage="test",
    )

    quantity_candidates = [
        candidate
        for candidate in trace.elements[0].candidates
        if candidate.semantic_role == "QUANTITY"
    ]

    assert quantity_candidates[0].field_path == "quantity"
    assert quantity_candidates[0].grounding_type == MODEL_EXPLICIT_QUANTITY
    assert quantity_candidates[1].field_path == "evidence[1]"
    assert quantity_candidates[1].grounding_type == MODEL_EVIDENCE_QUANTITY
    assert quantity_candidates[2].field_path == "evidence_notes[1]"
    assert quantity_candidates[2].source_type == MODEL_NOTE
    assert quantity_candidates[2].status == ExtractionStatus.INFERRED
    assert quantity_candidates[2].grounding_type == MODEL_INFERRED_QUANTITY


def _source_candidate(
    reference: str,
    value: int,
    *,
    source_id: str,
) -> QuantityGroundingCandidate:
    return QuantityGroundingCandidate(
        temporary_id="",
        reference=reference,
        value=value,
        source_id=source_id,
        source_file_name="cuadro.xlsx",
        page_number=None,
        sheet_name="Hoja1",
        cell_range="B2",
        region=None,
        raw_text=f"CANTIDAD {value}",
        field_path=None,
        origin=SPREADSHEET_CELL,
        source_type=SOURCE_INDEPENDENT_QUANTITY,
        status=ExtractionStatus.EXPLICIT,
    )
