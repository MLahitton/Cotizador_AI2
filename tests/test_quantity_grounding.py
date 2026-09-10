from pathlib import Path

from openpyxl import Workbook

from app.models.common import ExtractionStatus
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentComponent,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentMeasurement,
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
    COMMERCIAL_ROW_SINGLE_UNIT_INFERRED,
    COMPONENT_COUNT_COLLISION,
    NO_MODEL_QUANTITY,
    QUANTITY_GROUNDING_CONFLICT,
    SOURCE_INDEPENDENT_GROUNDED_WINS,
    SOURCE_INDEPENDENT_QUANTITY,
    SOURCE_MODEL_AGREE,
    SPREADSHEET_CELL,
    QuantityGroundingCandidate,
    build_source_independent_quantity_candidates,
    validate_enrichment_quantities,
)
from app.services.quantity_grounding import (
    MODEL_EVIDENCE_QUANTITY as GROUNDING_MODEL_EVIDENCE_QUANTITY,
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


def test_model_evidence_quantity_without_independent_source_stays_explicit() -> None:
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
    assert result.elements[0].quantity_status is None
    assert result.elements[0].quantity_confidence is None
    assert result.elements[0].missing_or_unknown == []
    assert decisions[0].action == "KEEP"
    assert decisions[0].reason == GROUNDING_MODEL_EVIDENCE_QUANTITY
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


def test_component_count_labels_reject_model_quantity_without_quantity_support() -> None:
    examples = [
        "PV-01 N° Cuerpos 5",
        "PV-01 N° Cuerpos: 5",
        "PV-01 Nº Cuerpos 5",
        "PV-01 Numero de cuerpos = 5",
        "PV-01 Número de cuerpos = 5",
        "PV-01 Cuerpos 5",
        "PV-01 5 cuerpos",
        "PV-01 Paneles 5",
        "PV-01 Módulos: 5",
        "PV-01 Hojas 5",
    ]

    for text in examples:
        result, decisions = validate_enrichment_quantities(
            GeminiEnrichmentResult(
                elements=[
                    GeminiElementEnrichment(
                        temporary_id="item",
                        reference="PV-01",
                        quantity=5,
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.95,
                        evidence=[GeminiEnrichmentEvidenceNote(type="table", text=text)],
                    )
                ]
            )
        )

        assert result.elements[0].quantity is None, text
        assert result.elements[0].quantity_status == ExtractionStatus.AMBIGUOUS
        assert result.elements[0].quantity_confidence == 0.5
        assert COMPONENT_COUNT_COLLISION in result.elements[0].missing_or_unknown
        assert decisions[0].action == "REJECT_QUANTITY"
        assert decisions[0].reason == COMPONENT_COUNT_COLLISION
        assert decisions[0].final_quantity is None


def test_strong_quantity_labels_keep_model_quantity_without_component_collision() -> None:
    examples = [
        ("PV-01 Cantidad 5", 5),
        ("PV-01 Cantidad: 5", 5),
        ("PV-01 Unidades 3", 3),
        ("PV-01 QTY 4", 4),
    ]

    for text, quantity in examples:
        result, decisions = validate_enrichment_quantities(
            GeminiEnrichmentResult(
                elements=[
                    GeminiElementEnrichment(
                        temporary_id="item",
                        reference="PV-01",
                        quantity=quantity,
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.95,
                        evidence=[GeminiEnrichmentEvidenceNote(type="table", text=text)],
                    )
                ]
            )
        )

        assert result.elements[0].quantity == quantity, text
        assert result.elements[0].quantity_status is None
        assert decisions[0].reason == GROUNDING_MODEL_EVIDENCE_QUANTITY


def test_quantity_label_wins_when_component_count_has_different_value() -> None:
    result, decisions = validate_enrichment_quantities(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="item",
                    reference="PV-01",
                    quantity=2,
                    status=ExtractionStatus.EXPLICIT,
                    confidence=0.95,
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            type="table",
                            text="PV-01 Cantidad 2 N° Cuerpos 5",
                        )
                    ],
                )
            ]
        )
    )

    assert result.elements[0].quantity == 2
    assert result.elements[0].quantity_status is None
    assert decisions[0].reason == GROUNDING_MODEL_EVIDENCE_QUANTITY


def test_component_count_quantity_is_rejected_even_when_other_quantity_label_exists() -> None:
    result, decisions = validate_enrichment_quantities(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="item",
                    reference="PV-01",
                    quantity=5,
                    status=ExtractionStatus.EXPLICIT,
                    confidence=0.95,
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            type="table",
                            text="PV-01 Cantidad 2 N° Cuerpos 5",
                        )
                    ],
                )
            ]
        )
    )

    assert result.elements[0].quantity is None
    assert result.elements[0].quantity_status == ExtractionStatus.AMBIGUOUS
    assert decisions[0].reason == COMPONENT_COUNT_COLLISION


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


def test_commercial_row_without_quantity_infers_single_unit_and_keeps_panel_count() -> None:
    result, decisions = validate_enrichment_quantities(
        GeminiEnrichmentResult(
            elements=[
                _commercial_row_element(
                    temporary_id="pv-01-sotano",
                    reference="PV-01",
                    context="SOTANO",
                    width=5.25,
                    height=2.50,
                    panel_count=5,
                    text="PV-01 SOTANO 5.25 x 2.50 N Cuerpos 5",
                )
            ]
        )
    )

    element = result.elements[0]
    assert element.quantity == 1
    assert element.quantity_status == ExtractionStatus.INFERRED
    assert element.quantity_confidence == 0.68
    assert element.panel_count == 5
    assert COMMERCIAL_ROW_SINGLE_UNIT_INFERRED in element.quantity_notes
    assert decisions[0].action == "INFER"
    assert decisions[0].reason == COMMERCIAL_ROW_SINGLE_UNIT_INFERRED
    assert decisions[0].selected_source_candidate is not None
    assert decisions[0].selected_source_candidate.value == 1

    trace = build_numeric_resolution_trace(result, stage="test")
    assert trace.elements[0].final_quantity.resolution_reason == (
        COMMERCIAL_ROW_SINGLE_UNIT_INFERRED
    )


def test_casa_pereira_commercial_rows_infer_one_without_using_component_counts() -> None:
    rows = [
        _commercial_row_element(
            temporary_id="pv-01-sotano",
            reference="PV-01",
            context="SOTANO",
            width=5.25,
            height=2.50,
            panel_count=5,
            text="PV-01 SOTANO 5.25 x 2.50 N Cuerpos 5",
        ),
        _commercial_row_element(
            temporary_id="v-01-sotano",
            reference="V-01",
            context="SOTANO",
            width=0.90,
            height=1.45,
            panel_count=2,
            text="V-01 SOTANO 0.90 x 1.45 N Cuerpos 2",
        ),
        _commercial_row_element(
            temporary_id="v-03-sotano",
            reference="V-03",
            context="SOTANO",
            width=0.80,
            height=0.40,
            panel_count=2,
            text="V-03 SOTANO 0.80 x 0.40 N Cuerpos 2 Apertura: 1 fijo + 1 rejilla",
            components=[
                GeminiEnrichmentComponent(name="fijo", quantity=1),
                GeminiEnrichmentComponent(name="rejilla", quantity=1),
            ],
        ),
    ]

    result, decisions = validate_enrichment_quantities(GeminiEnrichmentResult(elements=rows))

    assert [element.quantity for element in result.elements] == [1, 1, 1]
    assert [element.quantity_status for element in result.elements] == [
        ExtractionStatus.INFERRED,
        ExtractionStatus.INFERRED,
        ExtractionStatus.INFERRED,
    ]
    assert [element.panel_count for element in result.elements] == [5, 2, 2]
    assert result.elements[2].components[0].quantity == 1
    assert result.elements[2].components[1].quantity == 1
    assert [decision.reason for decision in decisions] == [
        COMMERCIAL_ROW_SINGLE_UNIT_INFERRED,
        COMMERCIAL_ROW_SINGLE_UNIT_INFERRED,
        COMMERCIAL_ROW_SINGLE_UNIT_INFERRED,
    ]


def test_commercial_row_without_formal_reference_can_infer_single_unit() -> None:
    result, decisions = validate_enrichment_quantities(
        GeminiEnrichmentResult(
            elements=[
                _commercial_row_element(
                    temporary_id="ventana-bano",
                    reference=None,
                    name="Ventana bano",
                    context="BANO",
                    width=0.80,
                    height=0.40,
                    panel_count=2,
                    text="Ventana bano 0.80 x 0.40 2 cuerpos",
                )
            ]
        )
    )

    assert result.elements[0].quantity == 1
    assert result.elements[0].quantity_status == ExtractionStatus.INFERRED
    assert decisions[0].reason == COMMERCIAL_ROW_SINGLE_UNIT_INFERRED


def test_explicit_quantity_evidence_wins_over_single_row_fallback() -> None:
    result, decisions = validate_enrichment_quantities(
        GeminiEnrichmentResult(
            elements=[
                _commercial_row_element(
                    temporary_id="pv-01",
                    reference="PV-01",
                    context="SOTANO",
                    width=5.25,
                    height=2.50,
                    panel_count=5,
                    quantity=None,
                    text="PV-01 SOTANO Cantidad 3 N Cuerpos 5",
                )
            ]
        )
    )

    assert result.elements[0].quantity == 3
    assert result.elements[0].quantity_status == ExtractionStatus.EXPLICIT
    assert result.elements[0].panel_count == 5
    assert decisions[0].reason == GROUNDING_MODEL_EVIDENCE_QUANTITY


def test_model_quantity_explicit_precedence_over_commercial_row_fallback() -> None:
    examples = [
        ("PV-01 Cantidad 2 N Cuerpos 5", 2, 5),
        ("PV-02 Cantidad 4 N Cuerpos 2", 4, 2),
    ]

    for index, (text, quantity, panel_count) in enumerate(examples, start=1):
        result, decisions = validate_enrichment_quantities(
            GeminiEnrichmentResult(
                elements=[
                    _commercial_row_element(
                        temporary_id=f"item-{index}",
                        reference=f"PV-0{index}",
                        context="SOTANO",
                        width=1.0,
                        height=1.0,
                        panel_count=panel_count,
                        quantity=quantity,
                        text=text,
                    )
                ]
            )
        )

        assert result.elements[0].quantity == quantity
        assert result.elements[0].quantity_status is None
        assert result.elements[0].panel_count == panel_count
        assert decisions[0].reason == GROUNDING_MODEL_EVIDENCE_QUANTITY


def test_ambiguous_group_text_does_not_infer_single_unit() -> None:
    result, decisions = validate_enrichment_quantities(
        GeminiEnrichmentResult(
            elements=[
                _commercial_row_element(
                    temporary_id="grupo",
                    reference=None,
                    name="Ventanas habitaciones",
                    context="HABITACIONES",
                    width=0.80,
                    height=1.20,
                    panel_count=2,
                    text="Ventanas habitaciones 0.80 x 1.20 2 cuerpos",
                )
            ]
        )
    )

    assert result.elements[0].quantity is None
    assert result.elements[0].quantity_status is None
    assert decisions[0].reason == NO_MODEL_QUANTITY


def test_component_count_alone_does_not_infer_single_unit() -> None:
    result, decisions = validate_enrichment_quantities(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="only-components",
                    panel_count=5,
                    evidence=[GeminiEnrichmentEvidenceNote(type="table", text="5 cuerpos")],
                )
            ]
        )
    )

    assert result.elements[0].quantity is None
    assert result.elements[0].quantity_status is None
    assert decisions[0].reason == NO_MODEL_QUANTITY


def test_casa_pereira_complete_nineteen_positions_remain_distinct_with_quantity_one() -> None:
    rows = [
        ("PV-01", "SOTANO", 5.25, 2.50, 5),
        ("V-01", "SOTANO", 0.90, 1.45, 2),
        ("V-03", "SOTANO", 0.80, 0.40, 2),
        ("V-01", "NIVEL 1", 1.20, 1.50, 2),
        ("V-01", "NIVEL 2", 1.20, 1.50, 2),
        ("V-02", "NIVEL 1", 1.00, 1.20, 2),
        ("V-02", "NIVEL 2", 1.00, 1.20, 2),
        ("V-04", "NIVEL 1", 1.40, 1.30, 3),
        ("V-04", "NIVEL 2", 1.40, 1.30, 3),
        ("P-01", "NIVEL 1", 0.90, 2.10, 1),
        ("P-01", "NIVEL 2", 0.90, 2.10, 1),
        ("F-01", "FACHADA", 1.50, 2.40, 1),
        ("F-02", "FACHADA", 1.60, 2.40, 1),
        ("F-03", "FACHADA", 1.70, 2.40, 1),
        ("V-05", "PATIO", 0.70, 1.00, 2),
        ("V-06", "PATIO", 0.75, 1.00, 2),
        ("V-07", "PATIO", 0.80, 1.00, 2),
        ("PV-02", "TERRAZA", 3.10, 2.30, 4),
        ("PV-03", "TERRAZA", 2.80, 2.30, 3),
    ]
    elements = [
        _commercial_row_element(
            temporary_id=f"item-{index}",
            reference=reference,
            context=context,
            width=width,
            height=height,
            panel_count=panel_count,
            text=f"{reference} {context} {width} x {height} N Cuerpos {panel_count}",
        )
        for index, (reference, context, width, height, panel_count) in enumerate(rows, start=1)
    ]

    result, decisions = validate_enrichment_quantities(GeminiEnrichmentResult(elements=elements))

    assert len(result.elements) == 19
    assert all(element.quantity == 1 for element in result.elements)
    assert all(element.quantity_status == ExtractionStatus.INFERRED for element in result.elements)
    assert [element.reference for element in result.elements].count("V-01") == 3
    v01_contexts = [
        element.occurrence_context for element in result.elements if element.reference == "V-01"
    ]
    assert v01_contexts == ["SOTANO", "NIVEL 1", "NIVEL 2"]
    assert [element.panel_count for element in result.elements] == [row[4] for row in rows]
    assert all(decision.reason == COMMERCIAL_ROW_SINGLE_UNIT_INFERRED for decision in decisions)


def _commercial_row_element(
    *,
    temporary_id: str,
    reference: str | None,
    context: str,
    width: float,
    height: float,
    panel_count: int,
    text: str,
    name: str | None = "Ventana",
    quantity: int | None = None,
    components: list[GeminiEnrichmentComponent] | None = None,
) -> GeminiElementEnrichment:
    return GeminiElementEnrichment(
        temporary_id=temporary_id,
        reference=reference,
        name=name,
        quantity=quantity,
        occurrence_context=context,
        panel_count=panel_count,
        measurements=[
            GeminiEnrichmentMeasurement(type="width", value=width, unit="m"),
            GeminiEnrichmentMeasurement(type="height", value=height, unit="m"),
        ],
        components=components or [],
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                type="table",
                text=text,
                sheet_name="Casa Pereira",
                cell_range="A2:G2",
            )
        ],
    )


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
