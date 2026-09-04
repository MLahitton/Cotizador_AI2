from pathlib import Path

from app.models.common import ExtractionStatus
from app.models.evidence import Region
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentComponent,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentMeasurement,
)
from app.providers.gemini_extraction import GeminiFullPipelineDebugCapture
from app.services.numeric_trace import build_numeric_resolution_trace
from app.services.semantic_review import (
    MODEL_EVIDENCE_QUANTITY,
    REVIEW_AMBIGUOUS_QUANTITY,
    REVIEW_CORRECTED_QUANTITY,
    REVIEW_CORRECTION_REJECTED_INSUFFICIENT_LOCAL_SUPPORT,
    SemanticFieldReview,
    SourceLocator,
    apply_quantity_review,
    build_quantity_review_numeric_context,
    evaluate_semantic_correction_support,
    parse_semantic_field_review_response,
    quantity_numeric_collision_summary,
    resolve_quantity_review_locator,
    should_review_quantity,
)
from tests.test_gemini_enrichment_pipeline import (
    _FakeResponse,
    _provider_with_responses,
    _scope_response,
    _xlsx_with_quantity,
)


def test_should_review_quantity_when_it_matches_level_signal() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-01",
        quantity=2,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="V-01 CANTIDAD: 2 N.P_2",
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    should_review, reason = should_review_quantity(element, None, trace)

    assert should_review is True
    assert reason == "QUANTITY_EQUALS_LEVEL"


def test_should_skip_quantity_without_suspicion() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-09",
        quantity=5,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="V-09 CANTIDAD: 5",
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    should_review, reason = should_review_quantity(element, None, trace)

    assert should_review is False
    assert reason == "NO_NUMERIC_SUSPICION"


def test_apply_quantity_review_changes_only_quantity_fields() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-04",
        quantity=3,
        geometry_raw="arched",
        functional_type_raw="FIXED",
        finish_raw="negro",
        glass=[],
        measurements=[GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m")],
        status=ExtractionStatus.EXPLICIT,
        confidence=0.95,
    )
    field_local_exclusions = {
        "quantity",
        "quantity_status",
        "quantity_confidence",
        "quantity_notes",
        "missing_or_unknown",
        "notes",
    }
    original_snapshot = element.model_dump(mode="json", exclude=field_local_exclusions)

    updated = apply_quantity_review(
        element,
        SemanticFieldReview(
            temporary_id="item",
            reference="V-04",
            original_value=3,
            reviewed_value=1,
            decision="CORRECTED",
            reason="Fuente original muestra CANTIDAD: 1.",
            confidence=0.9,
            source_ids=["source-1"],
        ),
        SourceLocator(text_context="V-04 CANTIDAD: 1", locator_used="ELEMENT_EVIDENCE"),
    )

    assert updated.quantity == 1
    assert updated.quantity_status == ExtractionStatus.EXPLICIT
    assert updated.quantity_confidence == 0.9
    assert REVIEW_CORRECTED_QUANTITY in updated.missing_or_unknown
    assert updated.status == ExtractionStatus.EXPLICIT
    assert updated.confidence == 0.95
    assert updated.geometry_raw == element.geometry_raw
    assert updated.functional_type_raw == element.functional_type_raw
    assert updated.finish_raw == element.finish_raw
    assert updated.measurements == element.measurements
    assert updated.model_dump(mode="json", exclude=field_local_exclusions) == original_snapshot


def test_apply_quantity_review_rejects_correction_without_local_support() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-07",
        quantity=1,
        functional_type_raw="PROJECTING",
        geometry_raw="rectangular",
        measurements=[GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m")],
        status=ExtractionStatus.EXPLICIT,
        confidence=0.95,
    )

    updated = apply_quantity_review(
        element,
        SemanticFieldReview(
            temporary_id="item",
            reference="V-07",
            original_value=1,
            reviewed_value=4,
            decision="CORRECTED",
            reason="Pagina amplia indica CANTIDAD: 4.",
            confidence=0.8,
            source_ids=["source-1"],
        ),
        SourceLocator(source_id="source-1", page_number=2, locator_used="NUMERIC_TRACE"),
    )

    assert updated.quantity == 1
    assert updated.quantity_status == ExtractionStatus.AMBIGUOUS
    assert updated.quantity_confidence == 0.5
    assert REVIEW_CORRECTION_REJECTED_INSUFFICIENT_LOCAL_SUPPORT in (
        updated.missing_or_unknown
    )
    assert updated.functional_type_raw == element.functional_type_raw
    assert updated.geometry_raw == element.geometry_raw
    assert updated.measurements == element.measurements
    assert updated.status == ExtractionStatus.EXPLICIT
    assert updated.confidence == 0.95


def test_apply_quantity_review_ambiguous_does_not_force_correction() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-08",
        quantity=1,
    )

    updated = apply_quantity_review(
        element,
        SemanticFieldReview(
            temporary_id="item",
            reference="V-08",
            original_value=1,
            reviewed_value=None,
            decision="AMBIGUOUS",
            reason="La fuente no deja claro si es 1 o 5.",
            confidence=0.4,
            source_ids=["source-1"],
        ),
    )

    assert updated.quantity == 1
    assert updated.status is None
    assert updated.quantity_status == ExtractionStatus.AMBIGUOUS
    assert REVIEW_AMBIGUOUS_QUANTITY in updated.missing_or_unknown


def test_correction_support_rejects_page_only_and_accepts_region() -> None:
    page_only = evaluate_semantic_correction_support(
        field="quantity",
        original_value=1,
        reviewed_value=4,
        locator=SourceLocator(
            source_id="source-1",
            page_number=2,
            locator_used="NUMERIC_TRACE",
        ),
    )
    with_region = evaluate_semantic_correction_support(
        field="quantity",
        original_value=1,
        reviewed_value=5,
        locator=SourceLocator(
            source_id="source-1",
            page_number=2,
            region=Region(x=0.1, y=0.2, width=0.3, height=0.4),
            locator_used="NUMERIC_TRACE",
        ),
    )

    assert page_only.support_level == "INSUFFICIENT"
    assert page_only.reason == "PAGE_ONLY"
    assert with_region.support_level == "STRONG"
    assert with_region.reason == "PAGE_REGION"


def test_correction_support_accepts_sheet_cell_and_local_quantity_label() -> None:
    sheet_cell = evaluate_semantic_correction_support(
        field="quantity",
        original_value=1,
        reviewed_value=5,
        locator=SourceLocator(
            source_id="source-1",
            sheet_name="Cantidades",
            cell_range="B2",
            locator_used="SOURCE_INDEPENDENT",
            authoritative_value=5,
        ),
    )
    text_context = evaluate_semantic_correction_support(
        field="quantity",
        original_value=7,
        reviewed_value=3,
        locator=SourceLocator(
            text_context="Elemento A - Cantidad: 3. Piso 7.",
            locator_used="ELEMENT_EVIDENCE",
        ),
    )

    assert sheet_cell.support_level == "STRONG"
    assert sheet_cell.reason == "SOURCE_INDEPENDENT"
    assert text_context.support_level == "STRONG"
    assert text_context.reason == "LOCAL_EXPLICIT_LABEL"


def test_correction_support_rejects_source_independent_disagreement() -> None:
    support = evaluate_semantic_correction_support(
        field="quantity",
        original_value=5,
        reviewed_value=3,
        locator=SourceLocator(
            source_id="source-1",
            sheet_name="Cantidades",
            cell_range="B2",
            locator_used="SOURCE_INDEPENDENT",
            authoritative_value=5,
        ),
    )

    assert support.support_level == "INSUFFICIENT"
    assert support.reason == "SOURCE_INDEPENDENT_DISAGREES"


def test_quantity_numeric_collision_detects_suspicious_roles() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-10",
        quantity=5,
        panel_count=5,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="V-10 CANTIDAD: 5 NIVELES 5 AL 9",
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    collision = quantity_numeric_collision_summary(5, trace)

    assert collision.detected is True
    assert "PANEL_COUNT" in collision.roles
    assert "LEVEL_RANGE" in collision.roles
    assert "REPETITION_COUNT" in collision.roles


def test_quantity_review_numeric_context_omits_first_pass_quantity_evidence_text() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-10",
        quantity=5,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="CANTIDAD: 5 NIVELES 5 AL 9",
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    context = build_quantity_review_numeric_context(trace)

    assert context.untrusted_first_pass_quantity == 5
    assert context.first_pass_quantity_evidence_included is False
    quantity_evidence_candidates = [
        candidate
        for candidate in context.candidates
        if candidate["semantic_role"] == "QUANTITY"
        and candidate["field_path"] == "evidence[1]"
    ]
    assert quantity_evidence_candidates[0]["evidence_text"] is None
    assert "First-pass quantity evidence text omitted" in (
        quantity_evidence_candidates[0]["note"]
    )
    assert any(
        candidate["semantic_role"] == "LEVEL_RANGE"
        and candidate["evidence_text"] == "NIVELES 5 AL 9"
        for candidate in context.candidates
    )


def test_confirmed_quantity_with_page_only_collision_is_not_trusted() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-07",
        quantity=1,
        components=[GeminiEnrichmentComponent(quantity=1)],
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="componente cantidad 1",
                page_number=2,
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    updated = apply_quantity_review(
        element,
        SemanticFieldReview(
            temporary_id="item",
            reference="V-07",
            original_value=1,
            reviewed_value=1,
            decision="CONFIRMED",
            reason="Confirma 1.",
            confidence=0.9,
            source_ids=["source-1"],
        ),
        SourceLocator(source_id="source-1", page_number=2, locator_used="NUMERIC_TRACE"),
        trace,
    )

    assert updated.quantity == 1
    assert updated.quantity_status == ExtractionStatus.AMBIGUOUS
    assert updated.quantity_confidence == 0.5
    assert REVIEW_AMBIGUOUS_QUANTITY in updated.missing_or_unknown


def test_confirmed_quantity_with_local_label_collision_stays_trusted() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-09",
        quantity=5,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="V-09 CANTIDAD: 5 N.P_5",
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    updated = apply_quantity_review(
        element,
        SemanticFieldReview(
            temporary_id="item",
            reference="V-09",
            original_value=5,
            reviewed_value=5,
            decision="CONFIRMED",
            reason="Cantidad comercial confirmada.",
            confidence=0.95,
            source_ids=["source-1"],
        ),
        SourceLocator(
            source_id="source-1",
            text_context="V-09 CANTIDAD: 5 N.P_5",
            locator_used="NUMERIC_TRACE",
        ),
        trace,
    )

    assert updated == element


def test_parse_semantic_review_accepts_direct_object() -> None:
    text = (
        '{"temporary_id": "item", "reference": "V-08", "field": "quantity", '
        '"original_value": 1, "observed_quantity": 5, '
        '"observed_text": "CANTIDAD: 5", '
        '"reviewed_value": 5, "decision": "CORRECTED", '
        '"reason": "Fuente original muestra cantidad 5.", '
        '"confidence": 0.9, "source_ids": ["source-1"]}'
    )

    review = parse_semantic_field_review_response(
        text,
        temporary_id="item",
        reference="V-08",
        original_value=1,
        source_ids=["source-1"],
    )

    assert review.decision == "CORRECTED"
    assert review.observed_quantity == 5
    assert review.observed_text == "CANTIDAD: 5"
    assert review.reviewed_value == 5
    assert review.temporary_id == "item"


def test_parse_semantic_review_accepts_single_item_list() -> None:
    object_text = (
        '{"temporary_id": "item", "reference": "V-08", "field": "quantity", '
        '"original_value": 1, "reviewed_value": 5, "decision": "CORRECTED", '
        '"reason": "Fuente original muestra cantidad 5.", '
        '"confidence": 0.9, "source_ids": ["source-1"]}'
    )

    direct = parse_semantic_field_review_response(
        object_text,
        temporary_id="item",
        reference="V-08",
        original_value=1,
        source_ids=["source-1"],
    )
    wrapped = parse_semantic_field_review_response(
        f"[{object_text}]",
        temporary_id="item",
        reference="V-08",
        original_value=1,
        source_ids=["source-1"],
    )

    assert wrapped == direct


def test_parse_semantic_review_empty_list_is_unresolved() -> None:
    review = parse_semantic_field_review_response(
        "[]",
        temporary_id="item",
        reference="V-08",
        original_value=1,
        source_ids=["source-1"],
    )

    assert review.decision == "UNRESOLVED"
    assert "exactly 1 item" in review.reason


def test_parse_semantic_review_multi_item_list_is_unresolved() -> None:
    text = (
        '[{"temporary_id": "item", "decision": "CONFIRMED", "reason": "a"}, '
        '{"temporary_id": "item", "decision": "CORRECTED", "reason": "b"}]'
    )

    review = parse_semantic_field_review_response(
        text,
        temporary_id="item",
        reference="V-08",
        original_value=1,
        source_ids=["source-1"],
    )

    assert review.decision == "UNRESOLVED"
    assert "exactly 1 item" in review.reason


def test_parse_semantic_review_invalid_shape_is_unresolved() -> None:
    review = parse_semantic_field_review_response(
        '"not an object"',
        temporary_id="item",
        reference="V-08",
        original_value=1,
        source_ids=["source-1"],
    )

    assert review.decision == "UNRESOLVED"
    assert "must be an object" in review.reason


def test_resolve_quantity_review_locator_prefers_pdf_page_region() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-08",
        quantity=1,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-2",
                type="visual",
                text="CANTIDAD: 5",
                page_number=4,
                region=Region(x=0.1, y=0.2, width=0.3, height=0.4),
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
        source_file_names_by_id={"source-2": "detalle.pdf"},
    ).elements[0]

    locator = resolve_quantity_review_locator(element, None, trace)

    assert locator.locator_used == "NUMERIC_TRACE"
    assert locator.source_id == "source-2"
    assert locator.source_file_name == "detalle.pdf"
    assert locator.page_number == 4
    assert locator.region == Region(x=0.1, y=0.2, width=0.3, height=0.4)
    assert locator.text_context is None
    assert locator.text_context_is_first_pass is True
    assert locator.locator_strength == "PAGE_REGION"
    assert locator.region_missing_origin is None


def test_resolve_quantity_review_locator_does_not_invent_missing_region() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-01",
        quantity=2,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="N.P_2",
                page_number=2,
            )
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    locator = resolve_quantity_review_locator(element, None, trace)

    assert locator.region is None
    assert locator.locator_strength == "PAGE_ONLY"
    assert locator.region_missing_origin == "MODEL_NOT_PROVIDED"


def test_resolve_quantity_review_locator_reuses_same_element_region_only() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="V-01",
        quantity=2,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                text="N.P_2",
                page_number=2,
            ),
            GeminiEnrichmentEvidenceNote(
                source_id="source-1",
                type="visual",
                visual_description="bloque V-01",
                page_number=2,
                region=Region(x=0.4, y=0.2, width=0.1, height=0.2),
            ),
            GeminiEnrichmentEvidenceNote(
                source_id="source-2",
                type="visual",
                visual_description="otro source",
                page_number=2,
                region=Region(x=0.1, y=0.1, width=0.1, height=0.1),
            ),
        ],
    )
    trace = build_numeric_resolution_trace(
        type("Result", (), {"elements": [element]})(),
        stage="test",
    ).elements[0]

    locator = resolve_quantity_review_locator(element, None, trace)

    assert locator.source_id == "source-1"
    assert locator.page_number == 2
    assert locator.region == Region(x=0.4, y=0.2, width=0.1, height=0.2)
    assert locator.locator_strength == "PAGE_REGION"


def test_resolve_quantity_review_locator_uses_text_context_without_file() -> None:
    element = GeminiElementEnrichment(
        temporary_id="item",
        reference="A-01",
        quantity=7,
        evidence=[
            GeminiEnrichmentEvidenceNote(
                text="Elemento A-01: cantidad 3. Piso 7.",
            )
        ],
    )

    locator = resolve_quantity_review_locator(element, None, None)

    assert locator.locator_used == "ELEMENT_EVIDENCE"
    assert locator.source_id is None
    assert locator.text_context == "Elemento A-01: cantidad 3. Piso 7."


def test_full_pipeline_semantic_review_corrects_pdf_quantity_with_original_source(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-01"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-01", "quantity": 2, '
                '"status": "explicit", "geometry_raw": "arched", '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "V-01 CANTIDAD: 1 N.P_2", "page_number": 1, '
                '"region": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}]}'
                "]}",
            ),
            _FakeResponse(
                '{"temporary_id": "d-1", "reference": "V-01", "field": "quantity", '
                '"original_value": 2, "reviewed_value": 1, "decision": "CORRECTED", '
                '"reason": "La fuente original muestra CANTIDAD: 1.", '
                '"confidence": 0.9, "source_ids": ["source-1"]}'
            ),
        ]
    )
    debug_capture = GeminiFullPipelineDebugCapture()

    result = provider.extract_with_discovery_from_files([pdf], debug_capture=debug_capture)

    assert result.elements[0].quantity is not None
    assert result.elements[0].quantity.value == 1
    assert result.elements[0].geometry is not None
    assert result.elements[0].geometry.raw_type == "arched"
    assert len(provider._provider._client.models.calls) == 4
    review_call = provider._provider._client.models.calls[-1]
    review_prompt = review_call["contents"][0].text
    assert review_call["contents"][1].inline_data.mime_type == "application/pdf"
    assert '"source_id": "source-1"' in review_prompt
    assert '"page_number": null' in review_prompt
    assert debug_capture.enrichment_debug is not None
    assert debug_capture.enrichment_debug.semantic_review_decisions[0].review_called is True
    assert (
        debug_capture.enrichment_debug.semantic_review_decisions[0].trigger_reason
        == MODEL_EVIDENCE_QUANTITY
    )


def test_full_pipeline_semantic_review_accepts_single_item_list_response(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-08"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-08", "quantity": 1, '
                '"status": "explicit", "confidence": 0.95, '
                '"functional_type_raw": "PROJECTING", '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "V-08 CANTIDAD: 5", "page_number": 1, '
                '"region": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}]}'
                "]}",
            ),
            _FakeResponse(
                '[{"temporary_id": "d-1", "reference": "V-08", "field": "quantity", '
                '"original_value": 1, "reviewed_value": 5, "decision": "CORRECTED", '
                '"reason": "La fuente original muestra CANTIDAD: 5.", '
                '"confidence": 0.9, "source_ids": ["source-1"]}]'
            ),
        ]
    )

    result = provider.extract_with_discovery_from_files([pdf])

    assert result.elements[0].quantity is not None
    assert result.elements[0].quantity.value == 5
    assert result.elements[0].functional_type is not None
    assert result.elements[0].functional_type.normalized == "PROJECTING"
    assert result.elements[0].functional_type.status == ExtractionStatus.EXPLICIT
    assert result.elements[0].functional_type.confidence == 0.95


def test_full_pipeline_semantic_review_invalid_shape_continues_as_unresolved(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-06"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-06", "quantity": 1, '
                '"status": "explicit", "confidence": 0.95, '
                '"functional_type_raw": "PROJECTING", '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "V-06 CANTIDAD: 1"}]}'
                "]}",
            ),
            _FakeResponse("[]"),
        ]
    )

    result = provider.extract_with_discovery_from_files([pdf])

    assert result.elements[0].quantity is not None
    assert result.elements[0].quantity.value == 1
    assert result.elements[0].quantity.status == ExtractionStatus.INFERRED
    assert result.elements[0].functional_type is not None
    assert result.elements[0].functional_type.normalized == "PROJECTING"
    assert result.elements[0].functional_type.status == ExtractionStatus.EXPLICIT


def test_full_pipeline_semantic_review_uses_region_locator_and_selected_source(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "planos.pdf"
    image = tmp_path / "detalle.png"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\r"
        + b"IHDR"
        + (1000).to_bytes(4, "big")
        + (1000).to_bytes(4, "big")
    )
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-08"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-08", "quantity": 1, '
                '"status": "explicit", "confidence": 0.95, '
                '"evidence": [{"source_id": "source-2", "type": "visual", '
                '"text": "V-08 CANTIDAD: 5", '
                '"region": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}]}'
                "]}",
            ),
            _FakeResponse(
                '{"temporary_id": "d-1", "reference": "V-08", "field": "quantity", '
                '"original_value": 1, "reviewed_value": 5, "decision": "CORRECTED", '
                '"reason": "La region muestra CANTIDAD: 5.", '
                '"confidence": 0.9, "source_ids": ["source-2"]}'
            ),
        ]
    )
    debug_capture = GeminiFullPipelineDebugCapture()

    result = provider.extract_with_discovery_from_files(
        [pdf, image],
        debug_capture=debug_capture,
    )

    assert result.elements[0].quantity is not None
    assert result.elements[0].quantity.value == 5
    review_call = provider._provider._client.models.calls[-1]
    review_prompt = review_call["contents"][0].text
    assert len(review_call["contents"]) == 2
    assert review_call["contents"][1].inline_data.mime_type == "image/png"
    assert '"source_id": "source-2"' in review_prompt
    assert '"region": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}' in review_prompt
    assert "UNTRUSTED_FIRST_PASS_VALUE: 1" in review_prompt
    assert "V-08 CANTIDAD: 5" not in review_prompt
    assert "First-pass quantity evidence text omitted" in review_prompt
    assert debug_capture.enrichment_debug is not None
    trace = debug_capture.enrichment_debug.semantic_review_decisions[0]
    assert trace.locator_used == "NUMERIC_TRACE"
    assert trace.source_id == "source-2"
    assert trace.region == Region(x=0.1, y=0.2, width=0.3, height=0.4)
    assert trace.confidence == 0.9


def test_full_pipeline_semantic_review_uses_xlsx_cell_locator_when_review_needed(
    tmp_path: Path,
) -> None:
    workbook_path = _xlsx_with_quantity(tmp_path, "V-08", 5)
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-08"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-08", "quantity": 5, '
                '"status": "explicit", "confidence": 0.95, '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "V-08 CANTIDAD: 5 N.P_5"}]}'
                "]}",
            ),
            _FakeResponse(
                '{"temporary_id": "d-1", "reference": "V-08", "field": "quantity", '
                '"original_value": 5, "reviewed_value": 5, "decision": "CONFIRMED", '
                '"reason": "La celda de cantidad muestra 5.", '
                '"confidence": 0.95, "source_ids": ["source-1"]}'
            ),
        ]
    )
    debug_capture = GeminiFullPipelineDebugCapture()

    result = provider.extract_with_discovery_from_files(
        [workbook_path],
        debug_capture=debug_capture,
    )

    assert result.elements[0].quantity is not None
    assert result.elements[0].quantity.value == 5
    review_call = provider._provider._client.models.calls[-1]
    review_prompt = review_call["contents"][0].text
    assert '"sheet_name": "Cantidades"' in review_prompt
    assert '"cell_range": "B2"' in review_prompt
    assert "NUMERIC ROLE COLLISIONS" in review_prompt
    assert '"detected": true' in review_prompt
    assert "LEVEL" in review_prompt
    assert debug_capture.enrichment_debug is not None
    trace = debug_capture.enrichment_debug.semantic_review_decisions[0]
    assert trace.locator_used == "SOURCE_INDEPENDENT"
    assert trace.sheet_name == "Cantidades"
    assert trace.cell_range == "B2"
    assert trace.numeric_collision_detected is True
    assert "LEVEL" in trace.collision_roles
    assert trace.confirmation_cross_check_called is True
    assert trace.confirmation_support == "STRONG"
    assert trace.effective_decision == "CONFIRMED"


def test_full_pipeline_skips_second_call_for_source_model_agreement(tmp_path: Path) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-09"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-09", "quantity": 5, '
                '"status": "explicit", '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "V-09 CANTIDAD: 5"}]}'
                "]}",
            ),
        ]
    )

    provider.extract_with_discovery_from_files([_xlsx_with_quantity(tmp_path, "V-09", 5)])

    assert len(provider._provider._client.models.calls) == 3
