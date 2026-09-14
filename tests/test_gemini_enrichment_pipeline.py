from pathlib import Path

from openpyxl import Workbook

import app.providers.gemini_extraction as provider_module
from app.models.common import ExtractionStatus
from app.models.gemini_discovery import GeminiDiscoveryResult, GeminiElementDiscovery
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentComponent,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentGlass,
    GeminiEnrichmentMeasurement,
    GeminiEnrichmentNamedItem,
    GeminiEnrichmentResult,
)
from app.models.requirement import ExtractionMetadata, Requirement, TokenUsage
from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini_extraction import (
    GeminiEnrichmentDebugCapture,
    GeminiEnrichmentParseError,
    GeminiExtractionProvider,
    GeminiFullPipelineDebugCapture,
)
from app.services.extraction_prompt import ELEMENT_SCOPE_PROMPT
from app.services.gemini_enrichment_pipeline import (
    build_discovery_batches,
    enrichment_to_gemini_extraction,
    merge_enrichment_batches,
)
from app.services.gemini_extraction_mapper import (
    map_gemini_extraction_to_requirement_extraction,
)
from app.services.inventory_reconciliation import (
    CONTEXT_INCOMPLETE_REASON,
    CONTEXT_LABEL,
    CONTEXT_LABEL_NOT_IDENTITY_REASON,
    DIFFERENT_CONTEXT_REASON,
    DUPLICATE_REFERENCE_REASON,
    GLASS_EXPLICIT_CONFLICT,
    GLASS_SCOPE_DOCUMENT_GENERAL,
    GLASS_SCOPE_ITEM_LOCAL,
    GLASS_SCOPE_SECTION_LEVEL,
    GLASS_SCOPE_UNKNOWN,
    ORPHAN_REFERENCE_REASON,
    PROFILE_EXPLICIT_CONFLICT,
    SOURCE_CONFLICT_REASON,
    InventoryDecision,
    reconcile_inventory_candidates,
)
from app.services.inventory_trace import (
    InventoryDebugTrace,
    InventoryElementTrace,
    enrichment_inventory_elements,
    final_inventory_elements,
)
from app.services.item_count_diagnostics import (
    DROPPED_AS_ORPHAN,
    DROPPED_BY_MAPPER,
    DROPPED_BY_SCOPE,
    DUPLICATE_DISCOVERY,
    DUPLICATE_ENRICHMENT,
    MERGED_IN_RECONCILIATION,
    MISSING_AT_DISCOVERY,
    ORPHAN_WITH_TECHNICAL_SUPPORT,
    RECONCILIATION_UNDERMERGE,
    build_item_count_diagnostic_report,
)
from app.services.numeric_trace import build_numeric_resolution_trace
from app.services.region_sanitizer import REGION_NORMALIZED


class _UploadedFile:
    def __init__(self, uri: str, mime_type: str) -> None:
        self.uri = uri
        self.mime_type = mime_type


class _FakeFilesClient:
    def __init__(self) -> None:
        self.uploads = []

    def upload(self, *, file: Path, config):
        self.uploads.append((Path(file), config))
        return _UploadedFile(
            uri=f"gemini://{len(self.uploads)}",
            mime_type=config.mime_type,
        )


class _UsageMetadata:
    def __init__(self, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self.prompt_token_count = input_tokens
        self.candidates_token_count = output_tokens
        self.total_token_count = total_tokens


class _FakeResponse:
    def __init__(self, text: str, usage_metadata: _UsageMetadata | None = None) -> None:
        self.text = text
        self.usage_metadata = usage_metadata


class _FakeModelsClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake Gemini response left.")
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.files = _FakeFilesClient()
        self.models = _FakeModelsClient(responses)


class _FakeSettings:
    gemini_enrichment_batch_size = 8


class _FakeProvider:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._client = _FakeClient(responses)
        self.model = "gemini-test"


def _provider_with_responses(responses: list[_FakeResponse]) -> GeminiExtractionProvider:
    provider = GeminiExtractionProvider.__new__(GeminiExtractionProvider)
    provider._provider = _FakeProvider(responses)
    provider._settings = _FakeSettings()
    return provider


def _pdf(tmp_path: Path, name: str = "planos.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.7\ncontent")
    return path


def _png(tmp_path: Path, name: str = "plano.png", width: int = 1000, height: int = 1000) -> Path:
    path = tmp_path / name
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\r"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )
    return path


def _xlsx_with_quantity(
    tmp_path: Path,
    reference: str,
    quantity: int,
    name: str = "cuadro.xlsx",
) -> Path:
    path = tmp_path / name
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Cantidades"
    sheet.append(["REF", "CANTIDAD", "N.P"])
    sheet.append([reference, quantity, 5])
    workbook.save(path)
    return path


def _discovery(count: int) -> GeminiDiscoveryResult:
    return GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(
                temporary_id=f"d-{index}",
                reference=f"V-{index:02d}",
            )
            for index in range(1, count + 1)
        ]
    )


def _enrichment_response(*temporary_ids: str, usage: tuple[int, int, int] | None = None):
    elements = ", ".join(
        f'{{"temporary_id": "{temporary_id}", "reference": "{temporary_id}", "quantity": 1}}'
        for temporary_id in temporary_ids
    )
    usage_metadata = _UsageMetadata(*usage) if usage else None
    return _FakeResponse(f'{{"elements": [{elements}]}}', usage_metadata)


def _scope_response(*items: tuple[str, str], usage: tuple[int, int, int] | None = None):
    elements = ", ".join(
        f'{{"temporary_id": "{temporary_id}", "scope": "{scope}"}}' for temporary_id, scope in items
    )
    usage_metadata = _UsageMetadata(*usage) if usage else None
    return _FakeResponse(f'{{"elements": [{elements}]}}', usage_metadata)


def _scope_response_with_sources(
    temporary_id: str,
    scope: str,
    source_id: str,
) -> _FakeResponse:
    return _FakeResponse(
        '{"elements": ['
        f'{{"temporary_id": "{temporary_id}", "scope": "{scope}", '
        f'"evidence_source_ids": ["{source_id}"]}}'
        "]}"
    )


def test_parse_enrichment_accepts_direct_object() -> None:
    result, events = provider_module._parse_gemini_enrichment_response(
        _FakeResponse('{"elements": [{"temporary_id": "d-1", "quantity": 1}]}')
    )

    assert events == []
    assert result.elements[0].temporary_id == "d-1"
    assert result.elements[0].quantity == 1


def test_parse_enrichment_accepts_singleton_wrapper_list() -> None:
    result, _ = provider_module._parse_gemini_enrichment_response(
        _FakeResponse('[{"elements": [{"temporary_id": "d-1"}]}]')
    )

    assert [element.temporary_id for element in result.elements] == ["d-1"]


def test_parse_enrichment_accepts_direct_elements_list() -> None:
    result, _ = provider_module._parse_gemini_enrichment_response(
        _FakeResponse('[{"temporary_id": "d-1"}, {"temporary_id": "d-2"}]')
    )

    assert [element.temporary_id for element in result.elements] == ["d-1", "d-2"]


def test_parse_enrichment_rejects_empty_list_with_diagnostic() -> None:
    try:
        provider_module._parse_gemini_enrichment_response(
            _FakeResponse("[]"),
            batch_number=2,
            requested_temporary_ids=["d-1", "d-2"],
        )
    except GeminiEnrichmentParseError as exc:
        diagnostic = exc.diagnostic
    else:
        raise AssertionError("Expected GeminiEnrichmentParseError.")

    assert diagnostic.stage == "ENRICHMENT"
    assert diagnostic.batch_number == 2
    assert diagnostic.requested_temporary_ids == ["d-1", "d-2"]
    assert diagnostic.response_top_level_shape == "list[len=0]"


def test_parse_enrichment_rejects_multiple_wrapper_list() -> None:
    try:
        provider_module._parse_gemini_enrichment_response(
            _FakeResponse('[{"elements": []}, {"elements": []}]')
        )
    except GeminiEnrichmentParseError as exc:
        diagnostic = exc.diagnostic
    else:
        raise AssertionError("Expected GeminiEnrichmentParseError.")

    assert diagnostic.response_top_level_shape == "list[len=2]"


def test_parse_enrichment_preserves_validation_error_for_elements_type() -> None:
    try:
        provider_module._parse_gemini_enrichment_response(
            _FakeResponse('{"elements": "not-a-list"}')
        )
    except GeminiEnrichmentParseError as exc:
        errors = exc.diagnostic.validation_error_summary
    else:
        raise AssertionError("Expected GeminiEnrichmentParseError.")

    assert errors[0]["loc"] == ["elements"]
    assert errors[0]["type"] == "list_type"


def test_parse_enrichment_preserves_validation_error_field_path() -> None:
    try:
        provider_module._parse_gemini_enrichment_response(
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "panel_count": 0}]}')
        )
    except GeminiEnrichmentParseError as exc:
        errors = exc.diagnostic.validation_error_summary
    else:
        raise AssertionError("Expected GeminiEnrichmentParseError.")

    assert errors[0]["loc"] == ["elements", 0, "panel_count"]


def test_parse_enrichment_accepts_fenced_json() -> None:
    result, _ = provider_module._parse_gemini_enrichment_response(
        _FakeResponse('```json\n{"elements": [{"temporary_id": "d-1"}]}\n```')
    )

    assert result.elements[0].temporary_id == "d-1"


def test_parse_enrichment_rejects_prose_plus_json() -> None:
    try:
        provider_module._parse_gemini_enrichment_response(
            _FakeResponse('Here is your result:\n{"elements": [{"temporary_id": "d-1"}]}')
        )
    except GeminiEnrichmentParseError as exc:
        diagnostic = exc.diagnostic
    else:
        raise AssertionError("Expected GeminiEnrichmentParseError.")

    assert diagnostic.response_top_level_shape == "invalid_json"


def test_parse_enrichment_accepts_one_layer_json_string() -> None:
    result, _ = provider_module._parse_gemini_enrichment_response(
        _FakeResponse('"{\\"elements\\": [{\\"temporary_id\\": \\"d-1\\"}]}"')
    )

    assert result.elements[0].temporary_id == "d-1"


def test_parse_enrichment_rejects_nested_json_string() -> None:
    try:
        provider_module._parse_gemini_enrichment_response(
            _FakeResponse('"\\"{\\\\\\"elements\\\\\\": []}\\""')
        )
    except GeminiEnrichmentParseError as exc:
        diagnostic = exc.diagnostic
    else:
        raise AssertionError("Expected GeminiEnrichmentParseError.")

    assert "json_string" in diagnostic.response_top_level_shape


def test_parse_enrichment_diagnostic_preview_is_capped_and_redacted() -> None:
    api_key = "AIza" + ("A" * 36)
    text = f'{{"foo": "{api_key}", "blob": "' + ("x" * 3000) + '"}'

    try:
        provider_module._parse_gemini_enrichment_response(_FakeResponse(text))
    except GeminiEnrichmentParseError as exc:
        diagnostic = exc.diagnostic
    else:
        raise AssertionError("Expected GeminiEnrichmentParseError.")

    assert api_key not in diagnostic.raw_text_prefix
    assert api_key not in diagnostic.raw_text_suffix
    assert len(diagnostic.raw_text_prefix) <= 800
    assert len(diagnostic.raw_text_suffix) <= 800


def test_batching_zero_elements() -> None:
    assert build_discovery_batches(GeminiDiscoveryResult(), 8) == []


def test_batching_one_element() -> None:
    batches = build_discovery_batches(_discovery(1), 8)

    assert len(batches) == 1
    assert [item.temporary_id for item in batches[0]] == ["d-1"]


def test_batching_exact_size() -> None:
    batches = build_discovery_batches(_discovery(8), 8)

    assert len(batches) == 1
    assert len(batches[0]) == 8


def test_batching_with_remainder() -> None:
    batches = build_discovery_batches(_discovery(10), 8)

    assert [len(batch) for batch in batches] == [8, 2]


def test_enrichment_preserves_order_and_repeated_references(tmp_path: Path) -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(temporary_id="a", reference="PV-02"),
            GeminiElementDiscovery(temporary_id="b", reference="PV-02"),
            GeminiElementDiscovery(temporary_id="c", reference="V-08"),
        ]
    )
    provider = _provider_with_responses([_enrichment_response("a", "b", "c")])

    result = provider.enrich_discoveries_from_files([_pdf(tmp_path)], discovery)

    assert [item.temporary_id for item in result.elements] == ["a", "b", "c"]
    assert [item.reference for item in result.elements] == ["a", "b", "c"]


def test_enrichment_duplicate_temporary_id_detected(tmp_path: Path) -> None:
    discovery = _discovery(1)
    provider = _provider_with_responses([_enrichment_response("d-1", "d-1")])

    result = provider.enrich_discoveries_from_files([_pdf(tmp_path)], discovery)

    assert any("duplicate temporary_id" in warning for warning in result.warnings)


def test_duplicate_discovery_temporary_id_detected(tmp_path: Path) -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(temporary_id="same", reference="V-01"),
            GeminiElementDiscovery(temporary_id="same", reference="V-02"),
        ]
    )
    provider = _provider_with_responses([_enrichment_response("same")])

    result = provider.enrich_discoveries_from_files([_pdf(tmp_path)], discovery)

    assert [item.temporary_id for item in result.elements] == ["same", "same"]
    assert any("duplicate discovery temporary_id" in warning for warning in result.warnings)


def test_enrichment_missing_temporary_id_warns_and_preserves_discovery(tmp_path: Path) -> None:
    discovery = _discovery(2)
    provider = _provider_with_responses([_enrichment_response("d-1")])

    result = provider.enrich_discoveries_from_files([_pdf(tmp_path)], discovery)

    assert [item.temporary_id for item in result.elements] == ["d-1", "d-2"]
    assert result.elements[1].reference == "V-02"
    assert result.elements[1].missing_or_unknown == ["technical_enrichment"]
    assert any("missing enrichment" in warning for warning in result.warnings)


def test_enrichment_merges_multiple_batches_and_accumulates_usage(tmp_path: Path) -> None:
    provider = _provider_with_responses(
        [
            _enrichment_response("d-1", "d-2", usage=(10, 5, 15)),
            _enrichment_response("d-3", usage=(20, 7, 27)),
        ]
    )
    debug_capture = GeminiEnrichmentDebugCapture()

    result = provider.enrich_discoveries_from_files(
        [_pdf(tmp_path)],
        _discovery(3),
        batch_size=2,
        debug_capture=debug_capture,
    )

    assert [item.temporary_id for item in result.elements] == ["d-1", "d-2", "d-3"]
    assert result.usage == TokenUsage(input_tokens=30, output_tokens=12, total_tokens=42)
    assert len(debug_capture.batch_results) == 2


def test_full_pipeline_mapper_receives_all_items_and_reuses_inline_parts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse(
                '{"elements": [{"temporary_id": "a", "reference": "PV-02"}, '
                '{"temporary_id": "b", "reference": "PV-02"}]}',
                _UsageMetadata(11, 4, 15),
            ),
            _scope_response(
                ("a", "in_scope_full"),
                ("b", "in_scope_partial"),
                usage=(8, 3, 11),
            ),
            _enrichment_response("a", usage=(10, 5, 15)),
            _enrichment_response("b", usage=(12, 6, 18)),
        ]
    )
    mapped_result = RequirementExtraction(
        requirement=Requirement(),
        extraction_metadata=ExtractionMetadata(),
    )
    mapper_calls = []

    def fake_mapper(
        gemini_extraction,
        *,
        model_provider,
        model,
        default_source_id="text-input",
        allowed_source_ids=None,
    ):
        mapper_calls.append(gemini_extraction)
        return mapped_result

    monkeypatch.setattr(
        provider_module,
        "map_gemini_extraction_to_requirement_extraction",
        fake_mapper,
    )
    debug_capture = GeminiFullPipelineDebugCapture()

    result = provider.extract_with_discovery_from_files(
        [_pdf(tmp_path), _pdf(tmp_path, "foto.pdf")],
        project_id="project",
        requirement_id="requirement",
        batch_size=1,
        debug_capture=debug_capture,
    )

    assert result is mapped_result
    assert len(mapper_calls[0].elements) == 2
    assert provider._provider._client.files.uploads == []
    assert len(provider._provider._client.models.calls) == 4
    assert all(len(call["contents"]) == 3 for call in provider._provider._client.models.calls)
    assert all(
        call["contents"][1].inline_data.mime_type == "application/pdf"
        and call["contents"][2].inline_data.mime_type == "application/pdf"
        for call in provider._provider._client.models.calls
    )
    assert result.extraction_metadata.token_usage == TokenUsage(
        input_tokens=41,
        output_tokens=18,
        total_tokens=59,
    )
    assert debug_capture.batch_count == 2
    assert debug_capture.scope is not None


def test_full_pipeline_scope_selects_only_allowed_items(tmp_path: Path, monkeypatch) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "full", "reference": "A"}, '
                '{"temporary_id": "partial", "reference": "A"}, '
                '{"temporary_id": "uncertain", "reference": "A"}, '
                '{"temporary_id": "out", "reference": "A"}'
                "]}"
            ),
            _scope_response(
                ("full", "in_scope_full"),
                ("partial", "in_scope_partial"),
                ("uncertain", "uncertain"),
                ("out", "out_of_scope"),
            ),
            _enrichment_response("full", "partial", "uncertain"),
        ]
    )
    mapped_result = RequirementExtraction(
        requirement=Requirement(),
        extraction_metadata=ExtractionMetadata(),
    )
    mapper_calls = []

    def fake_mapper(
        gemini_extraction,
        *,
        model_provider,
        model,
        default_source_id="text-input",
        allowed_source_ids=None,
    ):
        mapper_calls.append(gemini_extraction)
        return mapped_result

    monkeypatch.setattr(
        provider_module,
        "map_gemini_extraction_to_requirement_extraction",
        fake_mapper,
    )

    original_file = _pdf(tmp_path, "fachada_nin\u0303o.pdf")

    provider.extract_with_discovery_from_files([original_file], batch_size=8)

    assert [element.id for element in mapper_calls[0].elements] == [
        "full",
        "partial",
        "uncertain",
    ]
    assert original_file.exists() is True
    assert provider._provider._client.files.uploads == []
    part = provider._provider._client.models.calls[0]["contents"][1]
    assert part.inline_data.mime_type == "application/pdf"
    assert part.inline_data.data == b"%PDF-1.7\ncontent"


def test_scope_missing_defaults_to_uncertain_and_duplicate_is_warned(tmp_path: Path) -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(temporary_id="a", reference="V-01"),
            GeminiElementDiscovery(temporary_id="b", reference="V-01"),
        ]
    )
    provider = _provider_with_responses(
        [_scope_response(("a", "in_scope_full"), ("a", "out_of_scope"))]
    )

    scope = provider.classify_scope_from_files([_pdf(tmp_path)], discovery)

    assert [item.temporary_id for item in scope.elements] == ["a", "b"]
    assert [item.scope.value for item in scope.elements] == ["in_scope_full", "uncertain"]
    assert any("duplicate scope temporary_id" in warning for warning in scope.warnings)
    assert any("missing scope" in warning for warning in scope.warnings)


def test_discovery_and_scope_source_ids_survive_debug_models(tmp_path: Path) -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(
                temporary_id="a",
                reference="V-01",
                source_ids=["source-1", "source-3"],
            )
        ]
    )
    provider = _provider_with_responses(
        [_scope_response_with_sources("a", "in_scope_full", "source-3")]
    )

    scope = provider.classify_scope_from_files([_pdf(tmp_path)], discovery)

    assert discovery.elements[0].source_ids == ["source-1", "source-3"]
    assert scope.elements[0].evidence_source_ids == ["source-3"]


def test_scope_prompt_requires_positive_exclusion_for_out_of_scope() -> None:
    assert "OUT_OF_SCOPE REQUIERE EVIDENCIA POSITIVA DE EXCLUSION" in ELEMENT_SCOPE_PROMPT
    assert (
        "La mera ausencia de evidencia de vidrio NO equivale a evidencia de ausencia"
        in ELEMENT_SCOPE_PROMPT
    )
    assert "Prioriza evitar falsos negativos" in ELEMENT_SCOPE_PROMPT


def test_scope_prompt_does_not_hardcode_conceptual_example_names() -> None:
    prompt = ELEMENT_SCOPE_PROMPT.casefold()

    for term in (
        "pergola",
        "p\u00e9rgola",
        "marquesina",
        "cubierta",
        "puerta",
        "bbq",
        "ventana",
    ):
        assert term not in prompt


def test_scope_conservative_conceptual_cases_are_preserved(tmp_path: Path) -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(
                temporary_id="opaque-metal-door",
                name="puerta metalica completamente opaca",
                source_hint="descripcion explicita sin vidrio",
            ),
            GeminiElementDiscovery(
                temporary_id="metal-furniture",
                name="mueble metalico",
                source_hint="mobiliario sin participacion de vidrio",
            ),
            GeminiElementDiscovery(
                temporary_id="mixed-architectural-glass",
                name="elemento arquitectonico mixto",
                source_hint="incluye vidrio explicito",
            ),
            GeminiElementDiscovery(
                temporary_id="architectural-uncertain",
                name="conjunto arquitectonico",
                source_hint="sin vidrio explicito y sin evidencia de exclusion",
            ),
            GeminiElementDiscovery(
                temporary_id="no-glass-word-only",
                name="elemento sin palabra clave",
                source_hint="solo ausencia de la palabra vidrio",
            ),
        ]
    )
    provider = _provider_with_responses(
        [
            _scope_response(
                ("opaque-metal-door", "out_of_scope"),
                ("metal-furniture", "out_of_scope"),
                ("mixed-architectural-glass", "in_scope_partial"),
                ("architectural-uncertain", "uncertain"),
                ("no-glass-word-only", "uncertain"),
            )
        ]
    )

    scope = provider.classify_scope_from_files([_pdf(tmp_path)], discovery)

    assert {item.temporary_id: item.scope.value for item in scope.elements} == {
        "opaque-metal-door": "out_of_scope",
        "metal-furniture": "out_of_scope",
        "mixed-architectural-glass": "in_scope_partial",
        "architectural-uncertain": "uncertain",
        "no-glass-word-only": "uncertain",
    }


def test_uncertain_scope_continues_to_enrichment(tmp_path: Path, monkeypatch) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse(
                '{"elements": [{"temporary_id": "no-glass-word-only", '
                '"name": "elemento sin palabra clave"}]}'
            ),
            _scope_response(("no-glass-word-only", "uncertain")),
            _enrichment_response("no-glass-word-only"),
        ]
    )
    mapped_result = RequirementExtraction(
        requirement=Requirement(),
        extraction_metadata=ExtractionMetadata(),
    )
    mapper_calls = []

    def fake_mapper(
        gemini_extraction,
        *,
        model_provider,
        model,
        default_source_id="text-input",
        allowed_source_ids=None,
    ):
        mapper_calls.append(gemini_extraction)
        return mapped_result

    monkeypatch.setattr(
        provider_module,
        "map_gemini_extraction_to_requirement_extraction",
        fake_mapper,
    )

    provider.extract_with_discovery_from_files([_pdf(tmp_path)])

    assert [element.id for element in mapper_calls[0].elements] == ["no-glass-word-only"]


def test_merge_helper_preserves_missing_items_without_provider() -> None:
    merged = merge_enrichment_batches(_discovery(1), [GeminiEnrichmentResult()])

    assert len(merged.elements) == 1
    assert merged.elements[0].temporary_id == "d-1"
    assert any("missing enrichment" in warning for warning in merged.warnings)


def test_enrichment_to_gemini_extraction_preserves_context_as_structured_items() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="item-1",
                    occurrence_context="Habitacion de servicio (Page 1, Detail 1)",
                    variant_context="Alternativa con vidrio claro",
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-1",
                            type="visual",
                            text="VIDRIO 6mm",
                            page_number=2,
                            region={
                                "x": 0.15,
                                "y": 0.25,
                                "width": 0.35,
                                "height": 0.45,
                            },
                        )
                    ],
                    evidence_notes=["Nota: TODOS LOS VIDRIOS SON DE ESPESOR DE 6mm"],
                    status=ExtractionStatus.EXPLICIT,
                    confidence=0.8,
                )
            ]
        ),
    )
    element = extraction.elements[0]

    assert element.notes is None
    assert element.occurrences[0].location == "Habitacion de servicio (Page 1, Detail 1)"
    assert element.variants[0].label == "Alternativa con vidrio claro"
    assert element.evidence_items[0].source_id == "source-1"
    assert element.evidence_items[0].type == "visual"
    assert element.evidence_items[0].page_number == 2
    assert element.evidence_items[0].region is not None
    assert element.evidence_items[0].region.x == 0.15
    assert element.evidence == "Nota: TODOS LOS VIDRIOS SON DE ESPESOR DE 6mm"


def test_enrichment_to_gemini_extraction_preserves_structured_signals() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="item-1",
                    functional_type_raw="puerta corrediza",
                    operation_raw="corrediza",
                    panel_count=4,
                    movable_panel_count=2,
                    fixed_panel_count=2,
                    modulation_raw="OXXO",
                    opening_direction_raw="izquierda",
                    special_features=["POCKET"],
                    geometry_type_raw="estructura en L",
                    geometry_raw="estructura en L",
                    configuration_raw="puerta corrediza OXXO",
                    components=[
                        GeminiEnrichmentComponent(
                            name="fijo inferior",
                            type="panel",
                            geometry_raw="rectangular",
                            configuration_raw="fijo",
                            finish_raw="negro",
                            accessories=[],
                        )
                    ],
                )
            ]
        ),
    )
    element = extraction.elements[0]
    component = element.components[0]

    assert element.functional_type == "puerta corrediza"
    assert element.operation == "corrediza"
    assert element.panel_count == 4
    assert element.movable_panel_count == 2
    assert element.fixed_panel_count == 2
    assert element.modulation == "OXXO"
    assert element.opening_direction == "izquierda"
    assert element.special_features == ["POCKET"]
    assert element.geometry_type == "estructura en L"
    assert element.geometry == "estructura en L"
    assert element.configuration == "puerta corrediza OXXO"
    assert component.geometry == "rectangular"
    assert component.configuration == "fijo"
    assert component.finish == "negro"


def test_quantity_metadata_is_field_local_in_final_mapping() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="item-1",
                    reference="V-01",
                    quantity=1,
                    quantity_status=ExtractionStatus.INFERRED,
                    quantity_confidence=0.5,
                    quantity_notes="NO_SOURCE_GROUNDING_AVAILABLE",
                    functional_type_raw="PROJECTING",
                    measurements=[
                        GeminiEnrichmentMeasurement(
                            type="width",
                            raw_label="ANCHO",
                            value=2.8,
                            unit="m",
                            status=ExtractionStatus.EXPLICIT,
                            confidence=1.0,
                        ),
                        GeminiEnrichmentMeasurement(
                            type="height",
                            raw_label="ALTO",
                            value=2.9,
                            unit="m",
                            status=ExtractionStatus.EXPLICIT,
                            confidence=1.0,
                        ),
                    ],
                    geometry_raw="rectangular",
                    glass=[
                        GeminiEnrichmentGlass(
                            type="templado",
                            thickness="6 mm",
                            status=ExtractionStatus.EXPLICIT,
                            confidence=0.95,
                        )
                    ],
                    finish_raw="negro pintura al horno",
                    components=[
                        GeminiEnrichmentComponent(
                            name="nave proyectante",
                            role="PROJECTING",
                            status=ExtractionStatus.EXPLICIT,
                            confidence=0.95,
                        )
                    ],
                    status=ExtractionStatus.EXPLICIT,
                    confidence=0.95,
                )
            ]
        ),
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.quantity is not None
    assert element.quantity.value == 1
    assert element.quantity.status == ExtractionStatus.INFERRED
    assert element.quantity.confidence == 0.5
    assert element.functional_type is not None
    assert element.functional_type.normalized == "PROJECTING"
    assert element.functional_type.status == ExtractionStatus.EXPLICIT
    assert element.functional_type.confidence == 0.95
    assert element.geometry is not None
    assert element.geometry.status == ExtractionStatus.EXPLICIT
    assert element.geometry.confidence == 0.95
    assert element.measurements[0].status == ExtractionStatus.EXPLICIT
    assert element.measurements[0].confidence == 1.0
    assert element.measurements[1].status == ExtractionStatus.EXPLICIT
    assert element.measurements[1].confidence == 1.0
    assert element.glass[0].status == ExtractionStatus.EXPLICIT
    assert element.glass[0].confidence == 0.95
    assert element.finish is not None
    assert element.finish.status == ExtractionStatus.EXPLICIT
    assert element.finish.confidence == 0.95
    assert element.components[0].role is not None
    assert element.components[0].role.normalized == "PROJECTING"
    assert element.components[0].role.status == ExtractionStatus.EXPLICIT
    assert element.components[0].role.confidence == 0.95


def test_inventory_reconciliation_ignores_orphan_reference_tag() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[GeminiElementEnrichment(temporary_id="tag", reference="PV-07")]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert result.elements == []
    assert decisions[0].action == "DROP_AS_NON_COMMERCIAL"
    assert decisions[0].reason == ORPHAN_REFERENCE_REASON
    assert any(ORPHAN_REFERENCE_REASON in warning for warning in result.warnings)


def test_inventory_reconciliation_keeps_table_row_with_dimensions_and_quantity() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="row",
                reference="V-01",
                quantity=2,
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=1000, unit="mm"),
                    GeminiEnrichmentMeasurement(type="height", value=2000, unit="mm"),
                ],
            )
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["row"]
    assert decisions[0].action == "KEEP"


def test_inventory_reconciliation_keeps_drawing_with_geometry_and_operation() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="drawing",
                reference="V-02",
                operation_raw="corrediza",
                geometry_type_raw="rectangular",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=1200, unit="mm"),
                    GeminiEnrichmentMeasurement(type="height", value=1800, unit="mm"),
                ],
            )
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["drawing"]
    assert decisions[0].action == "KEEP"


def test_inventory_reconciliation_merges_canonical_duplicate_references() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="table",
                    reference="V-01",
                    quantity=1,
                    measurements=[
                        GeminiEnrichmentMeasurement(type="width", value=1000, unit="mm"),
                        GeminiEnrichmentMeasurement(type="height", value=2000, unit="mm"),
                    ],
                ),
                GeminiElementEnrichment(
                    temporary_id="drawing",
                    reference="V-1",
                    operation_raw="proyectante",
                    geometry_type_raw="rectangular",
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-1",
                            text="V-1 elevacion 1000 x 2000",
                            page_number=1,
                        )
                    ],
                ),
            ]
        ),
    )

    assert len(extraction.elements) == 1
    assert extraction.elements[0].reference == "V-01"
    assert extraction.elements[0].quantity == 1
    assert extraction.elements[0].operation == "proyectante"
    assert extraction.elements[0].evidence_items[0].text == "V-1 elevacion 1000 x 2000"
    assert DUPLICATE_REFERENCE_REASON in (extraction.notes or "")


def test_inventory_reconciliation_keeps_same_reference_with_distinct_commercial_context() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="sotano-v-01",
                reference="V-01",
                quantity=2,
                occurrence_context="SOTANO",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=0.9, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=1.45, unit="m"),
                    GeminiEnrichmentMeasurement(type="area", value=1.305, unit="m2"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-1-v-01",
                reference="V-01",
                quantity=5,
                occurrence_context="Nivel 1",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=5.8, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.1, unit="m"),
                    GeminiEnrichmentMeasurement(type="area", value=12.18, unit="m2"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-2-v-01",
                reference="V-01",
                quantity=3,
                occurrence_context="nivel_2",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=0.9, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.5, unit="m"),
                    GeminiEnrichmentMeasurement(type="area", value=2.25, unit="m2"),
                ],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == [
        "sotano-v-01",
        "nivel-1-v-01",
        "nivel-2-v-01",
    ]
    assert [item.quantity for item in result.elements] == [2, 5, 3]
    assert all(decision.action == "KEEP" for decision in decisions)
    assert {decision.reason for decision in decisions} == {DIFFERENT_CONTEXT_REASON}
    assert {decision.commercial_context for decision in decisions} == {
        "sotano",
        "nivel_1",
        "nivel_2",
    }


def test_inventory_reconciliation_keeps_repeated_pv_reference_by_commercial_context() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="sotano-pv-01",
                reference="PV-01",
                quantity=5,
                occurrence_context="SOTANO",
                measurements=[GeminiEnrichmentMeasurement(type="width", value=2.3, unit="m")],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-1-pv-01",
                reference="PV-01",
                quantity=5,
                occurrence_context="Nivel 1",
                measurements=[GeminiEnrichmentMeasurement(type="width", value=2.3, unit="m")],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-2-pv-01",
                reference="PV-01",
                quantity=5,
                occurrence_context="Nivel 2",
                measurements=[GeminiEnrichmentMeasurement(type="width", value=2.3, unit="m")],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == [
        "sotano-pv-01",
        "nivel-1-pv-01",
        "nivel-2-pv-01",
    ]
    assert [decision.reason for decision in decisions] == [
        DIFFERENT_CONTEXT_REASON,
        DIFFERENT_CONTEXT_REASON,
        DIFFERENT_CONTEXT_REASON,
    ]


def test_inventory_reconciliation_merges_same_reference_same_commercial_context() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="table",
                reference="V-01",
                quantity=2,
                occurrence_context="Nivel 1",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.1, unit="m"),
                ],
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="Nivel 1 V-01 1.20 x 2.10 cantidad 2",
                    )
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="detail",
                reference="V-1",
                occurrence_context="nivel_1",
                geometry_type_raw="rectangular",
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="visual",
                        text="Detalle V-01 Nivel 1",
                    )
                ],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert len(result.elements) == 1
    assert result.elements[0].temporary_id == "table"
    assert result.elements[0].reference == "V-01"
    assert result.elements[0].geometry_type_raw == "rectangular"
    assert decisions[0].action == "MERGE"
    assert decisions[0].commercial_context == "nivel_1"


def test_inventory_reconciliation_preserves_legacy_merge_when_context_is_missing() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="table",
                reference="V-01",
                quantity=2,
                measurements=[GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m")],
            ),
            GeminiElementEnrichment(
                temporary_id="detail",
                reference="V-1",
                geometry_type_raw="rectangular",
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert len(result.elements) == 1
    assert decisions[0].action == "MERGE"
    assert decisions[0].reason == DUPLICATE_REFERENCE_REASON
    assert decisions[0].commercial_context is None


def test_inventory_reconciliation_marks_same_reference_source_conflict_for_review() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="table",
                    reference="PV-03",
                    quantity=1,
                    operation_raw="corrediza",
                    measurements=[
                        GeminiEnrichmentMeasurement(type="width", value=1000, unit="mm"),
                    ],
                ),
                GeminiElementEnrichment(
                    temporary_id="drawing",
                    reference="PV-3",
                    quantity=1,
                    operation_raw="batiente",
                    geometry_type_raw="rectangular",
                ),
            ]
        ),
    )

    assert len(extraction.elements) == 1
    assert extraction.elements[0].reference == "PV-03"
    assert extraction.elements[0].status == ExtractionStatus.AMBIGUOUS
    assert SOURCE_CONFLICT_REASON in extraction.elements[0].missing_or_unknown
    assert SOURCE_CONFLICT_REASON in (extraction.notes or "")


def test_inventory_reconciliation_marks_quantity_conflict_without_silent_overwrite() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="table",
                    reference="V-10",
                    quantity=25,
                    quantity_status=ExtractionStatus.EXPLICIT,
                    measurements=[
                        GeminiEnrichmentMeasurement(type="width", value=2800, unit="mm"),
                        GeminiEnrichmentMeasurement(type="height", value=2900, unit="mm"),
                    ],
                ),
                GeminiElementEnrichment(
                    temporary_id="level-note",
                    reference="V-10",
                    quantity=5,
                    quantity_status=ExtractionStatus.EXPLICIT,
                    occurrence_context="N.P_5 / niveles 5 al 9",
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-2",
                            text="N.P_5 / niveles 5 al 9",
                        )
                    ],
                ),
            ]
        ),
    )

    element = extraction.elements[0]

    assert len(extraction.elements) == 1
    assert element.reference == "V-10"
    assert element.quantity == 25
    assert element.status == ExtractionStatus.AMBIGUOUS
    assert element.quantity_status == ExtractionStatus.AMBIGUOUS
    assert SOURCE_CONFLICT_REASON in element.missing_or_unknown
    assert SOURCE_CONFLICT_REASON in (extraction.notes or "")


def test_inventory_reconciliation_preserves_explicit_quantity_despite_other_conflicts() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="table",
                    reference="V-10",
                    quantity=2,
                    quantity_status=ExtractionStatus.EXPLICIT,
                    functional_type_raw="puerta",
                ),
                GeminiElementEnrichment(
                    temporary_id="level-note",
                    reference="V-10",
                    quantity=2,
                    quantity_status=ExtractionStatus.EXPLICIT,
                    functional_type_raw="ventana",
                    panel_count=3,
                ),
            ]
        ),
    )

    element = extraction.elements[0]

    assert len(extraction.elements) == 1
    assert element.quantity == 2
    assert element.quantity_status == ExtractionStatus.EXPLICIT
    assert element.status == ExtractionStatus.AMBIGUOUS
    assert SOURCE_CONFLICT_REASON in element.missing_or_unknown


def test_inventory_reconciliation_preserves_explicit_quantity_with_panel_counts() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="definition",
                    reference="X-01",
                    quantity=2,
                    quantity_status=ExtractionStatus.EXPLICIT,
                    panel_count=2,
                    components=[
                        GeminiEnrichmentComponent(role="FRAME", quantity=1),
                        GeminiEnrichmentComponent(role="LEAF", quantity=1),
                    ],
                ),
                GeminiElementEnrichment(
                    temporary_id="occurrence",
                    reference="X-1",
                    operation_raw="CORREDIZA",
                    components=[
                        GeminiEnrichmentComponent(role="FRAME", quantity=1),
                    ],
                ),
            ]
        ),
    )

    element = extraction.elements[0]

    assert len(extraction.elements) == 1
    assert element.reference == "X-01"
    assert element.quantity == 2
    assert element.quantity_status == ExtractionStatus.EXPLICIT
    assert element.panel_count == 2
    assert len(element.components) >= 1


def test_inventory_reconciliation_trace_shows_commercial_quantity_beats_repetition_count() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="definition",
                reference="V-10",
                quantity=25,
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=2.8, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.9, unit="m"),
                ],
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="V-10 CANTIDAD: 25 niveles 5 al 9",
                    )
                ],
                status=ExtractionStatus.EXPLICIT,
            ),
            GeminiElementEnrichment(
                temporary_id="placement",
                reference="V-10",
                quantity=5,
                occurrence_context="niveles 5 al 9",
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-2",
                        type="visual",
                        text="V-10 niveles 5 al 9",
                    )
                ],
                status=ExtractionStatus.EXPLICIT,
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)
    numeric_trace = build_numeric_resolution_trace(
        enrichment,
        stage="pre_reconciliation",
        source_file_names_by_id={"source-1": "cuadro.pdf", "source-2": "plano.pdf"},
    )

    assert len(result.elements) == 1
    assert result.elements[0].quantity == 25
    assert decisions[0].action == "MERGE"
    assert decisions[0].winner_temporary_id == "definition"
    decision_candidates = [
        (candidate.temporary_id, candidate.quantity, candidate.support_score)
        for candidate in decisions[0].candidates
    ]
    assert decision_candidates == [
        ("definition", 25, 2),
        ("placement", 5, 1),
    ]

    definition_trace = next(
        element
        for element in numeric_trace.elements
        if element.element_temporary_id == "definition"
    )
    roles = {
        (candidate.semantic_role, candidate.value) for candidate in definition_trace.candidates
    }

    assert ("QUANTITY", 25) in roles
    assert ("LEVEL_RANGE", "5-9") in roles
    assert ("REPETITION_COUNT", 5) in roles
    assert definition_trace.final_quantity.value == 25
    assert definition_trace.final_quantity.origin_candidate_field_path == "quantity"


def test_inventory_reconciliation_trace_shows_definition_beats_placement() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="definition",
                reference="V-12",
                quantity=5,
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=1.5, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.1, unit="m"),
                ],
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="V-12 CANTIDAD: 5",
                    )
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="placement",
                reference="V-12",
                quantity=1,
                occurrence_context="fachada norte",
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-2",
                        type="visual",
                        text="V-12 fachada norte",
                    )
                ],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert len(result.elements) == 1
    assert result.elements[0].quantity == 5
    assert decisions[0].action == "MERGE"
    assert decisions[0].winner_temporary_id == "definition"
    decision_candidates = [
        (candidate.temporary_id, candidate.quantity, candidate.support_score)
        for candidate in decisions[0].candidates
    ]
    assert decision_candidates == [
        ("definition", 5, 2),
        ("placement", 1, 1),
    ]


def test_inventory_reconciliation_keeps_single_quantity_one_when_no_conflict() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="single",
                reference="V-11",
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="width", value=1.1, unit="m")],
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        type="table",
                        text="V-11 CANTIDAD: 1",
                    )
                ],
            )
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)
    trace = build_numeric_resolution_trace(
        enrichment,
        stage="pre_reconciliation",
        source_file_names_by_id={"source-1": "cuadro.pdf"},
    )

    assert result.elements[0].quantity == 1
    assert decisions[0].action == "KEEP"
    assert trace.elements[0].final_quantity.value == 1


def test_inventory_reconciliation_keeps_context_labels_as_distinct_occurrences() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="room-a",
                reference="SALA",
                quantity=1,
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=4.10, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.85, unit="m"),
                ],
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-1",
                        text="SALA 4.10 x 2.85",
                    )
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="room-b",
                reference="SALA",
                quantity=1,
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=4.50, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
                evidence=[
                    GeminiEnrichmentEvidenceNote(
                        source_id="source-2",
                        text="SALA 4.50 x 2.50",
                    )
                ],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["room-a", "room-b"]
    assert [decision.action for decision in decisions] == ["KEEP", "KEEP"]
    assert {decision.reason for decision in decisions} == {CONTEXT_LABEL_NOT_IDENTITY_REASON}
    assert {decision.reference_semantics for decision in decisions} == {CONTEXT_LABEL}
    assert all(decision.normalized_reference is None for decision in decisions)
    assert [decision.winner_temporary_id for decision in decisions] == ["room-a", "room-b"]
    assert {
        candidate.source_ids for decision in decisions for candidate in decision.candidates
    } == {
        ("source-1",),
        ("source-2",),
    }
    assert any(
        "width=4.1m" in candidate.dimensions
        for decision in decisions
        for candidate in decision.candidates
    )
    assert any(
        "width=4.5m" in candidate.dimensions
        for decision in decisions
        for candidate in decision.candidates
    )


def test_inventory_reconciliation_preserves_reference_null_as_distinct_candidates() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="visual-a",
                reference=None,
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="width", value=6.70, unit="m")],
            ),
            GeminiElementEnrichment(
                temporary_id="visual-b",
                reference=None,
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="width", value=6.70, unit="m")],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["visual-a", "visual-b"]
    assert [decision.action for decision in decisions] == ["KEEP", "KEEP"]


def test_inventory_reconciliation_traces_same_reference_different_function_conflict() -> None:
    extraction = enrichment_to_gemini_extraction(
        GeminiDiscoveryResult(),
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="fixed",
                    reference="V-4",
                    quantity=1,
                    functional_type_raw="FIXED",
                    measurements=[GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m")],
                ),
                GeminiElementEnrichment(
                    temporary_id="door",
                    reference="V-04",
                    quantity=1,
                    functional_type_raw="SWING_DOOR",
                    measurements=[GeminiEnrichmentMeasurement(type="width", value=0.9, unit="m")],
                ),
            ]
        ),
    )

    assert len(extraction.elements) == 1
    assert extraction.elements[0].status == ExtractionStatus.AMBIGUOUS
    assert SOURCE_CONFLICT_REASON in extraction.elements[0].missing_or_unknown
    assert SOURCE_CONFLICT_REASON in (extraction.notes or "")


def test_numeric_trace_distinguishes_quantity_level_dimensions_and_component_count() -> None:
    trace = build_numeric_resolution_trace(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="v-12",
                    reference="V-12",
                    quantity=1,
                    panel_count=5,
                    measurements=[
                        GeminiEnrichmentMeasurement(type="width", value=11.68, unit="m"),
                        GeminiEnrichmentMeasurement(type="height", value=2.65, unit="m"),
                    ],
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-1",
                            type="table",
                            text="PLANTA 11.68 x 2.65 CANTIDAD: 1 N.P_3 5 cuerpos",
                            page_number=2,
                        )
                    ],
                    status=ExtractionStatus.EXPLICIT,
                )
            ]
        ),
        stage="test",
        source_file_names_by_id={"source-1": "planos.pdf"},
    )

    element_trace = trace.elements[0]
    roles = {(candidate.semantic_role, candidate.value) for candidate in element_trace.candidates}

    assert ("QUANTITY", 1) in roles
    assert ("LEVEL", 3) in roles
    assert ("WIDTH", 11.68) in roles
    assert ("HEIGHT", 2.65) in roles
    assert ("COMPONENT_COUNT", 5) in roles
    assert element_trace.final_quantity.value == 1
    assert element_trace.final_quantity.resolution_reason == "MODEL_OUTPUT"
    assert all(
        candidate.source_file_name == "planos.pdf"
        for candidate in element_trace.candidates
        if candidate.source_id == "source-1"
    )


def test_numeric_trace_recognizes_component_count_label_value_variants() -> None:
    examples = [
        "PV-01 N° Cuerpos 5",
        "PV-01 N° Cuerpos: 5",
        "PV-01 Nº Cuerpos 5",
        "PV-01 Numero de cuerpos = 5",
        "PV-01 Número de cuerpos = 5",
        "PV-01 Cuerpos 5",
        "PV-01 5 cuerpos",
        "PV-01 Paneles 5",
        "PV-01 Hojas 5",
    ]

    for text in examples:
        trace = build_numeric_resolution_trace(
            GeminiEnrichmentResult(
                elements=[
                    GeminiElementEnrichment(
                        temporary_id="pv-01",
                        reference="PV-01",
                        quantity=5,
                        evidence=[GeminiEnrichmentEvidenceNote(type="table", text=text)],
                    )
                ]
            ),
            stage="test",
        )

        roles = {
            (candidate.semantic_role, candidate.value) for candidate in trace.elements[0].candidates
        }
        assert ("COMPONENT_COUNT", 5) in roles, text


def test_numeric_trace_recognizes_section_count_label_value_variants() -> None:
    examples = [
        "PV-01 3 modulos",
        "PV-01 Módulos: 3",
        "PV-01 Modulos 3",
    ]

    for text in examples:
        trace = build_numeric_resolution_trace(
            GeminiEnrichmentResult(
                elements=[
                    GeminiElementEnrichment(
                        temporary_id="pv-01",
                        reference="PV-01",
                        quantity=3,
                        evidence=[GeminiEnrichmentEvidenceNote(type="table", text=text)],
                    )
                ]
            ),
            stage="test",
        )

        roles = {
            (candidate.semantic_role, candidate.value) for candidate in trace.elements[0].candidates
        }
        assert ("SECTION_COUNT", 3) in roles, text


def test_numeric_trace_distinguishes_level_range_and_repetition_count() -> None:
    trace = build_numeric_resolution_trace(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="v-20",
                    reference="V-20",
                    quantity=25,
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-2",
                            type="table",
                            text="CANTIDAD: 25. Se repite desde niveles 5 al 9",
                        )
                    ],
                )
            ]
        ),
        stage="test",
        source_file_names_by_id={"source-2": "cuadro.xlsx"},
    )

    candidates = trace.elements[0].candidates
    roles = {(candidate.semantic_role, candidate.value) for candidate in candidates}
    repetition = next(
        candidate for candidate in candidates if candidate.semantic_role == "REPETITION_COUNT"
    )

    assert ("QUANTITY", 25) in roles
    assert ("LEVEL_RANGE", "5-9") in roles
    assert ("REPETITION_COUNT", 5) in roles
    assert repetition.status == ExtractionStatus.INFERRED
    assert repetition.source_type == "MODEL_INFERRED"
    assert repetition.source_id == "source-2"
    assert repetition.source_file_name == "cuadro.xlsx"


def test_numeric_trace_keeps_multifile_candidate_provenance() -> None:
    trace = build_numeric_resolution_trace(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="pv-03",
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-1",
                            type="table",
                            text="PV-03 CANTIDAD: 3",
                        ),
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-2",
                            type="visual",
                            text="PISO 8",
                        ),
                    ],
                )
            ]
        ),
        stage="test",
        source_file_names_by_id={"source-1": "cuadro.pdf", "source-2": "plano.png"},
    )

    quantity = next(
        candidate
        for candidate in trace.elements[0].candidates
        if candidate.semantic_role == "QUANTITY"
    )
    floor = next(
        candidate
        for candidate in trace.elements[0].candidates
        if candidate.semantic_role == "FLOOR"
    )

    assert quantity.source_id == "source-1"
    assert quantity.source_file_name == "cuadro.pdf"
    assert quantity.source_type == "TABLE"
    assert floor.source_id == "source-2"
    assert floor.source_file_name == "plano.png"
    assert floor.source_type == "VISION"


def test_numeric_trace_observes_quantity_aliases_without_merging_other_roles() -> None:
    trace = build_numeric_resolution_trace(
        GeminiEnrichmentResult(
            elements=[
                GeminiElementEnrichment(
                    temporary_id="alias-case",
                    quantity=7,
                    evidence=[
                        GeminiEnrichmentEvidenceNote(
                            source_id="source-1",
                            type="table",
                            text="ITEM 10 QTY: 7 UND 7 4 modulos vidrio 10 mm",
                        )
                    ],
                )
            ]
        ),
        stage="test",
        source_file_names_by_id={"source-1": "cuadro.pdf"},
    )

    roles = [
        (candidate.semantic_role, candidate.value) for candidate in trace.elements[0].candidates
    ]

    assert ("QUANTITY", 7) in roles
    assert ("ITEM_NUMBER", 10) in roles
    assert ("SECTION_COUNT", 4) in roles
    assert ("GLASS_THICKNESS", 10) in roles
    assert trace.elements[0].final_quantity.value == 7


def test_enrichment_debug_capture_stores_batch_and_merged_numeric_traces(
    tmp_path: Path,
) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-01", "quantity": 5, '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "V-01 CANTIDAD: 5"}]}'
                "]}"
            )
        ]
    )
    debug_capture = GeminiEnrichmentDebugCapture()

    result = provider.enrich_discoveries_from_files(
        [_pdf(tmp_path)],
        _discovery(1),
        debug_capture=debug_capture,
    )

    assert result.elements[0].quantity == 5
    assert len(debug_capture.batch_numeric_traces) == 1
    assert debug_capture.batch_numeric_traces[0].elements[0].final_quantity.value == 5
    assert debug_capture.merged_numeric_trace is not None
    assert debug_capture.merged_numeric_trace.elements[0].final_quantity.value == 5


def test_enrichment_normalizes_absolute_image_region_without_losing_evidence(
    tmp_path: Path,
) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-01", '
                '"evidence": [{"source_id": "source-1", "type": "visual", '
                '"text": "Detalle V-01", '
                '"region": {"x": 50, "y": 100, "width": 200, "height": 300}}]}'
                "]}"
            )
        ]
    )
    debug_capture = GeminiEnrichmentDebugCapture()

    result = provider.enrich_discoveries_from_files(
        [_png(tmp_path)],
        _discovery(1),
        debug_capture=debug_capture,
    )

    evidence = result.elements[0].evidence[0]

    assert evidence.text == "Detalle V-01"
    assert evidence.region is not None
    assert evidence.region.x == 0.05
    assert evidence.region.y == 0.1
    assert evidence.region.width == 0.2
    assert evidence.region.height == 0.3
    assert debug_capture.region_sanitization_events[0].action == REGION_NORMALIZED


def test_full_pipeline_debug_capture_records_numeric_trace_without_changing_quantity(
    tmp_path: Path,
) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-01"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-01", "quantity": 5, '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "N.P_5 / CANTIDAD: 5"}]}'
                "]}"
            ),
        ]
    )
    debug_capture = GeminiFullPipelineDebugCapture()

    result = provider.extract_with_discovery_from_files(
        [_pdf(tmp_path)],
        debug_capture=debug_capture,
    )

    assert result.elements[0].quantity is not None
    assert result.elements[0].quantity.value == 5
    assert debug_capture.numeric_trace is not None
    assert debug_capture.numeric_trace.elements[0].final_quantity.value == 5
    assert {
        candidate.semantic_role for candidate in debug_capture.numeric_trace.elements[0].candidates
    } >= {"QUANTITY", "LEVEL"}
    assert debug_capture.reconciliation_decisions is not None


def test_full_pipeline_applies_source_grounded_quantity_before_mapping(
    tmp_path: Path,
) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse('{"elements": [{"temporary_id": "d-1", "reference": "V-10"}]}'),
            _scope_response(("d-1", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "d-1", "reference": "V-10", "quantity": 5, '
                '"status": "explicit", '
                '"evidence": [{"source_id": "source-1", "type": "table", '
                '"text": "V-10 CANTIDAD: 25 NIVELES 5 AL 9"}]}'
                "]}",
            ),
        ]
    )
    debug_capture = GeminiFullPipelineDebugCapture()

    result = provider.extract_with_discovery_from_files(
        [_xlsx_with_quantity(tmp_path, "V-10", 25)],
        debug_capture=debug_capture,
    )

    assert result.elements[0].quantity is not None
    assert result.elements[0].quantity.value == 25
    assert result.elements[0].quantity.status == ExtractionStatus.EXPLICIT
    assert debug_capture.enrichment is not None
    assert debug_capture.enrichment.elements[0].quantity == 25
    assert debug_capture.enrichment_debug is not None
    assert debug_capture.enrichment_debug.quantity_grounding_decisions[0].action == "REPLACE"


def test_full_pipeline_debug_capture_records_inventory_stage_counts(
    tmp_path: Path,
) -> None:
    provider = _provider_with_responses(
        [
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "a", "reference": "SALA"}, '
                '{"temporary_id": "b", "reference": "SALA"}'
                "]}"
            ),
            _scope_response(("a", "in_scope_full"), ("b", "in_scope_full")),
            _FakeResponse(
                '{"elements": ['
                '{"temporary_id": "a", "reference": "SALA", "quantity": 1, '
                '"measurements": [{"type": "width", "value": 4.10, "unit": "m"}]}, '
                '{"temporary_id": "b", "reference": "SALA", "quantity": 1, '
                '"measurements": [{"type": "width", "value": 4.50, "unit": "m"}]}'
                "]}"
            ),
        ]
    )
    debug_capture = GeminiFullPipelineDebugCapture()

    provider.extract_with_discovery_from_files(
        [_pdf(tmp_path)],
        debug_capture=debug_capture,
    )

    assert debug_capture.inventory_trace is not None
    stage_counts = {stage.stage: stage.count for stage in debug_capture.inventory_trace.stages}

    assert stage_counts["DISCOVERY"] == 2
    assert stage_counts["ENRICHMENT_BATCH_1"] == 2
    assert stage_counts["MERGED_ENRICHMENT"] == 2
    assert stage_counts["PRE_RECONCILIATION"] == 2
    assert stage_counts["POST_RECONCILIATION"] == 2
    assert stage_counts["FINAL_REQUIREMENT_EXTRACTION"] == 2
    assert debug_capture.reconciliation_decisions is not None
    assert [decision.action for decision in debug_capture.reconciliation_decisions] == [
        "KEEP",
        "KEEP",
    ]
    assert {decision.reason for decision in debug_capture.reconciliation_decisions} == {
        CONTEXT_LABEL_NOT_IDENTITY_REASON
    }


def test_item_count_diagnostics_reports_expected_item_present_until_final() -> None:
    trace = _inventory_trace(
        DISCOVERY=[_trace_item("V-1", temporary_id="d-1")],
        ENRICHMENT_BATCH_1=[_trace_item("V-01", temporary_id="d-1")],
        MERGED_ENRICHMENT=[_trace_item("V-01", temporary_id="d-1")],
        PRE_RECONCILIATION=[_trace_item("V-01", temporary_id="d-1")],
        POST_RECONCILIATION=[_trace_item("V-01", temporary_id="d-1")],
        FINAL_REQUIREMENT_EXTRACTION=[_trace_item("V-01", id="element-1")],
    )

    report = build_item_count_diagnostic_report(
        case_name="GOOD_CASE",
        expected_items=["V-01"],
        trace=trace,
        scoped_count=1,
    )

    assert report.stage_counts.discovered == 1
    assert report.stage_counts.scoped == 1
    assert report.stage_counts.enriched == 1
    assert report.stage_counts.final == 1
    assert report.metrics.precision == 1.0
    assert report.metrics.recall == 1.0
    assert report.metrics.item_count_error_percent == 0.0
    assert report.missing == ()
    assert report.unexpected == ()


def test_item_count_diagnostics_identifies_missing_at_discovery() -> None:
    trace = _inventory_trace(
        DISCOVERY=[_trace_item("V-01")],
        FINAL_REQUIREMENT_EXTRACTION=[_trace_item("V-01")],
    )

    report = build_item_count_diagnostic_report(
        case_name="BAD_CASE",
        expected_items=["V-01", "V-02"],
        trace=trace,
    )

    assert report.missing[0].identity == "V-02"
    assert report.missing[0].reason == MISSING_AT_DISCOVERY
    assert report.metrics.recall == 0.5
    assert report.metrics.missing_item_rate == 0.5


def test_item_count_diagnostics_identifies_scope_drop() -> None:
    trace = _inventory_trace(
        DISCOVERY=[_trace_item("V-01")],
        ENRICHMENT_BATCH_1=[],
        MERGED_ENRICHMENT=[],
        PRE_RECONCILIATION=[],
        POST_RECONCILIATION=[],
        FINAL_REQUIREMENT_EXTRACTION=[],
    )

    report = build_item_count_diagnostic_report(
        case_name="SCOPE_CASE",
        expected_items=["V-01"],
        trace=trace,
        scoped_count=0,
    )

    assert report.missing[0].reason == DROPPED_BY_SCOPE
    assert report.missing[0].last_seen_stage == "DISCOVERY"
    assert report.stage_counts.scoped == 0


def test_item_count_diagnostics_classifies_reconciliation_merge_and_orphan_drop() -> None:
    trace = _inventory_trace(
        DISCOVERY=[
            _trace_item(None, temporary_id="a"),
            _trace_item(None, temporary_id="b"),
        ],
        ENRICHMENT_BATCH_1=[
            _trace_item(None, temporary_id="a"),
            _trace_item(None, temporary_id="b"),
        ],
        MERGED_ENRICHMENT=[
            _trace_item(None, temporary_id="a"),
            _trace_item(None, temporary_id="b"),
        ],
        PRE_RECONCILIATION=[
            _trace_item(None, temporary_id="a"),
            _trace_item(None, temporary_id="b"),
        ],
        POST_RECONCILIATION=[_trace_item(None, temporary_id="a")],
        FINAL_REQUIREMENT_EXTRACTION=[_trace_item(None, id="a")],
    )
    merge_decision = InventoryDecision(
        action="MERGE",
        reason=DUPLICATE_REFERENCE_REASON,
        temporary_ids=("a", "b"),
        reference=None,
        losing_temporary_ids=("b",),
    )

    report = build_item_count_diagnostic_report(
        case_name="MERGE_CASE",
        expected_items=["a", "b"],
        trace=trace,
        reconciliation_decisions=[merge_decision],
    )

    assert report.missing[0].identity == "B"
    assert report.missing[0].reason == MERGED_IN_RECONCILIATION

    orphan_trace = _inventory_trace(
        DISCOVERY=[_trace_item("TAG-01", temporary_id="tag")],
        ENRICHMENT_BATCH_1=[_trace_item("TAG-01", temporary_id="tag")],
        MERGED_ENRICHMENT=[_trace_item("TAG-01", temporary_id="tag")],
        PRE_RECONCILIATION=[_trace_item("TAG-01", temporary_id="tag")],
        POST_RECONCILIATION=[],
        FINAL_REQUIREMENT_EXTRACTION=[],
    )
    orphan_decision = InventoryDecision(
        action="DROP_AS_NON_COMMERCIAL",
        reason=ORPHAN_REFERENCE_REASON,
        temporary_ids=("tag",),
        reference="TAG-01",
    )

    orphan_report = build_item_count_diagnostic_report(
        case_name="ORPHAN_CASE",
        expected_items=["TAG-01"],
        trace=orphan_trace,
        reconciliation_decisions=[orphan_decision],
    )

    assert orphan_report.missing[0].reason == DROPPED_AS_ORPHAN


def test_item_count_diagnostics_identifies_mapper_drop() -> None:
    trace = _inventory_trace(
        DISCOVERY=[_trace_item("V-01")],
        ENRICHMENT_BATCH_1=[_trace_item("V-01")],
        MERGED_ENRICHMENT=[_trace_item("V-01")],
        PRE_RECONCILIATION=[_trace_item("V-01")],
        POST_RECONCILIATION=[_trace_item("V-01")],
        FINAL_REQUIREMENT_EXTRACTION=[],
    )

    report = build_item_count_diagnostic_report(
        case_name="MAPPER_DROP_CASE",
        expected_items=["V-01"],
        trace=trace,
    )

    assert report.missing[0].reason == DROPPED_BY_MAPPER
    assert report.missing[0].last_seen_stage == "POST_RECONCILIATION"


def test_item_count_diagnostics_reports_duplicate_discovery_enrichment_and_final() -> None:
    trace = _inventory_trace(
        DISCOVERY=[_trace_item("V-01", temporary_id="a"), _trace_item("V-1", temporary_id="b")],
        ENRICHMENT_BATCH_1=[_trace_item("V-01", temporary_id="a")],
        ENRICHMENT_BATCH_2=[_trace_item("V-01", temporary_id="b")],
        FINAL_REQUIREMENT_EXTRACTION=[_trace_item("V-01", id="a"), _trace_item("V-01", id="b")],
    )

    report = build_item_count_diagnostic_report(
        case_name="DUPLICATE_CASE",
        expected_items=["V-01"],
        trace=trace,
    )

    assert ("V-01", DUPLICATE_DISCOVERY, "DISCOVERY") in _extra_rows(report.duplicates)
    assert ("V-01", DUPLICATE_ENRICHMENT, "ENRICHMENT") in _extra_rows(report.duplicates)
    assert ("V-01", RECONCILIATION_UNDERMERGE, "FINAL_REQUIREMENT_EXTRACTION") in _extra_rows(
        report.duplicates
    )
    assert report.metrics.duplicate_count == 1
    assert report.metrics.duplicate_rate == 0.5


def test_item_count_diagnostics_marks_unexpected_final_orphan_with_support() -> None:
    trace = _inventory_trace(
        DISCOVERY=[_trace_item("V-01"), _trace_item("TEMPORAL")],
        ENRICHMENT_BATCH_1=[_trace_item("V-01"), _trace_item("TEMPORAL")],
        MERGED_ENRICHMENT=[_trace_item("V-01"), _trace_item("TEMPORAL")],
        PRE_RECONCILIATION=[_trace_item("V-01"), _trace_item("TEMPORAL")],
        POST_RECONCILIATION=[_trace_item("V-01"), _trace_item("TEMPORAL")],
        FINAL_REQUIREMENT_EXTRACTION=[_trace_item("V-01"), _trace_item("TEMPORAL")],
    )

    report = build_item_count_diagnostic_report(
        case_name="EXTRA_CASE",
        expected_items=["V-01"],
        trace=trace,
    )

    assert report.unexpected[0].identity == "TEMPORAL"
    assert report.unexpected[0].reason == ORPHAN_WITH_TECHNICAL_SUPPORT
    assert report.metrics.precision == 0.5


def test_reconciliation_preserves_reference_null_and_non_formal_items_with_support() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="no-ref-a",
                reference=None,
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m")],
            ),
            GeminiElementEnrichment(
                temporary_id="context-label",
                reference="SALA",
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="height", value=2.5, unit="m")],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["no-ref-a", "context-label"]
    assert [decision.action for decision in decisions] == ["KEEP", "KEEP"]
    assert decisions[1].reason == CONTEXT_LABEL_NOT_IDENTITY_REASON


def test_reconciliation_drops_clear_orphan_but_keeps_commercial_supported_reference() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(temporary_id="orphan", reference="V-99"),
            GeminiElementEnrichment(
                temporary_id="supported",
                reference="V-100",
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="width", value=1.4, unit="m")],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.reference for item in result.elements] == ["V-100"]
    assert decisions[0].action == "DROP_AS_NON_COMMERCIAL"
    assert decisions[0].reason == ORPHAN_REFERENCE_REASON
    assert decisions[1].action == "KEEP"


def test_scope_pipeline_keeps_full_partial_uncertain_and_drops_out_of_scope() -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(temporary_id="full", reference="V-01"),
            GeminiElementDiscovery(temporary_id="partial", reference="V-02"),
            GeminiElementDiscovery(temporary_id="uncertain", reference="V-03"),
            GeminiElementDiscovery(temporary_id="out", reference="M-01"),
        ]
    )
    scope = provider_module.merge_scope_with_discovery(
        discovery,
        provider_module.GeminiScopeResult.model_validate(
            {
                "elements": [
                    {"temporary_id": "full", "scope": "in_scope_full"},
                    {"temporary_id": "partial", "scope": "in_scope_partial"},
                    {"temporary_id": "uncertain", "scope": "uncertain"},
                    {"temporary_id": "out", "scope": "out_of_scope"},
                ]
            }
        ),
    )

    selected = provider_module.select_discoveries_for_enrichment(discovery, scope)

    assert [item.temporary_id for item in selected.elements] == ["full", "partial", "uncertain"]


def test_reconciliation_distinct_contexts_keep_payloads_separate() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="north",
                reference="V-20",
                occurrence_context="fachada norte",
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="width", value=1.1, unit="m")],
                glass=[GeminiEnrichmentGlass(type="templado")],
                profiles=[GeminiEnrichmentNamedItem(code="K40")],
                components=[GeminiEnrichmentComponent(role="FIXED")],
                evidence=[GeminiEnrichmentEvidenceNote(source_id="source-1", text="V-20 norte")],
            ),
            GeminiElementEnrichment(
                temporary_id="south",
                reference="V-20",
                occurrence_context="fachada sur",
                quantity=1,
                measurements=[GeminiEnrichmentMeasurement(type="height", value=2.2, unit="m")],
                glass=[GeminiEnrichmentGlass(type="laminado")],
                profiles=[GeminiEnrichmentNamedItem(code="S50")],
                components=[GeminiEnrichmentComponent(role="SLIDING")],
                evidence=[GeminiEnrichmentEvidenceNote(source_id="source-2", text="V-20 sur")],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["north", "south"]
    assert {decision.reason for decision in decisions} == {DIFFERENT_CONTEXT_REASON}
    assert [item.measurements[0].type for item in result.elements] == ["width", "height"]
    assert [item.glass[0].type for item in result.elements] == ["templado", "laminado"]
    assert [item.profiles[0].code for item in result.elements] == ["K40", "S50"]
    assert [item.components[0].role for item in result.elements] == ["FIXED", "SLIDING"]
    assert [item.evidence[0].source_id for item in result.elements] == ["source-1", "source-2"]


def test_profile_resolution_preserves_explicit_and_inferred_profiles() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="K70",
                        role="system",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.92,
                    ),
                    GeminiEnrichmentNamedItem(
                        code="S50",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.7,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert [profile.code for profile in item.profiles] == ["S50", "K70"]
    assert [profile.status for profile in item.profiles] == [
        ExtractionStatus.EXPLICIT,
        ExtractionStatus.INFERRED,
    ]
    assert PROFILE_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_profile_resolution_preserves_inferred_alternatives_without_conflict() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="K70",
                        role="system",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.64,
                    ),
                    GeminiEnrichmentNamedItem(
                        code="K90",
                        role="system",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.81,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert [profile.code for profile in item.profiles] == ["K90", "K70"]
    assert PROFILE_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_profile_resolution_marks_explicit_primary_conflict_without_collapsing() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="K70",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.74,
                    ),
                    GeminiEnrichmentNamedItem(
                        code="S50",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.88,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert item.status == ExtractionStatus.AMBIGUOUS
    assert PROFILE_EXPLICIT_CONFLICT in item.missing_or_unknown
    assert [profile.code for profile in item.profiles] == ["S50", "K70"]


def test_profile_resolution_allows_complementary_explicit_profiles() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="K70",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.91,
                    ),
                    GeminiEnrichmentNamedItem(
                        code="MARCO-01",
                        role="frame",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.86,
                    ),
                    GeminiEnrichmentNamedItem(
                        code="HOJA-02",
                        role="sash",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.84,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert PROFILE_EXPLICIT_CONFLICT not in item.missing_or_unknown
    assert [profile.code for profile in item.profiles] == ["K70", "MARCO-01", "HOJA-02"]


def test_profile_resolution_keeps_inferred_when_no_explicit_system_exists() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="S80",
                        role="system",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.79,
                    )
                ],
                functional_type_raw="SLIDING_DOOR",
                operation_raw="SLIDING",
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert [profile.code for profile in item.profiles] == ["S80"]
    assert item.profiles[0].status == ExtractionStatus.INFERRED


def test_profile_resolution_general_note_does_not_displace_explicit_override() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="K70",
                        role="system",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.93,
                        notes="general note candidate",
                    ),
                    GeminiEnrichmentNamedItem(
                        code="S50",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.65,
                        notes="item row explicit override",
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    assert [profile.code for profile in result.elements[0].profiles] == ["S50", "K70"]


def test_profile_resolution_keeps_neighbor_profile_contexts_separate() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "north",
                [
                    GeminiEnrichmentNamedItem(
                        code="K40",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.9,
                    )
                ],
                reference="V-20",
                occurrence_context="fachada norte",
            ),
            _profile_element(
                "south",
                [
                    GeminiEnrichmentNamedItem(
                        code="S50",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.9,
                    )
                ],
                reference="V-20",
                occurrence_context="fachada sur",
            ),
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["north", "south"]
    assert [item.profiles[0].code for item in result.elements] == ["K40", "S50"]
    assert all(PROFILE_EXPLICIT_CONFLICT not in item.missing_or_unknown for item in result.elements)


def test_profile_resolution_preferred_profile_uses_status_confidence_before_input_order() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="K90",
                        role="system",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.99,
                    ),
                    GeminiEnrichmentNamedItem(
                        code="K70",
                        role="system",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.62,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    extraction = enrichment_to_gemini_extraction(GeminiDiscoveryResult(), result)
    mapped = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.elements[0].profiles[0].code == "K70"
    assert mapped.elements[0].profiles[0].code.value == "K70"


def test_inventory_trace_records_profile_resolution_by_stage() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _profile_element(
                "item-1",
                [
                    GeminiEnrichmentNamedItem(
                        code="K70",
                        name="Sistema K70",
                        role="system",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.82,
                    )
                ],
            )
        ]
    )
    result, _ = reconcile_inventory_candidates(enrichment)
    extraction = enrichment_to_gemini_extraction(GeminiDiscoveryResult(), result)
    mapped = map_gemini_extraction_to_requirement_extraction(extraction)

    enrichment_profile = enrichment_inventory_elements(result)[0].profiles[0]
    final_profile = final_inventory_elements(mapped)[0].profiles[0]

    assert enrichment_profile.code == "K70"
    assert enrichment_profile.role == "system"
    assert enrichment_profile.status == "inferred"
    assert enrichment_profile.confidence == 0.82
    assert enrichment_profile.resolution == "INFERRED"
    assert final_profile.code == "K70"
    assert final_profile.status == "inferred"
    assert final_profile.resolution == "INFERRED"


def test_glass_resolution_orders_explicit_glass_first() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="8 mm",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.99,
                    ),
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.61,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    assert [glass.type for glass in result.elements[0].glass] == ["laminado", "templado"]
    assert GLASS_EXPLICIT_CONFLICT not in result.elements[0].missing_or_unknown


def test_glass_resolution_preserves_inferred_alternatives_by_confidence() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="8 mm",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.7,
                    ),
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="4+4",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.82,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert [glass.type for glass in item.glass] == ["laminado", "templado"]
    assert GLASS_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_glass_resolution_marks_explicit_conflict_without_collapsing() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="8 mm",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.94,
                    ),
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.93,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert item.status == ExtractionStatus.AMBIGUOUS
    assert GLASS_EXPLICIT_CONFLICT in item.missing_or_unknown
    assert [glass.type for glass in item.glass] == ["templado", "laminado"]
    assert [glass.status for glass in item.glass] == [
        ExtractionStatus.AMBIGUOUS,
        ExtractionStatus.AMBIGUOUS,
    ]


def test_glass_resolution_treatment_does_not_displace_explicit_type() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        treatment="pelicula reduccion calor",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.98,
                    ),
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.74,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert item.glass[0].type == "laminado"
    assert item.glass[1].treatment == "pelicula reduccion calor"
    assert GLASS_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_glass_resolution_color_does_not_displace_type_or_composition() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        color="bronce",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.96,
                    ),
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="10 mm",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.76,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert item.glass[0].type == "templado"
    assert item.glass[1].color == "bronce"
    assert GLASS_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_glass_resolution_keeps_compatible_complementary_entries_without_conflict() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.88,
                    ),
                    GeminiEnrichmentGlass(
                        treatment="low-e",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.87,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)

    item = result.elements[0]
    assert [glass.composition or glass.treatment for glass in item.glass] == ["5+5", "low-e"]
    assert GLASS_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_glass_resolution_same_reference_different_context_does_not_mix_glass() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "north",
                [GeminiEnrichmentGlass(type="templado", status=ExtractionStatus.EXPLICIT)],
                reference="V-20",
                occurrence_context="fachada norte",
            ),
            _glass_element(
                "south",
                [GeminiEnrichmentGlass(type="laminado", status=ExtractionStatus.EXPLICIT)],
                reference="V-20",
                occurrence_context="fachada sur",
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["north", "south"]
    assert [item.glass[0].type for item in result.elements] == ["templado", "laminado"]
    assert {decision.reason for decision in decisions} == {DIFFERENT_CONTEXT_REASON}


def test_reconciliation_same_reference_different_context_does_not_mix_measurements() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="basement",
                reference="V-01",
                occurrence_context="SOTANO",
                quantity=1,
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=0.9, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=1.45, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="level-1",
                reference="V-01",
                occurrence_context="NIVEL 1",
                quantity=1,
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=3.0, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.1, unit="m"),
                ],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [item.temporary_id for item in result.elements] == ["basement", "level-1"]
    assert [
        [(measurement.type, measurement.value) for measurement in item.measurements]
        for item in result.elements
    ] == [
        [("width", 0.9), ("height", 1.45)],
        [("width", 3.0), ("height", 2.1)],
    ]
    assert {decision.reason for decision in decisions} == {DIFFERENT_CONTEXT_REASON}


def test_glass_scope_general_default_remains_applicable_without_override() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.72,
                        notes="Todos los vidrios seran laminados 5+5.",
                    )
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    trace = enrichment_inventory_elements(result)[0].glass[0]

    assert result.elements[0].glass[0].type == "laminado"
    assert trace.scope == GLASS_SCOPE_DOCUMENT_GENERAL


def test_glass_scope_item_local_override_precedes_general_default() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.95,
                        notes="Todos los vidrios seran laminados 5+5.",
                    ),
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="8 mm",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.61,
                        notes="V-01 vidrio templado 8 mm.",
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    item = result.elements[0]
    trace = enrichment_inventory_elements(result)[0].glass

    assert [glass.type for glass in item.glass] == ["templado", "laminado"]
    assert [glass.scope for glass in trace] == [
        GLASS_SCOPE_ITEM_LOCAL,
        GLASS_SCOPE_DOCUMENT_GENERAL,
    ]
    assert GLASS_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_glass_scope_section_defaults_follow_occurrence_context() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "level-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        status=ExtractionStatus.EXPLICIT,
                        notes="NIVEL 1 vidrio laminado.",
                    )
                ],
                reference="V-20",
                occurrence_context="NIVEL 1",
            ),
            _glass_element(
                "level-2",
                [
                    GeminiEnrichmentGlass(
                        type="templado",
                        status=ExtractionStatus.EXPLICIT,
                        notes="NIVEL 2 vidrio templado.",
                    )
                ],
                reference="V-20",
                occurrence_context="NIVEL 2",
            ),
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    trace = enrichment_inventory_elements(result)

    assert [item.glass[0].type for item in result.elements] == ["laminado", "templado"]
    assert [item.glass[0].scope for item in trace] == [
        GLASS_SCOPE_SECTION_LEVEL,
        GLASS_SCOPE_SECTION_LEVEL,
    ]


def test_glass_scope_local_treatment_overrides_general_treatment() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        treatment="pelicula reduccion calor",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.99,
                        notes="Todos los vidrios tendran pelicula de reduccion de calor.",
                    ),
                    GeminiEnrichmentGlass(
                        treatment="sin pelicula",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.7,
                        notes="V-01 sin pelicula.",
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    item = result.elements[0]
    trace = enrichment_inventory_elements(result)[0].glass

    assert [glass.treatment for glass in item.glass] == [
        "sin pelicula",
        "pelicula reduccion calor",
    ]
    assert [glass.scope for glass in trace] == [
        GLASS_SCOPE_ITEM_LOCAL,
        GLASS_SCOPE_DOCUMENT_GENERAL,
    ]


def test_glass_scope_general_type_does_not_conflict_with_local_explicit_type() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        notes="Todos los vidrios seran laminados 5+5.",
                    ),
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="8 mm",
                        status=ExtractionStatus.EXPLICIT,
                        notes="V-01 vidrio templado 8 mm.",
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    item = result.elements[0]

    assert [glass.type for glass in item.glass] == ["templado", "laminado"]
    assert [glass.status for glass in item.glass] == [
        ExtractionStatus.EXPLICIT,
        ExtractionStatus.EXPLICIT,
    ]
    assert GLASS_EXPLICIT_CONFLICT not in item.missing_or_unknown


def test_glass_scope_same_scope_explicit_incompatible_marks_conflict() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        notes="V-01 vidrio laminado 5+5.",
                    ),
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="8 mm",
                        status=ExtractionStatus.EXPLICIT,
                        notes="V-01 vidrio templado 8 mm.",
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    item = result.elements[0]

    assert GLASS_EXPLICIT_CONFLICT in item.missing_or_unknown
    assert [glass.status for glass in item.glass] == [
        ExtractionStatus.AMBIGUOUS,
        ExtractionStatus.AMBIGUOUS,
    ]


def test_glass_scope_neighbor_reference_does_not_become_item_local() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.99,
                        notes="V-02 vidrio laminado.",
                    ),
                    GeminiEnrichmentGlass(
                        type="templado",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.5,
                        notes="V-01 vidrio templado.",
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    trace = enrichment_inventory_elements(result)[0].glass

    assert [glass.type for glass in result.elements[0].glass] == ["templado", "laminado"]
    assert [glass.scope for glass in trace] == [
        GLASS_SCOPE_ITEM_LOCAL,
        GLASS_SCOPE_UNKNOWN,
    ]


def test_glass_scope_unknown_is_preserved_after_specific_scope() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="templado",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.99,
                    ),
                    GeminiEnrichmentGlass(
                        type="laminado",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.4,
                        notes="V-01 vidrio laminado.",
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    trace = enrichment_inventory_elements(result)[0].glass

    assert [glass.type for glass in result.elements[0].glass] == ["laminado", "templado"]
    assert [glass.scope for glass in trace] == [
        GLASS_SCOPE_ITEM_LOCAL,
        GLASS_SCOPE_UNKNOWN,
    ]


def test_glass_scope_preserves_inferred_when_no_explicit_or_default_exists() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="templado",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.74,
                    )
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    trace = enrichment_inventory_elements(result)[0].glass[0]

    assert result.elements[0].glass[0].type == "templado"
    assert result.elements[0].glass[0].status == ExtractionStatus.INFERRED
    assert trace.scope == GLASS_SCOPE_UNKNOWN


def test_glass_resolution_mapper_preserves_reconciled_order() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="templado",
                        thickness="8 mm",
                        status=ExtractionStatus.INFERRED,
                        confidence=0.95,
                    ),
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.75,
                    ),
                ],
            )
        ]
    )

    result, _ = reconcile_inventory_candidates(enrichment)
    extraction = enrichment_to_gemini_extraction(GeminiDiscoveryResult(), result)
    mapped = map_gemini_extraction_to_requirement_extraction(extraction)

    assert [glass.type.raw for glass in mapped.elements[0].glass] == ["laminado", "templado"]


def test_inventory_trace_records_glass_resolution_by_stage() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            _glass_element(
                "item-1",
                [
                    GeminiEnrichmentGlass(
                        type="laminado",
                        composition="5+5",
                        thickness="5+5",
                        treatment="low-e",
                        color="gris",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.86,
                    )
                ],
            )
        ]
    )
    result, _ = reconcile_inventory_candidates(enrichment)
    extraction = enrichment_to_gemini_extraction(GeminiDiscoveryResult(), result)
    mapped = map_gemini_extraction_to_requirement_extraction(extraction)

    enrichment_glass = enrichment_inventory_elements(result)[0].glass[0]
    final_glass = final_inventory_elements(mapped)[0].glass[0]

    assert enrichment_glass.type == "laminado"
    assert enrichment_glass.composition == "5+5"
    assert enrichment_glass.thickness == "5+5"
    assert enrichment_glass.treatment == "low-e"
    assert enrichment_glass.color == "gris"
    assert enrichment_glass.status == "explicit"
    assert enrichment_glass.confidence == 0.86
    assert enrichment_glass.scope == GLASS_SCOPE_UNKNOWN
    assert enrichment_glass.scopeReason == "No reliable glass evidence scope signal."
    assert enrichment_glass.resolution == "UNKNOWN_EXPLICIT"
    assert final_glass.type == "laminado"
    assert final_glass.composition == "5+5"
    assert final_glass.status == "explicit"
    assert final_glass.resolution == "EXPLICIT"


def test_item_count_diagnostics_preserves_casa_pereira_nineteen_to_nineteen_regression() -> None:
    expected = [f"P-{index:02d}" for index in range(1, 20)]
    trace = _inventory_trace(
        DISCOVERY=[_trace_item(reference) for reference in expected],
        ENRICHMENT_BATCH_1=[_trace_item(reference) for reference in expected[:10]],
        ENRICHMENT_BATCH_2=[_trace_item(reference) for reference in expected[10:]],
        MERGED_ENRICHMENT=[_trace_item(reference) for reference in expected],
        PRE_RECONCILIATION=[_trace_item(reference) for reference in expected],
        POST_RECONCILIATION=[_trace_item(reference) for reference in expected],
        FINAL_REQUIREMENT_EXTRACTION=[_trace_item(reference) for reference in expected],
    )

    report = build_item_count_diagnostic_report(
        case_name="Casa Pereira",
        expected_items=expected,
        trace=trace,
        scoped_count=19,
    )

    assert report.metrics.expected_count == 19
    assert report.metrics.actual_count == 19
    assert report.metrics.missing_count == 0
    assert report.metrics.unexpected_count == 0
    assert report.metrics.duplicate_count == 0
    assert report.metrics.recall == 1.0
    assert report.metrics.precision == 1.0
    assert report.stage_counts.enriched == 19
    assert report.stage_counts.post_reconciliation == 19


def test_discovery_context_is_propagated_when_enrichment_omits_it() -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(
                temporary_id="pv-01-level-a",
                reference="PV-01",
                occurrence_context="LEVEL A",
            )
        ]
    )
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="pv-01-level-a",
                reference="PV-01",
                measurements=[GeminiEnrichmentMeasurement(type="width", value=5.25, unit="m")],
            )
        ]
    )

    merged = merge_enrichment_batches(discovery, [enrichment])

    assert merged.elements[0].occurrence_context == "LEVEL A"


def test_discovery_context_conflict_marks_enrichment_ambiguous() -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(
                temporary_id="pv-01-level-a",
                reference="PV-01",
                occurrence_context="LEVEL A",
            )
        ]
    )
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="pv-01-level-a",
                reference="PV-01",
                occurrence_context="LEVEL B",
            )
        ]
    )

    merged = merge_enrichment_batches(discovery, [enrichment])

    assert merged.elements[0].occurrence_context == "LEVEL B"
    assert merged.elements[0].status == ExtractionStatus.AMBIGUOUS
    assert "occurrence_context_conflict" in merged.elements[0].missing_or_unknown
    assert any("occurrence_context_conflict" in warning for warning in merged.warnings)


def test_repeated_formal_reference_with_distinct_contexts_survives_final_extraction() -> None:
    discovery = GeminiDiscoveryResult(
        elements=[
            GeminiElementDiscovery(
                temporary_id="pv-a", reference="PV-01", occurrence_context="LEVEL A"
            ),
            GeminiElementDiscovery(
                temporary_id="pv-b", reference="PV-01", occurrence_context="LEVEL B"
            ),
            GeminiElementDiscovery(
                temporary_id="pv-c", reference="PV-01", occurrence_context="LEVEL C"
            ),
        ]
    )
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="pv-a",
                reference="PV-01",
                occurrence_context="LEVEL A",
                functional_type_raw="SLIDING_DOOR",
                operation_raw="SLIDING",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=5.25, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="pv-b",
                reference="PV-01",
                occurrence_context="LEVEL B",
                functional_type_raw="SLIDING_DOOR",
                operation_raw="SLIDING",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=4.10, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="pv-c",
                reference="PV-01",
                occurrence_context="LEVEL C",
                functional_type_raw="SLIDING_DOOR",
                operation_raw="SLIDING",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=2.60, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
        ]
    )

    extraction = enrichment_to_gemini_extraction(discovery, enrichment)

    assert [element.id for element in extraction.elements] == ["pv-a", "pv-b", "pv-c"]
    assert [element.reference for element in extraction.elements] == ["PV-01", "PV-01", "PV-01"]
    assert [element.occurrences[0].location for element in extraction.elements] == [
        "LEVEL A",
        "LEVEL B",
        "LEVEL C",
    ]


def test_repeated_reference_with_missing_context_and_distinct_dimensions_does_not_merge() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="pv-a",
                reference="PV-01",
                occurrence_context="LEVEL A",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=5.25, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="pv-unknown",
                reference="PV-01",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=4.10, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [element.temporary_id for element in result.elements] == ["pv-a", "pv-unknown"]
    assert result.elements[1].status == ExtractionStatus.AMBIGUOUS
    assert CONTEXT_INCOMPLETE_REASON in result.elements[1].missing_or_unknown
    assert [decision.action for decision in decisions] == ["KEEP", "KEEP"]
    assert {decision.reason for decision in decisions} == {CONTEXT_INCOMPLETE_REASON}


def test_v_like_duplicate_with_different_operation_and_context_stays_separate() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="v-level-1",
                reference="V-04",
                occurrence_context="NIVEL 1",
                functional_type_raw="SLIDING_WINDOW",
                operation_raw="SLIDING",
                measurements=[GeminiEnrichmentMeasurement(type="width", value=1.20, unit="m")],
            ),
            GeminiElementEnrichment(
                temporary_id="v-level-2",
                reference="V-04",
                occurrence_context="NIVEL 2",
                functional_type_raw="WINDOW",
                operation_raw="SWING",
                measurements=[GeminiEnrichmentMeasurement(type="width", value=0.90, unit="m")],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert [element.temporary_id for element in result.elements] == ["v-level-1", "v-level-2"]
    assert [element.operation_raw for element in result.elements] == ["SLIDING", "SWING"]
    assert all(decision.action == "KEEP" for decision in decisions)


def test_casa_pereira_repeated_references_keep_context_identity_regression() -> None:
    enrichment = GeminiEnrichmentResult(
        elements=[
            GeminiElementEnrichment(
                temporary_id="sotano-pv-01",
                reference="PV-01",
                occurrence_context="SOTANO",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=5.25, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-1-pv-01",
                reference="PV-01",
                occurrence_context="NIVEL 1",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=4.10, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-2-pv-01",
                reference="PV-01",
                occurrence_context="NIVEL 2",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=2.60, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-1-v-04",
                reference="V-04",
                occurrence_context="NIVEL 1",
                operation_raw="SLIDING",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=1.20, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=1.65, unit="m"),
                ],
            ),
            GeminiElementEnrichment(
                temporary_id="nivel-2-v-04",
                reference="V-04",
                occurrence_context="NIVEL 2",
                operation_raw="SWING",
                measurements=[
                    GeminiEnrichmentMeasurement(type="width", value=0.90, unit="m"),
                    GeminiEnrichmentMeasurement(type="height", value=2.50, unit="m"),
                ],
            ),
        ]
    )

    result, decisions = reconcile_inventory_candidates(enrichment)

    assert len(result.elements) == 5
    assert [element.temporary_id for element in result.elements] == [
        "sotano-pv-01",
        "nivel-1-pv-01",
        "nivel-2-pv-01",
        "nivel-1-v-04",
        "nivel-2-v-04",
    ]
    assert [element.occurrence_context for element in result.elements] == [
        "SOTANO",
        "NIVEL 1",
        "NIVEL 2",
        "NIVEL 1",
        "NIVEL 2",
    ]
    assert all(decision.action == "KEEP" for decision in decisions)


def _glass_element(
    temporary_id: str,
    glass: list[GeminiEnrichmentGlass],
    *,
    reference: str = "V-01",
    occurrence_context: str | None = None,
) -> GeminiElementEnrichment:
    return GeminiElementEnrichment(
        temporary_id=temporary_id,
        reference=reference,
        occurrence_context=occurrence_context,
        quantity=1,
        measurements=[GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m")],
        glass=glass,
    )


def _profile_element(
    temporary_id: str,
    profiles: list[GeminiEnrichmentNamedItem],
    *,
    reference: str = "V-01",
    occurrence_context: str | None = None,
    functional_type_raw: str | None = None,
    operation_raw: str | None = None,
) -> GeminiElementEnrichment:
    return GeminiElementEnrichment(
        temporary_id=temporary_id,
        reference=reference,
        occurrence_context=occurrence_context,
        quantity=1,
        functional_type_raw=functional_type_raw,
        operation_raw=operation_raw,
        measurements=[GeminiEnrichmentMeasurement(type="width", value=1.2, unit="m")],
        profiles=profiles,
    )


def _inventory_trace(**stages: list[InventoryElementTrace]) -> InventoryDebugTrace:
    trace = InventoryDebugTrace()
    for stage, elements in stages.items():
        trace.add_stage(stage, elements)
    return trace


def _trace_item(
    reference: str | None,
    *,
    temporary_id: str | None = None,
    id: str | None = None,
    description: str | None = None,
) -> InventoryElementTrace:
    return InventoryElementTrace(
        id=id,
        temporary_id=temporary_id,
        reference=reference,
        description=description,
    )


def _extra_rows(items) -> set[tuple[str, str, str]]:
    return {(item.identity, item.reason, item.stage) for item in items}
