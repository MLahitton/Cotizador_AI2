from pathlib import Path

import app.providers.gemini_extraction as provider_module
from app.models.common import ExtractionStatus
from app.models.gemini_discovery import GeminiDiscoveryResult, GeminiElementDiscovery
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentComponent,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentResult,
)
from app.models.requirement import ExtractionMetadata, Requirement, TokenUsage
from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini_extraction import (
    GeminiEnrichmentDebugCapture,
    GeminiExtractionProvider,
    GeminiFullPipelineDebugCapture,
)
from app.services.extraction_prompt import ELEMENT_SCOPE_PROMPT
from app.services.gemini_enrichment_pipeline import (
    build_discovery_batches,
    enrichment_to_gemini_extraction,
    merge_enrichment_batches,
)


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
        f'{{"temporary_id": "{temporary_id}", "reference": "{temporary_id}"}}'
        for temporary_id in temporary_ids
    )
    usage_metadata = _UsageMetadata(*usage) if usage else None
    return _FakeResponse(f'{{"elements": [{elements}]}}', usage_metadata)


def _scope_response(*items: tuple[str, str], usage: tuple[int, int, int] | None = None):
    elements = ", ".join(
        f'{{"temporary_id": "{temporary_id}", "scope": "{scope}"}}'
        for temporary_id, scope in items
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
    assert all(
        len(call["contents"]) == 3
        for call in provider._provider._client.models.calls
    )
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

    assert [element.id for element in mapper_calls[0].elements] == [
        "no-glass-word-only"
    ]


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
                            text="VIDRIO 6mm",
                            page_number=2,
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
    assert (
        element.occurrences[0].location
        == "Habitacion de servicio (Page 1, Detail 1)"
    )
    assert element.variants[0].label == "Alternativa con vidrio claro"
    assert element.evidence_items[0].source_id == "source-1"
    assert element.evidence_items[0].page_number == 2
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
