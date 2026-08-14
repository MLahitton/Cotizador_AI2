from app.models.common import ExtractionStatus
from app.models.gemini_extraction import (
    GeminiComponent,
    GeminiElement,
    GeminiEvidence,
    GeminiExtraction,
    GeminiGlass,
    GeminiMeasurement,
    GeminiNamedItem,
    GeminiOccurrence,
    GeminiRequirementInfo,
    GeminiVariant,
)
from app.services.gemini_extraction_mapper import map_gemini_extraction_to_requirement_extraction


def test_mapper_preserves_partial_element_without_required_specs() -> None:
    extraction = GeminiExtraction(
        requirement=GeminiRequirementInfo(description="Cotizar ventanas."),
        elements=[
            GeminiElement(
                reference="V1",
                description="Ventana mencionada sin categoria ni sistema.",
                missing_or_unknown=["categoria", "sistema"],
                status=ExtractionStatus.EXPLICIT,
                confidence=0.82,
            )
        ],
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction, model="gemini-test")

    assert len(result.elements) == 1
    assert result.elements[0].id == "element-1"
    assert result.elements[0].reference is not None
    assert result.elements[0].reference.value == "V1"
    assert result.elements[0].reference.status == ExtractionStatus.EXPLICIT
    assert result.elements[0].category is not None
    assert result.elements[0].category.status == ExtractionStatus.UNKNOWN
    assert result.elements[0].category.normalized is None
    assert result.elements[0].category.raw is None
    assert result.elements[0].missing_fields == ["categoria", "sistema"]
    assert result.extraction_metadata.model == "gemini-test"


def test_mapper_preserves_flexible_specs_components_and_evidence() -> None:
    extraction = GeminiExtraction(
        requirement=GeminiRequirementInfo(
            project_name="Edificio Norte",
            confidence=0.7,
            status=ExtractionStatus.INFERRED,
        ),
        evidence=[GeminiEvidence(text="Elemento V2 de 1.20 x 2.10 m")],
        elements=[
            GeminiElement(
                reference="V2",
                category="puerta corrediza",
                measurements=[
                    GeminiMeasurement(type="width", value=1.2, unit="m"),
                    GeminiMeasurement(type="height", value=2.1, unit="m"),
                ],
                geometry="rectangular",
                configuration="corrediza de dos hojas",
                quantity=3,
                glass=[
                    GeminiGlass(
                        type="laminado",
                        thickness="6 mm",
                        thickness_value=6,
                        thickness_unit="mm",
                    )
                ],
                materials=[GeminiNamedItem(description="aluminio")],
                profiles=[GeminiNamedItem(code="SNG-45", description="sistema mencionado")],
                finish="negro mate",
                accessories=[GeminiNamedItem(name="cerradura", quantity=1)],
                components=[GeminiComponent(name="hoja movil", type="panel", quantity=2)],
                status=ExtractionStatus.EXPLICIT,
            )
        ],
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id="source-1",
    )
    element = result.elements[0]

    assert result.requirement.project_name is not None
    assert result.requirement.project_name.status == ExtractionStatus.INFERRED
    assert element.category is not None
    assert element.category.raw == "puerta corrediza"
    assert element.measurements[0].value == 1.2
    assert element.geometry is not None
    assert element.geometry.description == "rectangular"
    assert element.configuration is not None
    assert element.configuration.raw_description == "corrediza de dos hojas"
    assert element.glass[0].thickness is not None
    assert element.glass[0].thickness.value == 6
    assert element.materials[0].raw_description == "aluminio"
    assert element.profiles[0].code is not None
    assert element.profiles[0].code.value == "SNG-45"
    assert element.finish is not None
    assert element.finish.raw_description == "negro mate"
    assert element.accessories[0].raw_description == "cerradura"
    assert element.components[0].name is not None
    assert element.components[0].name.value == "hoja movil"


def test_mapper_preserves_occurrence_context_when_structured() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="service-room-window",
                occurrences=[
                    GeminiOccurrence(
                        location="Habitacion de servicio (Page 1, Detail 1)",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.74,
                    )
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id="source-1",
    )
    occurrence = result.elements[0].occurrences[0]

    assert occurrence.location is not None
    assert occurrence.location.value == "Habitacion de servicio (Page 1, Detail 1)"
    assert occurrence.location.status == ExtractionStatus.EXPLICIT
    assert occurrence.confidence == 0.74


def test_mapper_does_not_create_occurrence_without_occurrence_data() -> None:
    extraction = GeminiExtraction(elements=[GeminiElement(id="item-1")])

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id="source-1",
    )

    assert result.elements[0].occurrences == []


def test_mapper_preserves_variant_context_when_structured() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="item-1",
                variants=[
                    GeminiVariant(
                        label="Opcion con vidrio claro",
                        reason="Alternativa indicada en nota",
                        status=ExtractionStatus.AMBIGUOUS,
                    )
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    variant = result.elements[0].variants[0]

    assert variant.label == "Opcion con vidrio claro"
    assert variant.reason == "Alternativa indicada en nota"
    assert variant.status == ExtractionStatus.AMBIGUOUS


def test_mapper_does_not_create_variant_without_variant_data() -> None:
    extraction = GeminiExtraction(elements=[GeminiElement(id="item-1")])

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.elements[0].variants == []


def test_mapper_preserves_complete_component_compatible_fields() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="item-1",
                components=[
                    GeminiComponent(
                        name="Hoja fija",
                        type="panel",
                        role="lateral",
                        description="Panel fijo lateral",
                        quantity=1,
                        measurements=[
                            GeminiMeasurement(
                                type="width",
                                value=0.9,
                                unit="m",
                                status=ExtractionStatus.EXPLICIT,
                            )
                        ],
                        geometry="rectangular",
                        configuration="fijo",
                        glass=[GeminiGlass(type="templado", thickness="6 mm")],
                        materials=[GeminiNamedItem(description="aluminio negro")],
                        profiles=[GeminiNamedItem(code="7038", name="perfil 4x4")],
                        finish="negro",
                        accessories=[GeminiNamedItem(name="empaque")],
                        status=ExtractionStatus.EXPLICIT,
                    )
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    component = result.elements[0].components[0]

    assert component.name is not None
    assert component.name.value == "Hoja fija"
    assert component.type is not None
    assert component.type.raw == "panel"
    assert component.role is not None
    assert component.role.raw == "lateral"
    assert component.quantity is not None
    assert component.quantity.value == 1
    assert component.measurements[0].value == 0.9
    assert component.geometry is not None
    assert component.geometry.description == "rectangular"
    assert component.configuration is not None
    assert component.configuration.raw_description == "fijo"
    assert component.glass[0].type is not None
    assert component.glass[0].type.raw == "templado"
    assert component.materials[0].raw_description == "aluminio negro"
    assert component.profiles[0].code is not None
    assert component.profiles[0].code.value == "7038"
    assert component.finish is not None
    assert component.finish.raw_description == "negro"
    assert component.accessories[0].raw_description == "empaque"
    assert component.notes == "Panel fijo lateral"


def test_mapper_promotes_element_evidence_to_root_evidence_and_ids() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-01",
                evidence=(
                    "Planos detalle 1 en pagina 1\n"
                    "Nota: TODOS LOS VIDRIOS SON DE ESPESOR DE 6mm"
                ),
                status=ExtractionStatus.EXPLICIT,
                confidence=0.81,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id="source-1",
    )
    element = result.elements[0]

    assert len(result.evidence) == 1
    assert result.evidence[0].id == "evidence-1"
    assert result.evidence[0].source_id == "source-1"
    assert result.evidence[0].extracted_text is not None
    assert "TODOS LOS VIDRIOS" in result.evidence[0].extracted_text
    assert result.evidence[0].page_number is None
    assert result.evidence[0].region is None
    assert element.evidence_ids == ["evidence-1"]
    assert element.technical_notes == []


def test_mapper_preserves_root_evidence_source_information() -> None:
    extraction = GeminiExtraction(
        evidence=[
            GeminiEvidence(
                id="ev-source",
                source_id="source-2",
                type="visual",
                text="Detalle de fachada",
                visual_description="Se observa vidrio",
                location="lamina A",
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.evidence[0].id == "ev-source"
    assert result.evidence[0].source_id == "source-2"
    assert result.evidence[0].type == "visual"
    assert result.evidence[0].extracted_text == "Detalle de fachada"
    assert result.evidence[0].visual_description == "Se observa vidrio"


def test_mapper_null_explicit_value_becomes_unknown_but_real_value_stays_explicit() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                reference=None,
                glass=[GeminiGlass(type=None, status=ExtractionStatus.EXPLICIT)],
                status=ExtractionStatus.EXPLICIT,
            ),
            GeminiElement(reference="V-01", status=ExtractionStatus.EXPLICIT),
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.elements[0].reference is not None
    assert result.elements[0].reference.value is None
    assert result.elements[0].reference.status == ExtractionStatus.UNKNOWN
    assert result.elements[0].glass[0].type is not None
    assert result.elements[0].glass[0].type.raw is None
    assert result.elements[0].glass[0].type.status == ExtractionStatus.UNKNOWN
    assert result.elements[1].reference is not None
    assert result.elements[1].reference.value == "V-01"
    assert result.elements[1].reference.status == ExtractionStatus.EXPLICIT


def test_mapper_preserves_missing_fields_repeated_references_order_and_metadata() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="first",
                reference="V-01",
                missing_or_unknown=["vidrio", "sistema"],
            ),
            GeminiElement(id="second", reference="V-01"),
        ],
        status=ExtractionStatus.AMBIGUOUS,
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        model_provider="google",
        model="gemini-test",
    )

    assert [element.id for element in result.elements] == ["first", "second"]
    assert [element.reference.value for element in result.elements if element.reference] == [
        "V-01",
        "V-01",
    ]
    assert result.elements[0].missing_fields == ["vidrio", "sistema"]
    assert result.extraction_metadata.model_provider == "google"
    assert result.extraction_metadata.model == "gemini-test"
    assert result.extraction_metadata.element_count == 2
    assert result.extraction_metadata.status == "ambiguous"
