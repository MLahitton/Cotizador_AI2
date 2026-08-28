import pytest

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
    GeminiRelationNote,
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


def test_mapper_preserves_structured_element_evidence_source_ids() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-01",
                evidence_items=[
                    GeminiEvidence(source_id="source-2", text="VIDRIO TEMPLADO 8mm"),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id=None,
        allowed_source_ids=["source-1", "source-2", "source-3"],
    )

    assert result.evidence[0].source_id == "source-2"
    assert result.evidence[0].extracted_text == "VIDRIO TEMPLADO 8mm"
    assert result.elements[0].evidence_ids == ["evidence-1"]


def test_mapper_preserves_two_element_evidences_from_two_sources() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-03",
                evidence_items=[
                    GeminiEvidence(source_id="source-1", text="Detalle V03"),
                    GeminiEvidence(source_id="source-2", text="Cuadro V03"),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id=None,
        allowed_source_ids=["source-1", "source-2"],
    )

    assert [evidence.source_id for evidence in result.evidence] == ["source-1", "source-2"]
    assert [evidence.extracted_text for evidence in result.evidence] == [
        "Detalle V03",
        "Cuadro V03",
    ]
    assert result.elements[0].evidence_ids == ["evidence-1", "evidence-2"]


def test_mapper_element_evidence_ids_only_include_own_evidence() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-01",
                evidence_items=[GeminiEvidence(source_id="source-1", text="Dato V01")],
            ),
            GeminiElement(
                id="v-02",
                evidence_items=[GeminiEvidence(source_id="source-2", text="Dato V02")],
            ),
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id=None,
        allowed_source_ids=["source-1", "source-2"],
    )

    assert result.elements[0].evidence_ids == ["evidence-1"]
    assert result.elements[1].evidence_ids == ["evidence-2"]


def test_mapper_unknown_source_id_warns_without_arbitrary_reassignment() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-01",
                evidence_items=[
                    GeminiEvidence(source_id="source-99", text="Dato con fuente desconocida"),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id=None,
        allowed_source_ids=["source-1", "source-2"],
    )

    assert result.evidence[0].source_id == "unknown"
    assert result.evidence[0].extracted_text == "Dato con fuente desconocida"
    assert any(warning.code == "unknown_evidence_source" for warning in result.warnings)


def test_mapper_multifile_missing_source_warns_without_source_1_fallback() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-01",
                evidence_items=[GeminiEvidence(text="Dato sin source_id")],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id=None,
        allowed_source_ids=["source-1", "source-2"],
    )

    assert result.evidence[0].source_id == "unknown"
    assert result.evidence[0].source_id != "source-1"
    assert any(warning.code == "missing_evidence_source" for warning in result.warnings)


def test_mapper_preserves_pdf_xlsx_and_image_evidence_fields() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-01",
                evidence_items=[
                    GeminiEvidence(source_id="source-1", text="PDF note", page_number=4),
                    GeminiEvidence(
                        source_id="source-2",
                        text="XLSX row",
                        sheet_name="APTO85",
                        cell_range="B14:H14",
                    ),
                    GeminiEvidence(
                        source_id="source-3",
                        text="Foto detalle",
                        visual_description="Se observa vidrio",
                    ),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id=None,
        allowed_source_ids=["source-1", "source-2", "source-3"],
    )

    assert result.evidence[0].page_number == 4
    assert result.evidence[0].sheet_name is None
    assert result.evidence[1].sheet_name == "APTO85"
    assert result.evidence[1].cell_range == "B14:H14"
    assert result.evidence[1].page_number is None
    assert result.evidence[2].visual_description == "Se observa vidrio"
    assert result.evidence[2].region is None


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


def test_mapper_normalizes_area_labels_without_losing_raw_values() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                measurements=[
                    GeminiMeasurement(type="custom", label=label, value=1.25, unit="m2")
                    for label in ["M2", "M²", "m2", "m²"]
                ]
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert [measurement.type for measurement in result.elements[0].measurements] == [
        "area",
        "area",
        "area",
        "area",
    ]
    assert [measurement.raw_label for measurement in result.elements[0].measurements] == [
        "M2",
        "M²",
        "m2",
        "m²",
    ]
    assert all(measurement.raw_value == 1.25 for measurement in result.elements[0].measurements)
    assert all(measurement.raw_unit == "m2" for measurement in result.elements[0].measurements)


def test_mapper_does_not_warn_when_reported_area_matches_dimensions() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                measurements=[
                    GeminiMeasurement(type="width", value=2000, unit="mm"),
                    GeminiMeasurement(type="height", value=2500, unit="mm"),
                    GeminiMeasurement(type="custom", label="M2", value=5.0, unit="m2"),
                ]
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert not any(warning.code == "MEASUREMENT_AREA_MISMATCH" for warning in result.warnings)


def test_mapper_accepts_small_reported_area_rounding_difference() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                measurements=[
                    GeminiMeasurement(type="width", value=3090, unit="mm"),
                    GeminiMeasurement(type="height", value=1900, unit="mm"),
                    GeminiMeasurement(type="custom", label="M2", value=5.87, unit="m2"),
                ]
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert not any(warning.code == "MEASUREMENT_AREA_MISMATCH" for warning in result.warnings)


def test_mapper_warns_for_large_reported_area_mismatch_without_overwriting_value() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="element-pv",
                evidence_items=[
                    GeminiEvidence(
                        id="evidence-123",
                        source_id="source-1",
                        text="ANCHO:5320 ALTO:2500 M2:1.33",
                    )
                ],
                measurements=[
                    GeminiMeasurement(type="width", value=5320, unit="mm"),
                    GeminiMeasurement(type="height", value=2500, unit="mm"),
                    GeminiMeasurement(type="custom", label="M2", value=1.33, unit="m2"),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id="source-1",
    )

    area = result.elements[0].measurements[2]
    assert area.type == "area"
    assert area.value == 1.33
    assert area.status == ExtractionStatus.EXPLICIT
    warning = next(
        warning
        for warning in result.warnings
        if warning.code == "MEASUREMENT_AREA_MISMATCH"
    )
    assert "13.30 m2" in warning.message
    assert warning.element_ids == ["element-pv"]
    assert warning.evidence_ids == ["evidence-123"]
    assert warning.source_ids == ["source-1"]


def test_mapper_does_not_warn_without_reported_area() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                measurements=[
                    GeminiMeasurement(type="width", value=2000, unit="mm"),
                    GeminiMeasurement(type="height", value=2500, unit="mm"),
                ]
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert [measurement.type for measurement in result.elements[0].measurements] == [
        "width",
        "height",
    ]
    assert not any(warning.code == "MEASUREMENT_AREA_MISMATCH" for warning in result.warnings)


def test_mapper_does_not_warn_for_reported_area_without_dimensions() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                measurements=[
                    GeminiMeasurement(type="custom", label="M2", value=1.33, unit="m2"),
                ]
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.elements[0].measurements[0].type == "area"
    assert not any(warning.code == "MEASUREMENT_AREA_MISMATCH" for warning in result.warnings)


def test_mapper_propagates_single_element_evidence_to_measurements() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                evidence_items=[
                    GeminiEvidence(
                        id="evidence-123",
                        source_id="source-1",
                        text="ALTO:1800 ANCHO:600 M2:1.08",
                    )
                ],
                measurements=[
                    GeminiMeasurement(type="height", value=1800, unit="mm"),
                    GeminiMeasurement(type="width", value=600, unit="mm"),
                    GeminiMeasurement(type="custom", label="M2", value=1.08, unit="m2"),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id="source-1",
    )

    assert [measurement.evidence_ids for measurement in result.elements[0].measurements] == [
        ["evidence-123"],
        ["evidence-123"],
        ["evidence-123"],
    ]


def test_mapper_does_not_assign_ambiguous_multiple_evidences_to_measurements() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                evidence_items=[
                    GeminiEvidence(id="evidence-1", source_id="source-1", text="ANCHO:600"),
                    GeminiEvidence(id="evidence-2", source_id="source-1", text="ALTO:1800"),
                ],
                measurements=[
                    GeminiMeasurement(type="height", value=1800, unit="mm"),
                    GeminiMeasurement(type="width", value=600, unit="mm"),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(
        extraction,
        default_source_id="source-1",
    )

    assert [measurement.evidence_ids for measurement in result.elements[0].measurements] == [
        [],
        [],
    ]


def test_mapper_structures_oxxo_modulation_counts_without_losing_raw() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="v-oxxo",
                configuration="corrediza OXXO",
                operation="corrediza",
                modulation="OXXO",
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    configuration = result.elements[0].configuration

    assert configuration is not None
    assert configuration.raw_description == "corrediza OXXO"
    assert configuration.modulation == "OXXO"
    assert configuration.arrangement == "OXXO"
    assert configuration.panel_count is not None
    assert configuration.panel_count.value == 4
    assert configuration.movable_panel_count is not None
    assert configuration.movable_panel_count.value == 2
    assert configuration.fixed_panel_count is not None
    assert configuration.fixed_panel_count.value == 2
    assert configuration.operation is not None
    assert configuration.operation.normalized == "SLIDING"


def test_mapper_structures_pocket_feature_and_preserves_raw_configuration() -> None:
    raw = "XX PARA GUARDARSE EN UN BOLSILLO"
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="pocket",
                configuration=raw,
                modulation="XX",
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    configuration = result.elements[0].configuration

    assert configuration is not None
    assert configuration.raw_description == raw
    assert configuration.modulation == "XX"
    assert configuration.panel_count is not None
    assert configuration.panel_count.value == 2
    assert "POCKET" in configuration.special_features
    assert result.elements[0].profiles == []


def test_mapper_structures_projecting_with_lower_fixed_panel() -> None:
    raw = "proyectante superior con fijo inferior"
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="projecting-fixed",
                functional_type=raw,
                operation=raw,
                configuration=raw,
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.functional_type is not None
    assert element.functional_type.normalized == "PROJECTING"
    assert element.configuration is not None
    assert element.configuration.operation is not None
    assert element.configuration.operation.normalized == "PROJECTING"
    assert "ASSOCIATED_FIXED_PANEL" in element.configuration.special_features
    assert "LOWER_FIXED_PANEL" in element.configuration.special_features


def test_mapper_normalizes_l_shape_and_triangular_geometry() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(id="l", geometry="estructura en L", geometry_type="estructura en L"),
            GeminiElement(id="tri", geometry="triangular", geometry_type="triangular"),
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.elements[0].geometry is not None
    assert result.elements[0].geometry.normalized_type == "L_SHAPE"
    assert result.elements[0].geometry.description == "estructura en L"
    assert result.elements[1].geometry is not None
    assert result.elements[1].geometry.normalized_type == "TRIANGULAR"
    assert result.elements[1].geometry.description == "triangular"


def test_mapper_structures_sliding_door_without_selecting_sg_system() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="door",
                category="puerta corrediza",
                functional_type="puerta corrediza",
                operation="corrediza",
                profiles=[GeminiNamedItem(code="3831", description="sistema solicitado")],
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.functional_type is not None
    assert element.functional_type.normalized == "SLIDING_DOOR"
    assert element.configuration is not None
    assert element.configuration.operation is not None
    assert element.configuration.operation.normalized == "SLIDING"
    assert element.profiles[0].code is not None
    assert element.profiles[0].code.value == "3831"
    assert element.profiles[0].name is None


def test_mapper_does_not_invent_functional_type_for_ambiguous_sliding() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="ambiguous",
                configuration="corrediza",
                operation="corrediza",
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.functional_type is None
    assert element.configuration is not None
    assert element.configuration.operation is not None
    assert element.configuration.operation.normalized == "SLIDING"


@pytest.mark.parametrize(
    ("raw_functional_type", "expected"),
    [
        ("fixed", "FIXED"),
        ("proyectante", "PROJECTING"),
        ("casement", "CASEMENT"),
        ("puerta batiente", "SWING_DOOR"),
        ("sliding window", "SLIDING_WINDOW"),
        ("sliding door", "SLIDING_DOOR"),
        ("folding window", "FOLDING_WINDOW"),
        ("folding door", "FOLDING_DOOR"),
        ("pergola", "PERGOLA"),
        ("division de bano", "SHOWER_DIVISION"),
        ("shower division", "SHOWER_DIVISION"),
        ("rejilla", "GRILLE"),
        ("louver", "GRILLE"),
        ("claraboya", "SKYLIGHT"),
    ],
)
def test_mapper_aligns_functional_type_vocabulary(
    raw_functional_type: str,
    expected: str,
) -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="item",
                functional_type=raw_functional_type,
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    functional_type = result.elements[0].functional_type

    assert functional_type is not None
    assert functional_type.raw == raw_functional_type
    assert functional_type.normalized == expected
    assert functional_type.status == ExtractionStatus.EXPLICIT


def test_mapper_keeps_pocket_as_special_feature_not_commercial_family() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="pocket-door",
                functional_type="puerta corrediza",
                operation="corrediza",
                configuration="XX PARA GUARDARSE EN UN BOLSILLO",
                modulation="XX",
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.functional_type is not None
    assert element.functional_type.normalized == "SLIDING_DOOR"
    assert element.configuration is not None
    assert "POCKET" in element.configuration.special_features
    assert element.profiles == []


@pytest.mark.parametrize(
    ("raw_finish", "finish_type", "color", "texture"),
    [
        ("negro pintura al horno", "PAINTED", "BLACK", None),
        ("blanco", None, "WHITE", None),
        ("gris", None, "GRAY", None),
        ("champaña", None, "CHAMPAGNE", None),
        ("anodizado blanco mate", "ANODIZED", "WHITE", "MATTE"),
        ("acero inoxidable", "STAINLESS_STEEL", None, None),
        ("inox", "STAINLESS_STEEL", None, None),
    ],
)
def test_mapper_structures_finish_without_losing_raw_or_inventing_codes(
    raw_finish: str,
    finish_type: str | None,
    color: str | None,
    texture: str | None,
) -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="finish",
                finish=raw_finish,
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    finish = result.elements[0].finish

    assert finish is not None
    assert finish.raw_description == raw_finish
    assert finish.normalized_type == finish_type
    assert (finish.color.normalized if finish.color else None) == color
    assert (finish.texture.normalized if finish.texture else None) == texture
    assert finish.code is None


def test_mapper_preserves_explicit_finish_code_without_inventing_catalog_code() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(id="without-code", finish="negro pintura al horno"),
            GeminiElement(id="with-code", finish="negro pintura al horno PP13"),
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.elements[0].finish is not None
    assert result.elements[0].finish.code is None
    assert result.elements[1].finish is not None
    assert result.elements[1].finish.code is not None
    assert result.elements[1].finish.code.value == "PP13"


def test_mapper_preserves_assembly_type_and_component_segments() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="corner-window",
                reference="V-25",
                category="ventana fija",
                geometry_type="esquina",
                assembly_type="corner",
                quantity=1,
                components=[
                    GeminiComponent(
                        role="FIXED",
                        quantity=1,
                        measurements=[
                            GeminiMeasurement(type="width", value=1.79, unit="m"),
                            GeminiMeasurement(type="height", value=2.8, unit="m"),
                        ],
                        geometry="rectangular",
                    ),
                    GeminiComponent(
                        role="FIXED",
                        quantity=1,
                        measurements=[
                            GeminiMeasurement(type="width", value=2.33, unit="m"),
                            GeminiMeasurement(type="height", value=2.8, unit="m"),
                        ],
                        geometry="rectangular",
                    ),
                ],
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.assembly_type == "CORNER"
    assert len(element.components) == 2
    assert element.quantity is not None
    assert element.quantity.value == 1
    assert element.components[0].role is not None
    assert element.components[0].role.normalized == "FIXED"
    assert element.components[0].measurements[0].value == 1.79
    assert element.components[1].measurements[0].value == 2.33


def test_mapper_normalizes_known_component_roles_and_preserves_unknown_raw() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="roles",
                components=[
                    GeminiComponent(role=" PROJECTING ", status=ExtractionStatus.EXPLICIT),
                    GeminiComponent(role="fixed", status=ExtractionStatus.INFERRED),
                    GeminiComponent(role="casement", status=ExtractionStatus.EXPLICIT),
                    GeminiComponent(role="folding", status=ExtractionStatus.EXPLICIT),
                    GeminiComponent(role="grille", status=ExtractionStatus.EXPLICIT),
                    GeminiComponent(role="louver", status=ExtractionStatus.EXPLICIT),
                    GeminiComponent(role="custom bracket", status=ExtractionStatus.EXPLICIT),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    roles = [component.role for component in result.elements[0].components]

    assert roles[0] is not None
    assert roles[0].raw == " PROJECTING "
    assert roles[0].normalized == "PROJECTING"
    assert roles[0].status == ExtractionStatus.EXPLICIT
    assert roles[1] is not None
    assert roles[1].raw == "fixed"
    assert roles[1].normalized == "FIXED"
    assert roles[1].status == ExtractionStatus.INFERRED
    assert roles[2] is not None
    assert roles[2].normalized == "CASEMENT"
    assert roles[3] is not None
    assert roles[3].normalized == "FOLDING"
    assert roles[4] is not None
    assert roles[4].normalized == "GRILLE"
    assert roles[5] is not None
    assert roles[5].normalized == "LOUVER"
    assert roles[6] is not None
    assert roles[6].raw == "custom bracket"
    assert roles[6].normalized is None


def test_mapper_infers_composite_assembly_from_projecting_and_fixed_components() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="projecting-with-fixed",
                reference="V-01",
                category="ventana proyectante con fijo",
                components=[
                    GeminiComponent(role="PROJECTING", quantity=1),
                    GeminiComponent(role="FIXED", quantity=1),
                ],
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert result.elements[0].assembly_type == "COMPOSITE"


@pytest.mark.parametrize(
    ("components", "expected_roles"),
    [
        (["SLIDING", "GRILLE"], ["SLIDING", "GRILLE"]),
        (["PROJECTING", "FIXED"], ["PROJECTING", "FIXED"]),
        (["SWING", "FIXED"], ["SWING", "FIXED"]),
        (["SLIDING", "FIXED"], ["SLIDING", "FIXED"]),
    ],
)
def test_mapper_preserves_composite_functional_components(
    components: list[str],
    expected_roles: list[str],
) -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="composite-item",
                category="ventana",
                components=[
                    GeminiComponent(role=role, quantity=1)
                    for role in components
                ],
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.assembly_type == "COMPOSITE"
    assert [
        component.role.normalized
        for component in element.components
        if component.role is not None
    ] == expected_roles


def test_mapper_preserves_grille_only_without_artificial_mobile_component() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="grille-only",
                category="rejilla",
                components=[GeminiComponent(role="GRILLE", quantity=1)],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.assembly_type is None
    assert len(element.components) == 1
    assert element.components[0].role is not None
    assert element.components[0].role.normalized == "GRILLE"


def test_mapper_preserves_two_fixed_components_with_distinct_geometry() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="two-fixed-panels",
                category="ventana fija",
                components=[
                    GeminiComponent(
                        role="FIXED",
                        measurements=[GeminiMeasurement(type="width", value=1.2, unit="m")],
                        geometry="left panel",
                    ),
                    GeminiComponent(
                        role="FIXED",
                        measurements=[GeminiMeasurement(type="width", value=1.8, unit="m")],
                        geometry="right panel",
                    ),
                ],
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.assembly_type == "MULTI_MODULE"
    assert len(element.components) == 2
    assert [component.measurements[0].value for component in element.components] == [1.2, 1.8]


def test_mapper_derives_components_from_explicit_sliding_and_grille_signals() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="sliding-grille-flat",
                category="ventana",
                functional_type="rejilla",
                operation="corrediza",
                description="Ventana de dos hojas corredizas con rejilla asociada",
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    roles = [
        component.role.normalized
        for component in result.elements[0].components
        if component.role is not None
    ]

    assert result.elements[0].assembly_type == "COMPOSITE"
    assert roles == ["GRILLE", "SLIDING"]


def test_mapper_derives_projecting_and_fixed_components_from_item_signals() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="projecting-fixed-flat",
                category="ventana proyectante",
                functional_type="proyectante",
                configuration="con pano fijo inferior",
                special_features=["LOWER_FIXED_PANEL"],
                description="Ventana proyectante con pano fijo inferior",
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    roles = [
        component.role.normalized
        for component in result.elements[0].components
        if component.role is not None
    ]

    assert result.elements[0].assembly_type == "COMPOSITE"
    assert roles == ["PROJECTING", "FIXED"]


def test_mapper_preserves_same_reference_as_distinct_elements() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(id="v-01-a", reference="V-01", description="Nivel 1"),
            GeminiElement(id="v-01-b", reference="V-01", description="Nivel 2"),
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)

    assert [element.id for element in result.elements] == ["v-01-a", "v-01-b"]
    assert [element.reference.value for element in result.elements if element.reference] == [
        "V-01",
        "V-01",
    ]


def test_mapper_derives_swing_and_fixed_components_from_item_signals() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="swing-fixed-flat",
                reference="P-01",
                category="puerta batiente",
                functional_type="puerta batiente",
                description="Puerta batiente de una hoja con pano fijo lateral",
                quantity=4,
                status=ExtractionStatus.EXPLICIT,
                confidence=0.78,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]
    roles = [
        component.role.normalized
        for component in element.components
        if component.role is not None
    ]

    assert len(result.elements) == 1
    assert element.assembly_type == "COMPOSITE"
    assert roles == ["SWING", "FIXED"]
    assert element.quantity is not None
    assert element.quantity.value == 4


def test_mapper_preserves_fixed_only_without_artificial_mobile_component() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="fixed-only",
                category="ventana fija",
                functional_type="fijo",
                description="Ventana fija en un solo pano",
                panel_count=2,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]
    roles = [
        component.role.normalized
        for component in element.components
        if component.role is not None
    ]

    assert roles == ["FIXED"]
    assert element.assembly_type is None


def test_mapper_preserves_swing_only_without_artificial_fixed_component() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="swing-only",
                category="puerta batiente",
                functional_type="puerta batiente",
                description="Puerta batiente de una hoja",
                panel_count=2,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]
    roles = [
        component.role.normalized
        for component in element.components
        if component.role is not None
    ]

    assert roles == ["SWING"]
    assert element.assembly_type is None


def test_mapper_preserves_conflicting_text_and_structure_as_reviewable_evidence() -> None:
    extraction = GeminiExtraction(
        evidence=[
            GeminiEvidence(id="ev-text", text="Texto: puerta batiente"),
            GeminiEvidence(id="ev-drawing", visual_description="Dibujo: pano fijo lateral"),
        ],
        elements=[
            GeminiElement(
                id="conflicting-assembly",
                reference="P-02",
                category="puerta",
                functional_type="puerta batiente",
                description="Puerta batiente con dibujo de pano fijo lateral",
                components=[
                    GeminiComponent(
                        role="SWING",
                        status=ExtractionStatus.EXPLICIT,
                        confidence=0.82,
                        evidence="Texto especifica hoja batiente.",
                    ),
                    GeminiComponent(
                        role="FIXED",
                        status=ExtractionStatus.AMBIGUOUS,
                        confidence=0.45,
                        evidence="Dibujo sugiere pano fijo lateral.",
                    ),
                ],
                conflicts=["Texto y dibujo no coinciden sobre el fijo lateral"],
                status=ExtractionStatus.AMBIGUOUS,
                confidence=0.52,
            )
        ],
        conflicts=[
            GeminiRelationNote(
                description="Texto y dibujo no coinciden sobre el fijo lateral",
                type="functional_structure_conflict",
                status=ExtractionStatus.AMBIGUOUS,
                confidence=0.52,
            )
        ],
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]
    roles = [
        component.role.normalized
        for component in element.components
        if component.role is not None
    ]

    assert element.assembly_type == "COMPOSITE"
    assert roles == ["SWING", "FIXED"]
    assert element.confidence == 0.52
    assert element.components[1].role is not None
    assert element.components[1].role.status == ExtractionStatus.AMBIGUOUS
    assert element.components[1].confidence == 0.45
    assert result.conflicts[0].field == "functional_structure_conflict"

def test_mapper_preserves_incomplete_reference_with_reviewable_unknowns() -> None:
    extraction = GeminiExtraction(
        elements=[
            GeminiElement(
                id="partial-window",
                reference="V-99",
                category="ventana",
                measurements=[GeminiMeasurement(type="width", value=0.6, unit="m")],
                missing_or_unknown=["height", "glass"],
                status=ExtractionStatus.EXPLICIT,
            )
        ]
    )

    result = map_gemini_extraction_to_requirement_extraction(extraction)
    element = result.elements[0]

    assert element.id == "partial-window"
    assert element.reference is not None
    assert element.reference.value == "V-99"
    assert element.measurements[0].value == 0.6
    assert element.missing_fields == ["height", "glass"]
