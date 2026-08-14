from app.models.common import ExtractionStatus, NormalizedValue, TraceableValue
from app.models.configuration import Configuration
from app.models.element import Element
from app.models.element_parts import Component, Occurrence, Variant
from app.models.evidence import Evidence
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
    GeminiVariant,
)
from app.models.geometry import Geometry
from app.models.measurement import EntityReference, Measurement
from app.models.relations import Conflict, Relationship
from app.models.requirement import ExtractionMetadata, Requirement, Warning
from app.models.requirement_extraction import RequirementExtraction
from app.models.specifications import (
    AccessorySpecification,
    FinishSpecification,
    GlassSpecification,
    MaterialSpecification,
    ProfileSpecification,
)


def map_gemini_extraction_to_requirement_extraction(
    extraction: GeminiExtraction,
    *,
    model_provider: str | None = "google",
    model: str | None = None,
    default_source_id: str = "text-input",
) -> RequirementExtraction:
    evidence = [
        _map_evidence(item, index, default_source_id)
        for index, item in enumerate(extraction.evidence, start=1)
    ]
    element_evidence = _build_element_evidence(
        extraction.elements,
        len(evidence),
        default_source_id,
    )
    evidence.extend(element_evidence)
    evidence_ids = [item.id for item in evidence]
    element_evidence_ids_by_element_id = _element_evidence_ids_by_element_id(
        extraction.elements,
        element_evidence,
    )
    fallback_evidence_ids = evidence_ids if len(evidence_ids) == 1 else []
    elements = [
        _map_element(
            item,
            index,
            element_evidence_ids_by_element_id.get(
                _element_id(item, index),
                fallback_evidence_ids,
            ),
        )
        for index, item in enumerate(extraction.elements, start=1)
    ]

    return RequirementExtraction(
        requirement=_map_requirement(extraction, evidence_ids),
        elements=elements,
        evidence=evidence,
        relationships=_map_relationship_notes(extraction.relationships),
        conflicts=_map_conflict_notes(extraction.conflicts),
        warnings=_map_unknowns_to_warnings(extraction.unknown_fields),
        extraction_metadata=ExtractionMetadata(
            model_provider=model_provider,
            model=model,
            element_count=len(elements),
            partial=bool(extraction.unknown_fields),
            status=extraction.status.value if extraction.status else "completed",
        ),
    )


def _status(status: ExtractionStatus | None, fallback: ExtractionStatus) -> ExtractionStatus:
    return status if status is not None else fallback


def _status_from_value(value: object | None) -> ExtractionStatus:
    return ExtractionStatus.EXPLICIT if value not in (None, "", []) else ExtractionStatus.UNKNOWN


def _status_for_value(
    value: object | None,
    status: ExtractionStatus | None,
) -> ExtractionStatus:
    if (
        value in (None, "", [])
        and status is not None
        and status != ExtractionStatus.NOT_APPLICABLE
    ):
        return ExtractionStatus.UNKNOWN

    return _status(status, _status_from_value(value))


def _is_marked_unknown(missing_fields: list[str], *aliases: str) -> bool:
    normalized_missing_fields = {_normalize_missing_field_name(field) for field in missing_fields}
    normalized_aliases = {_normalize_missing_field_name(alias) for alias in aliases}
    return bool(normalized_missing_fields.intersection(normalized_aliases))


def _normalize_missing_field_name(value: str) -> str:
    return value.casefold().replace("_", "").replace("-", "").replace(" ", "")


def _unknown_normalized(
    confidence: float | None,
    evidence_ids: list[str],
) -> NormalizedValue:
    return NormalizedValue(
        normalized=None,
        raw=None,
        status=ExtractionStatus.UNKNOWN,
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _unknown_traceable(
    confidence: float | None,
    evidence_ids: list[str],
) -> TraceableValue:
    return TraceableValue(
        value=None,
        status=ExtractionStatus.UNKNOWN,
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _traceable(
    value: str | int | float | bool | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str] | None = None,
    notes: str | None = None,
) -> TraceableValue | None:
    if value in (None, "") and status is None and confidence is None and notes is None:
        return None

    return TraceableValue(
        value=value,
        status=_status_for_value(value, status),
        confidence=confidence,
        evidence_ids=evidence_ids or [],
        notes=notes,
    )


def _normalized(
    raw: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str] | None = None,
    notes: str | None = None,
) -> NormalizedValue | None:
    if raw in (None, "") and status is None and confidence is None and notes is None:
        return None

    return NormalizedValue(
        normalized=None,
        raw=raw,
        status=_status_for_value(raw, status),
        confidence=confidence,
        evidence_ids=evidence_ids or [],
        notes=notes,
    )


def _map_requirement(extraction: GeminiExtraction, evidence_ids: list[str]) -> Requirement:
    requirement = extraction.requirement
    if requirement is None:
        return Requirement(confidence=extraction.confidence, evidence_ids=evidence_ids)

    notes = list(requirement.technical_notes)
    if requirement.notes:
        notes.append(requirement.notes)

    return Requirement(
        project_name=_traceable(
            requirement.project_name,
            requirement.status,
            requirement.confidence,
            evidence_ids,
        ),
        client_name=_traceable(
            requirement.client_name,
            requirement.status,
            requirement.confidence,
        ),
        location=_traceable(requirement.location, requirement.status, requirement.confidence),
        project_type=_traceable(
            requirement.project_type,
            requirement.status,
            requirement.confidence,
        ),
        description=requirement.description,
        dates=[
            TraceableValue(
                value=date.value if date.value is not None else date.text,
                status=_status_for_value(date.value or date.text, date.status),
                confidence=date.confidence,
                notes=date.notes or date.evidence,
            )
            for date in requirement.dates
        ],
        general_technical_notes=notes,
        evidence_ids=evidence_ids,
        confidence=requirement.confidence or extraction.confidence,
    )


def _map_element(
    item: GeminiElement,
    index: int,
    fallback_evidence_ids: list[str],
) -> Element:
    evidence_ids = fallback_evidence_ids
    missing_fields = list(item.missing_or_unknown)
    return Element(
        id=item.id or f"element-{index}",
        reference=_traceable(item.reference, item.status, item.confidence, evidence_ids),
        name=_traceable(item.name, item.status, item.confidence, evidence_ids),
        category=_map_optional_normalized_field(
            item.category,
            item.status,
            item.confidence,
            evidence_ids,
            missing_fields,
            "category",
            "categoria",
        ),
        description=item.description,
        occurrences=[
            _map_occurrence(occurrence, occurrence_index, evidence_ids)
            for occurrence_index, occurrence in enumerate(item.occurrences, start=1)
        ],
        variants=[
            _map_variant(variant, variant_index, evidence_ids)
            for variant_index, variant in enumerate(item.variants, start=1)
        ],
        components=[
            _map_component(component, component_index, evidence_ids)
            for component_index, component in enumerate(item.components, start=1)
        ],
        geometry=_map_geometry(item.geometry, item.status, item.confidence, evidence_ids),
        measurements=[_map_measurement(measurement) for measurement in item.measurements],
        quantity=_map_optional_traceable_field(
            item.quantity,
            item.status,
            item.confidence,
            evidence_ids,
            missing_fields,
            "quantity",
            "cantidad",
        ),
        configuration=_map_configuration(
            item.configuration,
            _optional_field_status(
                item.configuration,
                item.status,
                missing_fields,
                "configuration",
            ),
            item.confidence,
            evidence_ids,
        ),
        glass=[_map_glass(glass, evidence_ids) for glass in item.glass],
        materials=[_map_material(material, evidence_ids) for material in item.materials],
        profiles=[_map_profile(profile, evidence_ids) for profile in item.profiles],
        finish=_map_finish(
            item.finish,
            _optional_field_status(item.finish, item.status, missing_fields, "finish", "acabado"),
            item.confidence,
            evidence_ids,
        ),
        accessories=[_map_accessory(accessory, evidence_ids) for accessory in item.accessories],
        technical_notes=[],
        evidence_ids=evidence_ids,
        missing_fields=missing_fields,
        confidence=item.confidence,
        notes=item.notes,
    )


def _map_optional_normalized_field(
    raw: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
    missing_fields: list[str],
    *aliases: str,
) -> NormalizedValue | None:
    if raw in (None, "") and _is_marked_unknown(missing_fields, *aliases):
        return _unknown_normalized(confidence, evidence_ids)

    return _normalized(raw, status, confidence, evidence_ids)


def _map_optional_traceable_field(
    value: str | int | float | bool | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
    missing_fields: list[str],
    *aliases: str,
) -> TraceableValue | None:
    if value in (None, "") and _is_marked_unknown(missing_fields, *aliases):
        return _unknown_traceable(confidence, evidence_ids)

    return _traceable(value, status, confidence, evidence_ids)


def _optional_field_status(
    value: object | None,
    status: ExtractionStatus | None,
    missing_fields: list[str],
    *aliases: str,
) -> ExtractionStatus | None:
    if value in (None, "") and _is_marked_unknown(missing_fields, *aliases):
        return ExtractionStatus.UNKNOWN

    return status


def _map_measurement(item: GeminiMeasurement) -> Measurement:
    return Measurement(
        type=item.type or "unspecified",
        raw_label=item.label or item.text,
        value=item.value,
        unit=item.unit,
        raw_value=item.value,
        raw_unit=item.unit,
        status=_status_for_value(item.value or item.text, item.status),
        confidence=item.confidence,
        notes=item.notes or item.evidence,
    )


def _map_geometry(
    description: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> Geometry | None:
    if description in (None, "") and status is None and confidence is None:
        return None

    return Geometry(
        raw_type=description,
        description=description,
        status=_status_for_value(description, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _map_configuration(
    description: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> Configuration | None:
    if description in (None, "") and status is None and confidence is None:
        return None

    return Configuration(
        raw_description=description,
        status=_status_for_value(description, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _map_glass(item: GeminiGlass, evidence_ids: list[str]) -> GlassSpecification:
    raw_status = _status_for_value(
        item.type or item.thickness or item.color or item.description,
        item.status,
    )
    thickness = None
    if item.thickness_value is not None or item.thickness:
        thickness = Measurement(
            type="thickness",
            raw_label=item.thickness,
            value=item.thickness_value,
            unit=item.thickness_unit,
            raw_value=item.thickness_value,
            raw_unit=item.thickness_unit,
            status=raw_status,
            confidence=item.confidence,
            evidence_ids=evidence_ids,
        )

    return GlassSpecification(
        type=_normalized(item.type, item.status, item.confidence, evidence_ids),
        thickness=thickness,
        color=_normalized(item.color, item.status, item.confidence, evidence_ids),
        treatment=_normalized(item.treatment, item.status, item.confidence, evidence_ids),
        composition=item.composition or item.description,
        status=raw_status,
        confidence=item.confidence,
        evidence_ids=evidence_ids,
        notes=item.notes or item.evidence,
    )


def _map_material(item: GeminiNamedItem, evidence_ids: list[str]) -> MaterialSpecification:
    raw = item.description or item.name or item.type or item.code
    return MaterialSpecification(
        raw_description=raw,
        status=_status_for_value(raw, item.status),
        confidence=item.confidence,
        evidence_ids=evidence_ids,
        notes=item.notes or item.evidence,
    )


def _map_profile(item: GeminiNamedItem, evidence_ids: list[str]) -> ProfileSpecification:
    return ProfileSpecification(
        code=_traceable(item.code, item.status, item.confidence, evidence_ids),
        name=_traceable(item.name, item.status, item.confidence, evidence_ids),
        raw_description=item.description or item.type,
        role=_normalized(item.role, item.status, item.confidence, evidence_ids),
        status=_status_for_value(item.name or item.code or item.description, item.status),
        confidence=item.confidence,
        evidence_ids=evidence_ids,
        notes=item.notes or item.evidence,
    )


def _map_finish(
    description: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> FinishSpecification | None:
    if description in (None, "") and status is None and confidence is None:
        return None

    return FinishSpecification(
        raw_description=description,
        status=_status_for_value(description, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _map_accessory(item: GeminiNamedItem, evidence_ids: list[str]) -> AccessorySpecification:
    raw = item.description or item.name or item.type or item.code
    return AccessorySpecification(
        raw_description=raw,
        quantity=_traceable(item.quantity, item.status, item.confidence, evidence_ids),
        placement=item.role,
        status=_status_for_value(raw, item.status),
        confidence=item.confidence,
        evidence_ids=evidence_ids,
        notes=item.notes or item.evidence,
    )


def _map_occurrence(
    item: GeminiOccurrence,
    index: int,
    evidence_ids: list[str],
) -> Occurrence:
    return Occurrence(
        id=item.id or f"occurrence-{index}",
        location=_traceable(item.location, item.status, item.confidence, evidence_ids),
        level=_traceable(item.level, item.status, item.confidence, evidence_ids),
        typology=_traceable(item.typology, item.status, item.confidence, evidence_ids),
        quantity=_traceable(item.quantity, item.status, item.confidence, evidence_ids),
        measurements=[_map_measurement(measurement) for measurement in item.measurements],
        evidence_ids=evidence_ids,
        confidence=item.confidence,
        notes=item.notes or item.evidence,
    )


def _map_variant(item: GeminiVariant, index: int, evidence_ids: list[str]) -> Variant:
    return Variant(
        id=item.id or f"variant-{index}",
        label=item.label,
        reason=item.reason,
        measurements=[_map_measurement(measurement) for measurement in item.measurements],
        configuration=_map_configuration(
            item.configuration,
            item.status,
            item.confidence,
            evidence_ids,
        ),
        glass=[_map_glass(glass, evidence_ids) for glass in item.glass],
        materials=[_map_material(material, evidence_ids) for material in item.materials],
        profiles=[_map_profile(profile, evidence_ids) for profile in item.profiles],
        finish=_map_finish(item.finish, item.status, item.confidence, evidence_ids),
        accessories=[_map_accessory(accessory, evidence_ids) for accessory in item.accessories],
        occurrence_ids=list(item.occurrence_ids),
        evidence_ids=evidence_ids,
        status=item.status,
        confidence=item.confidence,
        notes=item.notes or item.evidence,
    )


def _map_component(item: GeminiComponent, index: int, evidence_ids: list[str]) -> Component:
    return Component(
        id=item.id or f"component-{index}",
        name=_traceable(item.name, item.status, item.confidence, evidence_ids),
        type=_normalized(item.type, item.status, item.confidence, evidence_ids),
        role=_normalized(item.role, item.status, item.confidence, evidence_ids),
        quantity=_traceable(item.quantity, item.status, item.confidence, evidence_ids),
        geometry=_map_geometry(item.geometry, item.status, item.confidence, evidence_ids),
        measurements=[_map_measurement(measurement) for measurement in item.measurements],
        configuration=_map_configuration(
            item.configuration,
            item.status,
            item.confidence,
            evidence_ids,
        ),
        glass=[_map_glass(glass, evidence_ids) for glass in item.glass],
        materials=[_map_material(material, evidence_ids) for material in item.materials],
        profiles=[_map_profile(profile, evidence_ids) for profile in item.profiles],
        finish=_map_finish(item.finish, item.status, item.confidence, evidence_ids),
        accessories=[_map_accessory(accessory, evidence_ids) for accessory in item.accessories],
        evidence_ids=evidence_ids,
        status=item.status,
        confidence=item.confidence,
        notes=item.notes or item.description or item.evidence,
    )


def _map_evidence(item: GeminiEvidence, index: int, default_source_id: str) -> Evidence:
    return Evidence(
        id=item.id or f"evidence-{index}",
        source_id=item.source_id or default_source_id,
        type=item.type or "text",
        extracted_text=item.text,
        visual_description=item.visual_description or item.location,
        status=_status_for_value(item.text or item.visual_description, item.status),
        confidence=item.confidence,
        notes=item.notes,
    )


def _build_element_evidence(
    elements: list[GeminiElement],
    existing_count: int,
    default_source_id: str,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for element in elements:
        if not element.evidence:
            continue

        evidence.append(
            Evidence(
                id=f"evidence-{existing_count + len(evidence) + 1}",
                source_id=default_source_id,
                type="text",
                extracted_text=element.evidence,
                status=_status_for_value(element.evidence, element.status),
                confidence=element.confidence,
            )
        )

    return evidence


def _element_evidence_ids_by_element_id(
    elements: list[GeminiElement],
    evidence: list[Evidence],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    evidence_index = 0
    for element_index, element in enumerate(elements, start=1):
        if not element.evidence:
            continue

        result[_element_id(element, element_index)] = [evidence[evidence_index].id]
        evidence_index += 1

    return result


def _element_id(item: GeminiElement, index: int) -> str:
    return item.id or f"element-{index}"


def _map_relationship_notes(items: list[GeminiRelationNote]) -> list[Relationship]:
    return [
        Relationship(
            id=f"relationship-{index}",
            type=NormalizedValue(
                raw=item.type or item.description,
                status=_status_for_value(item.type or item.description, item.status),
                confidence=item.confidence,
            ),
            from_entity=EntityReference(
                entity_type="element",
                entity_id=item.from_element or "unknown",
            ),
            to_entity=EntityReference(
                entity_type="element",
                entity_id=item.to_element or "unknown",
            ),
            status=_status_for_value(item.description, item.status),
            confidence=item.confidence,
            reason=item.description,
            notes=item.notes or item.evidence,
        )
        for index, item in enumerate(items, start=1)
    ]


def _map_conflict_notes(items: list[GeminiRelationNote]) -> list[Conflict]:
    return [
        Conflict(
            id=f"conflict-{index}",
            scope=EntityReference(entity_type="element", entity_id=item.from_element or "unknown"),
            field=item.type or "unspecified",
            severity="unknown",
            notes=item.notes or item.description or item.evidence,
        )
        for index, item in enumerate(items, start=1)
    ]


def _map_unknowns_to_warnings(items: list[str]) -> list[Warning]:
    return [
        Warning(
            id=f"warning-unknown-{index}",
            code="unknown_field",
            severity="info",
            message=item,
        )
        for index, item in enumerate(items, start=1)
    ]
