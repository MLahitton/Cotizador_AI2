import re
import unicodedata

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

AREA_ABSOLUTE_TOLERANCE_M2 = 0.02
AREA_RELATIVE_TOLERANCE = 0.02
AREA_MISMATCH_WARNING_CODE = "MEASUREMENT_AREA_MISMATCH"


def map_gemini_extraction_to_requirement_extraction(
    extraction: GeminiExtraction,
    *,
    model_provider: str | None = "google",
    model: str | None = None,
    default_source_id: str | None = "text-input",
    allowed_source_ids: list[str] | None = None,
) -> RequirementExtraction:
    warnings: list[Warning] = []
    evidence = [
        _map_evidence(item, index, default_source_id, allowed_source_ids, warnings)
        for index, item in enumerate(extraction.evidence, start=1)
    ]
    element_evidence = _build_element_evidence(
        extraction.elements,
        len(evidence),
        default_source_id,
        allowed_source_ids,
        warnings,
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
    warnings.extend(_measurement_area_mismatch_warnings(elements, evidence))

    return RequirementExtraction(
        requirement=_map_requirement(extraction, evidence_ids),
        elements=elements,
        evidence=evidence,
        relationships=_map_relationship_notes(extraction.relationships),
        conflicts=_map_conflict_notes(extraction.conflicts),
        warnings=[*warnings, *_map_unknowns_to_warnings(extraction.unknown_fields)],
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
    item = _suppress_unsupported_visual_function_signals(item)
    missing_fields = list(item.missing_or_unknown)
    component_candidates = _component_candidates(item)
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
        functional_type=_map_functional_type(
            item.functional_type,
            item.category,
            item.configuration,
            item.status,
            item.confidence,
            evidence_ids,
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
            for component_index, component in enumerate(component_candidates, start=1)
        ],
        geometry=_map_geometry(
            item.geometry,
            item.geometry_type,
            item.status,
            item.confidence,
            evidence_ids,
        ),
        assembly_type=_normalize_assembly_type(
            item.assembly_type,
            item.geometry_type,
            component_candidates,
        ),
        measurements=[
            _map_measurement(measurement, evidence_ids)
            for measurement in item.measurements
        ],
        quantity=_map_optional_traceable_field(
            item.quantity,
            item.quantity_status or item.status,
            item.quantity_confidence if item.quantity_confidence is not None else item.confidence,
            evidence_ids,
            missing_fields,
            "quantity",
            "cantidad",
            notes=item.quantity_notes,
        ),
        configuration=_map_configuration(
            item.configuration,
            operation=item.operation,
            panel_count=item.panel_count,
            movable_panel_count=item.movable_panel_count,
            fixed_panel_count=item.fixed_panel_count,
            modulation=item.modulation,
            opening_direction=item.opening_direction,
            special_features=item.special_features,
            status=_optional_field_status(
                item.configuration,
                item.status,
                missing_fields,
                "configuration",
            ),
            confidence=item.confidence,
            evidence_ids=evidence_ids,
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



def _suppress_unsupported_visual_function_signals(item: GeminiElement) -> GeminiElement:
    if not _has_visual_functional_signal(item):
        return item
    if not _has_only_table_measurement_evidence(item):
        return item

    missing = list(item.missing_or_unknown)
    if "visual_functional_evidence" not in missing:
        missing.append("visual_functional_evidence")

    return item.model_copy(
        update={
            "functional_type": None,
            "operation": None,
            "panel_count": None,
            "movable_panel_count": None,
            "fixed_panel_count": None,
            "modulation": None,
            "opening_direction": None,
            "components": [],
            "status": ExtractionStatus.AMBIGUOUS,
            "missing_or_unknown": missing,
            "notes": _append_note(
                item.notes,
                (
                    "Functional and panel signals require visual evidence; "
                    "table row evidence was preserved for measurements and quantity only."
                ),
            ),
        }
    )


def _has_visual_functional_signal(item: GeminiElement) -> bool:
    return any(
        value not in (None, "", [])
        for value in (
            item.functional_type,
            item.operation,
            item.panel_count,
            item.movable_panel_count,
            item.fixed_panel_count,
            item.modulation,
            item.opening_direction,
            item.components,
        )
    )


def _has_only_table_measurement_evidence(item: GeminiElement) -> bool:
    evidence_items = list(item.evidence_items)
    if not evidence_items and item.evidence:
        evidence_items = [GeminiEvidence(text=item.evidence)]
    if not evidence_items:
        return False
    return all(_is_table_measurement_only_evidence(evidence) for evidence in evidence_items)


def _is_table_measurement_only_evidence(evidence: GeminiEvidence) -> bool:
    if evidence.visual_description:
        return False
    evidence_type = _compact_words(evidence.type or "")
    if evidence_type and evidence_type in {
        "visual",
        "drawing",
        "image",
        "crop",
        "dibujo",
        "grafico",
    }:
        return False

    text = evidence.text or evidence.location or evidence.notes or ""
    compact = _compact_words(text)
    if any(
        token in compact
        for token in (
            "dibujo",
            "detalle",
            "elevacion",
            "alzado",
            "simbolo",
            "crop",
            "regionvisual",
        )
    ):
        return False
    if not compact:
        return False
    has_reference = re.search(r"\b(?:V|PV|P|F)-?\d+\b", text, flags=re.IGNORECASE) is not None
    has_measurement = (
        re.search(
            r"\d+(?:[.,]\d+)?\s*(?:x|×)\s*\d+(?:[.,]\d+)?",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
    has_function_word = any(
        token in compact
        for token in (
            "proyectante",
            "projecting",
            "corrediza",
            "corredizo",
            "sliding",
            "batiente",
            "casement",
            "fijo",
            "fixed",
            "plegable",
            "folding",
            "pivot",
            "pivote",
        )
    )
    return has_reference and has_measurement and not has_function_word


def _append_note(current: str | None, note: str) -> str:
    if not current:
        return note
    if note in current:
        return current
    return f"{current}\n{note}"

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
    notes: str | None = None,
) -> TraceableValue | None:
    if value in (None, "") and _is_marked_unknown(missing_fields, *aliases):
        return _unknown_traceable(confidence, evidence_ids)

    return _traceable(value, status, confidence, evidence_ids, notes=notes)


def _optional_field_status(
    value: object | None,
    status: ExtractionStatus | None,
    missing_fields: list[str],
    *aliases: str,
) -> ExtractionStatus | None:
    if value in (None, "") and _is_marked_unknown(missing_fields, *aliases):
        return ExtractionStatus.UNKNOWN

    return status


def _map_measurement(
    item: GeminiMeasurement,
    evidence_ids: list[str] | None = None,
) -> Measurement:
    return Measurement(
        type=_measurement_type(item),
        raw_label=item.label or item.text,
        value=item.value,
        unit=item.unit,
        raw_value=item.value,
        raw_unit=item.unit,
        status=_status_for_value(item.value or item.text, item.status),
        confidence=item.confidence,
        evidence_ids=_measurement_evidence_ids(evidence_ids),
        notes=item.notes or item.evidence,
    )


def _measurement_evidence_ids(evidence_ids: list[str] | None) -> list[str]:
    if evidence_ids is not None and len(evidence_ids) == 1:
        return list(evidence_ids)

    return []


def _measurement_type(item: GeminiMeasurement) -> str:
    raw_type = item.type or "unspecified"
    if _is_area_measurement_label(item.label) or _is_area_measurement_label(item.type):
        return "area"

    return raw_type


def _is_area_measurement_label(value: str | None) -> bool:
    if value is None:
        return False

    normalized = _compact_text(value)
    return normalized in {"m2", "m²", "mÂ²", "ma2", "area"}


def _compact_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(character) != "Mn"
    )
    return (
        without_accents.replace("²", "2")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("â", "")
        .replace("Â", "")
    )


def _measurement_area_mismatch_warnings(
    elements: list[Element],
    evidence: list[Evidence],
) -> list[Warning]:
    warnings: list[Warning] = []
    evidence_by_id = {item.id: item for item in evidence}
    for element in elements:
        width = _first_positive_measurement(element.measurements, "width")
        height = _first_positive_measurement(element.measurements, "height")
        reported_area = _first_positive_measurement(element.measurements, "area")
        if width is None or height is None or reported_area is None:
            continue

        width_m = _linear_measurement_to_meters(width)
        height_m = _linear_measurement_to_meters(height)
        reported_area_m2 = _area_measurement_to_square_meters(reported_area)
        if width_m is None or height_m is None or reported_area_m2 is None:
            continue

        derived_area_m2 = width_m * height_m
        difference = abs(derived_area_m2 - reported_area_m2)
        relative_difference = difference / derived_area_m2 if derived_area_m2 else 0
        if (
            difference <= AREA_ABSOLUTE_TOLERANCE_M2
            or relative_difference <= AREA_RELATIVE_TOLERANCE
        ):
            continue

        evidence_ids = _unique_ids(
            width.evidence_ids + height.evidence_ids + reported_area.evidence_ids
        )
        source_ids = _unique_ids(
            [
                evidence_by_id[evidence_id].source_id
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
            ]
        )
        warnings.append(
            Warning(
                id=f"warning-{AREA_MISMATCH_WARNING_CODE.casefold()}-{len(warnings) + 1}",
                code=AREA_MISMATCH_WARNING_CODE,
                severity="warning",
                message=(
                    f"Reported area {reported_area_m2:.2f} m2 differs from derived area "
                    f"{derived_area_m2:.2f} m2 using width "
                    f"{_measurement_value_with_unit(width)} and height "
                    f"{_measurement_value_with_unit(height)}."
                ),
                source_ids=source_ids,
                element_ids=[element.id],
                evidence_ids=evidence_ids,
            )
        )

    return warnings


def _first_positive_measurement(
    measurements: list[Measurement],
    measurement_type: str,
) -> Measurement | None:
    for measurement in measurements:
        if measurement.type == measurement_type and measurement.value is not None:
            if measurement.value > 0:
                return measurement

    return None


def _linear_measurement_to_meters(measurement: Measurement) -> float | None:
    if measurement.value is None or measurement.value <= 0:
        return None

    unit = _compact_text(measurement.unit or measurement.raw_unit or "mm")
    if unit in {"mm", "milimetro", "milimetros"}:
        return measurement.value / 1000
    if unit in {"cm", "centimetro", "centimetros"}:
        return measurement.value / 100
    if unit in {"m", "metro", "metros"}:
        return measurement.value

    return None


def _area_measurement_to_square_meters(measurement: Measurement) -> float | None:
    if measurement.value is None or measurement.value <= 0:
        return None

    unit = _compact_text(measurement.unit or measurement.raw_unit or "m2")
    if unit in {"m2", "m²", "mÂ²", "ma2", "metro2", "metros2", "metrocuadrado", "metroscuadrados"}:
        return measurement.value
    if unit in {"cm2", "cm²", "cmÂ²", "centimetro2", "centimetros2", "centimetroscuadrados"}:
        return measurement.value / 10_000
    if unit in {"mm2", "mm²", "mmÂ²", "milimetro2", "milimetros2", "milimetroscuadrados"}:
        return measurement.value / 1_000_000

    return None


def _measurement_value_with_unit(measurement: Measurement) -> str:
    unit = measurement.unit or measurement.raw_unit
    if unit:
        return f"{measurement.value:g} {unit}"
    return f"{measurement.value:g}"


def _unique_ids(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _map_geometry(
    description: str | None,
    geometry_type: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> Geometry | None:
    if (
        description in (None, "")
        and geometry_type in (None, "")
        and status is None
        and confidence is None
    ):
        return None

    normalized_type = _normalize_geometry_type(geometry_type or description)
    return Geometry(
        normalized_type=normalized_type,
        raw_type=geometry_type or description,
        description=description,
        status=_status_for_value(description, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _map_configuration(
    description: str | None,
    *,
    operation: str | None = None,
    panel_count: int | None = None,
    movable_panel_count: int | None = None,
    fixed_panel_count: int | None = None,
    modulation: str | None = None,
    opening_direction: str | None = None,
    special_features: list[str] | None = None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> Configuration | None:
    resolved_modulation = _normalize_modulation(modulation or description)
    resolved_panel_count = panel_count or _panel_count_from_modulation(resolved_modulation)
    normalized_features = _normalize_special_features(
        [*(special_features or []), description or ""]
    )
    resolved_movable_panel_count = (
        movable_panel_count
        if movable_panel_count is not None
        else _movable_count_from_modulation(resolved_modulation)
    )
    resolved_fixed_panel_count = (
        fixed_panel_count
        if fixed_panel_count is not None
        else _fixed_count_from_modulation(resolved_modulation)
    )
    if description in (None, "") and not any(
        [
            operation,
            resolved_panel_count,
            movable_panel_count,
            fixed_panel_count,
            resolved_modulation,
            opening_direction,
            normalized_features,
        ]
    ) and status is None and confidence is None:
        return None

    return Configuration(
        raw_description=description,
        operation=_normalized_signal(
            operation or description,
            _normalize_operation(operation or description),
            status,
            confidence,
            evidence_ids,
        ),
        panel_count=_traceable(
            resolved_panel_count,
            status,
            confidence,
            evidence_ids,
        ),
        movable_panel_count=_traceable(
            resolved_movable_panel_count,
            status,
            confidence,
            evidence_ids,
        ),
        fixed_panel_count=_traceable(
            resolved_fixed_panel_count,
            status,
            confidence,
            evidence_ids,
        ),
        arrangement=resolved_modulation,
        modulation=resolved_modulation,
        opening_direction=_normalized_signal(
            opening_direction,
            _normalize_opening_direction(opening_direction),
            status,
            confidence,
            evidence_ids,
        ),
        special_features=normalized_features,
        status=_status_for_value(description, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _map_functional_type(
    raw_functional_type: str | None,
    category: str | None,
    configuration: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> NormalizedValue | None:
    raw = raw_functional_type or category
    normalized = _normalize_functional_type(raw_functional_type, category, configuration)
    if normalized is None and raw in (None, ""):
        return None

    return NormalizedValue(
        normalized=normalized,
        raw=raw,
        status=_status_for_value(normalized or raw, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _normalized_signal(
    raw: str | None,
    normalized: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> NormalizedValue | None:
    if raw in (None, "") and normalized is None:
        return None

    return NormalizedValue(
        normalized=normalized,
        raw=raw,
        status=_status_for_value(normalized or raw, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _normalized_component_role(
    raw: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> NormalizedValue | None:
    return _normalized_signal(
        raw,
        _normalize_component_role(raw),
        status,
        confidence,
        evidence_ids,
    )


def _normalize_component_role(value: str | None) -> str | None:
    normalized = _compact_words(value or "").upper()
    canonical_roles = {
        "FIXED",
        "PROJECTING",
        "SLIDING",
        "SWING",
        "CASEMENT",
        "FOLDING",
        "GRILLE",
        "LOUVER",
        "LEG",
        "PANEL",
        "UNKNOWN",
    }
    return normalized if normalized in canonical_roles else None


def _component_candidates(item: GeminiElement) -> list[GeminiComponent]:
    candidates = list(item.components)
    existing_roles = {
        role
        for component in candidates
        if (role := _normalize_component_role(component.role or component.type)) is not None
    }

    for role in _component_roles_from_item(item):
        if role in existing_roles:
            continue

        candidates.append(
            GeminiComponent(
                role=role,
                description="Derived from explicit item-level assembly signal.",
                status=ExtractionStatus.INFERRED,
                confidence=item.confidence,
            )
        )
        existing_roles.add(role)

    return candidates


def _component_roles_from_item(item: GeminiElement) -> list[str]:
    roles: list[str] = []
    functional_type = _normalize_functional_type(
        item.functional_type,
        item.category,
        item.configuration,
    )
    _append_component_role(roles, _component_role_from_functional_type(functional_type))
    _append_component_role(roles, _component_role_from_operation(item.operation))

    text = _compact_words(
        " ".join(
            value
            for value in (
                item.functional_type,
                item.category,
                item.configuration,
                item.description,
                item.notes,
                *item.special_features,
            )
            if value
        )
    )
    features = _normalize_special_features(
        [*(item.special_features or []), item.configuration or "", item.description or ""]
    )

    if any(
        feature in features
        for feature in (
            "ASSOCIATED_FIXED_PANEL",
            "LOWER_FIXED_PANEL",
            "UPPER_FIXED_PANEL",
        )
    ) or _has_contextual_fixed_signal(text):
        _append_component_role(roles, "FIXED")

    if _has_grille_signal(text):
        _append_component_role(roles, "GRILLE")

    return roles


def _append_component_role(roles: list[str], role: str | None) -> None:
    if role is not None and role not in roles:
        roles.append(role)


def _component_role_from_functional_type(value: str | None) -> str | None:
    return {
        "SLIDING_WINDOW": "SLIDING",
        "SLIDING_DOOR": "SLIDING",
        "PROJECTING": "PROJECTING",
        "SWING_DOOR": "SWING",
        "CASEMENT": "CASEMENT",
        "FOLDING_WINDOW": "FOLDING",
        "FOLDING_DOOR": "FOLDING",
        "FIXED": "FIXED",
        "GRILLE": "GRILLE",
    }.get(value or "")


def _component_role_from_operation(value: str | None) -> str | None:
    text = _compact_words(value or "")
    if not text:
        return None
    if "corred" in text or "sliding" in text:
        return "SLIDING"
    if "proyect" in text or "projecting" in text:
        return "PROJECTING"
    if "batiente" in text or "swing" in text:
        return "SWING"
    if "casement" in text:
        return "CASEMENT"
    if "plegable" in text or "folding" in text:
        return "FOLDING"
    if "fijo" in text or "fixed" in text:
        return "FIXED"
    if _has_grille_signal(text):
        return "GRILLE"
    return None


def _has_contextual_fixed_signal(text: str) -> bool:
    if "fijo" not in text and "fixed" not in text:
        return False
    return any(
        term in text
        for term in (
            "asociado",
            "associated",
            "inferior",
            "lower",
            "superior",
            "upper",
            "lateral",
            "side",
            "con",
            "with",
        )
    )


def _has_grille_signal(text: str) -> bool:
    return any(term in text for term in ("rejilla", "grille", "louver", "celosia", "persiana"))


def _normalize_assembly_type(
    value: str | None,
    geometry_type: str | None,
    components: list[GeminiComponent],
) -> str | None:
    text = _compact_words(value or "")
    if text:
        if any(term in text for term in ("corner", "esquina", "escuadra")):
            return "CORNER"
        if any(term in text for term in ("composite", "compuesto", "mixto", "associatedfixed")):
            return "COMPOSITE"
        if any(term in text for term in ("multimodule", "modulos", "modules", "segmentos")):
            return "MULTI_MODULE"
        if any(term in text for term in ("single", "simple")):
            return "SINGLE"

    geometry = _normalize_geometry_type(geometry_type)
    if geometry == "CORNER":
        return "CORNER"
    roles = {
        role
        for component in components
        if (role := _normalize_component_role(component.role or component.type)) is not None
    }
    functional_roles = {
        role
        for role in roles
        if role
        in {
            "SLIDING",
            "PROJECTING",
            "SWING",
            "CASEMENT",
            "FOLDING",
            "FIXED",
            "GRILLE",
            "LOUVER",
        }
    }
    if len(functional_roles) > 1:
        return "COMPOSITE"
    return "MULTI_MODULE" if len(components) > 1 else None


def _normalize_functional_type(
    raw_functional_type: str | None,
    category: str | None,
    configuration: str | None,
) -> str | None:
    text = _compact_words(" ".join(item for item in [raw_functional_type, category] if item))
    config_text = _compact_words(configuration or "")
    combined = f"{text} {config_text}".strip()
    if not combined:
        return None

    if ("division" in combined and ("bano" in combined or "ducha" in combined)) or (
        "shower" in combined and "division" in combined
    ):
        return "SHOWER_DIVISION"
    if "pergola" in combined:
        return "PERGOLA"
    if any(term in combined for term in ("rejilla", "louver", "celosia", "grille")):
        return "GRILLE"
    if "claraboya" in combined or "skylight" in combined:
        return "SKYLIGHT"
    if "proyectante" in combined or "projecting" in combined:
        return "PROJECTING"
    if ("puerta" in combined or "door" in combined) and (
        "batiente" in combined or "swing" in combined
    ):
        return "SWING_DOOR"
    if "doblebatiente" in combined or ("doble" in combined and "batiente" in combined):
        return "DOUBLE_CASEMENT"
    if "batiente" in combined or "casement" in combined:
        return "CASEMENT"
    if ("fijo" in combined or "fixed" in combined) and not any(
        term in combined for term in ("corred", "proyect", "batiente", "sliding")
    ):
        return "FIXED"
    if "plegable" in combined or "plegadiza" in combined or "folding" in combined:
        if "puerta" in combined or "door" in combined:
            return "FOLDING_DOOR"
        if "ventana" in combined or "window" in combined:
            return "FOLDING_WINDOW"
        return None
    if "corred" in combined or "sliding" in combined:
        if "puerta" in combined or "door" in combined:
            return "SLIDING_DOOR"
        if "ventana" in combined or "window" in combined:
            return "SLIDING_WINDOW"
        return None
    if raw_functional_type and "otro" in text:
        return "OTHER"

    return None


def _normalize_operation(value: str | None) -> str | None:
    text = _compact_words(value or "")
    if not text:
        return None
    if "doblebatiente" in text or ("doble" in text and "batiente" in text):
        return "DOUBLE_CASEMENT"
    if "corred" in text or "sliding" in text:
        return "SLIDING"
    if "proyect" in text:
        return "PROJECTING"
    if "batiente" in text:
        return "CASEMENT"
    if "plegable" in text or "plegadiza" in text:
        return "FOLDING"
    if "pivot" in text or "pivote" in text:
        return "PIVOT"
    if "fijo" in text:
        return "FIXED"
    if "otro" in text:
        return "OTHER"
    return None


def _normalize_geometry_type(value: str | None) -> str | None:
    text = _compact_words(value or "")
    if not text:
        return None
    if "triangular" in text or "triangulo" in text:
        return "TRIANGULAR"
    if "trapezo" in text:
        return "TRAPEZOIDAL"
    if "estructuraenl" in text or "formadel" in text or "lshape" in text:
        return "L_SHAPE"
    if "esquina" in text or "corner" in text or "escuadra" in text:
        return "CORNER"
    if "arco" in text or "arc" in text:
        return "ARCH"
    if "curv" in text:
        return "CURVED"
    if "inclin" in text or "sloped" in text or "pendiente" in text:
        return "SLOPED"
    if "irregular" in text:
        return "IRREGULAR"
    if "rectangular" in text or "rectangulo" in text:
        return "RECTANGULAR"
    if "unknown" in text or "desconoc" in text:
        return "UNKNOWN"
    return None


def _normalize_opening_direction(value: str | None) -> str | None:
    text = _compact_words(value or "")
    if not text:
        return None
    if "izquierda" in text or text == "left":
        return "LEFT"
    if "derecha" in text or text == "right":
        return "RIGHT"
    if "arriba" in text or "superior" in text or text == "up":
        return "UP"
    if "abajo" in text or "inferior" in text or text == "down":
        return "DOWN"
    if "interior" in text or "haciaadentro" in text or "inward" in text:
        return "INWARD"
    if "exterior" in text or "haciaafuera" in text or "outward" in text:
        return "OUTWARD"
    return None


def _normalize_special_features(values: list[str]) -> list[str]:
    features: list[str] = []
    text = _compact_words(" ".join(value for value in values if value))
    explicit_values = {_normalize_feature_token(value) for value in values if value}
    for feature in explicit_values:
        if feature:
            _append_unique(features, feature)

    if any(
        term in text
        for term in (
            "bolsillo",
            "pocket",
            "empotradaenmuro",
            "guardarseenunbolsillo",
            "dentrodelmuro",
        )
    ):
        _append_unique(features, "POCKET")
    if "fijo" in text and any(
        term in text for term in ("asociado", "con", "inferior", "superior", "lateral")
    ):
        _append_unique(features, "ASSOCIATED_FIXED_PANEL")
    if "fijoinferior" in text or ("fijo" in text and "inferior" in text):
        _append_unique(features, "LOWER_FIXED_PANEL")
    if "fijosuperior" in text or ("fijo" in text and "superior" in text):
        _append_unique(features, "UPPER_FIXED_PANEL")
    if "mullion" in text or "divisor" in text:
        _append_unique(features, "MULLION")
    if "reticula" in text or "cuadricula" in text or "grid" in text:
        _append_unique(features, "GRID")
    if "chapa" in text and "reforz" in text:
        _append_unique(features, "REINFORCED_CATCHES")
    if any(
        term in text
        for term in (
            "conservarlasnaves",
            "mantenerdisenodenaves",
            "conservardisenodenaves",
        )
    ):
        _append_unique(features, "PRESERVE_MODULATION")

    return features


def _normalize_feature_token(value: str) -> str | None:
    text = value.strip().upper().replace(" ", "_").replace("-", "_")
    allowed = {
        "POCKET",
        "ASSOCIATED_FIXED_PANEL",
        "LOWER_FIXED_PANEL",
        "UPPER_FIXED_PANEL",
        "MULLION",
        "GRID",
        "REINFORCED_CATCHES",
        "PRESERVE_MODULATION",
    }
    return text if text in allowed else None


def _normalize_modulation(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\b[OX]{2,}\b", value.upper())
    return match.group(0) if match else None


def _panel_count_from_modulation(modulation: str | None) -> int | None:
    return len(modulation) if modulation else None


def _movable_count_from_modulation(modulation: str | None) -> int | None:
    if not modulation or not _has_fixed_and_movable_symbols(modulation):
        return None
    return modulation.count("X")


def _fixed_count_from_modulation(modulation: str | None) -> int | None:
    if not modulation or not _has_fixed_and_movable_symbols(modulation):
        return None
    return modulation.count("O")


def _has_fixed_and_movable_symbols(modulation: str) -> bool:
    symbols = set(modulation)
    return "O" in symbols and "X" in symbols


def _compact_words(value: str) -> str:
    return _compact_text(value).replace(".", "")


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


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
        normalized_type=_normalize_finish_type(description),
        color=_normalize_finish_color(description, status, confidence, evidence_ids),
        texture=_normalize_finish_texture(description, status, confidence, evidence_ids),
        code=_extract_finish_code(description, status, confidence, evidence_ids),
        raw_description=description,
        status=_status_for_value(description, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _normalize_finish_type(value: str | None) -> str | None:
    text = _compact_words(value or "")
    if not text:
        return None
    if "inox" in text or "aceroinoxidable" in text or "stainlesssteel" in text:
        return "STAINLESS_STEEL"
    if "anodiz" in text:
        return "ANODIZED"
    if "pintura" in text or "pintado" in text or "alhorno" in text or "poliester" in text:
        return "PAINTED"
    return None


def _normalize_finish_color(
    value: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> NormalizedValue | None:
    text = _compact_words(value or "")
    color_map = {
        "negro": "BLACK",
        "black": "BLACK",
        "blanco": "WHITE",
        "white": "WHITE",
        "gris": "GRAY",
        "gray": "GRAY",
        "grey": "GRAY",
        "champana": "CHAMPAGNE",
        "champaa": "CHAMPAGNE",
        "champagne": "CHAMPAGNE",
    }
    for token, normalized in color_map.items():
        if token in text:
            return NormalizedValue(
                normalized=normalized,
                raw=_raw_finish_token(value, token),
                status=_status_for_value(value, status),
                confidence=confidence,
                evidence_ids=evidence_ids,
            )
    return None


def _normalize_finish_texture(
    value: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> NormalizedValue | None:
    text = _compact_words(value or "")
    if "mate" not in text and "matte" not in text:
        return None
    return NormalizedValue(
        normalized="MATTE",
        raw="mate",
        status=_status_for_value(value, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _extract_finish_code(
    value: str | None,
    status: ExtractionStatus | None,
    confidence: float | None,
    evidence_ids: list[str],
) -> TraceableValue | None:
    if not value:
        return None
    match = re.search(r"\b(?:PP|AN)\d{2,4}\b", value.upper())
    if match is None:
        return None
    return TraceableValue(
        value=match.group(0),
        status=_status_for_value(value, status),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _raw_finish_token(value: str | None, token: str) -> str:
    if not value:
        return token
    for word in re.findall(r"\w+", value, flags=re.UNICODE):
        if _compact_words(word) == token:
            return word
    return token


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
        measurements=[
            _map_measurement(measurement, evidence_ids)
            for measurement in item.measurements
        ],
        evidence_ids=evidence_ids,
        confidence=item.confidence,
        notes=item.notes or item.evidence,
    )


def _map_variant(item: GeminiVariant, index: int, evidence_ids: list[str]) -> Variant:
    return Variant(
        id=item.id or f"variant-{index}",
        label=item.label,
        reason=item.reason,
        measurements=[
            _map_measurement(measurement, evidence_ids)
            for measurement in item.measurements
        ],
        configuration=_map_configuration(
            item.configuration,
            operation=None,
            panel_count=None,
            movable_panel_count=None,
            fixed_panel_count=None,
            modulation=None,
            opening_direction=None,
            special_features=[],
            status=item.status,
            confidence=item.confidence,
            evidence_ids=evidence_ids,
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
        role=_normalized_component_role(item.role, item.status, item.confidence, evidence_ids),
        quantity=_traceable(item.quantity, item.status, item.confidence, evidence_ids),
        geometry=_map_geometry(
            item.geometry,
            None,
            item.status,
            item.confidence,
            evidence_ids,
        ),
        measurements=[
            _map_measurement(measurement, evidence_ids)
            for measurement in item.measurements
        ],
        configuration=_map_configuration(
            item.configuration,
            operation=None,
            panel_count=None,
            movable_panel_count=None,
            fixed_panel_count=None,
            modulation=None,
            opening_direction=None,
            special_features=[],
            status=item.status,
            confidence=item.confidence,
            evidence_ids=evidence_ids,
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


def _map_evidence(
    item: GeminiEvidence,
    index: int,
    default_source_id: str | None,
    allowed_source_ids: list[str] | None,
    warnings: list[Warning],
) -> Evidence:
    source_id = _resolve_evidence_source_id(
        item.source_id,
        default_source_id,
        allowed_source_ids,
        warnings,
        f"evidence-{index}",
    )
    return Evidence(
        id=item.id or f"evidence-{index}",
        source_id=source_id,
        type=item.type or "text",
        page_number=item.page_number,
        sheet_name=item.sheet_name,
        cell_range=item.cell_range,
        region=item.region,
        extracted_text=item.text,
        visual_description=item.visual_description or item.location,
        status=_status_for_value(item.text or item.visual_description, item.status),
        confidence=item.confidence,
        notes=item.notes,
    )


def _build_element_evidence(
    elements: list[GeminiElement],
    existing_count: int,
    default_source_id: str | None,
    allowed_source_ids: list[str] | None,
    warnings: list[Warning],
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for element_index, element in enumerate(elements, start=1):
        for item in element.evidence_items:
            evidence.append(
                _map_evidence(
                    item,
                    existing_count + len(evidence) + 1,
                    default_source_id,
                    allowed_source_ids,
                    warnings,
                )
            )

        if not element.evidence:
            continue

        evidence_id = f"evidence-{existing_count + len(evidence) + 1}"
        evidence.append(
            Evidence(
                id=evidence_id,
                source_id=_resolve_evidence_source_id(
                    None,
                    default_source_id,
                    allowed_source_ids,
                    warnings,
                    f"element-{element_index}",
                ),
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
        element_evidence_count = len(element.evidence_items) + (1 if element.evidence else 0)
        if element_evidence_count == 0:
            continue

        result[_element_id(element, element_index)] = [
            item.id for item in evidence[evidence_index : evidence_index + element_evidence_count]
        ]
        evidence_index += element_evidence_count

    return result


def _element_id(item: GeminiElement, index: int) -> str:
    return item.id or f"element-{index}"


def _resolve_evidence_source_id(
    source_id: str | None,
    default_source_id: str | None,
    allowed_source_ids: list[str] | None,
    warnings: list[Warning],
    context: str,
) -> str:
    if source_id:
        if allowed_source_ids is not None and source_id not in allowed_source_ids:
            warnings.append(
                _warning(
                    "unknown_evidence_source",
                    f"Evidence {context} uses unknown source_id {source_id!r}.",
                )
            )
            return "unknown"
        return source_id

    if default_source_id:
        return default_source_id

    warnings.append(
        _warning(
            "missing_evidence_source",
            f"Evidence {context} has no source_id and no safe single-source fallback.",
        )
    )
    return "unknown"


def _warning(code: str, message: str) -> Warning:
    return Warning(
        id=f"warning-{code}",
        code=code,
        severity="warning",
        message=message,
    )


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
