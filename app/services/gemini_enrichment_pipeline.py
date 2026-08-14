from app.models.common import ExtractionStatus
from app.models.gemini_discovery import GeminiDiscoveryResult, GeminiElementDiscovery
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentComponent,
    GeminiEnrichmentGlass,
    GeminiEnrichmentMeasurement,
    GeminiEnrichmentNamedItem,
    GeminiEnrichmentResult,
)
from app.models.gemini_extraction import (
    GeminiComponent,
    GeminiElement,
    GeminiExtraction,
    GeminiGlass,
    GeminiMeasurement,
    GeminiNamedItem,
    GeminiOccurrence,
    GeminiVariant,
)
from app.models.requirement import TokenUsage


def build_discovery_batches(
    discovery: GeminiDiscoveryResult,
    batch_size: int,
) -> list[list[GeminiElementDiscovery]]:
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero.")

    return [
        discovery.elements[index : index + batch_size]
        for index in range(0, len(discovery.elements), batch_size)
    ]


def merge_enrichment_batches(
    discovery: GeminiDiscoveryResult,
    batch_results: list[GeminiEnrichmentResult],
) -> GeminiEnrichmentResult:
    warnings: list[str] = []
    merged_by_id: dict[str, GeminiElementEnrichment] = {}
    duplicate_ids: set[str] = set()

    for batch_index, batch_result in enumerate(batch_results, start=1):
        warnings.extend(batch_result.warnings)
        for enriched in batch_result.elements:
            if enriched.temporary_id in merged_by_id:
                duplicate_ids.add(enriched.temporary_id)
                warnings.append(
                    f"duplicate temporary_id {enriched.temporary_id!r} in batch {batch_index}"
                )
                continue
            merged_by_id[enriched.temporary_id] = enriched

    ordered_elements = []
    seen_discovery_ids: set[str] = set()
    for index, discovered in enumerate(discovery.elements, start=1):
        temporary_id = _discovery_temporary_id(discovered, index)
        if temporary_id in seen_discovery_ids:
            warnings.append(f"duplicate discovery temporary_id {temporary_id!r}")
        seen_discovery_ids.add(temporary_id)
        enriched = merged_by_id.get(temporary_id)
        if enriched is None:
            warnings.append(f"missing enrichment for temporary_id {temporary_id!r}")
            enriched = enrichment_from_discovery(discovered, index)
        ordered_elements.append(enriched)

    for duplicate_id in sorted(duplicate_ids):
        warnings.append(f"ignored duplicate enrichment for temporary_id {duplicate_id!r}")

    return GeminiEnrichmentResult(elements=ordered_elements, warnings=warnings)


def enrichment_from_discovery(
    discovery: GeminiElementDiscovery,
    index: int,
) -> GeminiElementEnrichment:
    return GeminiElementEnrichment(
        temporary_id=_discovery_temporary_id(discovery, index),
        reference=discovery.reference,
        name=discovery.name,
        category_raw=discovery.category_raw,
        evidence_notes=[discovery.source_hint] if discovery.source_hint else [],
        missing_or_unknown=["technical_enrichment"],
        status=ExtractionStatus.UNKNOWN,
        confidence=discovery.confidence,
        notes="Preserved from discovery because enrichment did not return this item.",
    )


def enrichment_to_gemini_extraction(
    discovery: GeminiDiscoveryResult,
    enrichment: GeminiEnrichmentResult,
) -> GeminiExtraction:
    elements = [
        _enriched_to_gemini_element(item, index)
        for index, item in enumerate(enrichment.elements, start=1)
    ]
    return GeminiExtraction(
        elements=elements,
        unknown_fields=[],
        notes="\n".join(discovery.notes + enrichment.warnings) or None,
    )


def sum_token_usage(items: list[TokenUsage | None]) -> TokenUsage | None:
    input_tokens = _sum_optional(item.input_tokens for item in items if item is not None)
    output_tokens = _sum_optional(item.output_tokens for item in items if item is not None)
    total_tokens = _sum_optional(item.total_tokens for item in items if item is not None)
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _sum_optional(values) -> int | None:
    materialized = [value for value in values if value is not None]
    return sum(materialized) if materialized else None


def _discovery_temporary_id(discovery: GeminiElementDiscovery, index: int) -> str:
    return discovery.temporary_id or f"discovery-{index}"


def _enriched_to_gemini_element(item: GeminiElementEnrichment, index: int) -> GeminiElement:
    return GeminiElement(
        id=item.temporary_id or f"element-{index}",
        reference=item.reference,
        name=item.name,
        category=item.category_raw,
        description=item.description,
        measurements=[_measurement(measurement) for measurement in item.measurements],
        geometry=item.geometry_raw,
        configuration=item.configuration_raw,
        quantity=item.quantity,
        glass=[_glass(glass) for glass in item.glass],
        materials=[_named_item(material) for material in item.materials],
        profiles=[_named_item(profile) for profile in item.profiles],
        finish=item.finish_raw,
        accessories=[_named_item(accessory) for accessory in item.accessories],
        components=[
            _component(component, component_index)
            for component_index, component in enumerate(item.components, start=1)
        ],
        occurrences=_occurrences_from_context(item),
        variants=_variants_from_context(item),
        evidence="\n".join(item.evidence_notes) or None,
        missing_or_unknown=list(item.missing_or_unknown),
        status=item.status,
        confidence=item.confidence,
        notes=item.notes,
    )


def _measurement(item: GeminiEnrichmentMeasurement) -> GeminiMeasurement:
    return GeminiMeasurement(
        type=item.type,
        label=item.raw_label,
        value=item.value,
        unit=item.unit,
        text=item.text,
        status=item.status,
        confidence=item.confidence,
        notes=item.notes,
    )


def _glass(item: GeminiEnrichmentGlass) -> GeminiGlass:
    return GeminiGlass(
        type=item.type,
        thickness=item.thickness,
        thickness_value=item.thickness_value,
        thickness_unit=item.thickness_unit,
        color=item.color,
        treatment=item.treatment,
        composition=item.composition,
        description=item.description,
        status=item.status,
        confidence=item.confidence,
        notes=item.notes,
    )


def _named_item(item: GeminiEnrichmentNamedItem) -> GeminiNamedItem:
    return GeminiNamedItem(
        name=item.name,
        type=item.type,
        code=item.code,
        role=item.role,
        description=item.description,
        quantity=item.quantity,
        status=item.status,
        confidence=item.confidence,
        notes=item.notes,
    )


def _component(item: GeminiEnrichmentComponent, index: int) -> GeminiComponent:
    return GeminiComponent(
        id=f"component-{index}",
        name=item.name,
        type=item.type,
        role=item.role,
        description=item.description,
        quantity=item.quantity,
        measurements=[_measurement(measurement) for measurement in item.measurements],
        geometry=None,
        configuration=None,
        glass=[_glass(glass) for glass in item.glass],
        materials=[_named_item(material) for material in item.materials],
        profiles=[_named_item(profile) for profile in item.profiles],
        finish=None,
        accessories=[],
        status=item.status,
        confidence=item.confidence,
        notes=item.notes,
    )


def _occurrences_from_context(item: GeminiElementEnrichment) -> list[GeminiOccurrence]:
    if not item.occurrence_context:
        return []

    return [
        GeminiOccurrence(
            id=f"{item.temporary_id}-occurrence-1",
            location=item.occurrence_context,
            status=item.status,
            confidence=item.confidence,
        )
    ]


def _variants_from_context(item: GeminiElementEnrichment) -> list[GeminiVariant]:
    if not item.variant_context:
        return []

    return [
        GeminiVariant(
            id=f"{item.temporary_id}-variant-1",
            label=item.variant_context,
            status=item.status,
            confidence=item.confidence,
        )
    ]
