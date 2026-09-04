from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.gemini_discovery import GeminiDiscoveryResult, GeminiElementDiscovery
from app.models.gemini_enrichment import GeminiElementEnrichment, GeminiEnrichmentResult
from app.models.requirement_extraction import RequirementExtraction


class InventoryElementTrace(BaseModel):
    id: str | None = None
    temporary_id: str | None = None
    reference: str | None = None
    description: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    quantity: str | int | float | bool | None = None
    status: str | None = None


class InventoryStageTrace(BaseModel):
    stage: str
    count: int
    elements: list[InventoryElementTrace] = Field(default_factory=list)


class InventoryDebugTrace(BaseModel):
    stages: list[InventoryStageTrace] = Field(default_factory=list)

    def add_stage(self, stage: str, elements: list[InventoryElementTrace]) -> None:
        self.stages.append(
            InventoryStageTrace(
                stage=stage,
                count=len(elements),
                elements=elements,
            )
        )


def discovery_inventory_elements(discovery: GeminiDiscoveryResult) -> list[InventoryElementTrace]:
    return [
        _discovery_element_trace(item, index)
        for index, item in enumerate(discovery.elements, start=1)
    ]


def enrichment_inventory_elements(
    enrichment: GeminiEnrichmentResult,
) -> list[InventoryElementTrace]:
    return [_enrichment_element_trace(item) for item in enrichment.elements]


def final_inventory_elements(
    extraction: RequirementExtraction,
) -> list[InventoryElementTrace]:
    return [
        InventoryElementTrace(
            id=item.id,
            reference=str(item.reference.value) if item.reference else None,
            description=item.description,
            source_ids=_final_source_ids(item.evidence_ids, extraction),
            dimensions=[
                _dimension_text(measurement.type, measurement.value, measurement.unit)
                for measurement in item.measurements
            ],
            quantity=item.quantity.value if item.quantity else None,
            status=None,
        )
        for item in extraction.elements
    ]


def _discovery_element_trace(
    item: GeminiElementDiscovery,
    index: int,
) -> InventoryElementTrace:
    return InventoryElementTrace(
        temporary_id=item.temporary_id or f"discovery-{index}",
        reference=item.reference,
        description=item.name or item.category_raw or item.source_hint,
        source_ids=list(item.source_ids),
        status=item.status.value if item.status else None,
    )


def _enrichment_element_trace(item: GeminiElementEnrichment) -> InventoryElementTrace:
    return InventoryElementTrace(
        temporary_id=item.temporary_id,
        reference=item.reference,
        description=item.description or item.name or item.category_raw,
        source_ids=_enrichment_source_ids(item),
        dimensions=[
            _dimension_text(
                measurement.type or measurement.raw_label,
                measurement.value,
                measurement.unit,
            )
            for measurement in item.measurements
        ],
        quantity=item.quantity,
        status=item.status.value if item.status else None,
    )


def _enrichment_source_ids(item: GeminiElementEnrichment) -> list[str]:
    values: list[str] = []
    for evidence in item.evidence:
        if evidence.source_id and evidence.source_id not in values:
            values.append(evidence.source_id)
    return values


def _final_source_ids(evidence_ids: list[str], extraction: RequirementExtraction) -> list[str]:
    evidence_by_id = {item.id: item for item in extraction.evidence}
    values: list[str] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence and evidence.source_id not in values:
            values.append(evidence.source_id)
    return values


def _dimension_text(
    kind: str | None,
    value: float | None,
    unit: str | None,
) -> str:
    label = kind or "unspecified"
    if value is None:
        return label
    return f"{label}={value}{unit or ''}"
