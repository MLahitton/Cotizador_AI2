from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus
from app.models.evidence import Region
from app.models.requirement import TokenUsage


class GeminiEnrichmentMeasurement(BaseModel):
    type: str | None = None
    raw_label: str | None = None
    value: float | None = None
    unit: str | None = None
    text: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None


class GeminiEnrichmentNamedItem(BaseModel):
    name: str | None = None
    type: str | None = None
    code: str | None = None
    role: str | None = None
    description: str | None = None
    quantity: str | int | float | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None


class GeminiEnrichmentGlass(BaseModel):
    type: str | None = None
    thickness: str | None = None
    thickness_value: float | None = None
    thickness_unit: str | None = None
    color: str | None = None
    treatment: str | None = None
    composition: str | None = None
    description: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None


class GeminiEnrichmentEvidenceNote(BaseModel):
    source_id: str | None = None
    type: str | None = None
    text: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    region: Region | None = None
    visual_description: str | None = None
    notes: str | None = None


class GeminiEnrichmentComponent(BaseModel):
    name: str | None = None
    type: str | None = None
    role: str | None = None
    description: str | None = None
    quantity: str | int | float | None = None
    measurements: list[GeminiEnrichmentMeasurement] = Field(default_factory=list)
    geometry_raw: str | None = None
    configuration_raw: str | None = None
    glass: list[GeminiEnrichmentGlass] = Field(default_factory=list)
    materials: list[GeminiEnrichmentNamedItem] = Field(default_factory=list)
    profiles: list[GeminiEnrichmentNamedItem] = Field(default_factory=list)
    finish_raw: str | None = None
    accessories: list[GeminiEnrichmentNamedItem] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None


class GeminiElementEnrichment(BaseModel):
    temporary_id: str
    reference: str | None = None
    name: str | None = None
    category_raw: str | None = None
    description: str | None = None
    quantity: str | int | float | None = None
    functional_type_raw: str | None = None
    operation_raw: str | None = None
    panel_count: int | None = Field(default=None, ge=1)
    movable_panel_count: int | None = Field(default=None, ge=0)
    fixed_panel_count: int | None = Field(default=None, ge=0)
    modulation_raw: str | None = None
    opening_direction_raw: str | None = None
    special_features: list[str] = Field(default_factory=list)
    measurements: list[GeminiEnrichmentMeasurement] = Field(default_factory=list)
    geometry_type_raw: str | None = None
    geometry_raw: str | None = None
    configuration_raw: str | None = None
    glass: list[GeminiEnrichmentGlass] = Field(default_factory=list)
    materials: list[GeminiEnrichmentNamedItem] = Field(default_factory=list)
    profiles: list[GeminiEnrichmentNamedItem] = Field(default_factory=list)
    finish_raw: str | None = None
    accessories: list[GeminiEnrichmentNamedItem] = Field(default_factory=list)
    components: list[GeminiEnrichmentComponent] = Field(default_factory=list)
    occurrence_context: str | None = None
    variant_context: str | None = None
    evidence: list[GeminiEnrichmentEvidenceNote] = Field(default_factory=list)
    evidence_notes: list[str] = Field(default_factory=list)
    missing_or_unknown: list[str] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None


class GeminiEnrichmentResult(BaseModel):
    elements: list[GeminiElementEnrichment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: TokenUsage | None = None
