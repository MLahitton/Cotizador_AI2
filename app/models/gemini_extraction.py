from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus


class GeminiValue(BaseModel):
    value: str | int | float | bool | None = None
    text: str | None = None
    unit: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiMeasurement(BaseModel):
    type: str | None = None
    label: str | None = None
    value: float | None = None
    unit: str | None = None
    text: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiGlass(BaseModel):
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
    evidence: str | None = None
    notes: str | None = None


class GeminiNamedItem(BaseModel):
    name: str | None = None
    type: str | None = None
    code: str | None = None
    role: str | None = None
    description: str | None = None
    quantity: str | int | float | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiOccurrence(BaseModel):
    id: str | None = None
    location: str | None = None
    level: str | None = None
    typology: str | None = None
    quantity: str | int | float | None = None
    measurements: list[GeminiMeasurement] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiVariant(BaseModel):
    id: str | None = None
    label: str | None = None
    reason: str | None = None
    measurements: list[GeminiMeasurement] = Field(default_factory=list)
    configuration: str | None = None
    glass: list[GeminiGlass] = Field(default_factory=list)
    materials: list[GeminiNamedItem] = Field(default_factory=list)
    profiles: list[GeminiNamedItem] = Field(default_factory=list)
    finish: str | None = None
    accessories: list[GeminiNamedItem] = Field(default_factory=list)
    occurrence_ids: list[str] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiComponent(BaseModel):
    id: str | None = None
    name: str | None = None
    type: str | None = None
    role: str | None = None
    quantity: str | int | float | None = None
    description: str | None = None
    measurements: list[GeminiMeasurement] = Field(default_factory=list)
    geometry: str | None = None
    configuration: str | None = None
    glass: list[GeminiGlass] = Field(default_factory=list)
    materials: list[GeminiNamedItem] = Field(default_factory=list)
    profiles: list[GeminiNamedItem] = Field(default_factory=list)
    finish: str | None = None
    accessories: list[GeminiNamedItem] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiElement(BaseModel):
    id: str | None = None
    reference: str | None = None
    name: str | None = None
    category: str | None = None
    functional_type: str | None = None
    description: str | None = None
    measurements: list[GeminiMeasurement] = Field(default_factory=list)
    geometry_type: str | None = None
    geometry: str | None = None
    operation: str | None = None
    configuration: str | None = None
    panel_count: int | None = Field(default=None, ge=1)
    movable_panel_count: int | None = Field(default=None, ge=0)
    fixed_panel_count: int | None = Field(default=None, ge=0)
    modulation: str | None = None
    opening_direction: str | None = None
    special_features: list[str] = Field(default_factory=list)
    quantity: str | int | float | None = None
    glass: list[GeminiGlass] = Field(default_factory=list)
    materials: list[GeminiNamedItem] = Field(default_factory=list)
    profiles: list[GeminiNamedItem] = Field(default_factory=list)
    finish: str | None = None
    accessories: list[GeminiNamedItem] = Field(default_factory=list)
    components: list[GeminiComponent] = Field(default_factory=list)
    occurrences: list[GeminiOccurrence] = Field(default_factory=list)
    variants: list[GeminiVariant] = Field(default_factory=list)
    evidence: str | None = None
    evidence_items: list[GeminiEvidence] = Field(default_factory=list)
    missing_or_unknown: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None


class GeminiRequirementInfo(BaseModel):
    project_name: str | None = None
    client_name: str | None = None
    location: str | None = None
    project_type: str | None = None
    description: str | None = None
    dates: list[GeminiValue] = Field(default_factory=list)
    technical_notes: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiEvidence(BaseModel):
    id: str | None = None
    source_id: str | None = None
    type: str | None = None
    text: str | None = None
    visual_description: str | None = None
    location: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None


class GeminiRelationNote(BaseModel):
    description: str | None = None
    from_element: str | None = None
    to_element: str | None = None
    type: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None
    evidence: str | None = None
    notes: str | None = None


class GeminiExtraction(BaseModel):
    requirement: GeminiRequirementInfo | None = None
    elements: list[GeminiElement] = Field(default_factory=list)
    evidence: list[GeminiEvidence] = Field(default_factory=list)
    relationships: list[GeminiRelationNote] = Field(default_factory=list)
    conflicts: list[GeminiRelationNote] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    status: ExtractionStatus | None = None
    confidence: float | None = None
    notes: str | None = None
