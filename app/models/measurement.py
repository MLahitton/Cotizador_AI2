from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus


class Measurement(BaseModel):
    type: str
    raw_label: str | None = None

    value: float | None = None
    unit: str | None = None

    raw_value: float | None = None
    raw_unit: str | None = None

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class EntityReference(BaseModel):
    entity_type: str
    entity_id: str