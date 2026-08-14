from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus, NormalizedValue
from app.models.measurement import EntityReference


class Relationship(BaseModel):
    id: str

    type: NormalizedValue

    from_entity: EntityReference
    to_entity: EntityReference

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class ConflictCandidate(BaseModel):
    value: str | int | float | bool | None = None
    unit: str | None = None

    source_entity_id: str | None = None

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class SuggestedResolution(BaseModel):
    candidate_index: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


class Conflict(BaseModel):
    id: str

    scope: EntityReference
    field: str

    candidates: list[ConflictCandidate] = Field(default_factory=list)

    severity: str
    status: str = "unresolved"

    suggested_resolution: SuggestedResolution | None = None

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None