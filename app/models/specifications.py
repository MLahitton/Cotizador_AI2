from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus, NormalizedValue, TraceableValue
from app.models.measurement import Measurement


class GlassSpecification(BaseModel):
    type: NormalizedValue | None = None
    thickness: Measurement | None = None
    color: NormalizedValue | None = None

    treatment: NormalizedValue | None = None
    composition: str | None = None
    coating: NormalizedValue | None = None
    transparency: NormalizedValue | None = None

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class MaterialSpecification(BaseModel):
    normalized_type: str | None = None
    raw_description: str | None = None

    grade: str | None = None
    dimensions: list[Measurement] = Field(default_factory=list)

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class ProfileSpecification(BaseModel):
    code: TraceableValue | None = None
    name: TraceableValue | None = None

    raw_description: str | None = None
    role: NormalizedValue | None = None
    material: NormalizedValue | None = None

    dimensions: list[Measurement] = Field(default_factory=list)

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class FinishSpecification(BaseModel):
    normalized_type: str | None = None
    color: NormalizedValue | None = None
    texture: NormalizedValue | None = None
    code: TraceableValue | None = None

    raw_description: str | None = None

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class AccessorySpecification(BaseModel):
    normalized_type: str | None = None
    raw_description: str | None = None

    brand: TraceableValue | None = None
    model: TraceableValue | None = None
    quantity: TraceableValue | None = None

    placement: str | None = None

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None