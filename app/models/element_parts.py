from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus, NormalizedValue, TraceableValue
from app.models.configuration import Configuration
from app.models.geometry import Geometry
from app.models.measurement import Measurement
from app.models.specifications import (
    AccessorySpecification,
    FinishSpecification,
    GlassSpecification,
    MaterialSpecification,
    ProfileSpecification,
)


class Occurrence(BaseModel):
    id: str
    source_id: str | None = None

    location: TraceableValue | None = None
    level: TraceableValue | None = None
    typology: TraceableValue | None = None

    quantity: TraceableValue | None = None
    measurements: list[Measurement] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class Variant(BaseModel):
    id: str

    label: str | None = None
    reason: str | None = None

    measurements: list[Measurement] = Field(default_factory=list)
    configuration: Configuration | None = None

    glass: list[GlassSpecification] = Field(default_factory=list)
    materials: list[MaterialSpecification] = Field(default_factory=list)
    profiles: list[ProfileSpecification] = Field(default_factory=list)
    finish: FinishSpecification | None = None
    accessories: list[AccessorySpecification] = Field(default_factory=list)

    occurrence_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    status: ExtractionStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    notes: str | None = None


class Component(BaseModel):
    id: str

    name: TraceableValue | None = None
    type: NormalizedValue | None = None
    role: NormalizedValue | None = None

    quantity: TraceableValue | None = None

    geometry: Geometry | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    configuration: Configuration | None = None

    glass: list[GlassSpecification] = Field(default_factory=list)
    materials: list[MaterialSpecification] = Field(default_factory=list)
    profiles: list[ProfileSpecification] = Field(default_factory=list)
    finish: FinishSpecification | None = None
    accessories: list[AccessorySpecification] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)

    status: ExtractionStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    notes: str | None = None