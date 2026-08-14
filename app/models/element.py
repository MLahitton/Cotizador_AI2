from pydantic import BaseModel, Field

from app.models.common import NormalizedValue, TraceableValue
from app.models.configuration import Configuration
from app.models.element_parts import Component, Occurrence, Variant
from app.models.geometry import Geometry
from app.models.measurement import Measurement
from app.models.specifications import (
    AccessorySpecification,
    FinishSpecification,
    GlassSpecification,
    MaterialSpecification,
    ProfileSpecification,
)


class Element(BaseModel):
    id: str

    reference: TraceableValue | None = None
    name: TraceableValue | None = None
    category: NormalizedValue | None = None
    description: str | None = None

    occurrences: list[Occurrence] = Field(default_factory=list)
    variants: list[Variant] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)

    geometry: Geometry | None = None
    measurements: list[Measurement] = Field(default_factory=list)

    quantity: TraceableValue | None = None
    configuration: Configuration | None = None

    glass: list[GlassSpecification] = Field(default_factory=list)
    materials: list[MaterialSpecification] = Field(default_factory=list)
    profiles: list[ProfileSpecification] = Field(default_factory=list)
    finish: FinishSpecification | None = None
    accessories: list[AccessorySpecification] = Field(default_factory=list)

    technical_notes: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None