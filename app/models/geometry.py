from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus
from app.models.measurement import Measurement


class GeometrySegment(BaseModel):
    id: str
    label: str | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    notes: str | None = None


class Geometry(BaseModel):
    normalized_type: str | None = None
    raw_type: str | None = None
    description: str | None = None

    segments: list[GeometrySegment] = Field(default_factory=list)

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None   