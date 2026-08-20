from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus, NormalizedValue, TraceableValue


class GridConfiguration(BaseModel):
    rows: int | None = Field(default=None, ge=1)
    columns: int | None = Field(default=None, ge=1)

    raw_description: str | None = None

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class Configuration(BaseModel):
    normalized_type: str | None = None
    raw_description: str | None = None

    operation: NormalizedValue | None = None

    panel_count: TraceableValue | None = None
    movable_panel_count: TraceableValue | None = None
    fixed_panel_count: TraceableValue | None = None

    arrangement: str | None = None
    modulation: str | None = None
    opening_direction: NormalizedValue | None = None

    tracks: TraceableValue | None = None
    leaves: TraceableValue | None = None

    grid: GridConfiguration | None = None

    special_features: list[str] = Field(default_factory=list)

    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
