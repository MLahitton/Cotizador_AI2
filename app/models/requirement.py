from pydantic import BaseModel, Field

from app.models.common import TraceableValue


class Requirement(BaseModel):
    project_id: str | None = None
    requirement_id: str | None = None

    project_name: TraceableValue | None = None
    client_name: TraceableValue | None = None
    location: TraceableValue | None = None
    project_type: TraceableValue | None = None

    description: str | None = None

    dates: list[TraceableValue] = Field(default_factory=list)
    general_technical_notes: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Warning(BaseModel):
    id: str
    code: str
    severity: str
    message: str

    source_ids: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    recoverable: bool = True


class TokenUsage(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ExtractionMetadata(BaseModel):
    schema_version: str = "1.0"

    model_provider: str | None = None
    model: str | None = None

    started_at: str | None = None
    completed_at: str | None = None
    processing_time_ms: int | None = Field(default=None, ge=0)

    source_count: int = Field(default=0, ge=0)
    element_count: int = Field(default=0, ge=0)

    partial: bool = False
    status: str = "completed"

    token_usage: TokenUsage | None = None

    pipeline_version: str = "ai2-v1"