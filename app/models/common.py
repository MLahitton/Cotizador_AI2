from enum import StrEnum

from pydantic import BaseModel, Field


class ExtractionStatus(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class EvidenceRef(BaseModel):
    evidence_ids: list[str] = Field(default_factory=list)


class TraceableValue(BaseModel):
    value: str | int | float | bool | None = None
    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class NormalizedValue(BaseModel):
    normalized: str | None = None
    raw: str | None = None
    status: ExtractionStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None