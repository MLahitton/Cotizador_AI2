from enum import StrEnum

from pydantic import BaseModel, Field


class ScopeStatus(StrEnum):
    IN_SCOPE_FULL = "in_scope_full"
    IN_SCOPE_PARTIAL = "in_scope_partial"
    OUT_OF_SCOPE = "out_of_scope"
    UNCERTAIN = "uncertain"


class GeminiScopeClassification(BaseModel):
    temporary_id: str
    scope: ScopeStatus
    reason: str | None = None
    in_scope_components: list[str] = Field(default_factory=list)
    out_of_scope_components: list[str] = Field(default_factory=list)
    evidence_notes: list[str] = Field(default_factory=list)
    confidence: float | None = None


class GeminiScopeResult(BaseModel):
    elements: list[GeminiScopeClassification] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
