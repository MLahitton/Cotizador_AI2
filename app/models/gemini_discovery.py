from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus


class GeminiElementDiscovery(BaseModel):
    temporary_id: str | None = None
    reference: str | None = None
    name: str | None = None
    category_raw: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    source_hint: str | None = None
    status: ExtractionStatus | None = None
    confidence: float | None = None


class GeminiDiscoveryResult(BaseModel):
    elements: list[GeminiElementDiscovery] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
