from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class SimilarityLevel(StrEnum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    REJECTED = "REJECTED"


class SimilarityBatchResultStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SimilarityElementInput(BaseModel):
    element_id: str
    reference: str | None = None
    category: str | None = None
    system: str | None = None
    glass_family: str | None = None
    glass_thickness: Decimal | None = None
    glass_composition: str | None = None
    configuration: str | None = None
    width_mm: Decimal | None = None
    height_mm: Decimal | None = None
    area_m2: Decimal | None = None
    quantity: Decimal | None = None
    finish: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class SimilarityHistoricalCandidateInput(BaseModel):
    candidate_id: str
    quote_id: str | None = None
    historical_item_id: str | None = None
    reference: str | None = None
    description: str | None = None
    category: str | None = None
    system: str | None = None
    glass_family: str | None = None
    glass_thickness: Decimal | None = None
    glass_composition: str | None = None
    configuration: str | None = None
    width_mm: Decimal | None = None
    height_mm: Decimal | None = None
    area_m2: Decimal | None = None
    quantity: Decimal | None = None
    finish: str | None = None
    public_unit_price: Decimal | None = None
    public_total: Decimal | None = None
    currency: str | None = None
    backend_preliminary_score: float | None = Field(default=None, ge=0, le=1)
    matched_signals: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)


class SimilarityCandidateResult(BaseModel):
    candidate_id: str
    similarity_score: float = Field(ge=0, le=1)
    similarity_level: SimilarityLevel
    matched_features: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    technical_explanation: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class SimilarityEvaluationResult(BaseModel):
    element_id: str
    evaluated_candidate_count: int = Field(ge=0)
    candidates: list[SimilarityCandidateResult] = Field(default_factory=list)
    overall_notes: list[str] = Field(default_factory=list)
    evaluation_source: str = "AI2_SIMILARITY"


class SimilarityBatchRequestItem(BaseModel):
    request_id: str
    element: SimilarityElementInput
    candidates: list[SimilarityHistoricalCandidateInput] = Field(min_length=1)


class SimilarityBatchEvaluationRequest(BaseModel):
    requests: list[SimilarityBatchRequestItem] = Field(min_length=1)


class SimilarityBatchResultItem(BaseModel):
    request_id: str
    status: SimilarityBatchResultStatus = SimilarityBatchResultStatus.COMPLETED
    candidates: list[SimilarityCandidateResult] = Field(default_factory=list)
    failure_code: str | None = None


class SimilarityBatchEvaluationResult(BaseModel):
    results: list[SimilarityBatchResultItem] = Field(default_factory=list)
    evaluation_source: str = "AI2_SIMILARITY_BATCH"
