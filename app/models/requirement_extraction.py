from pydantic import BaseModel, Field

from app.models.element import Element
from app.models.evidence import Evidence, Source
from app.models.relations import Conflict, Relationship
from app.models.requirement import (
    ExtractionMetadata,
    Requirement,
    Warning,
)


class RequirementExtraction(BaseModel):
    requirement: Requirement

    sources: list[Source] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)

    evidence: list[Evidence] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)

    extraction_metadata: ExtractionMetadata