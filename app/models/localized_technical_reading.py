"""Experimental localized technical reading artifacts; not a public API contract."""
from __future__ import annotations

import re
from types import UnionType
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.evidence import Region
from app.models.gemini_enrichment import GeminiElementEnrichment, GeminiEnrichmentResult

FieldEvidenceStatus = Literal[
    "SUPPORTED",
    "AMBIGUOUS",
    "CONFLICTING",
    "PENDING",
    "BLOCKED",
]


class LocalizedFieldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field_path: str
    input_image_id: str
    source_id: str
    region_id: str | None = None
    region: Region | None = None
    observed_text: str | None = None
    visual_description: str | None = None
    status: FieldEvidenceStatus
    notes: str | None = None


class LocalizedCandidateTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    technical_candidate_id: str
    document_id: str
    view_index: int
    localization_candidate_id: str
    reference_raw: str | None = None
    source_id: str
    linked_region_ids: list[str] = Field(default_factory=list)
    excluded_blocked_region_ids: list[str] = Field(default_factory=list)
    field_evidence: list[LocalizedFieldEvidence] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LocalizedTechnicalReadingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["LOCALIZED_TECHNICAL_READING_RECORDED_UNVERIFIED", "HAS_ERRORS"]
    enrichment: GeminiEnrichmentResult
    candidate_traces: list[LocalizedCandidateTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_map: dict[str, str] = Field(default_factory=dict)
    visual_association_validated: bool = False
    ready_for_pricing: bool = False
    experimental: bool = True


class LocalizedTechnicalReadEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enrichment: GeminiEnrichmentResult
    field_evidence: list[LocalizedFieldEvidence] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_field_paths(self):
        for evidence in self.field_evidence:
            evidence.field_path = _canonical_field_path(evidence.field_path)
        return self


_FULL_ELEMENT_PATH = re.compile(r"^enrichment\.elements\[(\d+)]\.(.+)$")
_PATH_SEGMENT = re.compile(r"^([A-Za-z_]\w*)(?:\[(\d+)])?$")


def _canonical_field_path(field_path: str) -> str:
    match = _FULL_ELEMENT_PATH.fullmatch(field_path)
    if match:
        if match.group(1) != "0":
            raise ValueError("field_path must reference a GeminiElementEnrichment field")
        field_path = match.group(2)
    elif field_path.startswith("enrichment."):
        raise ValueError("field_path must reference a GeminiElementEnrichment field")

    if not _field_path_targets_model(GeminiElementEnrichment, field_path):
        raise ValueError("field_path must reference a GeminiElementEnrichment field")
    return field_path


def _field_path_targets_model(model_type: type[BaseModel], field_path: str) -> bool:
    current_type: object = model_type
    segments = field_path.split(".")
    if not segments:
        return False
    for index, segment in enumerate(segments):
        if not isinstance(current_type, type) or not issubclass(current_type, BaseModel):
            return False
        match = _PATH_SEGMENT.fullmatch(segment)
        if not match:
            return False
        field_name, item_index = match.groups()
        field = current_type.model_fields.get(field_name)
        if field is None:
            return False
        annotation = _unwrap_optional(field.annotation)
        if item_index is not None:
            item_type = _list_item_type(annotation)
            if item_type is None:
                return False
            annotation = _unwrap_optional(item_type)
        elif index < len(segments) - 1 and _list_item_type(annotation) is not None:
            return False
        current_type = annotation
    return True


def _unwrap_optional(annotation: object) -> object:
    origin = get_origin(annotation)
    if origin in {UnionType, Union}:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _list_item_type(annotation: object) -> object | None:
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return args[0] if args else object
    return None
