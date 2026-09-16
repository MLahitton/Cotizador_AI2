"""Private page-localization proposals. NOT the AI2/Backend extraction contract.

Validating this schema checks shape/coordinates/references, not visual correctness.
There are intentionally no final width, height, quantity or catalog fields.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RegionKind = Literal[
    "REFERENCE_LABEL", "DRAWING", "DIMENSION_AREA", "TABLE", "TABLE_ROW",
    "LOCAL_NOTE", "GENERAL_NOTE", "FLOOR_PLAN", "OTHER",
]
PageRole = Literal[
    "DETAIL_SHEET", "SCHEDULE", "FLOOR_PLAN", "SPECIFICATION", "SKETCH",
    "PHOTO", "OTHER", "UNKNOWN",
]


class LocalizationRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    region_id: str = Field(pattern=r"^r[1-9][0-9]{0,3}$")
    kind: RegionKind
    # Conventional visual detection coordinates, relative to the WHOLE input preview.
    box_2d: list[float] = Field(min_length=4, max_length=4)
    observed_label: str | None = Field(max_length=500)
    transcription: str | None = Field(max_length=4000)
    legibility: Literal["READABLE", "SMALL_OR_UNCERTAIN", "UNREADABLE"]

    @model_validator(mode="after")
    def check_box(self):
        y0, x0, y1, x1 = self.box_2d
        if not all(math.isfinite(v) and 0 <= v <= 1000 for v in self.box_2d):
            raise ValueError("box_2d must use finite full-preview coordinates in [0,1000]")
        if y1 <= y0 or x1 <= x0:
            raise ValueError("box_2d must have strictly positive width and height")
        return self


class LocalizedElementProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_id: str = Field(pattern=r"^e[1-9][0-9]{0,3}$")
    reference_raw: str | None = Field(max_length=200)
    description_of_location: str = Field(max_length=1500)
    reference_region_ids: list[str] = Field(max_length=50)
    drawing_region_ids: list[str] = Field(max_length=50)
    table_region_ids: list[str] = Field(max_length=50)
    dimension_region_ids: list[str] = Field(max_length=50)
    note_region_ids: list[str] = Field(max_length=50)
    association: Literal["PROPOSED", "AMBIGUOUS"]
    association_basis: str = Field(min_length=1, max_length=1500)

    def linked_region_ids(self) -> list[str]:
        return list(dict.fromkeys(
            self.reference_region_ids + self.drawing_region_ids + self.table_region_ids
            + self.dimension_region_ids + self.note_region_ids
        ))


class PageLocalizationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    job_id: str = Field(max_length=100)
    page_roles: list[PageRole] = Field(min_length=1, max_length=8)
    # A suggestion only: coordinates ALWAYS refer to the original preview as sent.
    suggested_rotation_clockwise: Literal[0, 90, 180, 270] | None
    coverage: Literal["FULL_SCAN_CLAIMED", "PARTIAL", "UNREADABLE", "NO_ELEMENTS_SEEN"]
    regions: list[LocalizationRegion] = Field(max_length=300)
    elements: list[LocalizedElementProposal] = Field(max_length=150)
    issues: list[str] = Field(max_length=100)

    @model_validator(mode="after")
    def check_links(self):
        region_map = {region.region_id: region for region in self.regions}
        if len(region_map) != len(self.regions):
            raise ValueError("Duplicate region_id")
        if len({e.candidate_id for e in self.elements}) != len(self.elements):
            raise ValueError("Duplicate candidate_id")
        if len(set(self.page_roles)) != len(self.page_roles):
            raise ValueError("Duplicate page role")
        roles = {
            "reference_region_ids": {"REFERENCE_LABEL", "TABLE_ROW", "TABLE"},
            "drawing_region_ids": {"DRAWING", "FLOOR_PLAN", "OTHER"},
            "table_region_ids": {"TABLE", "TABLE_ROW"},
            "dimension_region_ids": {"DIMENSION_AREA"},
            "note_region_ids": {"LOCAL_NOTE", "GENERAL_NOTE"},
        }
        for element in self.elements:
            if not element.linked_region_ids():
                raise ValueError("Element must link to at least one visible region")
            for field, allowed in roles.items():
                ids = getattr(element, field)
                if len(ids) != len(set(ids)):
                    raise ValueError(f"Duplicate link in {field}")
                for region_id in ids:
                    if region_id not in region_map:
                        raise ValueError("Dangling region link")
                    if region_map[region_id].kind not in allowed:
                        raise ValueError("Region kind incompatible with link")
            if element.reference_raw is not None:
                if not element.reference_raw.strip() or not element.reference_region_ids:
                    raise ValueError("A nonempty reference requires its visible label/table region")
        if not self.elements and self.coverage == "FULL_SCAN_CLAIMED":
            raise ValueError(
                "Use NO_ELEMENTS_SEEN/UNREADABLE/PARTIAL, not a full success for no elements"
            )
        if self.coverage == "NO_ELEMENTS_SEEN" and self.elements:
            raise ValueError("NO_ELEMENTS_SEEN cannot contain element proposals")
        if (self.coverage != "FULL_SCAN_CLAIMED" or
                self.suggested_rotation_clockwise != 0) and not self.issues:
            raise ValueError("Partial, unreadable or orientation-uncertain views require issues")
        return self


Box2DSpace = Literal["IMAGE_LOCAL_0_1000_YX", "FULL_PAGE_0_1000_YX"]
LocalizationEvidenceStatus = Literal[
    "UNVERIFIED", "BLOCKED_NO_VISIBLE_CONTENT", "REVIEW_REQUIRED"
]
LocalizationCoverageStatus = Literal[
    "LOCATED", "NOT_VISIBLE", "UNREADABLE", "UNRESOLVED"
]


class ImageFrameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    image_id: str = Field(pattern=r"^img[0-9]{1,3}$")
    window_px: list[int] = Field(min_length=4, max_length=4)
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    mime_type: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def check_window(self):
        x0, y0, x1, y1 = self.window_px
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Image frame window must have positive width and height")
        if x1 - x0 != self.image_width_px or y1 - y0 != self.image_height_px:
            raise ValueError("Image frame dimensions must match its page window")
        return self


class ImageFrameLocalizationRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    region_id: str = Field(pattern=r"^r[1-9][0-9]{0,3}$")
    kind: RegionKind
    source_image_id: str = Field(pattern=r"^img[0-9]{1,3}$")
    box_2d: list[float] = Field(min_length=4, max_length=4)
    box_2d_space: Literal["IMAGE_LOCAL_0_1000_YX"] = "IMAGE_LOCAL_0_1000_YX"
    observed_label: str | None = Field(max_length=500)
    transcription: str | None = Field(max_length=4000)
    legibility: Literal["READABLE", "SMALL_OR_UNCERTAIN", "UNREADABLE"]

    @model_validator(mode="after")
    def check_box(self):
        y0, x0, y1, x1 = self.box_2d
        if not all(math.isfinite(v) and 0 <= v <= 1000 for v in self.box_2d):
            raise ValueError("box_2d must use finite source-image coordinates in [0,1000]")
        if y1 <= y0 or x1 <= x0:
            raise ValueError("box_2d must have strictly positive width and height")
        return self


class ImageFrameLocalizedElementProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_id: str = Field(pattern=r"^e[1-9][0-9]{0,3}$")
    reference_raw: str | None = Field(max_length=200)
    description_of_location: str = Field(max_length=1500)
    reference_region_ids: list[str] = Field(max_length=50)
    drawing_region_ids: list[str] = Field(max_length=50)
    table_region_ids: list[str] = Field(max_length=50)
    dimension_region_ids: list[str] = Field(max_length=50)
    note_region_ids: list[str] = Field(max_length=50)
    missing_dimension_area_links: list[str] = Field(max_length=50)
    dimension_area_status: LocalizationCoverageStatus
    table_status: LocalizationCoverageStatus
    localization_notes: str = Field(max_length=1500)
    association: Literal["PROPOSED", "AMBIGUOUS"]
    association_basis: str = Field(min_length=1, max_length=1500)

    def linked_region_ids(self) -> list[str]:
        return list(dict.fromkeys(
            self.reference_region_ids + self.drawing_region_ids + self.table_region_ids
            + self.dimension_region_ids + self.note_region_ids
        ))


class ImageFrameLocalizationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["page-localization-image-frames-v1.0"]
    job_id: str = Field(max_length=100)
    image_registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_roles: list[PageRole] = Field(min_length=1, max_length=8)
    suggested_rotation_clockwise: Literal[0, 90, 180, 270] | None
    coverage: Literal["FULL_SCAN_CLAIMED", "PARTIAL", "UNREADABLE", "NO_ELEMENTS_SEEN"]
    regions: list[ImageFrameLocalizationRegion] = Field(max_length=300)
    elements: list[ImageFrameLocalizedElementProposal] = Field(max_length=150)
    issues: list[str] = Field(max_length=100)

    @model_validator(mode="after")
    def check_links(self):
        region_map = {region.region_id: region for region in self.regions}
        if len(region_map) != len(self.regions):
            raise ValueError("Duplicate region_id")
        if len({e.candidate_id for e in self.elements}) != len(self.elements):
            raise ValueError("Duplicate candidate_id")
        if len(set(self.page_roles)) != len(self.page_roles):
            raise ValueError("Duplicate page role")
        roles = {
            "reference_region_ids": {"REFERENCE_LABEL", "TABLE_ROW", "TABLE"},
            "drawing_region_ids": {"DRAWING", "FLOOR_PLAN", "OTHER"},
            "table_region_ids": {"TABLE", "TABLE_ROW"},
            "dimension_region_ids": {"DIMENSION_AREA"},
            "note_region_ids": {"LOCAL_NOTE", "GENERAL_NOTE"},
        }
        for element in self.elements:
            if not element.linked_region_ids():
                raise ValueError("Element must link to at least one visible region")
            for field, allowed in roles.items():
                ids = getattr(element, field)
                if len(ids) != len(set(ids)):
                    raise ValueError(f"Duplicate link in {field}")
                for region_id in ids:
                    if region_id not in region_map:
                        raise ValueError("Dangling region link")
                    if region_map[region_id].kind not in allowed:
                        raise ValueError("Region kind incompatible with link")
            if element.reference_raw is not None:
                if not element.reference_raw.strip() or not element.reference_region_ids:
                    raise ValueError("A nonempty reference requires its visible label/table region")
            if element.dimension_area_status == "LOCATED" and not element.dimension_region_ids:
                raise ValueError("LOCATED dimension areas require dimension links")
            if element.dimension_area_status == "NOT_VISIBLE" and element.dimension_region_ids:
                raise ValueError("NOT_VISIBLE dimension areas cannot include dimension links")
            if element.table_status == "LOCATED" and not element.table_region_ids:
                raise ValueError("LOCATED tables require table links")
            if element.table_status == "NOT_VISIBLE" and element.table_region_ids:
                raise ValueError("NOT_VISIBLE tables cannot include table links")
            statuses_requiring_notes = {"UNREADABLE", "UNRESOLVED"}
            if (
                element.dimension_area_status in statuses_requiring_notes
                or element.table_status in statuses_requiring_notes
            ) and not element.localization_notes.strip():
                raise ValueError("Unresolved coverage states require notes")
        if not self.elements and self.coverage == "FULL_SCAN_CLAIMED":
            raise ValueError(
                "Use NO_ELEMENTS_SEEN/UNREADABLE/PARTIAL, not a full success for no elements"
            )
        if self.coverage == "NO_ELEMENTS_SEEN" and self.elements:
            raise ValueError("NO_ELEMENTS_SEEN cannot contain element proposals")
        if (self.coverage != "FULL_SCAN_CLAIMED" or
                self.suggested_rotation_clockwise != 0) and not self.issues:
            raise ValueError("Partial, unreadable or orientation-uncertain views require issues")
        return self
