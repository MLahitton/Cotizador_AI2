"""Private preparation artifacts, not the AI2/Backend extraction contract.

Coordinates and decoded PDF characters are observations only. This schema never
assigns a business meaning such as height, quantity or commercial system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourcePreparationOptions:
    render_scale: float = 2.0
    max_render_edge: int = 4096
    max_render_pixels: int = 12_000_000
    max_image_pixels: int = 30_000_000
    max_file_bytes: int = 50 * 1024 * 1024
    max_pages: int = 100
    max_chars_per_page: int = 100_000

    def __post_init__(self) -> None:
        if isinstance(self.render_scale, bool) or not math.isfinite(self.render_scale):
            raise ValueError("render_scale debe ser un numero finito positivo.")
        if self.render_scale <= 0 or self.render_scale > 8:
            raise ValueError("render_scale debe estar entre 0 (exclusivo) y 8.")
        for name in (
            "max_render_edge", "max_render_pixels", "max_image_pixels",
            "max_file_bytes", "max_pages", "max_chars_per_page",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} debe ser un entero positivo.")
        # These are safety ceilings, not claims about readable text resolution.
        if self.max_render_edge > 8192 or self.max_render_pixels > 32_000_000:
            raise ValueError("Limite de render demasiado alto para esta fase.")


@dataclass
class PreparedPage:
    view_index: int
    page_number: int | None   # Physical PDF page; NEVER fabricated for a JPG/PNG.
    pdf_page_label: str | None
    status: str
    preview_file: str | None
    observations_file: str | None
    width_px: int | None = None
    height_px: int | None = None
    native_char_count: int = 0
    located_char_count: int = 0
    text_state: str = "NOT_READ"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class PreparedSource:
    schema_version: int
    document_id: str
    content_sha256: str
    logical_name: str
    input_index: int
    size_bytes: int
    media_type: str
    status: str
    page_count: int | None
    pages: list[PreparedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    engine_versions: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
