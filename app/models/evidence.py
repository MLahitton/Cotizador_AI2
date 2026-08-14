from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus


class Region(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class Source(BaseModel):
    id: str
    file_name: str
    media_type: str

    source_type: str | None = None

    page_count: int | None = Field(default=None, ge=1)
    sheet_names: list[str] = Field(default_factory=list)

    description: str | None = None
    processing_status: str = "processed"


class Evidence(BaseModel):
    id: str
    source_id: str

    type: str

    page_number: int | None = Field(default=None, ge=1)
    sheet_name: str | None = None
    cell_range: str | None = None
    region: Region | None = None

    extracted_text: str | None = None
    visual_description: str | None = None

    status: ExtractionStatus = ExtractionStatus.EXPLICIT
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    notes: str | None = None