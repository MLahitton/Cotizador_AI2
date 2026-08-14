from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class HistoricalQuoteIssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class HistoricalQuoteSource(BaseModel):
    file_name: str
    sha256: str
    file_format: str
    workbook_type: str | None = None
    source_path: str | None = None
    source_index: int | None = Field(default=None, ge=1)


class HistoricalQuoteIssue(BaseModel):
    code: str
    message: str
    severity: HistoricalQuoteIssueSeverity = HistoricalQuoteIssueSeverity.WARNING
    item_id: str | None = None
    source_sheet: str | None = None
    source_cells: list[str] = Field(default_factory=list)


class HistoricalQuoteItem(BaseModel):
    id: str | None = None
    reference: str | None = None
    name: str | None = None
    category_raw: str | None = None
    category_normalized: str | None = None

    location_raw: str | None = None
    occurrence_context: str | None = None

    width: Decimal | None = None
    height: Decimal | None = None
    area: Decimal | None = None
    quantity: Decimal | None = None
    dimension_unit: str | None = None
    area_unit: str | None = None
    quantity_unit: str | None = None

    system_raw: str | None = None
    system_normalized: str | None = None

    glass_raw: str | None = None
    glass_family: str | None = None
    glass_type: str | None = None
    glass_thickness_mm: Decimal | None = None
    glass_composition: str | None = None
    glass_color: str | None = None
    glass_treatment: str | None = None

    finish_raw: str | None = None
    finish_normalized: str | None = None

    hardware: list[str] = Field(default_factory=list)
    protections: list[str] = Field(default_factory=list)
    lock_raw: str | None = None

    public_unit_price: Decimal | None = None
    public_total: Decimal | None = None
    currency: str | None = None

    source_sheet: str | None = None
    source_cells: list[str] = Field(default_factory=list)

    notes: str | None = None


class HistoricalQuote(BaseModel):
    id: str | None = None
    quote_id: str | None = None

    customer_name: str | None = None
    project_name: str | None = None
    location: str | None = None

    commercial_variant: str | None = None
    commercial_line: str | None = None
    sales_owner: str | None = None
    validity_raw: str | None = None
    currency: str | None = None

    revision_family: str | None = None
    revision_label: str | None = None
    variant_label: str | None = None

    source: HistoricalQuoteSource
    items: list[HistoricalQuoteItem] = Field(default_factory=list)
    issues: list[HistoricalQuoteIssue] = Field(default_factory=list)

    notes: str | None = None
