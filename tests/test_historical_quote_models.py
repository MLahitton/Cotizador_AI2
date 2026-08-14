from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.historical_quote import (
    HistoricalQuote,
    HistoricalQuoteIssue,
    HistoricalQuoteIssueSeverity,
    HistoricalQuoteItem,
    HistoricalQuoteSource,
)


def test_historical_quote_source_valid_construction_and_serialization() -> None:
    source = HistoricalQuoteSource(
        file_name="cotizacion.xlsx",
        sha256="a" * 64,
        file_format="xlsx",
        workbook_type="steel_and_glass_quote",
        source_path="historical/2024/cotizacion.xlsx",
        source_index=3,
    )

    data = source.model_dump(mode="json")

    assert data == {
        "file_name": "cotizacion.xlsx",
        "sha256": "a" * 64,
        "file_format": "xlsx",
        "workbook_type": "steel_and_glass_quote",
        "source_path": "historical/2024/cotizacion.xlsx",
        "source_index": 3,
    }


def test_historical_quote_source_optional_fields() -> None:
    source = HistoricalQuoteSource(
        file_name="cotizacion.pdf",
        sha256="b" * 64,
        file_format="pdf",
    )

    assert source.workbook_type is None
    assert source.source_path is None
    assert source.source_index is None


def test_historical_quote_item_minimal_valid_construction() -> None:
    item = HistoricalQuoteItem(reference="V-01")

    assert item.reference == "V-01"
    assert item.category_normalized is None
    assert item.source_cells == []


def test_historical_quote_item_rich_construction_preserves_raw_and_normalized() -> None:
    item = HistoricalQuoteItem(
        id="item-1",
        reference="PV-01",
        name="Puerta vidriera",
        category_raw="PUERTA VIDRIERA",
        category_normalized="door_glass",
        location_raw="Nivel 1",
        occurrence_context="Habitacion principal",
        width=Decimal("1.20"),
        height=Decimal("2.40"),
        area=Decimal("2.880"),
        quantity=Decimal("2"),
        dimension_unit="m",
        area_unit="m2",
        system_raw="Sistema especial",
        system_normalized="special_system",
        glass_raw="4 + PVB 0.38 + 4 templado",
        glass_family="laminated",
        glass_type="laminated_tempered",
        glass_thickness_mm=Decimal("8.38"),
        glass_composition="4 + PVB 0.38 + 4",
        glass_color="claro",
        glass_treatment="templado",
        finish_raw="Pintura negra",
        finish_normalized="black",
        hardware=["bisagra hidraulica"],
        protections=["protector de canto"],
        lock_raw="cerradura tipo pico loro",
        public_unit_price=Decimal("1234567.89"),
        public_total=Decimal("2469135.78"),
        currency="COP",
        source_sheet="COTIZACION",
        source_cells=["COTIZACION!A17", "COTIZACION!B20:G26"],
    )

    assert item.category_raw == "PUERTA VIDRIERA"
    assert item.category_normalized == "door_glass"
    assert item.glass_raw == "4 + PVB 0.38 + 4 templado"
    assert item.glass_composition == "4 + PVB 0.38 + 4"
    assert item.system_raw == "Sistema especial"
    assert item.system_normalized == "special_system"
    assert item.source_cells == ["COTIZACION!A17", "COTIZACION!B20:G26"]


def test_historical_quote_item_money_uses_decimal_without_trivial_precision_loss() -> None:
    item = HistoricalQuoteItem(
        public_unit_price=Decimal("0.10"),
        public_total=Decimal("0.30"),
        currency="COP",
    )

    assert item.public_unit_price + item.public_total == Decimal("0.40")
    assert item.model_dump(mode="json")["public_unit_price"] == "0.10"


def test_historical_quote_full_serialization_with_metadata_revision_and_variant() -> None:
    quote = HistoricalQuote(
        id="historical-quote-1",
        quote_id="commercial-quote-1",
        customer_name="Cliente",
        project_name="Proyecto",
        location="Bogota",
        commercial_variant="base",
        commercial_line="ventaneria",
        sales_owner="asesor",
        validity_raw="15 dias",
        currency="COP",
        revision_family="commercial-quote-1",
        revision_label="R1",
        variant_label="Alternativa A",
        source=HistoricalQuoteSource(
            file_name="cotizacion.xlsx",
            sha256="c" * 64,
            file_format="xlsx",
        ),
        items=[HistoricalQuoteItem(id="item-1", reference="V-01")],
        issues=[
            HistoricalQuoteIssue(
                code="missing_total",
                message="No se encontro total publico.",
                severity=HistoricalQuoteIssueSeverity.WARNING,
                item_id="item-1",
                source_sheet="COTIZACION",
                source_cells=["COTIZACION!G20"],
            )
        ],
    )

    data = quote.model_dump(mode="json")

    assert data["quote_id"] == "commercial-quote-1"
    assert data["revision_label"] == "R1"
    assert data["variant_label"] == "Alternativa A"
    assert data["source"]["file_name"] == "cotizacion.xlsx"
    assert data["items"][0]["reference"] == "V-01"
    assert data["issues"][0]["severity"] == "warning"


def test_historical_quote_issue_validates_severity_and_optional_context() -> None:
    issue = HistoricalQuoteIssue(
        code="unknown_glass",
        message="Vidrio no normalizado.",
        severity=HistoricalQuoteIssueSeverity.INFO,
    )

    assert issue.item_id is None
    assert issue.source_cells == []

    with pytest.raises(ValidationError):
        HistoricalQuoteIssue(
            code="bad",
            message="bad severity",
            severity="critical",
        )


def test_historical_quote_does_not_expose_internal_cost_fields() -> None:
    forbidden_fields = {
        "commission",
        "profit",
        "margin",
        "benefit",
        "backoffice",
        "internal_cost",
        "labor_cost",
    }

    assert forbidden_fields.isdisjoint(HistoricalQuoteItem.model_fields)
    assert forbidden_fields.isdisjoint(HistoricalQuote.model_fields)
