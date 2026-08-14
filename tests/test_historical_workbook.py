from pathlib import Path

from openpyxl import Workbook

from app.services.historical_workbook import (
    HistoricalWorkbookIssueCode,
    HistoricalWorkbookType,
    calculate_sha256,
    inspect_historical_workbook,
)


def test_sha256_is_stable_for_same_content_and_independent_of_name(tmp_path: Path) -> None:
    first = tmp_path / "first.xlsx"
    second = tmp_path / "second.xlsx"
    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    assert calculate_sha256(first) == calculate_sha256(second)
    assert len(calculate_sha256(first)) == 64
    assert calculate_sha256(first) == calculate_sha256(first).lower()


def test_sha256_changes_for_different_content(tmp_path: Path) -> None:
    first = tmp_path / "first.xlsx"
    second = tmp_path / "second.xlsx"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    assert calculate_sha256(first) != calculate_sha256(second)


def test_valid_ooxml_workbook_is_processable_and_lists_sheets(tmp_path: Path) -> None:
    workbook_path = tmp_path / "quote.xlsx"
    _save_workbook(workbook_path, ["COTIZACIÓN", "BD GN"])

    inspection = inspect_historical_workbook(workbook_path, source_index=1)

    assert inspection.source.file_name == "quote.xlsx"
    assert inspection.source.file_format == "xlsx"
    assert inspection.source.workbook_type == HistoricalWorkbookType.OOXML
    assert inspection.source.source_index == 1
    assert inspection.is_processable is True
    assert inspection.sheet_names == ["COTIZACIÓN", "BD GN"]
    assert inspection.has_quote_sheet is True
    assert inspection.issues == []


def test_missing_quote_sheet_produces_controlled_warning(tmp_path: Path) -> None:
    workbook_path = tmp_path / "quote.xlsx"
    _save_workbook(workbook_path, ["BD GN", "P&G"])

    inspection = inspect_historical_workbook(workbook_path)

    assert inspection.is_processable is True
    assert inspection.has_quote_sheet is False
    assert [issue.code for issue in inspection.issues] == [
        HistoricalWorkbookIssueCode.MISSING_QUOTE_SHEET
    ]
    assert inspection.issues[0].severity == "warning"


def test_empty_file_produces_controlled_error(tmp_path: Path) -> None:
    workbook_path = tmp_path / "empty.xlsx"
    workbook_path.write_bytes(b"")

    inspection = inspect_historical_workbook(workbook_path)

    assert inspection.source.sha256 == calculate_sha256(workbook_path)
    assert inspection.source.workbook_type == HistoricalWorkbookType.UNKNOWN
    assert inspection.is_processable is False
    assert [issue.code for issue in inspection.issues] == [
        HistoricalWorkbookIssueCode.EMPTY_FILE
    ]


def test_unknown_content_is_classified_without_stacktrace(tmp_path: Path) -> None:
    workbook_path = tmp_path / "not-a-workbook.xlsx"
    workbook_path.write_bytes(b"plain text")

    inspection = inspect_historical_workbook(workbook_path)

    assert inspection.source.file_format == "xlsx"
    assert inspection.source.workbook_type == HistoricalWorkbookType.UNKNOWN
    assert inspection.is_processable is False
    assert [issue.code for issue in inspection.issues] == [
        HistoricalWorkbookIssueCode.UNKNOWN_FORMAT
    ]


def test_ole_cdfv2_container_is_detected_but_not_processable(tmp_path: Path) -> None:
    workbook_path = tmp_path / "legacy.xlsx"
    workbook_path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1legacy")

    inspection = inspect_historical_workbook(workbook_path)

    assert inspection.source.file_format == "xlsx"
    assert inspection.source.workbook_type == HistoricalWorkbookType.OLE_CDFV2
    assert inspection.is_processable is False
    assert [issue.code for issue in inspection.issues] == [
        HistoricalWorkbookIssueCode.UNSUPPORTED_CONTAINER
    ]


def test_corrupt_zip_candidate_produces_invalid_ooxml_issue(tmp_path: Path) -> None:
    workbook_path = tmp_path / "corrupt.xlsx"
    workbook_path.write_bytes(b"PKnot a real zip")

    inspection = inspect_historical_workbook(workbook_path)

    assert inspection.source.workbook_type == HistoricalWorkbookType.UNKNOWN
    assert inspection.is_processable is False
    assert [issue.code for issue in inspection.issues] == [
        HistoricalWorkbookIssueCode.INVALID_OOXML
    ]


def test_zip_without_xlsx_structure_is_invalid_ooxml(tmp_path: Path) -> None:
    workbook_path = tmp_path / "archive.xlsx"
    _save_zip_like_file(workbook_path)

    inspection = inspect_historical_workbook(workbook_path)

    assert inspection.source.workbook_type == HistoricalWorkbookType.UNKNOWN
    assert inspection.is_processable is False
    assert [issue.code for issue in inspection.issues] == [
        HistoricalWorkbookIssueCode.INVALID_OOXML
    ]


def test_misleading_extension_is_classified_by_content(tmp_path: Path) -> None:
    workbook_path = tmp_path / "misleading.xlsx"
    workbook_path.write_bytes(b"not xlsx content")

    inspection = inspect_historical_workbook(workbook_path)

    assert inspection.source.file_format == "xlsx"
    assert inspection.source.workbook_type == HistoricalWorkbookType.UNKNOWN
    assert inspection.is_processable is False


def _save_workbook(path: Path, sheet_names: list[str]) -> None:
    workbook = Workbook()
    default_sheet = workbook.active
    default_sheet.title = sheet_names[0]
    for sheet_name in sheet_names[1:]:
        workbook.create_sheet(sheet_name)
    workbook.save(path)
    workbook.close()


def _save_zip_like_file(path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("not-workbook.txt", "content")
