from pathlib import Path

import pytest

from app.providers.gemini_extraction import (
    GeminiDiscoveryDebugCapture,
    GeminiExtractionProvider,
)


class _UploadedFile:
    def __init__(self, uri: str, mime_type: str) -> None:
        self.uri = uri
        self.mime_type = mime_type


class _FakeFilesClient:
    def __init__(self) -> None:
        self.uploads = []

    def upload(self, *, file: Path, config):
        path = Path(file)
        assert path.exists()
        self.uploads.append((path, config))
        return _UploadedFile(
            uri=f"gemini://{len(self.uploads)}",
            mime_type=config.mime_type,
        )


class _FakeModelsClient:
    def __init__(self, response_text: str) -> None:
        self.calls = []
        self.response_text = response_text

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.response_text)


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeClient:
    def __init__(self, response_text: str) -> None:
        self.files = _FakeFilesClient()
        self.models = _FakeModelsClient(response_text)


class _FakeProvider:
    def __init__(self, response_text: str) -> None:
        self._client = _FakeClient(response_text)
        self.model = "gemini-test"


def _provider_with_fake_client(response_text: str) -> GeminiExtractionProvider:
    provider = GeminiExtractionProvider.__new__(GeminiExtractionProvider)
    provider._provider = _FakeProvider(response_text)
    return provider


def test_discovery_from_single_file_returns_valid_json_result(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client(
        '{"elements": [{"temporary_id": "e1", "reference": "PV-01"}]}'
    )

    result = provider.discover_elements_from_files([pdf])

    assert len(result.elements) == 1
    assert result.elements[0].temporary_id == "e1"
    assert result.elements[0].reference == "PV-01"
    call = provider._provider._client.models.calls[0]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is None
    assert "PASS 1: ELEMENT DISCOVERY" in call["contents"][0].text


def test_discovery_from_multiple_files_sends_all_files_together(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    image = tmp_path / "foto.png"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    image.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    provider = _provider_with_fake_client('{"elements": []}')

    result = provider.discover_elements_from_files([pdf, image])

    assert result.elements == []
    assert len(provider._provider._client.files.uploads) == 2
    call = provider._provider._client.models.calls[0]
    assert len(call["contents"]) == 3
    assert "AVAILABLE SOURCES" in call["contents"][0].text
    assert "source-1 | planos.pdf | application/pdf" in call["contents"][0].text
    assert "source-2 | foto.png | image/png" in call["contents"][0].text


def test_discovery_debug_capture_stores_raw_and_validated_result(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    raw_json = '{"elements": [{"reference": "V-01"}], "notes": ["ok"]}'
    provider = _provider_with_fake_client(raw_json)
    debug_capture = GeminiDiscoveryDebugCapture()

    result = provider.discover_elements_from_files([pdf], debug_capture=debug_capture)

    assert debug_capture.raw_response_text == raw_json
    assert debug_capture.discovery_result is result
    assert debug_capture.model == "gemini-test"


def test_discovery_rejects_invalid_json(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client('{"elements": "invalid"}')

    with pytest.raises(ValueError, match="GeminiDiscoveryResult"):
        provider.discover_elements_from_files([pdf])


def test_discovery_preserves_empty_element_list(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client('{"elements": [], "notes": ["no elements"]}')

    result = provider.discover_elements_from_files([pdf])

    assert result.elements == []
    assert result.notes == ["no elements"]


def test_discovery_preserves_elements_without_reference(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client(
        '{"elements": [{"temporary_id": "sin-ref-1", "reference": null}]}'
    )

    result = provider.discover_elements_from_files([pdf])

    assert len(result.elements) == 1
    assert result.elements[0].temporary_id == "sin-ref-1"
    assert result.elements[0].reference is None


def test_discovery_does_not_deduplicate_distinct_or_repeated_references(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client(
        """
        {
          "elements": [
            {"temporary_id": "a", "reference": "V-01"},
            {"temporary_id": "b", "reference": "V-02"},
            {"temporary_id": "c", "reference": "V-01"}
          ]
        }
        """
    )

    result = provider.discover_elements_from_files([pdf])

    assert [element.temporary_id for element in result.elements] == ["a", "b", "c"]
    assert [element.reference for element in result.elements] == ["V-01", "V-02", "V-01"]


def test_discovery_rejects_empty_input_file_list() -> None:
    provider = _provider_with_fake_client('{"elements": []}')

    with pytest.raises(ValueError, match="discovery"):
        provider.discover_elements_from_files([])
