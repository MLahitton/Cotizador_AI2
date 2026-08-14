import zipfile
from pathlib import Path

import pytest

import app.providers.gemini_extraction as provider_module
from app.models.requirement import ExtractionMetadata, Requirement
from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini_extraction import GeminiExtractionProvider

_DEFAULT_RESPONSE_TEXT = '{"elements": [{"reference": "V1"}]}'


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
    def __init__(
        self,
        response_text: str | None = None,
        usage_metadata=None,
    ) -> None:
        self.calls = []
        self.response_text = _DEFAULT_RESPONSE_TEXT if response_text is None else response_text
        self.usage_metadata = usage_metadata

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.response_text, self.usage_metadata)


class _FakeResponse:
    def __init__(self, text: str | None, usage_metadata=None) -> None:
        self.text = text
        if usage_metadata is not _MISSING_USAGE_METADATA:
            self.usage_metadata = usage_metadata


class _FakeClient:
    def __init__(
        self,
        response_text: str | None = None,
        usage_metadata=None,
    ) -> None:
        self.files = _FakeFilesClient()
        self.models = _FakeModelsClient(response_text, usage_metadata)


class _FakeProvider:
    def __init__(
        self,
        response_text: str | None = None,
        usage_metadata=None,
    ) -> None:
        self._client = _FakeClient(response_text, usage_metadata)
        self.model = "gemini-test"


class _UsageMetadata:
    def __init__(
        self,
        prompt_token_count: int | None = None,
        candidates_token_count: int | None = None,
        total_token_count: int | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count


_MISSING_USAGE_METADATA = object()


def _provider_with_fake_client(
    response_text: str | None = None,
    usage_metadata=None,
) -> GeminiExtractionProvider:
    provider = GeminiExtractionProvider.__new__(GeminiExtractionProvider)
    provider._provider = _FakeProvider(response_text, usage_metadata)
    return provider


def _minimal_pdf_bytes(page_count: int) -> bytes:
    page_objects = "\n".join(
        f"{index + 3} 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj"
        for index in range(page_count)
    )
    return (
        "%PDF-1.7\n"
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Count 3 >>\nendobj\n"
        f"{page_objects}\n"
        "%%EOF\n"
    ).encode()


def test_extract_from_files_uploads_all_files_in_one_model_call(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    png = tmp_path / "foto.png"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    png.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    provider = _provider_with_fake_client()

    result = provider.extract_from_files(
        [pdf, png],
        project_id="project-1",
        requirement_id="requirement-1",
    )

    fake_client = provider._provider._client
    assert len(fake_client.files.uploads) == 2
    assert len(fake_client.models.calls) == 1
    call = fake_client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["config"].response_schema is None
    assert len(call["contents"]) == 3
    assert "AVAILABLE SOURCES" in call["contents"][0].text
    assert "source-1 | planos.pdf | application/pdf" in call["contents"][0].text
    assert "source-2 | foto.png | image/png" in call["contents"][0].text
    assert result.requirement.project_id == "project-1"
    assert result.requirement.requirement_id == "requirement-1"
    assert [source.id for source in result.sources] == ["source-1", "source-2"]
    assert [source.media_type for source in result.sources] == ["application/pdf", "image/png"]
    assert result.extraction_metadata.source_count == 2


def test_extract_from_files_populates_pdf_page_count_when_available(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(_minimal_pdf_bytes(page_count=3))
    provider = _provider_with_fake_client()

    result = provider.extract_from_files([pdf])

    assert result.sources[0].media_type == "application/pdf"
    assert result.sources[0].page_count == 3


def test_extract_from_files_preserves_three_sources_order_and_evidence_source(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "planos.pdf"
    xlsx = tmp_path / "cuadro.xlsx"
    png = tmp_path / "boceto.png"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    with zipfile.ZipFile(xlsx, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("xl/workbook.xml", "<workbook></workbook>")
    png.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    provider = _provider_with_fake_client(
        '{"elements": [{"id": "v-01", "evidence_items": ['
        '{"source_id": "source-2", "text": "Evidencia desde foto"}'
        "]}]}"
    )

    result = provider.extract_from_files([pdf, xlsx, png])
    prompt = provider._provider._client.models.calls[0]["contents"][0].text

    assert [source.id for source in result.sources] == ["source-1", "source-2", "source-3"]
    assert [source.file_name for source in result.sources] == [
        "planos.pdf",
        "cuadro.xlsx",
        "boceto.png",
    ]
    assert [source.source_type for source in result.sources] == [
        "document",
        "spreadsheet",
        "image",
    ]
    assert "source-1 | planos.pdf | application/pdf" in prompt
    assert (
        "source-2 | cuadro.xlsx | "
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ) in prompt
    assert "source-3 | boceto.png | image/png" in prompt
    assert result.evidence[0].source_id == "source-2"
    assert result.elements[0].evidence_ids == ["evidence-1"]


def test_extract_from_text_uses_json_text_without_response_schema() -> None:
    provider = _provider_with_fake_client('{"elements": [{"reference": "T1"}]}')

    result = provider.extract_from_text("Elemento T1")

    call = provider._provider._client.models.calls[0]
    assert "Elemento T1" in call["contents"]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is None
    assert result.elements[0].reference is not None
    assert result.elements[0].reference.value == "T1"


def test_extract_from_text_propagates_complete_usage_metadata() -> None:
    provider = _provider_with_fake_client(
        '{"elements": [{"reference": "T1"}]}',
        usage_metadata=_UsageMetadata(
            prompt_token_count=100,
            candidates_token_count=25,
            total_token_count=125,
        ),
    )

    result = provider.extract_from_text("Elemento T1")

    assert result.extraction_metadata.token_usage is not None
    assert result.extraction_metadata.token_usage.input_tokens == 100
    assert result.extraction_metadata.token_usage.output_tokens == 25
    assert result.extraction_metadata.token_usage.total_tokens == 125


def test_extract_from_text_leaves_token_usage_none_when_usage_metadata_absent() -> None:
    provider = _provider_with_fake_client(
        '{"elements": [{"reference": "T1"}]}',
        usage_metadata=_MISSING_USAGE_METADATA,
    )

    result = provider.extract_from_text("Elemento T1")

    assert result.extraction_metadata.token_usage is None


def test_extract_from_text_propagates_partial_usage_metadata() -> None:
    provider = _provider_with_fake_client(
        '{"elements": [{"reference": "T1"}]}',
        usage_metadata=_UsageMetadata(
            prompt_token_count=100,
            total_token_count=130,
        ),
    )

    result = provider.extract_from_text("Elemento T1")

    assert result.extraction_metadata.token_usage is not None
    assert result.extraction_metadata.token_usage.input_tokens == 100
    assert result.extraction_metadata.token_usage.output_tokens is None
    assert result.extraction_metadata.token_usage.total_tokens == 130


def test_extract_from_text_calls_mapper_after_local_json_parse(monkeypatch) -> None:
    provider = _provider_with_fake_client('{"elements": [{"reference": "T2"}]}')
    calls = []
    mapped_result = RequirementExtraction(
        requirement=Requirement(),
        extraction_metadata=ExtractionMetadata(),
    )

    def fake_mapper(
        gemini_extraction,
        *,
        model_provider,
        model,
        default_source_id="text-input",
        allowed_source_ids=None,
    ):
        calls.append((gemini_extraction, model_provider, model))
        return mapped_result

    monkeypatch.setattr(
        provider_module,
        "map_gemini_extraction_to_requirement_extraction",
        fake_mapper,
    )

    result = provider.extract_from_text("Elemento T2")

    assert result is mapped_result
    assert len(calls) == 1
    assert calls[0][0].elements[0].reference == "T2"
    assert calls[0][1] == "google"
    assert calls[0][2] == "gemini-test"


def test_extract_from_files_parses_json_text_and_maps_to_requirement(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client(
        '{"requirement": {"project_name": "Demo"}, "elements": [{"reference": "V7"}]}'
    )

    result = provider.extract_from_files([pdf])

    assert result.requirement.project_name is not None
    assert result.requirement.project_name.value == "Demo"
    assert result.elements[0].reference is not None
    assert result.elements[0].reference.value == "V7"


def test_extract_from_files_rejects_invalid_json_response(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client('{"elements": "invalid"}')

    with pytest.raises(ValueError, match="JSON invalido"):
        provider.extract_from_files([pdf])


def test_extract_from_files_rejects_empty_response_text(tmp_path: Path) -> None:
    pdf = tmp_path / "planos.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")
    provider = _provider_with_fake_client("")

    with pytest.raises(ValueError, match="no devolvio texto JSON"):
        provider.extract_from_files([pdf])


def test_extract_from_files_uses_ascii_temp_copy_for_unicode_names(tmp_path: Path) -> None:
    image = tmp_path / "fachada_nin\u0303o.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    provider = _provider_with_fake_client()

    provider.extract_from_files([image])

    upload_path, upload_config = provider._provider._client.files.uploads[0]
    upload_path.name.encode("ascii")
    assert upload_path.name == "source-1.png"
    assert upload_config.display_name == image.name
    assert upload_config.mime_type == "image/png"


def test_extract_from_files_rejects_missing_files(tmp_path: Path) -> None:
    provider = _provider_with_fake_client()

    with pytest.raises(FileNotFoundError):
        provider.extract_from_files([tmp_path / "missing.pdf"])


def test_extract_from_files_rejects_unsupported_mime(tmp_path: Path) -> None:
    text_file = tmp_path / "not-supported.pdf"
    text_file.write_bytes(b"not really a pdf")
    provider = _provider_with_fake_client()

    with pytest.raises(ValueError, match="MIME type soportado"):
        provider.extract_from_files([text_file])
