import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.requirements import get_gemini_extraction_provider
from app.main import app
from app.models.common import ExtractionStatus, NormalizedValue, TraceableValue
from app.models.configuration import Configuration
from app.models.element import Element
from app.models.evidence import Source
from app.models.geometry import Geometry
from app.models.requirement import ExtractionMetadata, Requirement
from app.models.requirement_extraction import RequirementExtraction


class FakeExtractionProvider:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = []
        self.error = error

    def extract_with_discovery_from_files(
        self,
        files: list[Path],
        project_id: str | None = None,
        requirement_id: str | None = None,
        batch_size: int | None = None,
        debug_capture=None,
    ) -> RequirementExtraction:
        self.calls.append(
            {
                "files": list(files),
                "file_names": [path.name for path in files],
                "project_id": project_id,
                "requirement_id": requirement_id,
                "batch_size": batch_size,
                "debug_capture": debug_capture,
                "existence_during_call": [path.exists() for path in files],
            }
        )
        if self.error:
            raise self.error

        return RequirementExtraction(
            requirement=Requirement(
                project_id=project_id,
                requirement_id=requirement_id,
            ),
            elements=[
                Element(
                    id="element-1",
                    reference=TraceableValue(
                        value="V1",
                        status=ExtractionStatus.EXPLICIT,
                    ),
                    functional_type=NormalizedValue(
                        normalized="SLIDING_DOOR",
                        raw="puerta corrediza",
                        status=ExtractionStatus.EXPLICIT,
                    ),
                    configuration=Configuration(
                        raw_description="puerta corrediza OXXO",
                        modulation="OXXO",
                        status=ExtractionStatus.EXPLICIT,
                    ),
                    geometry=Geometry(
                        normalized_type="L_SHAPE",
                        raw_type="estructura en L",
                        description="estructura en L",
                        status=ExtractionStatus.EXPLICIT,
                    ),
                )
            ],
            extraction_metadata=ExtractionMetadata(
                model_provider="test",
                model="fake",
                source_count=len(files),
                element_count=1,
            ),
        )


def _client_with_provider(provider: FakeExtractionProvider) -> TestClient:
    app.dependency_overrides[get_gemini_extraction_provider] = lambda: provider
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def test_health_still_responds() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_exposes_requirements_extract() -> None:
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/requirements/extract" in schema["paths"]
    request_body = schema["paths"]["/requirements/extract"]["post"]["requestBody"]
    assert "multipart/form-data" in request_body["content"]
    multipart_schema = _resolve_schema_ref(
        schema,
        request_body["content"]["multipart/form-data"]["schema"],
    )
    files_schema = multipart_schema["properties"]["files"]
    assert files_schema["type"] == "array"
    assert files_schema["items"] == {
        "type": "string",
        "format": "binary",
    }
    assert files_schema["description"] == "PDF/XLSX/JPG/PNG files"
    assert "project_id" in multipart_schema["properties"]
    assert "requirement_id" in multipart_schema["properties"]
    assert "files" not in multipart_schema.get("required", [])


def test_extract_without_files_returns_400() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post("/requirements/extract")

    assert response.status_code == 400
    assert provider.calls == []


def test_extract_single_file_reaches_provider_and_serializes_result() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf"))],
        )

    assert response.status_code == 200
    assert len(provider.calls) == 1
    assert provider.calls[0]["existence_during_call"] == [True]
    assert response.json()["elements"][0]["reference"]["value"] == "V1"
    assert response.json()["elements"][0]["functional_type"]["normalized"] == "SLIDING_DOOR"
    assert response.json()["elements"][0]["configuration"]["modulation"] == "OXXO"
    assert response.json()["elements"][0]["geometry"]["normalized_type"] == "L_SHAPE"
    assert response.json()["extraction_metadata"]["source_count"] == 1


def test_extract_multi_file_passes_all_files_and_ids_to_full_pipeline() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            data={"project_id": "project-1", "requirement_id": "requirement-1"},
            files=[
                ("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf")),
                (
                    "files",
                    (
                        "cuadro.xlsx",
                        _xlsx_bytes(),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ),
                ),
                ("files", ("foto.jpg", b"\xff\xd8\xff\xe0content", "image/jpeg")),
                ("files", ("foto.png", b"\x89PNG\r\n\x1a\ncontent", "image/png")),
            ],
        )

    assert response.status_code == 200
    assert len(provider.calls) == 1
    assert len(provider.calls[0]["files"]) == 4
    assert provider.calls[0]["project_id"] == "project-1"
    assert provider.calls[0]["requirement_id"] == "requirement-1"
    assert [path.name[:4] for path in provider.calls[0]["files"]] == [
        "001_",
        "002_",
        "003_",
        "004_",
    ]
    assert response.json()["requirement"]["project_id"] == "project-1"
    assert response.json()["requirement"]["requirement_id"] == "requirement-1"


def test_extract_accepts_xlsx_only() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[
                (
                    "files",
                    (
                        "cuadro.xlsx",
                        _xlsx_bytes(),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ),
                )
            ],
        )

    assert response.status_code == 200
    assert len(provider.calls) == 1
    assert provider.calls[0]["file_names"][0].endswith("_cuadro.xlsx")


def test_extract_removes_temporaries_after_success() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf"))],
        )

    assert response.status_code == 200
    assert provider.calls[0]["files"][0].exists() is False


def test_extract_removes_temporaries_after_error() -> None:
    provider = FakeExtractionProvider(error=RuntimeError("provider exploded"))
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf"))],
        )

    assert response.status_code == 502
    assert provider.calls[0]["files"][0].exists() is False


def test_extract_unsupported_declared_content_type_returns_400() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("notes.txt", b"hello", "text/plain"))],
        )

    assert response.status_code == 400
    assert provider.calls == []


def test_extract_provider_unsupported_file_error_returns_400() -> None:
    provider = FakeExtractionProvider(
        error=ValueError("No se pudo detectar un MIME type soportado")
    )
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("bad.pdf", b"not a pdf", "application/pdf"))],
        )

    assert response.status_code == 400
    assert provider.calls[0]["files"][0].exists() is False


def test_extract_empty_file_returns_400() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("empty.pdf", b"", "application/pdf"))],
        )

    assert response.status_code == 400
    assert provider.calls == []


def test_provider_error_does_not_leak_sensitive_details() -> None:
    provider = FakeExtractionProvider(error=RuntimeError("API_KEY=secret-token"))
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf"))],
        )

    assert response.status_code == 502
    assert "secret-token" not in response.text
    assert response.json()["detail"] == "Error del proveedor de IA."


def test_invalid_ai_response_error_returns_502() -> None:
    provider = FakeExtractionProvider(error=ValueError("Gemini devolvio JSON invalido"))
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf"))],
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Respuesta de IA invalida."


def test_extract_same_filename_keeps_two_distinct_temporaries_in_order() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[
                ("files", ("plano.pdf", b"%PDF-1.7\none", "application/pdf")),
                ("files", ("plano.pdf", b"%PDF-1.7\ntwo", "application/pdf")),
            ],
        )

    assert response.status_code == 200
    assert provider.calls[0]["file_names"] == ["001_plano.pdf", "002_plano.pdf"]
    assert provider.calls[0]["files"][0] != provider.calls[0]["files"][1]


def test_extract_path_traversal_filename_stays_inside_temp_dir() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("..\\secret.pdf", b"%PDF-1.7\ncontent", "application/pdf"))],
        )

    assert response.status_code == 200
    temp_path = provider.calls[0]["files"][0]
    assert temp_path.name == "001_secret.pdf"
    assert temp_path.parent.name.startswith("ai2-requirement-upload-")


def test_extract_unexpected_error_returns_500_without_sensitive_details() -> None:
    provider = FakeExtractionProvider(error=Exception("file_uri=gemini://secret"))
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf"))],
        )

    assert response.status_code == 500
    assert "gemini://secret" not in response.text
    assert response.json()["detail"] == "Error interno inesperado."


def test_extract_serializes_requirement_with_multiple_sources() -> None:
    class SourceProvider(FakeExtractionProvider):
        def extract_with_discovery_from_files(
            self,
            files: list[Path],
            project_id: str | None = None,
            requirement_id: str | None = None,
            batch_size: int | None = None,
            debug_capture=None,
        ) -> RequirementExtraction:
            result = super().extract_with_discovery_from_files(
                files,
                project_id=project_id,
                requirement_id=requirement_id,
                batch_size=batch_size,
                debug_capture=debug_capture,
            )
            result.sources = [
                Source(
                    id="source-1",
                    file_name="a.pdf",
                    media_type="application/pdf",
                    source_type="document",
                ),
                Source(
                    id="source-2",
                    file_name="b.png",
                    media_type="image/png",
                    source_type="image",
                ),
            ]
            return result

    provider = SourceProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[
                ("files", ("a.pdf", b"%PDF-1.7\ncontent", "application/pdf")),
                ("files", ("b.png", b"\x89PNG\r\n\x1a\ncontent", "image/png")),
            ],
        )

    assert response.status_code == 200
    assert [source["id"] for source in response.json()["sources"]] == ["source-1", "source-2"]


def _xlsx_bytes() -> bytes:
    import io

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("xl/workbook.xml", "<workbook></workbook>")
    return output.getvalue()


def _resolve_schema_ref(openapi_schema: dict, schema: dict) -> dict:
    ref = schema.get("$ref")
    if ref is None:
        return schema

    _, section, subsection, name = ref.split("/")
    return openapi_schema[section][subsection][name]
