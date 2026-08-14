from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.requirements import get_gemini_extraction_provider
from app.main import app
from app.models.common import ExtractionStatus, TraceableValue
from app.models.element import Element
from app.models.requirement import ExtractionMetadata, Requirement
from app.models.requirement_extraction import RequirementExtraction


class FakeExtractionProvider:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = []
        self.error = error

    def extract_from_files(
        self,
        files: list[Path],
        project_id: str | None = None,
        requirement_id: str | None = None,
    ) -> RequirementExtraction:
        self.calls.append(
            {
                "files": list(files),
                "project_id": project_id,
                "requirement_id": requirement_id,
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
    assert "/requirements/extract" in response.json()["paths"]


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
    assert response.json()["extraction_metadata"]["source_count"] == 1


def test_extract_multi_file_passes_all_files_and_ids_to_provider() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            data={"project_id": "project-1", "requirement_id": "requirement-1"},
            files=[
                ("files", ("v1.pdf", b"%PDF-1.7\ncontent", "application/pdf")),
                ("files", ("foto.png", b"\x89PNG\r\n\x1a\ncontent", "image/png")),
            ],
        )

    assert response.status_code == 200
    assert len(provider.calls[0]["files"]) == 2
    assert provider.calls[0]["project_id"] == "project-1"
    assert provider.calls[0]["requirement_id"] == "requirement-1"
    assert response.json()["requirement"]["project_id"] == "project-1"
    assert response.json()["requirement"]["requirement_id"] == "requirement-1"


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


def test_extract_unsupported_declared_content_type_returns_415() -> None:
    provider = FakeExtractionProvider()
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("notes.txt", b"hello", "text/plain"))],
        )

    assert response.status_code == 415
    assert provider.calls == []


def test_extract_provider_unsupported_file_error_returns_415() -> None:
    provider = FakeExtractionProvider(
        error=ValueError("No se pudo detectar un MIME type soportado")
    )
    with _client_with_provider(provider) as client:
        response = client.post(
            "/requirements/extract",
            files=[("files", ("bad.pdf", b"not a pdf", "application/pdf"))],
        )

    assert response.status_code == 415
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
