from pathlib import Path
from types import SimpleNamespace

import pytest

import app.providers.gemini_extraction as provider_module
import app.services.localized_extraction_pipeline as pipeline_module
from app.models.gemini_enrichment import GeminiElementEnrichment, GeminiEnrichmentResult
from app.models.requirement import ExtractionMetadata, Requirement, TokenUsage
from app.models.requirement_extraction import RequirementExtraction
from app.models.source_preparation_v1 import PreparedPage, PreparedSource
from app.providers.gemini_extraction import GeminiExtractionProvider
from app.services.localized_technical_reader import LocalizedTechnicalReaderError


class _FakeSettings:
    def __init__(self, pipeline: str) -> None:
        self.ai2_extraction_pipeline = pipeline
        self.gemini_api_key = "test-key"
        self.gemini_enrichment_batch_size = 8


class _FakeGeminiProvider:
    model = "gemini-test"


class _FakeLocalizationClient:
    model = "gemini-test"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def generate_image_frames(self, prompt, images):
        self.calls += 1
        assert prompt == "localization-prompt"
        assert images == [("img0", "image/png", b"image")]
        return {
            "text": "{}",
            "finish_reason": "STOP",
            "usage": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        }

    def close(self) -> None:
        self.closed = True


class _FakeTechnicalClient:
    model = "gemini-test"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def generate_localized_technical(self, prompt, images):
        self.calls += 1
        assert prompt == "technical-prompt"
        assert images == [("page_context", "image/png", b"context")]
        return {
            "text": "{}",
            "finish_reason": "STOP",
            "usage": {
                "prompt_token_count": 20,
                "candidates_token_count": 8,
                "total_token_count": 28,
            },
        }

    def close(self) -> None:
        self.closed = True


def _provider(pipeline: str) -> GeminiExtractionProvider:
    provider = GeminiExtractionProvider.__new__(GeminiExtractionProvider)
    provider._settings = _FakeSettings(pipeline)
    provider._provider = _FakeGeminiProvider()
    return provider


def _extraction() -> RequirementExtraction:
    return RequirementExtraction(
        requirement=Requirement(),
        elements=[],
        extraction_metadata=ExtractionMetadata(model_provider="test", model="fake"),
    )


def _prepared_source() -> PreparedSource:
    return PreparedSource(
        schema_version=1,
        document_id="doc-" + "1" * 24,
        content_sha256="a" * 64,
        logical_name="planos.pdf",
        input_index=1,
        size_bytes=10,
        media_type="application/pdf",
        status="PREPARED",
        page_count=1,
        pages=[
            PreparedPage(
                view_index=1,
                page_number=1,
                pdf_page_label=None,
                status="PREPARED",
                preview_file="page-0001.png",
                observations_file="page-0001.json",
                width_px=100,
                height_px=100,
            )
        ],
    )


def _install_pipeline_fakes(monkeypatch, elements: list[GeminiElementEnrichment]) -> dict:
    calls = {"prepared_files": [], "mapped_elements": []}

    def fake_prepare_source(path, output_root, *, logical_name, input_index):
        calls["prepared_files"].append((Path(path), Path(output_root), logical_name, input_index))
        return _prepared_source()

    def fake_write_json(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("{}", encoding="utf-8")

    def fake_build_plan(path, *, tiles):
        assert Path(path).name == "source_preparation_report.json"
        assert tiles == "auto"
        return {
            "jobs": [{"job_id": "job-1"}],
            "prompt_sha256": "localization",
            "response_schema_sha256": "schema",
        }

    def fake_localization_request(plan, job):
        assert job["job_id"] == "job-1"
        return "localization-prompt", [("img0", "image/png", b"image")], {}

    def fake_save_localization(plan, job, text, output_dir):
        assert text == "{}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "localization.json").write_text("{}", encoding="utf-8")
        return {"status": "PROPOSALS_RECORDED_UNVERIFIED"}

    def fake_build_technical_plan(report_path, *, preparation_report_path):
        assert Path(report_path).name == "localization.json"
        assert Path(preparation_report_path).name == "source_preparation_report.json"
        return {
            "candidates": [
                {
                    "technical_candidate_id": "doc|1|e1",
                    "candidate_id": "e1",
                }
            ]
        }

    def fake_select_candidate_jobs(plan):
        return list(plan["candidates"])

    def fake_technical_request(plan, job):
        return "technical-prompt", [("page_context", "image/png", b"context")], {}

    def fake_record_technical(plan, job, response, output_dir):
        assert response["text"] == "{}"
        return SimpleNamespace(
            enrichment=GeminiEnrichmentResult(elements=list(elements)),
            warnings=["technical warning"],
        )

    def fake_mapper(gemini_extraction, **kwargs):
        calls["mapped_elements"].append(len(gemini_extraction.elements))
        return _extraction()

    monkeypatch.setattr(pipeline_module, "prepare_source", fake_prepare_source)
    monkeypatch.setattr(pipeline_module, "write_json", fake_write_json)
    monkeypatch.setattr(pipeline_module, "build_plan", fake_build_plan)
    monkeypatch.setattr(pipeline_module, "image_frame_request_for_job", fake_localization_request)
    monkeypatch.setattr(
        pipeline_module,
        "save_image_frame_proposal_artifacts",
        fake_save_localization,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_localized_technical_plan",
        fake_build_technical_plan,
    )
    monkeypatch.setattr(pipeline_module, "select_candidate_jobs", fake_select_candidate_jobs)
    monkeypatch.setattr(pipeline_module, "request_for_candidate", fake_technical_request)
    monkeypatch.setattr(
        pipeline_module,
        "record_localized_technical_response",
        fake_record_technical,
    )
    monkeypatch.setattr(
        pipeline_module,
        "map_gemini_extraction_to_requirement_extraction",
        fake_mapper,
    )
    return calls


def test_legacy_flag_preserves_current_path(monkeypatch, tmp_path: Path) -> None:
    provider = _provider("legacy")
    expected = _extraction()
    localized_called = False

    def fake_legacy(self, files, **kwargs):
        assert files == [tmp_path / "planos.pdf"]
        assert kwargs["project_id"] == "p1"
        return expected

    def fake_localized(*args, **kwargs):
        nonlocal localized_called
        localized_called = True
        raise AssertionError("localized_v1 must not run for legacy flag")

    monkeypatch.setattr(
        GeminiExtractionProvider,
        "_extract_with_discovery_legacy_from_files",
        fake_legacy,
    )
    monkeypatch.setattr(
        provider_module,
        "extract_requirement_with_localized_v1",
        fake_localized,
    )

    result = provider.extract_with_discovery_from_files(
        [tmp_path / "planos.pdf"],
        project_id="p1",
    )

    assert result is expected
    assert localized_called is False


def test_localized_v1_flag_selects_new_pipeline(monkeypatch, tmp_path: Path) -> None:
    provider = _provider("localized_v1")
    expected = _extraction()
    call = {}

    def fake_localized(files, **kwargs):
        call["files"] = files
        call["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(provider_module, "extract_requirement_with_localized_v1", fake_localized)

    result = provider.extract_with_discovery_from_files(
        [tmp_path / "planos.pdf"],
        project_id="p1",
        requirement_id="r1",
    )

    assert result is expected
    assert call["files"] == [tmp_path / "planos.pdf"]
    assert call["kwargs"]["project_id"] == "p1"
    assert call["kwargs"]["requirement_id"] == "r1"
    assert call["kwargs"]["api_key"] == "test-key"
    assert call["kwargs"]["model"] == "gemini-test"


def test_localized_pipeline_prepares_files_maps_results_and_returns_requirement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "planos.pdf"
    input_file.write_bytes(b"%PDF-1.7")
    element = GeminiElementEnrichment(
        temporary_id="doc|1|e1",
        reference="V-01",
        quantity=1,
    )
    calls = _install_pipeline_fakes(monkeypatch, [element])

    result = pipeline_module.extract_requirement_with_localized_v1(
        [input_file],
        project_id="p1",
        requirement_id="r1",
        api_key="key",
        model="gemini-test",
        localization_client=_FakeLocalizationClient(),
        technical_client=_FakeTechnicalClient(),
    )

    assert isinstance(result, RequirementExtraction)
    assert calls["prepared_files"][0][0] == input_file
    assert calls["prepared_files"][0][2] == "planos.pdf"
    assert calls["mapped_elements"] == [1]
    assert result.requirement.project_id == "p1"
    assert result.requirement.requirement_id == "r1"
    assert result.sources[0].id == "source-1"
    assert result.extraction_metadata.pipeline_version == "localized-v1"
    assert result.extraction_metadata.token_usage == TokenUsage(
        input_tokens=30,
        output_tokens=13,
        total_tokens=43,
    )


def test_localized_pipeline_partial_technical_failure_keeps_valid_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "planos.pdf"
    input_file.write_bytes(b"%PDF-1.7")
    element = GeminiElementEnrichment(temporary_id="doc|1|e2", reference="V-02")
    _install_pipeline_fakes(monkeypatch, [element])

    def two_jobs(plan):
        return [
            {"technical_candidate_id": "doc|1|e1", "candidate_id": "e1"},
            {"technical_candidate_id": "doc|1|e2", "candidate_id": "e2"},
        ]

    def partial_record(plan, job, response, output_dir):
        if job["candidate_id"] == "e1":
            raise LocalizedTechnicalReaderError("SYNTHETIC_PARTIAL_FAILURE")
        return SimpleNamespace(
            enrichment=GeminiEnrichmentResult(elements=[element]),
            warnings=[],
        )

    monkeypatch.setattr(pipeline_module, "select_candidate_jobs", two_jobs)
    monkeypatch.setattr(pipeline_module, "record_localized_technical_response", partial_record)

    result = pipeline_module.extract_requirement_with_localized_v1(
        [input_file],
        project_id=None,
        requirement_id=None,
        api_key="key",
        model="gemini-test",
        localization_client=_FakeLocalizationClient(),
        technical_client=_FakeTechnicalClient(),
    )

    assert isinstance(result, RequirementExtraction)
    assert result.warnings
    assert any("technical read failed" in warning.message for warning in result.warnings)


def test_localized_pipeline_requires_useful_results(monkeypatch, tmp_path: Path) -> None:
    input_file = tmp_path / "planos.pdf"
    input_file.write_bytes(b"%PDF-1.7")
    _install_pipeline_fakes(monkeypatch, [])

    with pytest.raises(
        pipeline_module.LocalizedExtractionPipelineError,
        match="LOCALIZED_V1_NO_USEFUL_EXTRACTION",
    ):
        pipeline_module.extract_requirement_with_localized_v1(
            [input_file],
            project_id=None,
            requirement_id=None,
            api_key="key",
            model="gemini-test",
            localization_client=_FakeLocalizationClient(),
            technical_client=_FakeTechnicalClient(),
        )


def test_product_localized_integration_does_not_reference_corpus_or_manual_tests() -> None:
    product_files = [
        Path("app/providers/gemini_extraction.py"),
        Path("app/services/localized_extraction_pipeline.py"),
    ]

    combined = "\n".join(path.read_text(encoding="utf-8") for path in product_files)

    assert "manual_tests" not in combined
    assert "corpus_v1" not in combined
