import logging
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from google.genai import types

from app.core.settings import get_settings
from app.models.evidence import Source
from app.models.gemini_discovery import GeminiDiscoveryResult
from app.models.gemini_enrichment import GeminiEnrichmentResult
from app.models.gemini_extraction import GeminiExtraction
from app.models.gemini_scope import GeminiScopeResult
from app.models.requirement import TokenUsage
from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini import GeminiProvider
from app.services.extraction_prompt import (
    build_file_discovery_prompt,
    build_file_enrichment_prompt,
    build_file_extraction_prompt,
    build_file_scope_prompt,
    build_text_extraction_prompt,
)
from app.services.gemini_enrichment_pipeline import (
    build_discovery_batches,
    enrichment_to_gemini_extraction,
    merge_enrichment_batches,
    sum_token_usage,
)
from app.services.gemini_extraction_mapper import map_gemini_extraction_to_requirement_extraction
from app.services.gemini_scope_pipeline import (
    merge_scope_with_discovery,
    scope_lookup,
    select_discoveries_for_enrichment,
)

logger = logging.getLogger(__name__)

SUPPORTED_FILE_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class GeminiExtractionProvider:
    def __init__(self) -> None:
        self._provider = GeminiProvider()
        self._settings = get_settings()

    def extract_from_text(
        self,
        prompt: str,
        debug_capture: GeminiExtractionDebugCapture | None = None,
    ) -> RequirementExtraction:
        response = self._provider._client.models.generate_content(
            model=self._provider.model,
            contents=build_text_extraction_prompt(prompt),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        gemini_extraction = _parse_gemini_extraction_response(response, debug_capture)
        extraction = map_gemini_extraction_to_requirement_extraction(
            gemini_extraction,
            model_provider="google",
            model=self._provider.model,
        )
        _apply_usage_metadata(extraction, response)
        return extraction

    def discover_elements_from_files(
        self,
        files: list[Path],
        debug_capture: GeminiDiscoveryDebugCapture | None = None,
    ) -> GeminiDiscoveryResult:
        if not files:
            raise ValueError("Debes proporcionar al menos un archivo para discovery.")

        paths = [Path(file) for file in files]
        file_specs = [_prepare_file_spec(path) for path in paths]
        prompt = build_file_discovery_prompt(
            paths,
            media_types=_media_types(file_specs),
        )

        contents = _build_multifile_contents(prompt, file_specs)
        response = self._provider._client.models.generate_content(
            model=self._provider.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        discovery_result = _parse_gemini_discovery_response(response, debug_capture)
        if debug_capture is not None:
            debug_capture.model = self._provider.model
            debug_capture.token_usage = _extract_token_usage(response)
        return discovery_result

    def enrich_discoveries_from_files(
        self,
        files: list[Path],
        discovery: GeminiDiscoveryResult,
        batch_size: int | None = None,
        debug_capture: GeminiEnrichmentDebugCapture | None = None,
    ) -> GeminiEnrichmentResult:
        if not files:
            raise ValueError("Debes proporcionar al menos un archivo para enrichment.")

        with _RequirementFileContext(self, files) as context:
            return self._enrich_discovery_with_context(
                context,
                discovery,
                batch_size=batch_size,
                debug_capture=debug_capture,
            )

    def classify_scope_from_files(
        self,
        files: list[Path],
        discovery: GeminiDiscoveryResult,
        debug_capture: GeminiScopeDebugCapture | None = None,
    ) -> GeminiScopeResult:
        if not files:
            raise ValueError("Debes proporcionar al menos un archivo para scope.")

        with _RequirementFileContext(self, files) as context:
            response = self._generate_json(
                _build_multifile_contents(
                    build_file_scope_prompt(
                        discovery.elements,
                        context.paths,
                        media_types=_media_types(context.file_specs),
                    ),
                    context.file_specs,
                )
            )

        raw_scope = _parse_gemini_scope_response(response, debug_capture)
        scope = merge_scope_with_discovery(discovery, raw_scope)
        if debug_capture is not None:
            debug_capture.scope_result = scope
            debug_capture.model = self._provider.model
            debug_capture.token_usage = _extract_token_usage(response)
        return scope

    def extract_with_discovery_from_files(
        self,
        files: list[Path],
        project_id: str | None = None,
        requirement_id: str | None = None,
        batch_size: int | None = None,
        debug_capture: GeminiFullPipelineDebugCapture | None = None,
    ) -> RequirementExtraction:
        if not files:
            raise ValueError("Debes proporcionar al menos un archivo para extraer.")

        total_started = time.perf_counter()
        file_load_started = time.perf_counter()
        with _RequirementFileContext(self, files) as context:
            _log_perf(
                requirement_id,
                "FILE_LOAD",
                _elapsed_ms(file_load_started),
                file_count=len(context.paths),
                file_names=",".join(path.name for path in context.paths),
            )
            discovery_debug = GeminiDiscoveryDebugCapture()
            discovery_started = time.perf_counter()
            discovery_response = self._generate_json(
                _build_multifile_contents(
                    build_file_discovery_prompt(
                        context.paths,
                        media_types=_media_types(context.file_specs),
                    ),
                    context.file_specs,
                )
            )
            discovery = _parse_gemini_discovery_response(discovery_response, discovery_debug)
            _log_perf(
                requirement_id,
                "LLM_STRUCTURED_EXTRACTION",
                _elapsed_ms(discovery_started),
                substage="discovery",
                element_count=len(discovery.elements),
            )
            discovery_debug.model = self._provider.model
            discovery_debug.token_usage = _extract_token_usage(discovery_response)

            scope_debug = GeminiScopeDebugCapture()
            scope_started = time.perf_counter()
            scope_response = self._generate_json(
                _build_multifile_contents(
                    build_file_scope_prompt(
                        discovery.elements,
                        context.paths,
                        media_types=_media_types(context.file_specs),
                    ),
                    context.file_specs,
                )
            )
            raw_scope = _parse_gemini_scope_response(scope_response, scope_debug)
            scope = merge_scope_with_discovery(discovery, raw_scope)
            _log_perf(
                requirement_id,
                "LLM_STRUCTURED_EXTRACTION",
                _elapsed_ms(scope_started),
                substage="scope",
                element_count=len(discovery.elements),
            )
            scope_debug.scope_result = scope
            scope_debug.model = self._provider.model
            scope_debug.token_usage = _extract_token_usage(scope_response)
            scoped_discovery = select_discoveries_for_enrichment(discovery, scope)

            enrichment_debug = GeminiEnrichmentDebugCapture()
            enrichment_started = time.perf_counter()
            enrichment = self._enrich_discovery_with_context(
                context,
                scoped_discovery,
                batch_size=batch_size,
                debug_capture=enrichment_debug,
                scope=scope,
            )
            _log_perf(
                requirement_id,
                "LLM_STRUCTURED_EXTRACTION",
                _elapsed_ms(enrichment_started),
                substage="enrichment",
                batch_count=len(enrichment_debug.batch_results or []),
            )
            postprocess_started = time.perf_counter()
            gemini_extraction = enrichment_to_gemini_extraction(scoped_discovery, enrichment)
            extraction = map_gemini_extraction_to_requirement_extraction(
                gemini_extraction,
                model_provider="google",
                model=self._provider.model,
                default_source_id=_default_evidence_source_id(context.file_specs),
                allowed_source_ids=_source_ids(context.file_specs),
            )
            _log_perf(
                requirement_id,
                "POSTPROCESS",
                _elapsed_ms(postprocess_started),
                element_count=len(extraction.elements),
            )
            extraction.requirement.project_id = project_id
            extraction.requirement.requirement_id = requirement_id
            extraction.sources = _build_sources(context.file_specs)
            extraction.warnings.extend(_warnings_from_messages(enrichment.warnings))
            extraction.warnings.extend(_warnings_from_messages(scope.warnings))
            extraction.extraction_metadata.source_count = len(context.paths)
            extraction.extraction_metadata.token_usage = sum_token_usage(
                [discovery_debug.token_usage, scope_debug.token_usage, enrichment.usage]
            )

            if debug_capture is not None:
                debug_capture.discovery = discovery
                debug_capture.scope = scope
                debug_capture.enrichment = enrichment
                debug_capture.gemini_extraction = gemini_extraction
                debug_capture.discovery_debug = discovery_debug
                debug_capture.scope_debug = scope_debug
                debug_capture.enrichment_debug = enrichment_debug
                debug_capture.model = self._provider.model
                debug_capture.batch_size = enrichment_debug.batch_size
                debug_capture.batch_count = len(enrichment_debug.batch_results or [])

            _log_perf(
                requirement_id,
                "TOTAL_AI2_EXTRACTION",
                _elapsed_ms(total_started),
                element_count=len(extraction.elements),
                warning_count=len(extraction.warnings),
            )
            return extraction

    def extract_from_files(
        self,
        files: list[Path],
        project_id: str | None = None,
        requirement_id: str | None = None,
        debug_capture: GeminiExtractionDebugCapture | None = None,
    ) -> RequirementExtraction:
        if not files:
            raise ValueError("Debes proporcionar al menos un archivo para extraer.")

        paths = [Path(file) for file in files]
        file_specs = [_prepare_file_spec(path) for path in paths]
        prompt = build_file_extraction_prompt(
            paths,
            project_id=project_id,
            requirement_id=requirement_id,
            media_types=_media_types(file_specs),
        )

        contents = _build_multifile_contents(prompt, file_specs)
        response = self._provider._client.models.generate_content(
            model=self._provider.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        gemini_extraction = _parse_gemini_extraction_response(response, debug_capture)
        extraction = map_gemini_extraction_to_requirement_extraction(
            gemini_extraction,
            model_provider="google",
            model=self._provider.model,
            default_source_id=_default_evidence_source_id(file_specs),
            allowed_source_ids=_source_ids(file_specs),
        )
        extraction.requirement.project_id = project_id
        extraction.requirement.requirement_id = requirement_id
        extraction.sources = _build_sources(file_specs)
        extraction.extraction_metadata.source_count = len(paths)
        _apply_usage_metadata(extraction, response)
        return extraction

    def _enrich_discovery_with_context(
        self,
        context: _RequirementFileContext,
        discovery: GeminiDiscoveryResult,
        batch_size: int | None = None,
        debug_capture: GeminiEnrichmentDebugCapture | None = None,
        scope: GeminiScopeResult | None = None,
    ) -> GeminiEnrichmentResult:
        resolved_batch_size = _resolve_batch_size(
            batch_size,
            self._settings.gemini_enrichment_batch_size,
        )
        batches = build_discovery_batches(discovery, resolved_batch_size)
        batch_results = []
        batch_usage = []

        for batch in batches:
            response = self._generate_json(
                _build_multifile_contents(
                    build_file_enrichment_prompt(
                        batch,
                        scope_lookup(scope) if scope else None,
                        context.paths,
                        media_types=_media_types(context.file_specs),
                    ),
                    context.file_specs,
                )
            )
            batch_result = _parse_gemini_enrichment_response(response)
            usage = _extract_token_usage(response)
            batch_results.append(batch_result)
            batch_usage.append(usage)

            if debug_capture is not None:
                debug_capture.raw_responses.append(getattr(response, "text", None))
                debug_capture.batch_results.append(batch_result)
                debug_capture.batch_usage.append(usage)

        merged = merge_enrichment_batches(discovery, batch_results)
        merged.usage = sum_token_usage(batch_usage)
        if debug_capture is not None:
            debug_capture.merged_result = merged
            debug_capture.batch_size = resolved_batch_size
            debug_capture.model = self._provider.model
        return merged

    def _generate_json(self, contents):
        return self._provider._client.models.generate_content(
            model=self._provider.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )


class _LocalFileSpec:
    def __init__(self, path: Path, mime_type: str) -> None:
        self.path = path
        self.mime_type = mime_type


class _RequirementFileContext:
    def __init__(self, owner: GeminiExtractionProvider, files: list[Path]) -> None:
        self._owner = owner
        self.paths = [Path(file) for file in files]
        self.file_specs: list[_LocalFileSpec] = []

    def __enter__(self) -> _RequirementFileContext:
        _ = self._owner
        self.file_specs = [_prepare_file_spec(path) for path in self.paths]
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


@dataclass
class GeminiExtractionDebugCapture:
    raw_response_text: str | None = None
    gemini_extraction: GeminiExtraction | None = None


@dataclass
class GeminiDiscoveryDebugCapture:
    raw_response_text: str | None = None
    discovery_result: GeminiDiscoveryResult | None = None
    token_usage: TokenUsage | None = None
    model: str | None = None


@dataclass
class GeminiScopeDebugCapture:
    raw_response_text: str | None = None
    scope_result: GeminiScopeResult | None = None
    token_usage: TokenUsage | None = None
    model: str | None = None


@dataclass
class GeminiEnrichmentDebugCapture:
    raw_responses: list[str | None] | None = None
    batch_results: list[GeminiEnrichmentResult] | None = None
    batch_usage: list[TokenUsage | None] | None = None
    merged_result: GeminiEnrichmentResult | None = None
    batch_size: int | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        self.raw_responses = [] if self.raw_responses is None else self.raw_responses
        self.batch_results = [] if self.batch_results is None else self.batch_results
        self.batch_usage = [] if self.batch_usage is None else self.batch_usage


@dataclass
class GeminiFullPipelineDebugCapture:
    discovery: GeminiDiscoveryResult | None = None
    scope: GeminiScopeResult | None = None
    enrichment: GeminiEnrichmentResult | None = None
    gemini_extraction: GeminiExtraction | None = None
    discovery_debug: GeminiDiscoveryDebugCapture | None = None
    scope_debug: GeminiScopeDebugCapture | None = None
    enrichment_debug: GeminiEnrichmentDebugCapture | None = None
    model: str | None = None
    batch_size: int | None = None
    batch_count: int | None = None


def _prepare_file_spec(path: Path) -> _LocalFileSpec:
    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe: {path}")
    if not path.is_file():
        raise ValueError(f"La ruta no es un archivo: {path}")

    mime_type = _detect_supported_mime_type(path)
    if mime_type not in SUPPORTED_FILE_MIME_TYPES:
        raise ValueError(f"Tipo de archivo no soportado para GeminiExtractionProvider: {path}")

    return _LocalFileSpec(path=path, mime_type=mime_type)


def _build_sources(file_specs: list[_LocalFileSpec]) -> list[Source]:
    return [
        Source(
            id=f"source-{index}",
            file_name=spec.path.name,
            media_type=spec.mime_type,
            source_type=_source_type_for_mime_type(spec.mime_type),
            page_count=_pdf_page_count(spec.path) if spec.mime_type == "application/pdf" else None,
        )
        for index, spec in enumerate(file_specs, start=1)
    ]


def _source_type_for_mime_type(mime_type: str) -> str:
    if mime_type == "application/pdf":
        return "document"
    if mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return "spreadsheet"
    return "image"


def _default_evidence_source_id(file_specs: list[_LocalFileSpec]) -> str | None:
    return "source-1" if len(file_specs) == 1 else None


def _source_ids(file_specs: list[_LocalFileSpec]) -> list[str]:
    return [f"source-{index}" for index, _ in enumerate(file_specs, start=1)]


def _media_types(file_specs: list[_LocalFileSpec]) -> list[str]:
    return [spec.mime_type for spec in file_specs]


def _parse_gemini_extraction_response(
    response,
    debug_capture: GeminiExtractionDebugCapture | None = None,
) -> GeminiExtraction:
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini no devolvio texto JSON para validar como GeminiExtraction.")

    try:
        if debug_capture is not None:
            debug_capture.raw_response_text = text
        gemini_extraction = GeminiExtraction.model_validate_json(text)
        if debug_capture is not None:
            debug_capture.gemini_extraction = gemini_extraction
        return gemini_extraction
    except Exception as exc:
        raise ValueError("Gemini devolvio JSON invalido para GeminiExtraction.") from exc


def _parse_gemini_discovery_response(
    response,
    debug_capture: GeminiDiscoveryDebugCapture | None = None,
) -> GeminiDiscoveryResult:
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini no devolvio texto JSON para validar GeminiDiscoveryResult.")

    try:
        if debug_capture is not None:
            debug_capture.raw_response_text = text
        discovery_result = GeminiDiscoveryResult.model_validate_json(text)
        if debug_capture is not None:
            debug_capture.discovery_result = discovery_result
        return discovery_result
    except Exception as exc:
        raise ValueError("Gemini devolvio JSON invalido para GeminiDiscoveryResult.") from exc


def _parse_gemini_scope_response(
    response,
    debug_capture: GeminiScopeDebugCapture | None = None,
) -> GeminiScopeResult:
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini no devolvio texto JSON para validar GeminiScopeResult.")

    try:
        if debug_capture is not None:
            debug_capture.raw_response_text = text
        scope_result = GeminiScopeResult.model_validate_json(text)
        if debug_capture is not None:
            debug_capture.scope_result = scope_result
        return scope_result
    except Exception as exc:
        raise ValueError("Gemini devolvio JSON invalido para GeminiScopeResult.") from exc


def _parse_gemini_enrichment_response(response) -> GeminiEnrichmentResult:
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini no devolvio texto JSON para validar GeminiEnrichmentResult.")

    try:
        return GeminiEnrichmentResult.model_validate_json(text)
    except Exception as exc:
        raise ValueError("Gemini devolvio JSON invalido para GeminiEnrichmentResult.") from exc


def _apply_usage_metadata(
    extraction: RequirementExtraction,
    response,
) -> None:
    token_usage = _extract_token_usage(response)
    if token_usage is not None:
        extraction.extraction_metadata.token_usage = token_usage


def _extract_token_usage(response) -> TokenUsage | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return None

    input_tokens = getattr(usage_metadata, "prompt_token_count", None)
    output_tokens = getattr(usage_metadata, "candidates_token_count", None)
    total_tokens = getattr(usage_metadata, "total_token_count", None)

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _resolve_batch_size(batch_size: int | None, default_batch_size: int) -> int:
    resolved = batch_size if batch_size is not None else default_batch_size
    if resolved <= 0:
        raise ValueError("batch_size debe ser mayor que cero.")
    return resolved


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log_perf(
    requirement_id: str | None,
    stage: str,
    elapsed_ms: int,
    **fields: object,
) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(
        "[NEWPIPE-PERF] RequirementId=%s Stage=%s ElapsedMs=%s %s",
        requirement_id,
        stage,
        elapsed_ms,
        details,
    )


def _warnings_from_messages(messages: list[str]):
    from app.models.requirement import Warning

    return [
        Warning(
            id=f"enrichment-warning-{index}",
            code="enrichment_warning",
            severity="warning",
            message=message,
        )
        for index, message in enumerate(messages, start=1)
    ]


def _detect_supported_mime_type(path: Path) -> str:
    with path.open("rb") as file:
        header = file.read(16)

    if header.startswith(b"%PDF-"):
        return "application/pdf"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        if "[Content_Types].xml" in names and "xl/workbook.xml" in names:
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    raise ValueError(f"No se pudo detectar un MIME type soportado para: {path}")


def _pdf_page_count(path: Path) -> int | None:
    try:
        content = path.read_bytes()
    except OSError:
        return None

    count = len(re.findall(rb"/Type\s*/Page\b", content))
    return count or None


def _build_multifile_contents(prompt: str, file_specs: list[_LocalFileSpec]) -> list[types.Part]:
    parts = [types.Part.from_text(text=prompt)]
    parts.extend(_build_inline_file_parts(file_specs))

    return parts


def _build_inline_file_parts(file_specs: list[_LocalFileSpec]) -> list[types.Part]:
    return [
        types.Part.from_bytes(
            data=spec.path.read_bytes(),
            mime_type=spec.mime_type,
        )
        for spec in file_specs
    ]
