import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.genai.errors import APIError
from pydantic import ValidationError

from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini_extraction import GeminiExtractionProvider

ALLOWED_UPLOAD_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/requirements", tags=["requirements"])


def get_gemini_extraction_provider() -> Callable[[], GeminiExtractionProvider]:
    return GeminiExtractionProvider


@router.post("/extract", response_model=RequirementExtraction)
async def extract_requirement(
    provider_dependency: Annotated[
        GeminiExtractionProvider | Callable[[], GeminiExtractionProvider],
        Depends(get_gemini_extraction_provider),
    ],
    files: Annotated[
        list[UploadFile],
        File(
            description="PDF/XLSX/JPG/PNG files",
            json_schema_extra={"items": {"type": "string", "format": "binary"}},
        ),
    ] = None,
    project_id: Annotated[str | None, Form()] = None,
    requirement_id: Annotated[str | None, Form()] = None,
) -> RequirementExtraction:
    total_started = time.perf_counter()
    failed_stage = "REQUEST_RECEIVED"
    file_count = len(files or [])
    _log_perf(
        requirement_id,
        "REQUEST_RECEIVED",
        0,
        project_id=project_id,
        file_count=file_count,
    )

    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un archivo.")

    _validate_declared_content_types(files)
    failed_stage = "FILE_LOAD"
    with tempfile.TemporaryDirectory(prefix="ai2-requirement-upload-") as temp_dir:
        temp_paths = []
        for index, upload in enumerate(files, start=1):
            file_started = time.perf_counter()
            temp_paths.append(await _save_upload_file(upload, Path(temp_dir), index))
            _log_perf(
                requirement_id,
                "FILE_LOAD",
                _elapsed_ms(file_started),
                project_id=project_id,
                file_index=index,
                file_name=upload.filename,
                content_type=upload.content_type,
            )

        try:
            failed_stage = "PROVIDER_INIT"
            provider = (
                provider_dependency()
                if callable(provider_dependency)
                else provider_dependency
            )
            failed_stage = "LLM_STRUCTURED_EXTRACTION"
            extraction_started = time.perf_counter()
            result = provider.extract_with_discovery_from_files(
                temp_paths,
                project_id=project_id,
                requirement_id=requirement_id,
            )
            _log_perf(
                requirement_id,
                "LLM_STRUCTURED_EXTRACTION",
                _elapsed_ms(extraction_started),
                project_id=project_id,
                item_count=len(result.elements),
            )
            _log_perf(
                requirement_id,
                "RESPONSE_READY",
                0,
                project_id=project_id,
                item_count=len(result.elements),
                warning_count=len(result.warnings),
                conflict_count=len(result.conflicts),
            )
            _log_perf(
                requirement_id,
                "TOTAL_AI2_EXTRACTION",
                _elapsed_ms(total_started),
                project_id=project_id,
                file_count=file_count,
                item_count=len(result.elements),
            )
            return result
        except ValidationError as exc:
            logger.exception(
                "[NEWPIPE-PERF] REQUIREMENTS_EXTRACT_FAILED failedStage=%s",
                failed_stage,
            )
            raise HTTPException(status_code=502, detail="Respuesta de IA invalida.") from exc
        except ValueError as exc:
            logger.exception(
                "[NEWPIPE-PERF] REQUIREMENTS_EXTRACT_FAILED failedStage=%s",
                failed_stage,
            )
            raise _http_error_from_value_error(exc) from exc
        except APIError as exc:
            logger.exception(
                "[NEWPIPE-PERF] REQUIREMENTS_EXTRACT_FAILED failedStage=%s",
                failed_stage,
            )
            raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
        except RuntimeError as exc:
            logger.exception(
                "[NEWPIPE-PERF] REQUIREMENTS_EXTRACT_FAILED failedStage=%s",
                failed_stage,
            )
            raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
        except Exception as exc:
            logger.exception(
                "[NEWPIPE-PERF] REQUIREMENTS_EXTRACT_FAILED failedStage=%s",
                failed_stage,
            )
            raise HTTPException(status_code=500, detail="Error interno inesperado.") from exc


def _validate_declared_content_types(files: list[UploadFile]) -> None:
    for upload in files:
        if upload.content_type and upload.content_type not in ALLOWED_UPLOAD_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Tipo de archivo no soportado.")


async def _save_upload_file(upload: UploadFile, temp_dir: Path, index: int) -> Path:
    original_name = _safe_upload_file_name(upload.filename or f"source-{index}")
    destination = temp_dir / f"{index:03d}_{original_name}"
    bytes_written = 0

    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            bytes_written += len(chunk)
            output.write(chunk)

    if bytes_written == 0:
        raise HTTPException(status_code=400, detail="No se permiten archivos vacios.")

    return destination


def _safe_upload_file_name(file_name: str) -> str:
    name = Path(file_name).name or "upload"
    forbidden = '<>:"/\\|?*'
    safe_name = "".join(
        "_" if character in forbidden or ord(character) < 32 else character
        for character in name
    )
    return safe_name or "upload"


def _http_error_from_value_error(exc: ValueError) -> HTTPException:
    message = str(exc)
    if "MIME type soportado" in message or "Tipo de archivo no soportado" in message:
        return HTTPException(status_code=400, detail="Tipo de archivo no soportado.")
    if "JSON invalido" in message or "texto JSON" in message:
        return HTTPException(status_code=502, detail="Respuesta de IA invalida.")

    return HTTPException(status_code=400, detail="Solicitud invalida.")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log_perf(requirement_id: str | None, stage: str, elapsed_ms: int, **values) -> None:
    details = " ".join(f"{key}={value}" for key, value in values.items())
    logger.info(
        "[NEWPIPE-PERF] RequirementId=%s Stage=%s ElapsedMs=%s %s",
        requirement_id,
        stage,
        elapsed_ms,
        details,
    )
