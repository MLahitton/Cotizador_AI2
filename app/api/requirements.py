import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.genai.errors import APIError
from pydantic import ValidationError

from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini_extraction import GeminiExtractionProvider

ALLOWED_UPLOAD_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
}

router = APIRouter(prefix="/requirements", tags=["requirements"])


def get_gemini_extraction_provider() -> GeminiExtractionProvider:
    return GeminiExtractionProvider()


@router.post("/extract", response_model=RequirementExtraction)
async def extract_requirement(
    provider: Annotated[GeminiExtractionProvider, Depends(get_gemini_extraction_provider)],
    files: Annotated[list[UploadFile] | None, File(description="PDF/JPG/PNG files")] = None,
    project_id: Annotated[str | None, Form()] = None,
    requirement_id: Annotated[str | None, Form()] = None,
) -> RequirementExtraction:
    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un archivo.")

    _validate_declared_content_types(files)
    with tempfile.TemporaryDirectory(prefix="ai2-requirement-upload-") as temp_dir:
        temp_paths = []
        for index, upload in enumerate(files, start=1):
            temp_paths.append(await _save_upload_file(upload, Path(temp_dir), index))

        try:
            return provider.extract_from_files(
                temp_paths,
                project_id=project_id,
                requirement_id=requirement_id,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="Respuesta de IA invalida.") from exc
        except ValueError as exc:
            raise _http_error_from_value_error(exc) from exc
        except APIError as exc:
            raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Error interno inesperado.") from exc


def _validate_declared_content_types(files: list[UploadFile]) -> None:
    for upload in files:
        if upload.content_type and upload.content_type not in ALLOWED_UPLOAD_MIME_TYPES:
            raise HTTPException(status_code=415, detail="Tipo de archivo no soportado.")


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
        return HTTPException(status_code=415, detail="Tipo de archivo no soportado.")
    if "JSON invalido" in message or "texto JSON" in message:
        return HTTPException(status_code=422, detail="Respuesta de IA invalida.")

    return HTTPException(status_code=400, detail="Solicitud invalida.")
