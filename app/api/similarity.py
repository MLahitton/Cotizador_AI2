import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from app.models.similarity import (
    SimilarityBatchEvaluationRequest,
    SimilarityBatchEvaluationResult,
    SimilarityElementInput,
    SimilarityEvaluationResult,
    SimilarityHistoricalCandidateInput,
)
from app.services.similarity_evaluator import (
    SimilarityEvaluationError,
    SimilarityEvaluator,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/similarity", tags=["similarity"])


class SimilarityEvaluationRequest(BaseModel):
    element: SimilarityElementInput
    candidates: list[SimilarityHistoricalCandidateInput] = Field(min_length=1)


def get_similarity_evaluator() -> SimilarityEvaluator:
    return SimilarityEvaluator()


@router.post("/evaluate", response_model=SimilarityEvaluationResult)
def evaluate_similarity(
    request: SimilarityEvaluationRequest,
    evaluator: Annotated[SimilarityEvaluator, Depends(get_similarity_evaluator)],
) -> SimilarityEvaluationResult:
    total_started = time.perf_counter()
    _log_perf("REQUEST_RECEIVED", 0, candidate_count=len(request.candidates))
    try:
        result = evaluator.evaluate(request.element, request.candidates)
        _log_perf(
            "POSTPROCESS",
            0,
            candidate_count=len(result.candidates),
        )
        _log_perf(
            "TOTAL_SIMILARITY",
            _elapsed_ms(total_started),
            candidate_count=len(request.candidates),
            evaluated_candidate_count=len(result.candidates),
        )
        return result
    except SimilarityEvaluationError as exc:
        raise _http_error_from_similarity_error(exc) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc


@router.post("/evaluate-batch", response_model=SimilarityBatchEvaluationResult)
def evaluate_similarity_batch(
    request: SimilarityBatchEvaluationRequest,
    evaluator: Annotated[SimilarityEvaluator, Depends(get_similarity_evaluator)],
) -> SimilarityBatchEvaluationResult:
    total_started = time.perf_counter()
    candidate_count = sum(len(item.candidates) for item in request.requests)
    _log_perf(
        "BATCH_REQUEST_RECEIVED",
        0,
        request_count=len(request.requests),
        candidate_count=candidate_count,
    )
    try:
        result = evaluator.evaluate_batch(request.requests)
        _log_perf(
            "BATCH_POSTPROCESS",
            0,
            request_count=len(result.results),
            candidate_count=sum(len(item.candidates) for item in result.results),
        )
        _log_perf(
            "TOTAL_SIMILARITY_BATCH",
            _elapsed_ms(total_started),
            request_count=len(request.requests),
            candidate_count=candidate_count,
            result_count=len(result.results),
        )
        return result
    except SimilarityEvaluationError as exc:
        raise _http_error_from_similarity_error(exc) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc


def _http_error_from_similarity_error(exc: SimilarityEvaluationError) -> HTTPException:
    message = str(exc)
    if "duplicado en candidatos de entrada" in message:
        return HTTPException(status_code=400, detail="candidate_id duplicado.")
    if "request_id duplicado" in message:
        return HTTPException(status_code=400, detail="request_id duplicado.")
    if "candidate_id duplicado en request_id" in message:
        return HTTPException(status_code=400, detail="candidate_id duplicado.")
    if "JSON invalido" in message or "texto JSON" in message:
        return HTTPException(status_code=502, detail="Respuesta de IA invalida.")
    if "candidate_id" in message or "request_id" in message:
        return HTTPException(status_code=502, detail="Respuesta de IA inconsistente.")

    return HTTPException(status_code=400, detail="Solicitud invalida.")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log_perf(stage: str, elapsed_ms: int, **values) -> None:
    details = " ".join(f"{key}={value}" for key, value in values.items())
    logger.info(
        "[NEWPIPE-PERF] Stage=%s ElapsedMs=%s %s",
        stage,
        elapsed_ms,
        details,
    )
