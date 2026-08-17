from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from app.models.similarity import (
    SimilarityElementInput,
    SimilarityEvaluationResult,
    SimilarityHistoricalCandidateInput,
)
from app.services.similarity_evaluator import (
    SimilarityEvaluationError,
    SimilarityEvaluator,
)

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
    try:
        return evaluator.evaluate(request.element, request.candidates)
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
    if "JSON invalido" in message or "texto JSON" in message:
        return HTTPException(status_code=502, detail="Respuesta de IA invalida.")
    if "candidate_id" in message:
        return HTTPException(status_code=502, detail="Respuesta de IA inconsistente.")

    return HTTPException(status_code=400, detail="Solicitud invalida.")
