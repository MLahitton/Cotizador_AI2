import logging
import time

from google.genai import types

from app.models.similarity import (
    SimilarityBatchEvaluationResult,
    SimilarityBatchRequestItem,
    SimilarityBatchResultStatus,
    SimilarityElementInput,
    SimilarityEvaluationResult,
    SimilarityHistoricalCandidateInput,
)
from app.providers.gemini import GeminiProvider
from app.services.similarity_prompt import (
    build_similarity_batch_prompt,
    build_similarity_prompt,
)


logger = logging.getLogger(__name__)


class SimilarityEvaluationError(ValueError):
    pass


class SimilarityEvaluator:
    def __init__(self, provider: GeminiProvider | None = None) -> None:
        self._provider = provider or GeminiProvider()

    def evaluate(
        self,
        element: SimilarityElementInput,
        candidates: list[SimilarityHistoricalCandidateInput],
    ) -> SimilarityEvaluationResult:
        expected_ids = [candidate.candidate_id for candidate in candidates]
        if len(set(expected_ids)) != len(expected_ids):
            raise SimilarityEvaluationError("candidate_id duplicado en candidatos de entrada.")

        llm_started = time.perf_counter()
        response = self._provider._client.models.generate_content(
            model=self._provider.model,
            contents=build_similarity_prompt(element, candidates),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        logger.info(
            "[NEWPIPE-PERF] Stage=LLM_CALL ElapsedMs=%s candidateCount=%s",
            int((time.perf_counter() - llm_started) * 1000),
            len(candidates),
        )
        result = _parse_similarity_response(response)
        _validate_candidate_ids(result, expected_ids)
        result.candidates.sort(key=lambda candidate: candidate.similarity_score, reverse=True)
        result.evaluated_candidate_count = len(result.candidates)
        return result

    def evaluate_batch(
        self,
        requests: list[SimilarityBatchRequestItem],
    ) -> SimilarityBatchEvaluationResult:
        request_ids = [request.request_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise SimilarityEvaluationError("request_id duplicado en similarity batch.")

        expected_by_request: dict[str, list[str]] = {}
        for request in requests:
            expected_ids = [candidate.candidate_id for candidate in request.candidates]
            if len(set(expected_ids)) != len(expected_ids):
                raise SimilarityEvaluationError(
                    f"candidate_id duplicado en request_id {request.request_id}."
                )
            expected_by_request[request.request_id] = expected_ids

        candidate_count = sum(len(request.candidates) for request in requests)
        llm_started = time.perf_counter()
        response = self._provider._client.models.generate_content(
            model=self._provider.model,
            contents=build_similarity_batch_prompt(requests),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        logger.info(
            "[NEWPIPE-PERF] Stage=LLM_CALL_BATCH ElapsedMs=%s requestCount=%s candidateCount=%s",
            int((time.perf_counter() - llm_started) * 1000),
            len(requests),
            candidate_count,
        )
        result = _parse_similarity_batch_response(response)
        _validate_batch_result_ids(result, expected_by_request)
        for item in result.results:
            item.candidates.sort(
                key=lambda candidate: candidate.similarity_score,
                reverse=True,
            )
        return result


def _parse_similarity_response(response) -> SimilarityEvaluationResult:
    text = getattr(response, "text", None)
    if not text:
        raise SimilarityEvaluationError("Gemini no devolvio texto JSON para similarity.")
    try:
        return SimilarityEvaluationResult.model_validate_json(text)
    except Exception as exc:
        raise SimilarityEvaluationError("Gemini devolvio JSON invalido para similarity.") from exc


def _parse_similarity_batch_response(response) -> SimilarityBatchEvaluationResult:
    text = getattr(response, "text", None)
    if not text:
        raise SimilarityEvaluationError("Gemini no devolvio texto JSON para similarity batch.")
    try:
        return SimilarityBatchEvaluationResult.model_validate_json(text)
    except Exception as exc:
        raise SimilarityEvaluationError(
            "Gemini devolvio JSON invalido para similarity batch."
        ) from exc


def _validate_candidate_ids(
    result: SimilarityEvaluationResult,
    expected_ids: list[str],
) -> None:
    expected = set(expected_ids)
    returned = [candidate.candidate_id for candidate in result.candidates]
    returned_set = set(returned)

    if len(returned_set) != len(returned):
        raise SimilarityEvaluationError("Gemini devolvio candidate_id duplicado.")

    missing = sorted(expected - returned_set)
    unknown = sorted(returned_set - expected)
    if missing:
        raise SimilarityEvaluationError(
            "Gemini no devolvio evaluacion para candidate_id esperado: "
            + ", ".join(missing)
        )
    if unknown:
        raise SimilarityEvaluationError(
            "Gemini devolvio candidate_id desconocido: " + ", ".join(unknown)
        )


def _validate_batch_result_ids(
    result: SimilarityBatchEvaluationResult,
    expected_by_request: dict[str, list[str]],
) -> None:
    expected_request_ids = set(expected_by_request)
    returned_request_ids = [item.request_id for item in result.results]
    returned_request_set = set(returned_request_ids)

    if len(returned_request_set) != len(returned_request_ids):
        raise SimilarityEvaluationError("Gemini devolvio request_id duplicado.")

    missing_requests = sorted(expected_request_ids - returned_request_set)
    unknown_requests = sorted(returned_request_set - expected_request_ids)
    if missing_requests:
        raise SimilarityEvaluationError(
            "Gemini no devolvio resultado para request_id esperado: "
            + ", ".join(missing_requests)
        )
    if unknown_requests:
        raise SimilarityEvaluationError(
            "Gemini devolvio request_id desconocido: " + ", ".join(unknown_requests)
        )

    for item in result.results:
        if item.status == SimilarityBatchResultStatus.FAILED:
            continue
        expected = set(expected_by_request[item.request_id])
        returned = [candidate.candidate_id for candidate in item.candidates]
        returned_set = set(returned)
        if len(returned_set) != len(returned):
            raise SimilarityEvaluationError(
                f"Gemini devolvio candidate_id duplicado en request_id {item.request_id}."
            )
        missing = sorted(expected - returned_set)
        unknown = sorted(returned_set - expected)
        if missing:
            raise SimilarityEvaluationError(
                f"Gemini no devolvio candidate_id esperado en request_id {item.request_id}: "
                + ", ".join(missing)
            )
        if unknown:
            raise SimilarityEvaluationError(
                f"Gemini devolvio candidate_id desconocido en request_id {item.request_id}: "
                + ", ".join(unknown)
            )
