from google.genai import types

from app.models.similarity import (
    SimilarityElementInput,
    SimilarityEvaluationResult,
    SimilarityHistoricalCandidateInput,
)
from app.providers.gemini import GeminiProvider
from app.services.similarity_prompt import build_similarity_prompt


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

        response = self._provider._client.models.generate_content(
            model=self._provider.model,
            contents=build_similarity_prompt(element, candidates),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        result = _parse_similarity_response(response)
        _validate_candidate_ids(result, expected_ids)
        result.candidates.sort(key=lambda candidate: candidate.similarity_score, reverse=True)
        result.evaluated_candidate_count = len(result.candidates)
        return result


def _parse_similarity_response(response) -> SimilarityEvaluationResult:
    text = getattr(response, "text", None)
    if not text:
        raise SimilarityEvaluationError("Gemini no devolvio texto JSON para similarity.")
    try:
        return SimilarityEvaluationResult.model_validate_json(text)
    except Exception as exc:
        raise SimilarityEvaluationError("Gemini devolvio JSON invalido para similarity.") from exc


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
