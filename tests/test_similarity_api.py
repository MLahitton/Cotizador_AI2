from fastapi.testclient import TestClient

from app.api.similarity import get_similarity_evaluator
from app.main import app
from app.models.similarity import (
    SimilarityBatchEvaluationResult,
    SimilarityEvaluationResult,
)
from app.services.similarity_evaluator import SimilarityEvaluationError


class _FakeSimilarityEvaluator:
    def __init__(self, result=None, batch_result=None, error: Exception | None = None) -> None:
        self.result = SimilarityEvaluationResult.model_validate(result or _result_payload())
        self.batch_result = SimilarityBatchEvaluationResult.model_validate(
            batch_result or _batch_result_payload()
        )
        self.error = error
        self.calls = []
        self.batch_calls = []

    def evaluate(self, element, candidates):
        self.calls.append((element, candidates))
        if self.error is not None:
            raise self.error
        return self.result

    def evaluate_batch(self, requests):
        self.batch_calls.append(requests)
        if self.error is not None:
            raise self.error
        return self.batch_result


def test_similarity_evaluate_valid_request_returns_200() -> None:
    fake = _FakeSimilarityEvaluator()

    response = _post_with_fake(fake, "/similarity/evaluate", _request_payload())

    assert response.status_code == 200
    assert response.json()["element_id"] == "new-1"
    assert response.json()["candidates"][0]["candidate_id"] == "c-1"
    assert len(fake.calls) == 1
    element, candidates = fake.calls[0]
    assert element.element_id == "new-1"
    assert candidates[0].candidate_id == "c-1"


def test_similarity_evaluate_multiple_candidates_returns_200() -> None:
    fake = _FakeSimilarityEvaluator(
        result=_result_payload(
            candidates=[
                _candidate_result("c-2", 0.95, "VERY_HIGH"),
                _candidate_result("c-1", 0.75, "HIGH"),
            ]
        )
    )
    payload = _request_payload(candidates=[_candidate_payload("c-1"), _candidate_payload("c-2")])

    response = _post_with_fake(fake, "/similarity/evaluate", payload)

    assert response.status_code == 200
    assert [candidate["candidate_id"] for candidate in response.json()["candidates"]] == [
        "c-2",
        "c-1",
    ]


def test_similarity_evaluate_empty_candidates_returns_controlled_error() -> None:
    response = _post_with_fake(
        _FakeSimilarityEvaluator(),
        "/similarity/evaluate",
        _request_payload(candidates=[]),
    )

    assert response.status_code == 422


def test_similarity_evaluate_duplicate_candidate_id_returns_controlled_error() -> None:
    fake = _FakeSimilarityEvaluator(
        error=SimilarityEvaluationError("candidate_id duplicado en candidatos de entrada.")
    )
    payload = _request_payload(candidates=[_candidate_payload("c-1"), _candidate_payload("c-1")])

    response = _post_with_fake(fake, "/similarity/evaluate", payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "candidate_id duplicado."


def test_similarity_evaluate_similarity_error_returns_controlled_error() -> None:
    fake = _FakeSimilarityEvaluator(
        error=SimilarityEvaluationError("Gemini devolvio JSON invalido para similarity.")
    )

    response = _post_with_fake(fake, "/similarity/evaluate", _request_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "Respuesta de IA invalida."


def test_similarity_evaluate_provider_error_returns_502_without_sensitive_details() -> None:
    fake = _FakeSimilarityEvaluator(error=RuntimeError("GEMINI_API_KEY=secret-value"))

    response = _post_with_fake(fake, "/similarity/evaluate", _request_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "Error del proveedor de IA."
    assert "secret-value" not in response.text


def test_similarity_evaluate_batch_valid_request_returns_200_without_mixing_candidates() -> None:
    fake = _FakeSimilarityEvaluator(batch_result=_batch_result_payload())

    response = _post_with_fake(fake, "/similarity/evaluate-batch", _batch_request_payload())

    assert response.status_code == 200
    body = response.json()
    assert [item["request_id"] for item in body["results"]] == ["r-1", "r-2"]
    assert body["results"][0]["candidates"][0]["candidate_id"] == "c-1"
    assert body["results"][1]["candidates"][0]["candidate_id"] == "c-2"
    assert len(fake.batch_calls) == 1
    assert [request.request_id for request in fake.batch_calls[0]] == ["r-1", "r-2"]


def test_similarity_evaluate_batch_empty_requests_returns_controlled_error() -> None:
    response = _post_with_fake(
        _FakeSimilarityEvaluator(),
        "/similarity/evaluate-batch",
        {"requests": []},
    )

    assert response.status_code == 422


def test_similarity_evaluate_batch_duplicate_request_id_returns_controlled_error() -> None:
    fake = _FakeSimilarityEvaluator(
        error=SimilarityEvaluationError("request_id duplicado en similarity batch.")
    )

    response = _post_with_fake(fake, "/similarity/evaluate-batch", _batch_request_payload())

    assert response.status_code == 400
    assert response.json()["detail"] == "request_id duplicado."


def test_similarity_evaluate_batch_partial_failure_returns_200() -> None:
    fake = _FakeSimilarityEvaluator(
        batch_result={
            "results": [
                _batch_result_item("r-1", [_candidate_result("c-1")]),
                {
                    "request_id": "r-2",
                    "status": "FAILED",
                    "candidates": [],
                    "failure_code": "SIMILARITY_ITEM_FAILED",
                },
            ],
            "evaluation_source": "AI2_SIMILARITY_BATCH",
        }
    )

    response = _post_with_fake(fake, "/similarity/evaluate-batch", _batch_request_payload())

    assert response.status_code == 200
    assert response.json()["results"][1]["status"] == "FAILED"
    assert response.json()["results"][1]["failure_code"] == "SIMILARITY_ITEM_FAILED"


def test_similarity_evaluate_batch_similarity_error_returns_controlled_error() -> None:
    fake = _FakeSimilarityEvaluator(
        error=SimilarityEvaluationError("Gemini devolvio request_id desconocido: r-3")
    )

    response = _post_with_fake(fake, "/similarity/evaluate-batch", _batch_request_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "Respuesta de IA inconsistente."


def test_similarity_evaluate_batch_provider_error_returns_502_without_sensitive_details() -> None:
    fake = _FakeSimilarityEvaluator(error=RuntimeError("GEMINI_API_KEY=secret-value"))

    response = _post_with_fake(fake, "/similarity/evaluate-batch", _batch_request_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "Error del proveedor de IA."
    assert "secret-value" not in response.text


def test_similarity_openapi_exposes_response_schema() -> None:
    client = TestClient(app)

    openapi = client.get("/openapi.json").json()
    operation = openapi["paths"]["/similarity/evaluate"]["post"]
    batch_operation = openapi["paths"]["/similarity/evaluate-batch"]["post"]

    assert "application/json" in operation["requestBody"]["content"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/SimilarityEvaluationResult")
    assert "application/json" in batch_operation["requestBody"]["content"]
    batch_response_schema = batch_operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert batch_response_schema["$ref"].endswith("/SimilarityBatchEvaluationResult")


def _post_with_fake(fake: _FakeSimilarityEvaluator, path: str, payload: dict):
    app.dependency_overrides[get_similarity_evaluator] = lambda: fake
    try:
        client = TestClient(app)
        return client.post(path, json=payload)
    finally:
        app.dependency_overrides.clear()


def _request_payload(candidates: list[dict] | None = None) -> dict:
    return {
        "element": {
            "element_id": "new-1",
            "reference": "PV-06",
            "category": "puerta vidriera",
            "system": "3831",
            "glass_family": "templado",
            "glass_thickness": "6",
            "configuration": "corrediza",
            "area_m2": "9.35",
            "finish": "negro",
        },
        "candidates": candidates if candidates is not None else [_candidate_payload("c-1")],
    }


def _batch_request_payload() -> dict:
    return {
        "requests": [
            {
                "request_id": "r-1",
                **_request_payload(candidates=[_candidate_payload("c-1")]),
            },
            {
                "request_id": "r-2",
                **_request_payload(candidates=[_candidate_payload("c-2")]),
            },
        ]
    }


def _candidate_payload(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "quote_id": "SG943",
        "historical_item_id": "hist-1",
        "reference": "PV-01",
        "description": "Puerta corrediza Venecia Napoles",
        "category": "puerta vidriera",
        "system": "Venecia Napoles",
        "glass_family": "templado",
        "glass_thickness": "6",
        "configuration": "corrediza",
        "area_m2": "9.35",
        "finish": "negro",
        "backend_preliminary_score": 0.82,
        "matched_signals": ["category", "glass"],
        "missing_signals": [],
    }


def _result_payload(candidates: list[dict] | None = None) -> dict:
    return {
        "element_id": "new-1",
        "evaluated_candidate_count": len(candidates or [_candidate_result("c-1")]),
        "candidates": candidates or [_candidate_result("c-1")],
        "overall_notes": [],
        "evaluation_source": "AI2_SIMILARITY",
    }


def _batch_result_payload() -> dict:
    return {
        "results": [
            _batch_result_item("r-1", [_candidate_result("c-1")]),
            _batch_result_item("r-2", [_candidate_result("c-2", 0.82, "MEDIUM")]),
        ],
        "evaluation_source": "AI2_SIMILARITY_BATCH",
    }


def _batch_result_item(request_id: str, candidates: list[dict]) -> dict:
    return {
        "request_id": request_id,
        "status": "COMPLETED",
        "candidates": candidates,
        "failure_code": None,
    }


def _candidate_result(
    candidate_id: str,
    score: float = 0.91,
    level: str = "HIGH",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "similarity_score": score,
        "similarity_level": level,
        "matched_features": ["category", "glass"],
        "differences": ["Sistema 3831 vs Venecia Napoles"],
        "technical_explanation": "Comparable tecnico.",
        "confidence": 0.8,
    }
