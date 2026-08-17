from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.similarity import (
    SimilarityCandidateResult,
    SimilarityElementInput,
    SimilarityEvaluationResult,
    SimilarityHistoricalCandidateInput,
    SimilarityLevel,
)


def test_similarity_element_input_accepts_minimal_valid_element() -> None:
    element = SimilarityElementInput(element_id="new-1")

    assert element.element_id == "new-1"
    assert element.warnings == []
    assert element.system is None


def test_similarity_historical_candidate_accepts_valid_candidate() -> None:
    candidate = SimilarityHistoricalCandidateInput(
        candidate_id="candidate-1",
        quote_id="SG943",
        historical_item_id="item-31",
        reference="V-31",
        description="CUERPO BATIENTE",
        category="ventana",
        system="SERIE 40",
        glass_family="templado",
        glass_thickness=Decimal("6"),
        glass_composition="templado 6 mm",
        configuration="batiente",
        width_mm=Decimal("730"),
        height_mm=Decimal("2200"),
        area_m2=Decimal("1.606"),
        quantity=Decimal("3"),
        finish="negro mate",
        public_unit_price=Decimal("1724647.627"),
        public_total=Decimal("5173942.88"),
        currency="COP",
        backend_preliminary_score=0.82,
        matched_signals=["category", "system"],
        missing_signals=["finish"],
    )

    assert candidate.candidate_id == "candidate-1"
    assert candidate.public_total == Decimal("5173942.88")
    assert candidate.matched_signals == ["category", "system"]
    assert candidate.missing_signals == ["finish"]


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_similarity_score_outside_zero_to_one_is_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        SimilarityCandidateResult(
            candidate_id="candidate-1",
            similarity_score=score,
            similarity_level=SimilarityLevel.MEDIUM,
        )


def test_candidate_result_preserves_matched_features_and_differences() -> None:
    result = SimilarityCandidateResult(
        candidate_id="candidate-1",
        similarity_score=0.91,
        similarity_level=SimilarityLevel.HIGH,
        matched_features=["category", "system", "configuration"],
        differences=[
            "vidrio distinto",
            "espesor distinto",
            "diferencia de area",
            "informacion faltante",
        ],
        technical_explanation="Buen sistema, pero el vidrio no coincide.",
        confidence=0.77,
    )

    assert "system" in result.matched_features
    assert "vidrio distinto" in result.differences
    assert "espesor distinto" in result.differences
    assert "diferencia de area" in result.differences
    assert "informacion faltante" in result.differences


def test_evaluation_result_accepts_multiple_candidates() -> None:
    evaluation = SimilarityEvaluationResult(
        element_id="new-1",
        evaluated_candidate_count=2,
        candidates=[
            SimilarityCandidateResult(
                candidate_id="candidate-1",
                similarity_score=0.95,
                similarity_level=SimilarityLevel.VERY_HIGH,
            ),
            SimilarityCandidateResult(
                candidate_id="candidate-2",
                similarity_score=0.35,
                similarity_level=SimilarityLevel.LOW,
                differences=["sistema distinto"],
            ),
        ],
        overall_notes=["Comparacion tecnica solamente."],
    )

    assert evaluation.evaluation_source == "AI2_SIMILARITY"
    assert evaluation.evaluated_candidate_count == 2
    assert [candidate.candidate_id for candidate in evaluation.candidates] == [
        "candidate-1",
        "candidate-2",
    ]


def test_decimal_prices_serialize_safely_in_json_mode() -> None:
    candidate = SimilarityHistoricalCandidateInput(
        candidate_id="candidate-1",
        public_unit_price=Decimal("1234567.89"),
        public_total=Decimal("2469135.78"),
        currency="COP",
    )

    dumped = candidate.model_dump(mode="json")

    assert dumped["public_unit_price"] == "1234567.89"
    assert dumped["public_total"] == "2469135.78"
    assert dumped["currency"] == "COP"


def test_similarity_levels_are_valid_enum_values() -> None:
    assert [level.value for level in SimilarityLevel] == [
        "VERY_HIGH",
        "HIGH",
        "MEDIUM",
        "LOW",
        "REJECTED",
    ]
