from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.models.common import ExtractionStatus
from app.models.evidence import Region
from app.models.gemini_enrichment import GeminiElementEnrichment
from app.services.numeric_trace import NumericElementTrace
from app.services.quantity_grounding import (
    MODEL_EVIDENCE_QUANTITY,
    NO_SOURCE_GROUNDING_AVAILABLE,
    QUANTITY_GROUNDING_CONFLICT,
    QuantityGroundingDecision,
)

ReviewDecision = Literal["CONFIRMED", "CORRECTED", "AMBIGUOUS", "UNRESOLVED", "SKIPPED"]
CorrectionSupport = Literal["STRONG", "INSUFFICIENT", "AMBIGUOUS", "NOT_NEEDED"]
ConfirmationSupport = Literal["STRONG", "INSUFFICIENT", "NOT_NEEDED"]

FIELD_QUANTITY = "quantity"
TRIGGER_MODEL_UNVERIFIED = "MODEL_UNVERIFIED"
TRIGGER_QUANTITY_EQUALS_LEVEL = "QUANTITY_EQUALS_LEVEL"
TRIGGER_QUANTITY_EQUALS_REPETITION_COUNT = "QUANTITY_EQUALS_REPETITION_COUNT"
TRIGGER_QUANTITY_EQUALS_COMPONENT_COUNT = "QUANTITY_EQUALS_COMPONENT_COUNT"
TRIGGER_QUANTITY_EQUALS_PANEL_COUNT = "QUANTITY_EQUALS_PANEL_COUNT"
TRIGGER_SOURCE_CONFLICT = "SOURCE_CONFLICT"
REVIEW_CORRECTED_QUANTITY = "SEMANTIC_REVIEW_CORRECTED_QUANTITY"
REVIEW_AMBIGUOUS_QUANTITY = "SEMANTIC_REVIEW_AMBIGUOUS_QUANTITY"
REVIEW_UNRESOLVED_QUANTITY = "SEMANTIC_REVIEW_UNRESOLVED_QUANTITY"
REVIEW_CORRECTION_REJECTED_INSUFFICIENT_LOCAL_SUPPORT = (
    "SEMANTIC_REVIEW_CORRECTION_REJECTED_INSUFFICIENT_LOCAL_SUPPORT"
)

_SUSPICIOUS_ROLES = {
    "LEVEL",
    "FLOOR",
    "LEVEL_RANGE",
    "REPETITION_COUNT",
    "COMPONENT_COUNT",
    "PANEL_COUNT",
    "ITEM_NUMBER",
}


class SemanticFieldReview(BaseModel):
    element_id: str | None = None
    temporary_id: str
    reference: str | None = None
    field: str = FIELD_QUANTITY
    original_value: str | int | float | None = None
    observed_quantity: str | int | float | None = None
    observed_text: str | None = None
    reviewed_value: str | int | float | None = None
    decision: ReviewDecision
    reason: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_ids: list[str] = Field(default_factory=list)


class SourceLocator(BaseModel):
    source_id: str | None = None
    source_file_name: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    region: Region | None = None
    text_context: str | None = None
    text_context_is_first_pass: bool = False
    authoritative_value: str | int | float | None = None
    locator_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    locator_used: str = "NONE"
    locator_strength: str = "NONE"
    region_missing_origin: str | None = None


@dataclass(frozen=True)
class SemanticReviewTrace:
    reference: str | None
    field: str
    original_value: str | int | float | None
    trigger_reason: str
    review_called: bool
    locator_used: str
    source_id: str | None
    page_number: int | None
    sheet_name: str | None
    cell_range: str | None
    region: Region | None
    region_missing_origin: str | None
    locator_strength: str
    review_decision: str
    independent_reread: bool
    first_pass_quantity: str | int | float | None
    first_pass_quantity_evidence_included: bool
    observed_quantity: str | int | float | None
    observed_text: str | None
    reviewed_value: str | int | float | None
    final_value: str | int | float | None
    confidence: float | None
    correction_proposed: bool
    correction_applied: bool
    correction_support: CorrectionSupport
    correction_support_reason: str
    effective_decision: str
    numeric_collision_detected: bool
    collision_roles: tuple[str, ...]
    confirmation_cross_check_called: bool
    confirmation_support: ConfirmationSupport
    confirmation_support_reason: str
    source_ids: tuple[str, ...]


def should_review_quantity(
    element: GeminiElementEnrichment,
    grounding_decision: QuantityGroundingDecision | None,
    numeric_trace: NumericElementTrace | None = None,
) -> tuple[bool, str]:
    if element.quantity in (None, ""):
        return False, "NO_QUANTITY"

    if grounding_decision is not None:
        if grounding_decision.reason == QUANTITY_GROUNDING_CONFLICT:
            return True, TRIGGER_SOURCE_CONFLICT
        if (
            not grounding_decision.source_candidates
            and grounding_decision.model_evidence_candidates
        ):
            return True, MODEL_EVIDENCE_QUANTITY

    numeric_value = _numeric_value(element.quantity)
    if numeric_value is None or numeric_trace is None:
        return False, "NO_NUMERIC_SUSPICION"

    for candidate in numeric_trace.candidates:
        if candidate.semantic_role not in _SUSPICIOUS_ROLES:
            continue
        candidate_value = _numeric_value(candidate.value)
        if candidate_value is None:
            continue
        if _same_number(candidate_value, numeric_value):
            return True, _trigger_for_role(candidate.semantic_role)

    if (
        grounding_decision is not None
        and grounding_decision.reason == NO_SOURCE_GROUNDING_AVAILABLE
        and element.status == ExtractionStatus.AMBIGUOUS
    ):
        return True, TRIGGER_MODEL_UNVERIFIED

    return False, "NO_NUMERIC_SUSPICION"


def resolve_quantity_review_locator(
    element: GeminiElementEnrichment,
    grounding_decision: QuantityGroundingDecision | None,
    numeric_trace: NumericElementTrace | None = None,
) -> SourceLocator:
    if grounding_decision is not None:
        if grounding_decision.selected_source_candidate is not None:
            return _locator_from_grounding_candidate(
                grounding_decision.selected_source_candidate,
                locator_used="SOURCE_INDEPENDENT",
                locator_confidence=1.0,
            )
        if len(grounding_decision.source_candidates) == 1:
            return _locator_from_grounding_candidate(
                grounding_decision.source_candidates[0],
                locator_used="SOURCE_INDEPENDENT",
                locator_confidence=0.95,
            )

    if numeric_trace is not None:
        for candidate in _ranked_numeric_candidates(numeric_trace):
            if candidate.source_id or candidate.page_number or candidate.region:
                region = candidate.region or _same_element_region_for_candidate(
                    element,
                    source_id=candidate.source_id,
                    page_number=candidate.page_number,
                )
                return SourceLocator(
                    source_id=candidate.source_id,
                    source_file_name=candidate.source_file_name,
                    page_number=candidate.page_number,
                    sheet_name=candidate.sheet_name,
                    cell_range=candidate.cell_range,
                    region=region,
                    text_context=(
                        None
                        if candidate.grounding_type == MODEL_EVIDENCE_QUANTITY
                        else candidate.evidence_text
                    ),
                    text_context_is_first_pass=(
                        candidate.grounding_type == MODEL_EVIDENCE_QUANTITY
                    ),
                    locator_confidence=0.85,
                    locator_used="NUMERIC_TRACE",
                    locator_strength=_locator_strength(
                        source_id=candidate.source_id,
                        page_number=candidate.page_number,
                        sheet_name=candidate.sheet_name,
                        cell_range=candidate.cell_range,
                        region=region,
                    ),
                    region_missing_origin=_region_missing_origin(
                        source_id=candidate.source_id,
                        page_number=candidate.page_number,
                        region=region,
                    ),
                )

    for evidence in element.evidence:
        text_context = evidence.text or evidence.visual_description or evidence.notes
        if (
            evidence.source_id
            or evidence.page_number
            or evidence.sheet_name
            or evidence.cell_range
            or evidence.region
            or text_context
        ):
            return SourceLocator(
                source_id=evidence.source_id,
                page_number=evidence.page_number,
                sheet_name=evidence.sheet_name,
                cell_range=evidence.cell_range,
                region=evidence.region,
                text_context=text_context,
                text_context_is_first_pass=True,
                locator_confidence=0.7,
                locator_used="ELEMENT_EVIDENCE",
                locator_strength=_locator_strength(
                    source_id=evidence.source_id,
                    page_number=evidence.page_number,
                    sheet_name=evidence.sheet_name,
                    cell_range=evidence.cell_range,
                    region=evidence.region,
                ),
                region_missing_origin=_region_missing_origin(
                    source_id=evidence.source_id,
                    page_number=evidence.page_number,
                    region=evidence.region,
                ),
            )

    if element.reference:
        return SourceLocator(
            text_context=f"reference={element.reference}",
            locator_confidence=0.3,
            locator_used="REFERENCE_CONTEXT",
            locator_strength="REFERENCE_ONLY",
            region_missing_origin="MODEL_NOT_PROVIDED",
        )

    return SourceLocator()


def apply_quantity_review(
    element: GeminiElementEnrichment,
    review: SemanticFieldReview,
    locator: SourceLocator | None = None,
    numeric_trace: NumericElementTrace | None = None,
) -> GeminiElementEnrichment:
    if review.field != FIELD_QUANTITY:
        return element
    if review.temporary_id != element.temporary_id:
        return element

    if review.decision == "CORRECTED":
        support = evaluate_semantic_correction_support(
            field=FIELD_QUANTITY,
            original_value=element.quantity,
            reviewed_value=review.reviewed_value,
            locator=locator,
        )
        if support.support_level != "STRONG":
            return _replace_quantity(
                element,
                element.quantity,
                ExtractionStatus.AMBIGUOUS,
                _capped_uncertain_confidence(review.confidence, 0.5),
                REVIEW_CORRECTION_REJECTED_INSUFFICIENT_LOCAL_SUPPORT,
                support.reason,
            )
        return _replace_quantity(
            element,
            review.reviewed_value,
            ExtractionStatus.EXPLICIT,
            review.confidence,
            REVIEW_CORRECTED_QUANTITY,
            review.reason,
        )
    if review.decision == "AMBIGUOUS":
        return _replace_quantity(
            element,
            element.quantity,
            ExtractionStatus.AMBIGUOUS,
            _capped_uncertain_confidence(review.confidence, 0.5),
            REVIEW_AMBIGUOUS_QUANTITY,
            review.reason,
        )
    if review.decision == "UNRESOLVED":
        return _replace_quantity(
            element,
            element.quantity,
            ExtractionStatus.INFERRED,
            _capped_uncertain_confidence(review.confidence, 0.4),
            REVIEW_UNRESOLVED_QUANTITY,
            review.reason,
        )
    if review.decision == "CONFIRMED":
        support = evaluate_confirmed_quantity_support(
            original_value=element.quantity,
            locator=locator,
            numeric_trace=numeric_trace,
        )
        if support.support_level == "INSUFFICIENT":
            return _replace_quantity(
                element,
                element.quantity,
                ExtractionStatus.AMBIGUOUS,
                _capped_uncertain_confidence(review.confidence, 0.5),
                REVIEW_AMBIGUOUS_QUANTITY,
                support.reason,
            )
    return element


class SemanticCorrectionSupport(BaseModel):
    support_level: CorrectionSupport
    reason: str


class SemanticConfirmationSupport(BaseModel):
    support_level: ConfirmationSupport
    reason: str


class NumericCollisionSummary(BaseModel):
    detected: bool = False
    roles: list[str] = Field(default_factory=list)


class QuantityReviewNumericContext(BaseModel):
    element_temporary_id: str
    reference: str | None = None
    untrusted_first_pass_quantity: str | int | float | None = None
    candidates: list[dict[str, object]] = Field(default_factory=list)
    first_pass_quantity_evidence_included: bool = False


def build_quantity_review_numeric_context(
    numeric_trace: NumericElementTrace | None,
) -> QuantityReviewNumericContext:
    if numeric_trace is None:
        return QuantityReviewNumericContext(element_temporary_id="")

    candidates: list[dict[str, object]] = []
    for candidate in numeric_trace.candidates:
        item = candidate.model_dump(mode="json")
        if (
            candidate.semantic_role == "QUANTITY"
            and candidate.grounding_type == MODEL_EVIDENCE_QUANTITY
        ):
            item["evidence_text"] = None
            item["note"] = (
                "First-pass quantity evidence text omitted; re-read original source."
            )
        candidates.append(item)

    return QuantityReviewNumericContext(
        element_temporary_id=numeric_trace.element_temporary_id,
        reference=numeric_trace.reference,
        untrusted_first_pass_quantity=numeric_trace.final_quantity.value,
        candidates=candidates,
        first_pass_quantity_evidence_included=False,
    )


def quantity_numeric_collision_summary(
    value: str | int | float | None,
    numeric_trace: NumericElementTrace | None,
) -> NumericCollisionSummary:
    value_number = _numeric_value(value)
    if value_number is None or numeric_trace is None:
        return NumericCollisionSummary()

    roles: list[str] = []
    for candidate in numeric_trace.candidates:
        if candidate.field_path == "quantity":
            continue
        for role_value in _suspicious_role_values(candidate):
            if _same_number(role_value, value_number):
                _append_unique(roles, candidate.semantic_role)
                break

    return NumericCollisionSummary(detected=bool(roles), roles=roles)


def should_cross_check_confirmed_quantity(
    review: SemanticFieldReview,
    numeric_trace: NumericElementTrace | None,
) -> bool:
    if review.decision != "CONFIRMED":
        return False
    return quantity_numeric_collision_summary(review.reviewed_value, numeric_trace).detected


def evaluate_confirmed_quantity_support(
    *,
    original_value: str | int | float | None,
    locator: SourceLocator | None,
    numeric_trace: NumericElementTrace | None,
) -> SemanticConfirmationSupport:
    if not quantity_numeric_collision_summary(original_value, numeric_trace).detected:
        return SemanticConfirmationSupport(
            support_level="NOT_NEEDED",
            reason="NO_NUMERIC_COLLISION",
        )
    if locator is None:
        return SemanticConfirmationSupport(
            support_level="INSUFFICIENT",
            reason="NO_LOCATOR",
        )
    if locator.locator_used == "SOURCE_INDEPENDENT":
        if _same_optional_value(locator.authoritative_value, original_value):
            return SemanticConfirmationSupport(
                support_level="STRONG",
                reason="SOURCE_INDEPENDENT",
            )
        return SemanticConfirmationSupport(
            support_level="INSUFFICIENT",
            reason="SOURCE_INDEPENDENT_DISAGREES",
        )
    locator_strength = _effective_locator_strength(locator)
    if locator_strength == "SHEET_CELL":
        return SemanticConfirmationSupport(
            support_level="STRONG",
            reason="SHEET_CELL",
        )
    if locator_strength == "PAGE_REGION":
        return SemanticConfirmationSupport(
            support_level="STRONG",
            reason="PAGE_REGION",
        )
    if locator.text_context_is_first_pass and locator.text_context:
        return SemanticConfirmationSupport(
            support_level="INSUFFICIENT",
            reason="FIRST_PASS_TEXT_CONTEXT",
        )
    if _text_context_has_explicit_quantity(locator.text_context, original_value):
        return SemanticConfirmationSupport(
            support_level="STRONG",
            reason="LOCAL_EXPLICIT_LABEL",
        )
    if locator.page_number is not None:
        return SemanticConfirmationSupport(
            support_level="INSUFFICIENT",
            reason="PAGE_ONLY",
        )
    if locator.source_id:
        return SemanticConfirmationSupport(
            support_level="INSUFFICIENT",
            reason="SOURCE_ONLY",
        )
    if locator.locator_used == "REFERENCE_CONTEXT":
        return SemanticConfirmationSupport(
            support_level="INSUFFICIENT",
            reason="REFERENCE_ONLY",
        )
    return SemanticConfirmationSupport(
        support_level="INSUFFICIENT",
        reason="NO_LOCAL_SUPPORT",
    )


def evaluate_semantic_correction_support(
    *,
    field: str,
    original_value: str | int | float | None,
    reviewed_value: str | int | float | None,
    locator: SourceLocator | None,
) -> SemanticCorrectionSupport:
    if field != FIELD_QUANTITY:
        return SemanticCorrectionSupport(
            support_level="INSUFFICIENT",
            reason="UNSUPPORTED_FIELD",
        )
    if _same_optional_value(original_value, reviewed_value):
        return SemanticCorrectionSupport(
            support_level="NOT_NEEDED",
            reason="NO_VALUE_CHANGE",
        )
    if locator is None:
        return SemanticCorrectionSupport(
            support_level="INSUFFICIENT",
            reason="NO_LOCATOR",
        )
    if locator.locator_used == "SOURCE_INDEPENDENT":
        if _same_optional_value(locator.authoritative_value, reviewed_value):
            return SemanticCorrectionSupport(
                support_level="STRONG",
                reason="SOURCE_INDEPENDENT",
            )
        return SemanticCorrectionSupport(
            support_level="INSUFFICIENT",
            reason="SOURCE_INDEPENDENT_DISAGREES",
        )
    locator_strength = _effective_locator_strength(locator)
    if locator_strength == "SHEET_CELL":
        return SemanticCorrectionSupport(
            support_level="STRONG",
            reason="SHEET_CELL",
        )
    if locator_strength == "PAGE_REGION":
        return SemanticCorrectionSupport(
            support_level="STRONG",
            reason="PAGE_REGION",
        )
    if locator.text_context_is_first_pass and locator.text_context:
        return SemanticCorrectionSupport(
            support_level="INSUFFICIENT",
            reason="FIRST_PASS_TEXT_CONTEXT",
        )
    if _text_context_has_explicit_quantity(locator.text_context, reviewed_value):
        return SemanticCorrectionSupport(
            support_level="STRONG",
            reason="LOCAL_EXPLICIT_LABEL",
        )
    if locator.page_number is not None:
        return SemanticCorrectionSupport(
            support_level="INSUFFICIENT",
            reason="PAGE_ONLY",
        )
    if locator.source_id:
        return SemanticCorrectionSupport(
            support_level="INSUFFICIENT",
            reason="SOURCE_ONLY",
        )
    if locator.locator_used == "REFERENCE_CONTEXT":
        return SemanticCorrectionSupport(
            support_level="INSUFFICIENT",
            reason="REFERENCE_ONLY",
        )
    return SemanticCorrectionSupport(
        support_level="INSUFFICIENT",
        reason="NO_LOCAL_SUPPORT",
    )


def parse_semantic_field_review_response(
    text: str,
    *,
    temporary_id: str,
    reference: str | None,
    original_value: str | int | float | None,
    source_ids: list[str],
) -> SemanticFieldReview:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return _unresolved_review(
            temporary_id,
            reference,
            original_value,
            source_ids,
            f"Invalid SemanticFieldReview JSON: {exc.msg}.",
        )

    if isinstance(data, list):
        if len(data) != 1:
            return _unresolved_review(
                temporary_id,
                reference,
                original_value,
                source_ids,
                f"SemanticFieldReview response list must contain exactly 1 item; got {len(data)}.",
            )
        data = data[0]

    if not isinstance(data, dict):
        return _unresolved_review(
            temporary_id,
            reference,
            original_value,
            source_ids,
            f"SemanticFieldReview response must be an object; got {type(data).__name__}.",
        )

    try:
        return SemanticFieldReview.model_validate(data)
    except Exception as exc:
        return _unresolved_review(
            temporary_id,
            reference,
            original_value,
            source_ids,
            f"Invalid SemanticFieldReview shape: {type(exc).__name__}.",
        )


def skipped_review_trace(
    element: GeminiElementEnrichment,
    trigger_reason: str,
    locator: SourceLocator | None = None,
) -> SemanticReviewTrace:
    locator = locator or SourceLocator()
    return SemanticReviewTrace(
        reference=element.reference,
        field=FIELD_QUANTITY,
        original_value=element.quantity,
        trigger_reason=trigger_reason,
        review_called=False,
        locator_used=locator.locator_used,
        source_id=locator.source_id,
        page_number=locator.page_number,
        sheet_name=locator.sheet_name,
        cell_range=locator.cell_range,
        region=locator.region,
        region_missing_origin=locator.region_missing_origin,
        locator_strength=locator.locator_strength,
        review_decision="SKIPPED",
        independent_reread=False,
        first_pass_quantity=element.quantity,
        first_pass_quantity_evidence_included=False,
        observed_quantity=None,
        observed_text=None,
        reviewed_value=None,
        final_value=element.quantity,
        confidence=None,
        correction_proposed=False,
        correction_applied=False,
        correction_support="NOT_NEEDED",
        correction_support_reason="NO_REVIEW",
        effective_decision="SKIPPED",
        numeric_collision_detected=False,
        collision_roles=(),
        confirmation_cross_check_called=False,
        confirmation_support="NOT_NEEDED",
        confirmation_support_reason="NO_REVIEW",
        source_ids=tuple(_element_source_ids(element)),
    )


def completed_review_trace(
    element_before: GeminiElementEnrichment,
    element_after: GeminiElementEnrichment,
    trigger_reason: str,
    review: SemanticFieldReview,
    locator: SourceLocator | None = None,
    numeric_trace: NumericElementTrace | None = None,
) -> SemanticReviewTrace:
    locator = locator or SourceLocator()
    support = evaluate_semantic_correction_support(
        field=FIELD_QUANTITY,
        original_value=element_before.quantity,
        reviewed_value=review.reviewed_value,
        locator=locator,
    )
    correction_proposed = (
        review.decision == "CORRECTED"
        and not _same_optional_value(element_before.quantity, review.reviewed_value)
    )
    correction_applied = (
        correction_proposed
        and support.support_level == "STRONG"
        and not _same_optional_value(element_before.quantity, element_after.quantity)
    )
    effective_decision = (
        "AMBIGUOUS"
        if correction_proposed and not correction_applied
        else review.decision
    )
    collision = quantity_numeric_collision_summary(element_before.quantity, numeric_trace)
    confirmation_support = evaluate_confirmed_quantity_support(
        original_value=element_before.quantity,
        locator=locator,
        numeric_trace=numeric_trace,
    )
    confirmation_cross_check_called = should_cross_check_confirmed_quantity(
        review,
        numeric_trace,
    )
    if (
        confirmation_cross_check_called
        and confirmation_support.support_level == "INSUFFICIENT"
    ):
        effective_decision = "AMBIGUOUS"
    return SemanticReviewTrace(
        reference=element_before.reference,
        field=FIELD_QUANTITY,
        original_value=element_before.quantity,
        trigger_reason=trigger_reason,
        review_called=True,
        locator_used=locator.locator_used,
        source_id=locator.source_id,
        page_number=locator.page_number,
        sheet_name=locator.sheet_name,
        cell_range=locator.cell_range,
        region=locator.region,
        region_missing_origin=locator.region_missing_origin,
        locator_strength=locator.locator_strength,
        review_decision=review.decision,
        independent_reread=True,
        first_pass_quantity=element_before.quantity,
        first_pass_quantity_evidence_included=False,
        observed_quantity=review.observed_quantity,
        observed_text=review.observed_text,
        reviewed_value=review.reviewed_value,
        final_value=element_after.quantity,
        confidence=review.confidence,
        correction_proposed=correction_proposed,
        correction_applied=correction_applied,
        correction_support=support.support_level,
        correction_support_reason=support.reason,
        effective_decision=effective_decision,
        numeric_collision_detected=collision.detected,
        collision_roles=tuple(collision.roles),
        confirmation_cross_check_called=confirmation_cross_check_called,
        confirmation_support=confirmation_support.support_level,
        confirmation_support_reason=confirmation_support.reason,
        source_ids=tuple(review.source_ids),
    )


def _replace_quantity(
    element: GeminiElementEnrichment,
    quantity: str | int | float | None,
    status: ExtractionStatus,
    confidence: float | None,
    marker: str,
    reason: str,
) -> GeminiElementEnrichment:
    missing = list(element.missing_or_unknown)
    if marker not in missing:
        missing.append(marker)
    return element.model_copy(
        update={
            "quantity": quantity,
            "quantity_status": status,
            "quantity_confidence": confidence
            if confidence is not None
            else (
                element.quantity_confidence
                if element.quantity_confidence is not None
                else element.confidence
            ),
            "quantity_notes": _append_note(element.quantity_notes, f"{marker}: {reason}"),
            "missing_or_unknown": missing,
            "notes": _append_note(element.notes, f"{marker}: {reason}"),
        }
    )


def _unresolved_review(
    temporary_id: str,
    reference: str | None,
    original_value: str | int | float | None,
    source_ids: list[str],
    reason: str,
) -> SemanticFieldReview:
    return SemanticFieldReview(
        temporary_id=temporary_id,
        reference=reference,
        original_value=original_value,
        reviewed_value=None,
        decision="UNRESOLVED",
        reason=reason,
        source_ids=source_ids,
    )


def _locator_from_grounding_candidate(
    candidate,
    *,
    locator_used: str,
    locator_confidence: float,
) -> SourceLocator:
    return SourceLocator(
        source_id=candidate.source_id,
        source_file_name=candidate.source_file_name,
        page_number=candidate.page_number,
        sheet_name=candidate.sheet_name,
        cell_range=candidate.cell_range,
        region=candidate.region,
        text_context=candidate.raw_text,
        authoritative_value=candidate.value,
        locator_confidence=locator_confidence,
        locator_used=locator_used,
        locator_strength=_locator_strength(
            source_id=candidate.source_id,
            page_number=candidate.page_number,
            sheet_name=candidate.sheet_name,
            cell_range=candidate.cell_range,
            region=candidate.region,
        ),
        region_missing_origin=_region_missing_origin(
            source_id=candidate.source_id,
            page_number=candidate.page_number,
            region=candidate.region,
        ),
    )


def _ranked_numeric_candidates(numeric_trace: NumericElementTrace):
    candidates = list(numeric_trace.candidates)

    def score(candidate) -> int:
        value = 0
        if candidate.semantic_role == "QUANTITY":
            value += 30
        if candidate.region is not None:
            value += 20
        if candidate.cell_range is not None:
            value += 20
        if candidate.page_number is not None:
            value += 10
        if candidate.source_id is not None:
            value += 5
        if candidate.evidence_text is not None:
            value += 3
        return value

    return sorted(candidates, key=score, reverse=True)


def _suspicious_role_values(candidate) -> list[int | float]:
    if candidate.semantic_role not in _SUSPICIOUS_ROLES:
        return []
    if candidate.semantic_role == "LEVEL_RANGE":
        return _level_range_values(candidate.value)
    value = _numeric_value(candidate.value)
    return [] if value is None else [value]


def _level_range_values(value: str | int | float | None) -> list[int | float]:
    if not isinstance(value, str):
        numeric = _numeric_value(value)
        return [] if numeric is None else [numeric]
    parts = value.replace(" ", "").split("-", 1)
    if len(parts) != 2:
        numeric = _numeric_value(value)
        return [] if numeric is None else [numeric]
    start = _numeric_value(parts[0])
    end = _numeric_value(parts[1])
    values = [item for item in (start, end) if item is not None]
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        values.append(end - start + 1)
    return values


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _same_element_region_for_candidate(
    element: GeminiElementEnrichment,
    *,
    source_id: str | None,
    page_number: int | None,
) -> Region | None:
    if source_id is None:
        return None
    for evidence in element.evidence:
        if evidence.region is None:
            continue
        if evidence.source_id != source_id:
            continue
        if page_number is not None and evidence.page_number != page_number:
            continue
        return evidence.region
    return None


def _locator_strength(
    *,
    source_id: str | None,
    page_number: int | None,
    sheet_name: str | None,
    cell_range: str | None,
    region: Region | None,
) -> str:
    if source_id and sheet_name and cell_range:
        return "SHEET_CELL"
    if source_id and region is not None:
        return "PAGE_REGION"
    if source_id and page_number is not None:
        return "PAGE_ONLY"
    if source_id:
        return "SOURCE_ONLY"
    return "NONE"


def _region_missing_origin(
    *,
    source_id: str | None,
    page_number: int | None,
    region: Region | None,
) -> str | None:
    if region is not None:
        return None
    if source_id or page_number is not None:
        return "MODEL_NOT_PROVIDED"
    return "UNKNOWN"


def _effective_locator_strength(locator: SourceLocator) -> str:
    if locator.locator_strength != "NONE":
        return locator.locator_strength
    return _locator_strength(
        source_id=locator.source_id,
        page_number=locator.page_number,
        sheet_name=locator.sheet_name,
        cell_range=locator.cell_range,
        region=locator.region,
    )


def _capped_uncertain_confidence(confidence: float | None, maximum: float) -> float | None:
    if confidence is None:
        return None
    return min(confidence, maximum)


def _trigger_for_role(role: str) -> str:
    if role in {"LEVEL", "FLOOR", "ITEM_NUMBER"}:
        return TRIGGER_QUANTITY_EQUALS_LEVEL
    if role in {"LEVEL_RANGE", "REPETITION_COUNT"}:
        return TRIGGER_QUANTITY_EQUALS_REPETITION_COUNT
    if role == "COMPONENT_COUNT":
        return TRIGGER_QUANTITY_EQUALS_COMPONENT_COUNT
    if role == "PANEL_COUNT":
        return TRIGGER_QUANTITY_EQUALS_PANEL_COUNT
    return TRIGGER_MODEL_UNVERIFIED


def _element_source_ids(element: GeminiElementEnrichment) -> list[str]:
    values: list[str] = []
    for evidence in element.evidence:
        if evidence.source_id and evidence.source_id not in values:
            values.append(evidence.source_id)
    return values


def _numeric_value(value: str | int | float | None) -> int | float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    try:
        parsed = float(value.replace(",", "."))
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _same_number(left: int | float, right: int | float) -> bool:
    return abs(float(left) - float(right)) < 0.000001


def _same_optional_value(
    left: str | int | float | None,
    right: str | int | float | None,
) -> bool:
    left_number = _numeric_value(left)
    right_number = _numeric_value(right)
    if left_number is not None and right_number is not None:
        return _same_number(left_number, right_number)
    return left == right


def _text_context_has_explicit_quantity(
    text: str | None,
    reviewed_value: str | int | float | None,
) -> bool:
    if not text:
        return False
    reviewed_number = _numeric_value(reviewed_value)
    if reviewed_number is None:
        return False
    for match in _QUANTITY_RE.finditer(text):
        match_number = _numeric_value(match.group(1))
        if match_number is not None and _same_number(match_number, reviewed_number):
            return True
    return False


_NUMBER = r"\d+(?:[.,]\d+)?"
_QUANTITY_RE = re.compile(
    rf"\b(?:cantidad(?:\s+total)?|cant\.?|cnt|qty|unidades|und)\s*:?\s*({_NUMBER})\b",
    flags=re.IGNORECASE,
)


def _append_note(current: str | None, note: str) -> str:
    if not current:
        return note
    if note in current:
        return current
    return f"{current}\n{note}"
