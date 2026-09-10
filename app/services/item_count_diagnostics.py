from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.services.inventory_reconciliation import InventoryDecision
from app.services.inventory_trace import InventoryDebugTrace, InventoryElementTrace

MISSING_AT_DISCOVERY = "MISSING_AT_DISCOVERY"
DROPPED_BY_SCOPE = "DROPPED_BY_SCOPE"
LOST_DURING_ENRICHMENT = "LOST_DURING_ENRICHMENT"
LOST_DURING_MERGE = "LOST_DURING_MERGE"
MERGED_IN_RECONCILIATION = "MERGED_IN_RECONCILIATION"
DROPPED_AS_ORPHAN = "DROPPED_AS_ORPHAN"
DROPPED_BY_MAPPER = "DROPPED_BY_MAPPER"
UNKNOWN_LOSS_STAGE = "UNKNOWN_LOSS_STAGE"

DUPLICATE_DISCOVERY = "DUPLICATE_DISCOVERY"
DUPLICATE_ENRICHMENT = "DUPLICATE_ENRICHMENT"
SAME_ITEM_MULTIPLE_CONTEXTS = "SAME_ITEM_MULTIPLE_CONTEXTS"
NON_COMMERCIAL_ITEM = "NON_COMMERCIAL_ITEM"
ORPHAN_WITH_TECHNICAL_SUPPORT = "ORPHAN_WITH_TECHNICAL_SUPPORT"
RECONCILIATION_UNDERMERGE = "RECONCILIATION_UNDERMERGE"
UNEXPECTED_FINAL_ITEM = "UNEXPECTED_FINAL_ITEM"

_DISCOVERY = "DISCOVERY"
_MERGED = "MERGED_ENRICHMENT"
_PRE_RECONCILIATION = "PRE_RECONCILIATION"
_POST_RECONCILIATION = "POST_RECONCILIATION"
_FINAL = "FINAL_REQUIREMENT_EXTRACTION"
_FORMAL_REFERENCE_RE = re.compile(r"^([A-Z]{1,4})[\s._-]*(\d{1,4})([A-Z]?)$")


@dataclass(frozen=True)
class ItemCountMetrics:
    expected_count: int
    actual_count: int
    missing_count: int
    unexpected_count: int
    duplicate_count: int
    precision: float
    recall: float
    f1: float
    item_count_error_percent: float
    duplicate_rate: float
    missing_item_rate: float


@dataclass(frozen=True)
class PipelineStageCounts:
    discovered: int = 0
    scoped: int | None = None
    enriched: int = 0
    merged: int = 0
    pre_reconciliation: int = 0
    post_reconciliation: int = 0
    final: int = 0
    enrichment_batches: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MissingItemDiagnostic:
    identity: str
    reason: str
    last_seen_stage: str | None


@dataclass(frozen=True)
class ExtraItemDiagnostic:
    identity: str
    reason: str
    stage: str


@dataclass(frozen=True)
class ItemCountDiagnosticReport:
    case_name: str
    expected_items: tuple[str, ...]
    actual_items: tuple[str, ...]
    missing: tuple[MissingItemDiagnostic, ...]
    unexpected: tuple[ExtraItemDiagnostic, ...]
    duplicates: tuple[ExtraItemDiagnostic, ...]
    metrics: ItemCountMetrics
    stage_counts: PipelineStageCounts


def build_item_count_diagnostic_report(
    *,
    case_name: str,
    expected_items: list[str] | tuple[str, ...],
    trace: InventoryDebugTrace,
    reconciliation_decisions: list[InventoryDecision] | None = None,
    scoped_count: int | None = None,
) -> ItemCountDiagnosticReport:
    expected = tuple(_normalize_identity(item) for item in expected_items)
    actual = tuple(_stage_identities(trace, _FINAL))
    expected_counter = Counter(expected)
    actual_counter = Counter(actual)

    missing = tuple(
        _missing_diagnostic(identity, trace, reconciliation_decisions or [])
        for identity in expected_counter
        if actual_counter[identity] < expected_counter[identity]
    )
    unexpected = tuple(
        ExtraItemDiagnostic(identity, _unexpected_reason(identity, trace), _FINAL)
        for identity in actual_counter
        if actual_counter[identity] > expected_counter[identity]
    )
    duplicates = tuple(_duplicate_diagnostics(trace))
    metrics = _metrics(expected_counter, actual_counter)

    return ItemCountDiagnosticReport(
        case_name=case_name,
        expected_items=expected,
        actual_items=actual,
        missing=missing,
        unexpected=unexpected,
        duplicates=duplicates,
        metrics=metrics,
        stage_counts=pipeline_stage_counts(trace, scoped_count=scoped_count),
    )


def pipeline_stage_counts(
    trace: InventoryDebugTrace,
    *,
    scoped_count: int | None = None,
) -> PipelineStageCounts:
    enrichment_batches = {
        stage.stage: stage.count
        for stage in trace.stages
        if stage.stage.startswith("ENRICHMENT_BATCH_")
    }
    return PipelineStageCounts(
        discovered=_stage_count(trace, _DISCOVERY),
        scoped=scoped_count,
        enriched=sum(enrichment_batches.values()),
        merged=_stage_count(trace, _MERGED),
        pre_reconciliation=_stage_count(trace, _PRE_RECONCILIATION),
        post_reconciliation=_stage_count(trace, _POST_RECONCILIATION),
        final=_stage_count(trace, _FINAL),
        enrichment_batches=enrichment_batches,
    )


def _metrics(
    expected_counter: Counter[str],
    actual_counter: Counter[str],
) -> ItemCountMetrics:
    expected_total = expected_counter.total()
    actual_total = actual_counter.total()
    true_positives = sum(
        min(expected_counter[identity], actual_counter[identity]) for identity in expected_counter
    )
    missing_count = max(expected_total - true_positives, 0)
    unexpected_count = sum(
        max(actual_counter[identity] - expected_counter[identity], 0) for identity in actual_counter
    )
    duplicate_count = sum(max(count - 1, 0) for count in actual_counter.values())
    precision = (
        true_positives / actual_total if actual_total else (1.0 if expected_total == 0 else 0.0)
    )
    recall = true_positives / expected_total if expected_total else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    count_error = abs(actual_total - expected_total) / expected_total if expected_total else 0.0
    return ItemCountMetrics(
        expected_count=expected_total,
        actual_count=actual_total,
        missing_count=missing_count,
        unexpected_count=unexpected_count,
        duplicate_count=duplicate_count,
        precision=precision,
        recall=recall,
        f1=f1,
        item_count_error_percent=count_error * 100,
        duplicate_rate=duplicate_count / actual_total if actual_total else 0.0,
        missing_item_rate=missing_count / expected_total if expected_total else 0.0,
    )


def _missing_diagnostic(
    identity: str,
    trace: InventoryDebugTrace,
    reconciliation_decisions: list[InventoryDecision],
) -> MissingItemDiagnostic:
    if not _identity_seen(trace, _DISCOVERY, identity):
        return MissingItemDiagnostic(identity, MISSING_AT_DISCOVERY, None)
    if not _identity_seen_any_enrichment_batch(trace, identity):
        return MissingItemDiagnostic(identity, DROPPED_BY_SCOPE, _DISCOVERY)
    if not _identity_seen(trace, _MERGED, identity):
        return MissingItemDiagnostic(identity, LOST_DURING_MERGE, _last_seen_stage(trace, identity))
    if not _identity_seen(trace, _PRE_RECONCILIATION, identity):
        return MissingItemDiagnostic(
            identity,
            LOST_DURING_ENRICHMENT,
            _last_seen_stage(trace, identity),
        )
    if not _identity_seen(trace, _POST_RECONCILIATION, identity):
        reason = _reconciliation_loss_reason(identity, reconciliation_decisions)
        return MissingItemDiagnostic(identity, reason, _PRE_RECONCILIATION)
    if not _identity_seen(trace, _FINAL, identity):
        return MissingItemDiagnostic(identity, DROPPED_BY_MAPPER, _POST_RECONCILIATION)
    return MissingItemDiagnostic(identity, UNKNOWN_LOSS_STAGE, _last_seen_stage(trace, identity))


def _reconciliation_loss_reason(
    identity: str,
    decisions: list[InventoryDecision],
) -> str:
    for decision in decisions:
        identities = {_normalize_identity(value) for value in decision.temporary_ids}
        if decision.reference:
            identities.add(_normalize_identity(decision.reference))
        identities.update(_normalize_identity(value) for value in decision.losing_temporary_ids)
        if identity not in identities:
            continue
        if decision.action == "DROP_AS_NON_COMMERCIAL":
            return DROPPED_AS_ORPHAN
        if decision.action == "MERGE":
            return MERGED_IN_RECONCILIATION
    return UNKNOWN_LOSS_STAGE


def _unexpected_reason(identity: str, trace: InventoryDebugTrace) -> str:
    if _is_duplicate_in_stage(trace, _FINAL, identity):
        return RECONCILIATION_UNDERMERGE
    if _identity_seen(trace, _POST_RECONCILIATION, identity):
        return ORPHAN_WITH_TECHNICAL_SUPPORT
    return UNEXPECTED_FINAL_ITEM


def _duplicate_diagnostics(trace: InventoryDebugTrace) -> list[ExtraItemDiagnostic]:
    diagnostics: list[ExtraItemDiagnostic] = []
    for stage_name, reason in (
        (_DISCOVERY, DUPLICATE_DISCOVERY),
        ("ENRICHMENT", DUPLICATE_ENRICHMENT),
        (_FINAL, RECONCILIATION_UNDERMERGE),
    ):
        identities = (
            _all_enrichment_identities(trace)
            if stage_name == "ENRICHMENT"
            else _stage_identities(trace, stage_name)
        )
        for identity, count in Counter(identities).items():
            if count > 1:
                diagnostics.append(ExtraItemDiagnostic(identity, reason, stage_name))
    return diagnostics


def _stage_count(trace: InventoryDebugTrace, stage_name: str) -> int:
    stage = _stage(trace, stage_name)
    return stage.count if stage is not None else 0


def _stage_identities(trace: InventoryDebugTrace, stage_name: str) -> list[str]:
    stage = _stage(trace, stage_name)
    if stage is None:
        return []
    return [_element_identity(element) for element in stage.elements]


def _all_enrichment_identities(trace: InventoryDebugTrace) -> list[str]:
    values: list[str] = []
    for stage in trace.stages:
        if stage.stage.startswith("ENRICHMENT_BATCH_"):
            values.extend(_element_identity(element) for element in stage.elements)
    return values


def _identity_seen(trace: InventoryDebugTrace, stage_name: str, identity: str) -> bool:
    return identity in _stage_identities(trace, stage_name)


def _identity_seen_any_enrichment_batch(trace: InventoryDebugTrace, identity: str) -> bool:
    return identity in _all_enrichment_identities(trace)


def _is_duplicate_in_stage(trace: InventoryDebugTrace, stage_name: str, identity: str) -> bool:
    return Counter(_stage_identities(trace, stage_name))[identity] > 1


def _last_seen_stage(trace: InventoryDebugTrace, identity: str) -> str | None:
    last_seen = None
    for stage in trace.stages:
        identities = [_element_identity(element) for element in stage.elements]
        if identity in identities:
            last_seen = stage.stage
    return last_seen


def _stage(trace: InventoryDebugTrace, stage_name: str):
    for stage in trace.stages:
        if stage.stage == stage_name:
            return stage
    return None


def _element_identity(element: InventoryElementTrace) -> str:
    for value in (element.reference, element.temporary_id, element.id, element.description):
        if value:
            return _normalize_identity(str(value))
    return "UNKNOWN_ITEM"


def _normalize_identity(value: str) -> str:
    normalized = value.strip().upper()
    match = _FORMAL_REFERENCE_RE.fullmatch(normalized)
    if match is None:
        return normalized
    prefix, number, suffix = match.groups()
    return f"{prefix}-{int(number):02d}{suffix}"
