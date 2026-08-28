from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.models.common import ExtractionStatus
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentResult,
)

ORPHAN_REFERENCE_REASON = "INVENTORY_ORPHAN_REFERENCE_IGNORED"
DUPLICATE_REFERENCE_REASON = "INVENTORY_REFERENCE_DUPLICATE_MERGED"
SOURCE_CONFLICT_REASON = "INVENTORY_SOURCE_CONFLICT_REQUIRES_REVIEW"


@dataclass(frozen=True)
class InventoryDecision:
    action: str
    reason: str
    temporary_ids: tuple[str, ...]
    reference: str | None


def reconcile_inventory_candidates(
    enrichment: GeminiEnrichmentResult,
) -> tuple[GeminiEnrichmentResult, list[InventoryDecision]]:
    """Resolve AI2 enriched candidates into commercial inventory elements.

    Discovery intentionally over-collects potential references. This pass prevents an
    isolated drawing/text tag from becoming a commercial item unless it has physical or
    commercial support. It also merges canonical duplicate references like V-1/V-01.
    """

    decisions: list[InventoryDecision] = []
    kept: list[GeminiElementEnrichment] = []
    warnings = list(enrichment.warnings)
    groups: dict[str, list[GeminiElementEnrichment]] = defaultdict(list)
    passthrough: list[GeminiElementEnrichment] = []

    for element in enrichment.elements:
        if _is_orphan_reference(element):
            decisions.append(
                InventoryDecision(
                    "DROP_AS_NON_COMMERCIAL",
                    ORPHAN_REFERENCE_REASON,
                    (element.temporary_id,),
                    element.reference,
                )
            )
            warnings.append(
                f"{ORPHAN_REFERENCE_REASON}: ignored isolated reference "
                f"{element.reference!r} from {element.temporary_id!r}."
            )
            continue

        canonical = _canonical_reference(element.reference)
        if canonical is None:
            passthrough.append(element)
            decisions.append(
                InventoryDecision(
                    "KEEP",
                    "INVENTORY_CANDIDATE_HAS_COMMERCIAL_SUPPORT",
                    (element.temporary_id,),
                    element.reference,
                )
            )
            continue

        groups[canonical].append(element)

    for canonical, items in groups.items():
        if len(items) == 1:
            kept.append(items[0])
            decisions.append(
                InventoryDecision(
                    "KEEP",
                    "INVENTORY_CANDIDATE_HAS_COMMERCIAL_SUPPORT",
                    (items[0].temporary_id,),
                    items[0].reference,
                )
            )
            continue

        merged, merge_warnings = _merge_reference_group(canonical, items)
        kept.append(merged)
        warnings.extend(merge_warnings)
        decisions.append(
            InventoryDecision(
                "MERGE",
                DUPLICATE_REFERENCE_REASON,
                tuple(item.temporary_id for item in items),
                canonical,
            )
        )

    kept.extend(passthrough)
    result = GeminiEnrichmentResult(
        elements=kept,
        warnings=warnings,
        usage=enrichment.usage,
    )
    return result, decisions


def _is_orphan_reference(element: GeminiElementEnrichment) -> bool:
    if not _text(element.reference):
        return False

    return _commercial_support_score(element) == 0


def _commercial_support_score(element: GeminiElementEnrichment) -> int:
    score = 0
    if _has_quantity(element):
        score += 1
    if _has_measurements(element):
        score += 1
    if _text(element.functional_type_raw) or _text(element.operation_raw):
        score += 1
    if _text(element.geometry_type_raw) or _text(element.geometry_raw):
        score += 1
    if _text(element.configuration_raw) or element.panel_count is not None:
        score += 1
    if element.glass:
        score += 1
    if element.profiles:
        score += 1
    if _text(element.finish_raw):
        score += 1
    if element.components:
        score += 1
    return score


def _has_quantity(element: GeminiElementEnrichment) -> bool:
    value = element.quantity
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return value > 0
    return True


def _has_measurements(element: GeminiElementEnrichment) -> bool:
    return any(
        measurement.value is not None or _text(measurement.text) or _text(measurement.raw_label)
        for measurement in element.measurements
    )


def _canonical_reference(reference: str | None) -> str | None:
    if not reference:
        return None
    value = reference.strip().upper()
    match = re.fullmatch(r"([A-Z]{1,4})[\s._-]*(\d{1,4})([A-Z]?)", value)
    if match is None:
        return value if value else None
    prefix, number, suffix = match.groups()
    return f"{prefix}-{int(number):02d}{suffix}"


def _merge_reference_group(
    canonical: str,
    items: list[GeminiElementEnrichment],
) -> tuple[GeminiElementEnrichment, list[str]]:
    ordered = sorted(items, key=lambda item: _commercial_support_score(item), reverse=True)
    base = ordered[0].model_copy(deep=True)
    warnings = [
        f"{DUPLICATE_REFERENCE_REASON}: merged {len(items)} candidates for {canonical}."
    ]
    conflicts: list[str] = []

    base.reference = canonical
    for item in ordered[1:]:
        _fill_scalar(base, item, "name", conflicts)
        _fill_scalar(base, item, "category_raw", conflicts)
        _fill_scalar(base, item, "description", conflicts)
        _fill_scalar(base, item, "quantity", conflicts)
        _fill_scalar(base, item, "functional_type_raw", conflicts)
        _fill_scalar(base, item, "operation_raw", conflicts)
        _fill_scalar(base, item, "geometry_type_raw", conflicts)
        _fill_scalar(base, item, "geometry_raw", conflicts)
        _fill_scalar(base, item, "configuration_raw", conflicts)
        _fill_scalar(base, item, "finish_raw", conflicts)
        _extend_unique(base.measurements, item.measurements)
        _extend_unique(base.glass, item.glass)
        _extend_unique(base.materials, item.materials)
        _extend_unique(base.profiles, item.profiles)
        _extend_unique(base.accessories, item.accessories)
        _extend_unique(base.components, item.components)
        _extend_unique(base.evidence, item.evidence)
        _extend_text(base.evidence_notes, item.evidence_notes)
        _extend_text(base.missing_or_unknown, item.missing_or_unknown)
        if _text(item.notes):
            base.notes = _join_notes(base.notes, item.notes)

    if conflicts:
        base.status = ExtractionStatus.AMBIGUOUS
        _extend_text(base.missing_or_unknown, [SOURCE_CONFLICT_REASON])
        warnings.append(
            f"{SOURCE_CONFLICT_REASON}: {canonical} has conflicting fields "
            f"{', '.join(sorted(set(conflicts)))}."
        )

    return base, warnings


def _fill_scalar(
    target: GeminiElementEnrichment,
    source: GeminiElementEnrichment,
    field: str,
    conflicts: list[str],
) -> None:
    current = getattr(target, field)
    incoming = getattr(source, field)
    if _is_empty(current) and not _is_empty(incoming):
        setattr(target, field, incoming)
        return
    if not _is_empty(current) and not _is_empty(incoming) and current != incoming:
        conflicts.append(field)


def _extend_unique(target: list, incoming: list) -> None:
    existing = {_fingerprint(item) for item in target}
    for item in incoming:
        fingerprint = _fingerprint(item)
        if fingerprint not in existing:
            target.append(item)
            existing.add(fingerprint)


def _extend_text(target: list[str], incoming: list[str]) -> None:
    existing = set(target)
    for value in incoming:
        if value not in existing:
            target.append(value)
            existing.add(value)


def _fingerprint(value) -> str:
    if isinstance(value, GeminiEnrichmentEvidenceNote):
        return "|".join(
            str(part or "")
            for part in (
                value.source_id,
                value.text,
                value.page_number,
                value.sheet_name,
                value.cell_range,
                value.visual_description,
                value.notes,
            )
        )
    return value.model_dump_json() if hasattr(value, "model_dump_json") else repr(value)


def _join_notes(current: str | None, incoming: str | None) -> str | None:
    if not current:
        return incoming
    if not incoming or incoming in current:
        return current
    return f"{current}\n{incoming}"


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == []


def _text(value: str | None) -> bool:
    return bool(value and value.strip())
