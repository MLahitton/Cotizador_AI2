from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.models.common import ExtractionStatus
from app.models.gemini_enrichment import (
    GeminiElementEnrichment,
    GeminiEnrichmentEvidenceNote,
    GeminiEnrichmentGlass,
    GeminiEnrichmentNamedItem,
    GeminiEnrichmentResult,
)

ORPHAN_REFERENCE_REASON = "INVENTORY_ORPHAN_REFERENCE_IGNORED"
DUPLICATE_REFERENCE_REASON = "INVENTORY_REFERENCE_DUPLICATE_MERGED"
SOURCE_CONFLICT_REASON = "INVENTORY_SOURCE_CONFLICT_REQUIRES_REVIEW"
FORMAL_REFERENCE = "FORMAL_REFERENCE"
CONTEXT_LABEL = "CONTEXT_LABEL"
UNKNOWN_REFERENCE = "UNKNOWN_REFERENCE"
CONTEXT_LABEL_NOT_IDENTITY_REASON = "INVENTORY_CONTEXT_LABEL_NOT_IDENTITY"
INSUFFICIENT_IDENTITY_REASON = "INVENTORY_INSUFFICIENT_IDENTITY_EVIDENCE"
DIFFERENT_CONTEXT_REASON = "INVENTORY_REFERENCE_CONTEXT_DISTINCT"
SAME_CONTEXT_REASON = "INVENTORY_REFERENCE_SAME_CONTEXT_MERGED"
CONTEXT_INCOMPLETE_REASON = "INVENTORY_REFERENCE_CONTEXT_INCOMPLETE_REQUIRES_REVIEW"
PROFILE_EXPLICIT_CONFLICT = "PROFILE_EXPLICIT_CONFLICT"
GLASS_EXPLICIT_CONFLICT = "GLASS_EXPLICIT_CONFLICT"
GLASS_SCOPE_ITEM_LOCAL = "ITEM_LOCAL"
GLASS_SCOPE_SECTION_LEVEL = "SECTION_LEVEL"
GLASS_SCOPE_DOCUMENT_GENERAL = "DOCUMENT_GENERAL"
GLASS_SCOPE_UNKNOWN = "UNKNOWN"

_FORMAL_REFERENCE_RE = re.compile(r"([A-Z]{1,4})[\s._-]*(\d{1,4})([A-Z]?)")
_CONTEXT_SEPARATOR_RE = re.compile(r"[\s._-]+")
_MENTIONED_REFERENCE_RE = re.compile(r"\b([A-Z]{1,4})[\s._-]+(\d{1,4})([A-Z]?)\b")
_GENERAL_GLASS_SCOPE_RE = re.compile(
    r"\b("
    r"todos|todas|por defecto|aplica a toda|toda la casa|en general|"
    r"salvo indicaci[oó]n contraria|unless otherwise noted|typical|default"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InventoryCandidateSnapshot:
    temporary_id: str
    reference: str | None
    commercial_context: str | None = None
    source_ids: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    quantity: str | int | float | None = None
    support_score: int = 0


@dataclass(frozen=True)
class InventoryDecision:
    action: str
    reason: str
    temporary_ids: tuple[str, ...]
    reference: str | None
    normalized_reference: str | None = None
    commercial_context: str | None = None
    reference_semantics: str | None = None
    winner_temporary_id: str | None = None
    losing_temporary_ids: tuple[str, ...] = ()
    candidates: tuple[InventoryCandidateSnapshot, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GlassScopeResolution:
    scope: str
    reason: str


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
        reference_semantics = _classify_reference_semantics(element.reference)
        if _is_orphan_reference(element):
            decisions.append(
                InventoryDecision(
                    "DROP_AS_NON_COMMERCIAL",
                    ORPHAN_REFERENCE_REASON,
                    (element.temporary_id,),
                    element.reference,
                    normalized_reference=_canonical_reference(element.reference),
                    reference_semantics=reference_semantics,
                    candidates=(_candidate_snapshot(element),),
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
            reason = (
                CONTEXT_LABEL_NOT_IDENTITY_REASON
                if reference_semantics == CONTEXT_LABEL
                else INSUFFICIENT_IDENTITY_REASON
            )
            decisions.append(
                InventoryDecision(
                    "KEEP",
                    reason,
                    (element.temporary_id,),
                    element.reference,
                    normalized_reference=None,
                    reference_semantics=reference_semantics,
                    winner_temporary_id=element.temporary_id,
                    candidates=(_candidate_snapshot(element),),
                )
            )
            continue

        groups[canonical].append(element)

    for canonical, items in groups.items():
        for context, context_items in _group_by_commercial_context(items):
            if context is None and len(context_items) > 1:
                distinct_items = _split_ambiguous_duplicate_reference(
                    context_items, canonical
                )
                if distinct_items is not None:
                    for item in distinct_items:
                        kept.append(item)
                        warnings.extend(_resolution_warnings(item, canonical))
                        decisions.append(
                            InventoryDecision(
                                "KEEP",
                                CONTEXT_INCOMPLETE_REASON,
                                (item.temporary_id,),
                                item.reference,
                                normalized_reference=canonical,
                                commercial_context=_canonical_commercial_context(item),
                                reference_semantics=FORMAL_REFERENCE,
                                winner_temporary_id=item.temporary_id,
                                candidates=(_candidate_snapshot(item),),
                            )
                        )
                    warnings.append(
                        f"{CONTEXT_INCOMPLETE_REASON}: preserved "
                        f"{len(distinct_items)} candidates for {canonical} because "
                        "duplicate identity was ambiguous."
                    )
                    continue
            if len(context_items) == 1:
                item = _resolve_item_preference(context_items[0])
                kept.append(item)
                warnings.extend(_resolution_warnings(item, canonical))
                decisions.append(
                    InventoryDecision(
                        "KEEP",
                        _keep_reason(items, context),
                        (item.temporary_id,),
                        item.reference,
                        normalized_reference=canonical,
                        commercial_context=context,
                        reference_semantics=FORMAL_REFERENCE,
                        winner_temporary_id=item.temporary_id,
                        candidates=(_candidate_snapshot(item),),
                    )
                )
                continue

            reason = SAME_CONTEXT_REASON if context is not None else DUPLICATE_REFERENCE_REASON
            merged, merge_warnings, ordered = _merge_reference_group(
                canonical, context_items, reason
            )
            kept.append(merged)
            warnings.extend(merge_warnings)
            decisions.append(
                InventoryDecision(
                    "MERGE",
                    reason,
                    tuple(item.temporary_id for item in context_items),
                    canonical,
                    normalized_reference=canonical,
                    commercial_context=context,
                    reference_semantics=FORMAL_REFERENCE,
                    winner_temporary_id=ordered[0].temporary_id,
                    losing_temporary_ids=tuple(item.temporary_id for item in ordered[1:]),
                    candidates=tuple(_candidate_snapshot(item) for item in ordered),
                )
            )

    resolved_passthrough: list[GeminiElementEnrichment] = []
    for item in passthrough:
        resolved = _resolve_item_preference(item)
        resolved_passthrough.append(resolved)
        warnings.extend(_resolution_warnings(resolved, item.reference))

    kept.extend(resolved_passthrough)
    result = GeminiEnrichmentResult(
        elements=kept,
        warnings=warnings,
        usage=enrichment.usage,
    )
    return result, decisions



def _split_ambiguous_duplicate_reference(
    items: list[GeminiElementEnrichment],
    canonical: str,
) -> list[GeminiElementEnrichment] | None:
    contexts = [_canonical_commercial_context(item) for item in items]
    has_context = any(context is not None for context in contexts)
    has_missing_context = any(context is None for context in contexts)
    complete_dimensions = [item for item in items if _has_complete_dimensions(item)]
    has_distinct_physical_signature = len(
        {_physical_identity_signature(item) for item in complete_dimensions}
    ) > 1
    all_missing_contexts = not has_context and has_missing_context
    mixed_contexts = has_context and has_missing_context
    physically_distinct_inventory_rows = (
        (mixed_contexts and len(complete_dimensions) == len(items))
        or (all_missing_contexts and len(complete_dimensions) == len(items))
    )

    if not physically_distinct_inventory_rows or not has_distinct_physical_signature:
        return None

    resolved: list[GeminiElementEnrichment] = []
    for item in items:
        updated = item.model_copy(deep=True)
        updated.reference = canonical
        if _canonical_commercial_context(updated) is None:
            updated.status = ExtractionStatus.AMBIGUOUS
            if CONTEXT_INCOMPLETE_REASON not in updated.missing_or_unknown:
                updated.missing_or_unknown.append(CONTEXT_INCOMPLETE_REASON)
        resolved.append(_resolve_item_preference(updated))

    return resolved



def _has_complete_dimensions(item: GeminiElementEnrichment) -> bool:
    types = {
        _normalized_text(measurement.type)
        for measurement in item.measurements
        if measurement.value is not None or _text(measurement.text)
    }
    return "width" in types and "height" in types

def _physical_identity_signature(item: GeminiElementEnrichment) -> tuple[object, ...]:
    dimensions = tuple(
        (
            _normalized_text(measurement.type),
            measurement.value,
            _normalized_text(measurement.unit),
            _normalized_text(measurement.text),
        )
        for measurement in item.measurements
        if measurement.value is not None or _text(measurement.text)
    )
    return (
        dimensions,
        item.panel_count,
        item.movable_panel_count,
        item.fixed_panel_count,
        _normalized_text(item.geometry_type_raw),
        _normalized_text(item.geometry_raw),
    )



def _normalized_text(value: str | None) -> str | None:
    if not _text(value):
        return None
    return value.strip().casefold()

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
    match = _FORMAL_REFERENCE_RE.fullmatch(value)
    if match is None:
        return None
    prefix, number, suffix = match.groups()
    return f"{prefix}-{int(number):02d}{suffix}"


def _classify_reference_semantics(reference: str | None) -> str:
    if not _text(reference):
        return UNKNOWN_REFERENCE
    if _canonical_reference(reference) is not None:
        return FORMAL_REFERENCE
    return CONTEXT_LABEL


def _group_by_commercial_context(
    items: list[GeminiElementEnrichment],
) -> list[tuple[str | None, list[GeminiElementEnrichment]]]:
    contexts = [_canonical_commercial_context(item) for item in items]
    if any(context is None for context in contexts):
        return [(None, items)]

    grouped: dict[str, list[GeminiElementEnrichment]] = defaultdict(list)
    for item, context in zip(items, contexts, strict=True):
        grouped[context].append(item)

    return list(grouped.items())


def _keep_reason(items: list[GeminiElementEnrichment], context: str | None) -> str:
    if context is not None and len(items) > 1:
        return DIFFERENT_CONTEXT_REASON
    return "INVENTORY_CANDIDATE_HAS_COMMERCIAL_SUPPORT"


def _canonical_commercial_context(element: GeminiElementEnrichment) -> str | None:
    value = element.occurrence_context
    if not _text(value):
        return None

    normalized = _CONTEXT_SEPARATOR_RE.sub("_", value.strip().casefold())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or None


def _merge_reference_group(
    canonical: str,
    items: list[GeminiElementEnrichment],
    reason: str = DUPLICATE_REFERENCE_REASON,
) -> tuple[GeminiElementEnrichment, list[str], list[GeminiElementEnrichment]]:
    ordered = sorted(items, key=lambda item: _commercial_support_score(item), reverse=True)
    base = ordered[0].model_copy(deep=True)
    warnings = [
        f"{reason}: merged {len(items)} candidates for {canonical}."
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

    base = _resolve_item_preference(base)
    warnings.extend(_resolution_warnings(base, canonical))

    return base, warnings, ordered




def _resolve_item_preference(element: GeminiElementEnrichment) -> GeminiElementEnrichment:
    resolved = _resolve_profile_preference(element)
    return _resolve_glass_preference(resolved)


def _resolution_warnings(
    element: GeminiElementEnrichment,
    reference: str | None,
) -> list[str]:
    return [
        *_profile_resolution_warnings(element, reference),
        *_glass_resolution_warnings(element, reference),
    ]


def _resolve_glass_preference(element: GeminiElementEnrichment) -> GeminiElementEnrichment:
    if not element.glass:
        return element

    resolved = element.model_copy(deep=True)
    scoped_glass = [(glass, _classify_glass_scope(resolved, glass)) for glass in resolved.glass]
    sorted_scoped_glass = sorted(scoped_glass, key=lambda item: _glass_sort_key(*item))
    if _has_explicit_glass_conflict(sorted_scoped_glass):
        conflicting_scopes = _conflicting_explicit_glass_scopes(sorted_scoped_glass)
        sorted_glass = [
            glass.model_copy(update={"status": ExtractionStatus.AMBIGUOUS})
            if _is_explicit_principal_glass(glass) and scope.scope in conflicting_scopes
            else glass
            for glass, scope in sorted_scoped_glass
        ]
        resolved.status = ExtractionStatus.AMBIGUOUS
        _extend_text(resolved.missing_or_unknown, [GLASS_EXPLICIT_CONFLICT])
        resolved.notes = _join_notes(
            resolved.notes,
            "Conflicting explicit glass candidates require review.",
        )
    else:
        sorted_glass = [glass for glass, _scope in sorted_scoped_glass]

    resolved.glass = sorted_glass
    return resolved


def _glass_resolution_warnings(
    element: GeminiElementEnrichment,
    reference: str | None,
) -> list[str]:
    if GLASS_EXPLICIT_CONFLICT not in element.missing_or_unknown:
        return []
    label = reference or element.reference or element.temporary_id
    return [f"{GLASS_EXPLICIT_CONFLICT}: {label} has conflicting explicit glass candidates."]


def _glass_sort_key(
    glass: GeminiEnrichmentGlass,
    scope: GlassScopeResolution,
) -> tuple[int, int, int, int, int, str]:
    return (
        -_glass_scope_rank(scope.scope),
        -_glass_status_rank(glass),
        -_glass_principal_rank(glass),
        -_glass_confidence_rank(glass),
        -_glass_completeness_rank(glass),
        _glass_identity(glass),
    )


def _glass_scope_rank(scope: str) -> int:
    if scope == GLASS_SCOPE_ITEM_LOCAL:
        return 4
    if scope == GLASS_SCOPE_SECTION_LEVEL:
        return 3
    if scope == GLASS_SCOPE_DOCUMENT_GENERAL:
        return 2
    return 1


def classify_glass_scope(
    element: GeminiElementEnrichment,
    glass: GeminiEnrichmentGlass,
) -> GlassScopeResolution:
    return _classify_glass_scope(element, glass)


def _classify_glass_scope(
    element: GeminiElementEnrichment,
    glass: GeminiEnrichmentGlass,
) -> GlassScopeResolution:
    corpus = _glass_scope_corpus(element, glass)
    normalized_corpus = _normalize_scope_text(corpus)
    reference = _canonical_reference(element.reference)
    mentioned_references = set(_mentioned_references(corpus))

    if (
        reference
        and reference in mentioned_references
        and not (mentioned_references - {reference})
    ):
        return GlassScopeResolution(
            GLASS_SCOPE_ITEM_LOCAL,
            "Evidence text names the current item reference.",
        )

    if reference and mentioned_references and reference not in mentioned_references:
        return GlassScopeResolution(
            GLASS_SCOPE_UNKNOWN,
            "Evidence text names a different item reference.",
        )

    context = _canonical_commercial_context(element)
    if context and context in normalized_corpus:
        return GlassScopeResolution(
            GLASS_SCOPE_SECTION_LEVEL,
            "Evidence text matches the item occurrence context.",
        )

    if _GENERAL_GLASS_SCOPE_RE.search(corpus):
        return GlassScopeResolution(
            GLASS_SCOPE_DOCUMENT_GENERAL,
            "Evidence text contains a document-general marker.",
        )

    return GlassScopeResolution(
        GLASS_SCOPE_UNKNOWN,
        "No reliable glass evidence scope signal.",
    )


def _glass_scope_corpus(
    element: GeminiElementEnrichment,
    glass: GeminiEnrichmentGlass,
) -> str:
    values = [
        glass.type,
        glass.composition,
        glass.thickness,
        glass.color,
        glass.treatment,
        glass.description,
        glass.notes,
        element.notes,
        *element.evidence_notes,
    ]
    for evidence in element.evidence:
        values.extend(
            [
                evidence.text,
                evidence.visual_description,
                evidence.notes,
                evidence.cell_range,
                evidence.sheet_name,
            ]
        )
    return " ".join(
        value.strip() for value in values if isinstance(value, str) and value.strip()
    )


def _mentioned_references(value: str) -> list[str]:
    references: list[str] = []
    for match in _MENTIONED_REFERENCE_RE.finditer(value.upper()):
        prefix, number, suffix = match.groups()
        references.append(f"{prefix}-{int(number):02d}{suffix}")
    return references


def _normalize_scope_text(value: str) -> str:
    normalized = value.strip().casefold()
    normalized = _CONTEXT_SEPARATOR_RE.sub("_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def _glass_status_rank(glass: GeminiEnrichmentGlass) -> int:
    status = glass.status
    if status == ExtractionStatus.EXPLICIT:
        return 3
    if status == ExtractionStatus.INFERRED:
        return 2
    if status == ExtractionStatus.AMBIGUOUS:
        return 1
    return 0


def _glass_principal_rank(glass: GeminiEnrichmentGlass) -> int:
    if _has_glass_type_or_composition(glass):
        return 3
    if _has_glass_thickness(glass):
        return 2
    if _has_glass_treatment_or_color(glass):
        return 1
    return 0


def _glass_confidence_rank(glass: GeminiEnrichmentGlass) -> int:
    if glass.confidence is None:
        return -1
    return int(glass.confidence * 1000)


def _glass_completeness_rank(glass: GeminiEnrichmentGlass) -> int:
    score = 0
    if _has_glass_type_or_composition(glass):
        score += 3
    if _has_glass_thickness(glass):
        score += 2
    if _has_glass_treatment_or_color(glass):
        score += 1
    if _text(glass.description):
        score += 1
    return score


def _has_explicit_glass_conflict(
    glass_items: list[tuple[GeminiEnrichmentGlass, GlassScopeResolution]],
) -> bool:
    return bool(_conflicting_explicit_glass_scopes(glass_items))


def _conflicting_explicit_glass_scopes(
    glass_items: list[tuple[GeminiEnrichmentGlass, GlassScopeResolution]],
) -> set[str]:
    identities_by_scope: dict[str, set[str]] = defaultdict(set)
    for glass, scope in glass_items:
        if not _is_explicit_principal_glass(glass):
            continue
        identity = _glass_identity(glass)
        if identity:
            identities_by_scope[scope.scope].add(identity)
    return {
        scope
        for scope, identities in identities_by_scope.items()
        if len(identities) > 1
    }


def _is_explicit_principal_glass(glass: GeminiEnrichmentGlass) -> bool:
    return glass.status == ExtractionStatus.EXPLICIT and _has_glass_type_or_composition(glass)


def _has_glass_type_or_composition(glass: GeminiEnrichmentGlass) -> bool:
    return _text(glass.type) or _text(glass.composition)


def _has_glass_thickness(glass: GeminiEnrichmentGlass) -> bool:
    return _text(glass.thickness) or glass.thickness_value is not None


def _has_glass_treatment_or_color(glass: GeminiEnrichmentGlass) -> bool:
    return _text(glass.treatment) or _text(glass.color)


def _glass_identity(glass: GeminiEnrichmentGlass) -> str:
    return "|".join(
        part for part in (
            _normalize_profile_text(glass.type),
            _normalize_profile_text(glass.composition),
            _normalize_profile_text(glass.thickness),
            _normalize_float(glass.thickness_value),
            _normalize_profile_text(glass.thickness_unit),
        ) if part
    )


def _normalize_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:g}"

def _resolve_profile_preference(element: GeminiElementEnrichment) -> GeminiElementEnrichment:
    if not element.profiles:
        return element

    resolved = element.model_copy(deep=True)
    resolved.profiles = sorted(resolved.profiles, key=_profile_sort_key)
    if _has_explicit_profile_conflict(resolved.profiles):
        resolved.status = ExtractionStatus.AMBIGUOUS
        _extend_text(resolved.missing_or_unknown, [PROFILE_EXPLICIT_CONFLICT])
        resolved.notes = _join_notes(
            resolved.notes,
            "Conflicting explicit primary profiles require review.",
        )
    return resolved


def _profile_resolution_warnings(
    element: GeminiElementEnrichment,
    reference: str | None,
) -> list[str]:
    if PROFILE_EXPLICIT_CONFLICT not in element.missing_or_unknown:
        return []
    label = reference or element.reference or element.temporary_id
    return [f"{PROFILE_EXPLICIT_CONFLICT}: {label} has conflicting explicit primary profiles."]


def _profile_sort_key(profile: GeminiEnrichmentNamedItem) -> tuple[int, int, int, str]:
    # Explicit local evidence should be exposed before inferred candidates because
    # downstream currently treats the first coded profile as preferred.
    return (
        -_profile_status_rank(profile),
        -_profile_primary_rank(profile),
        -_profile_confidence_rank(profile),
        _profile_identity(profile),
    )


def _profile_status_rank(profile: GeminiEnrichmentNamedItem) -> int:
    status = profile.status
    if status == ExtractionStatus.EXPLICIT:
        return 3
    if status == ExtractionStatus.INFERRED:
        return 2
    if status == ExtractionStatus.AMBIGUOUS:
        return 1
    return 0


def _profile_primary_rank(profile: GeminiEnrichmentNamedItem) -> int:
    return 1 if _is_primary_profile(profile) else 0


def _profile_confidence_rank(profile: GeminiEnrichmentNamedItem) -> int:
    if profile.confidence is None:
        return -1
    return int(profile.confidence * 1000)


def _has_explicit_profile_conflict(profiles: list[GeminiEnrichmentNamedItem]) -> bool:
    explicit_primary: dict[str, set[str]] = defaultdict(set)
    for profile in profiles:
        if profile.status != ExtractionStatus.EXPLICIT or not _is_primary_profile(profile):
            continue
        identity = _profile_identity(profile)
        if identity:
            explicit_primary[_profile_role_group(profile)].add(identity)
    return any(len(identities) > 1 for identities in explicit_primary.values())


def _is_primary_profile(profile: GeminiEnrichmentNamedItem) -> bool:
    role = _normalize_profile_text(profile.role or profile.type)
    if not role:
        return True
    complementary_markers = {
        "ACCESSORY",
        "ACCESORIO",
        "COMPLEMENTARY",
        "COMPLEMENTARIO",
        "FRAME",
        "HOJA",
        "LEAF",
        "MARCO",
        "RAIL",
        "RIEL",
        "SASH",
    }
    if role in complementary_markers:
        return False
    primary_markers = {
        "MAIN",
        "PERFIL",
        "PRIMARY",
        "PRINCIPAL",
        "PROFILE",
        "SYSTEM",
        "SISTEMA",
    }
    return role in primary_markers


def _profile_role_group(profile: GeminiEnrichmentNamedItem) -> str:
    if _is_primary_profile(profile):
        return "PRIMARY"
    return _normalize_profile_text(profile.role or profile.type)


def _profile_identity(profile: GeminiEnrichmentNamedItem) -> str:
    return _normalize_profile_text(
        profile.code or profile.name or profile.description or profile.type or ""
    )


def _normalize_profile_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip()).upper()

def _candidate_snapshot(element: GeminiElementEnrichment) -> InventoryCandidateSnapshot:
    return InventoryCandidateSnapshot(
        temporary_id=element.temporary_id,
        reference=element.reference,
        commercial_context=_canonical_commercial_context(element),
        source_ids=tuple(_source_ids(element)),
        dimensions=tuple(_dimensions(element)),
        quantity=element.quantity,
        support_score=_commercial_support_score(element),
    )


def _source_ids(element: GeminiElementEnrichment) -> list[str]:
    values: list[str] = []
    for evidence in element.evidence:
        if evidence.source_id and evidence.source_id not in values:
            values.append(evidence.source_id)
    return values


def _dimensions(element: GeminiElementEnrichment) -> list[str]:
    values: list[str] = []
    for measurement in element.measurements:
        label = measurement.type or measurement.raw_label or "unspecified"
        if measurement.value is None:
            values.append(label)
        else:
            values.append(f"{label}={measurement.value}{measurement.unit or ''}")
    return values


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
