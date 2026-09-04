from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

REGION_ALREADY_NORMALIZED = "REGION_ALREADY_NORMALIZED"
REGION_NORMALIZED = "REGION_NORMALIZED"
REGION_NORMALIZED_X2Y2 = "REGION_NORMALIZED_X2Y2"
REGION_DROPPED_INVALID = "REGION_DROPPED_INVALID"
REGION_SOURCE_DIMENSIONS_MISSING = "REGION_SOURCE_DIMENSIONS_MISSING"


@dataclass(frozen=True)
class SourceRegionFrame:
    width: int
    height: int


@dataclass(frozen=True)
class RegionSanitizationEvent:
    action: str
    element_temporary_id: str | None
    evidence_index: int
    source_id: str | None
    original_region: dict[str, Any]
    sanitized_region: dict[str, float] | None = None
    reason: str | None = None


def sanitize_enrichment_regions_json(
    text: str,
    *,
    source_frames: dict[str, SourceRegionFrame] | None = None,
) -> tuple[str, list[RegionSanitizationEvent]]:
    payload = json.loads(text)
    events = sanitize_enrichment_regions_payload(payload, source_frames=source_frames)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), events


def sanitize_enrichment_regions_payload(
    payload: dict[str, Any],
    *,
    source_frames: dict[str, SourceRegionFrame] | None = None,
) -> list[RegionSanitizationEvent]:
    source_frames = source_frames or {}
    events: list[RegionSanitizationEvent] = []
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return events

    for element in elements:
        if not isinstance(element, dict):
            continue
        temporary_id = _optional_str(element.get("temporary_id"))
        evidence_items = element.get("evidence")
        if not isinstance(evidence_items, list):
            continue
        for index, evidence in enumerate(evidence_items, start=1):
            if not isinstance(evidence, dict):
                continue
            region = evidence.get("region")
            if region is None:
                continue
            if not isinstance(region, dict):
                evidence["region"] = None
                events.append(
                    _event(
                        REGION_DROPPED_INVALID,
                        temporary_id,
                        index,
                        evidence,
                        {},
                        reason="region is not an object",
                    )
                )
                continue

            sanitized, event = _sanitize_region(
                region,
                temporary_id=temporary_id,
                evidence_index=index,
                source_id=_optional_str(evidence.get("source_id")),
                source_frames=source_frames,
            )
            if event is not None:
                events.append(event)
            if sanitized is None:
                evidence["region"] = None
            else:
                evidence["region"] = sanitized

    if events:
        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.extend(_warning_message(event) for event in events)

    return events


def _sanitize_region(
    region: dict[str, Any],
    *,
    temporary_id: str | None,
    evidence_index: int,
    source_id: str | None,
    source_frames: dict[str, SourceRegionFrame],
) -> tuple[dict[str, float] | None, RegionSanitizationEvent | None]:
    numeric = _numeric_region(region)
    if numeric is None:
        return None, RegionSanitizationEvent(
            action=REGION_DROPPED_INVALID,
            element_temporary_id=temporary_id,
            evidence_index=evidence_index,
            source_id=source_id,
            original_region=dict(region),
            reason="region fields must be numeric x/y/width/height",
        )

    if _is_valid_normalized_region(numeric):
        return numeric, RegionSanitizationEvent(
            action=REGION_ALREADY_NORMALIZED,
            element_temporary_id=temporary_id,
            evidence_index=evidence_index,
            source_id=source_id,
            original_region=dict(region),
            sanitized_region=numeric,
        )
    if _has_normalized_individual_bounds(numeric):
        return None, RegionSanitizationEvent(
            action=REGION_DROPPED_INVALID,
            element_temporary_id=temporary_id,
            evidence_index=evidence_index,
            source_id=source_id,
            original_region=dict(region),
            reason="normalized region exceeds x/y bounds",
        )

    frame = source_frames.get(source_id or "")
    if frame is None:
        return None, RegionSanitizationEvent(
            action=REGION_SOURCE_DIMENSIONS_MISSING,
            element_temporary_id=temporary_id,
            evidence_index=evidence_index,
            source_id=source_id,
            original_region=dict(region),
            reason="absolute or invalid region without deterministic source dimensions",
        )

    normalized = _normalize_absolute_region(numeric, frame)
    if normalized is not None:
        return normalized, RegionSanitizationEvent(
            action=REGION_NORMALIZED,
            element_temporary_id=temporary_id,
            evidence_index=evidence_index,
            source_id=source_id,
            original_region=dict(region),
            sanitized_region=normalized,
        )

    normalized_x2y2 = _normalize_absolute_x2y2_region(numeric, frame)
    if normalized_x2y2 is not None:
        return normalized_x2y2, RegionSanitizationEvent(
            action=REGION_NORMALIZED_X2Y2,
            element_temporary_id=temporary_id,
            evidence_index=evidence_index,
            source_id=source_id,
            original_region=dict(region),
            sanitized_region=normalized_x2y2,
        )

    return None, RegionSanitizationEvent(
        action=REGION_DROPPED_INVALID,
        element_temporary_id=temporary_id,
        evidence_index=evidence_index,
        source_id=source_id,
        original_region=dict(region),
        reason="region cannot be normalized within source bounds",
    )


def _numeric_region(region: dict[str, Any]) -> dict[str, float] | None:
    try:
        numeric = {
            "x": float(region["x"]),
            "y": float(region["y"]),
            "width": float(region["width"]),
            "height": float(region["height"]),
        }
    except (KeyError, TypeError, ValueError):
        return None

    return numeric


def _is_valid_normalized_region(region: dict[str, float]) -> bool:
    return (
        0 <= region["x"] <= 1
        and 0 <= region["y"] <= 1
        and 0 <= region["width"] <= 1
        and 0 <= region["height"] <= 1
        and region["x"] + region["width"] <= 1
        and region["y"] + region["height"] <= 1
    )


def _has_normalized_individual_bounds(region: dict[str, float]) -> bool:
    return (
        0 <= region["x"] <= 1
        and 0 <= region["y"] <= 1
        and 0 <= region["width"] <= 1
        and 0 <= region["height"] <= 1
    )


def _normalize_absolute_region(
    region: dict[str, float],
    frame: SourceRegionFrame,
) -> dict[str, float] | None:
    if (
        region["x"] < 0
        or region["y"] < 0
        or region["width"] < 0
        or region["height"] < 0
        or region["x"] + region["width"] > frame.width
        or region["y"] + region["height"] > frame.height
    ):
        return None

    normalized = {
        "x": region["x"] / frame.width,
        "y": region["y"] / frame.height,
        "width": region["width"] / frame.width,
        "height": region["height"] / frame.height,
    }
    return normalized if _is_valid_normalized_region(normalized) else None


def _normalize_absolute_x2y2_region(
    region: dict[str, float],
    frame: SourceRegionFrame,
) -> dict[str, float] | None:
    if (
        region["x"] < 0
        or region["y"] < 0
        or region["width"] <= region["x"]
        or region["height"] <= region["y"]
        or region["width"] > frame.width
        or region["height"] > frame.height
    ):
        return None

    normalized = {
        "x": region["x"] / frame.width,
        "y": region["y"] / frame.height,
        "width": (region["width"] - region["x"]) / frame.width,
        "height": (region["height"] - region["y"]) / frame.height,
    }
    return normalized if _is_valid_normalized_region(normalized) else None


def _event(
    action: str,
    temporary_id: str | None,
    evidence_index: int,
    evidence: dict[str, Any],
    original_region: dict[str, Any],
    *,
    reason: str,
) -> RegionSanitizationEvent:
    return RegionSanitizationEvent(
        action=action,
        element_temporary_id=temporary_id,
        evidence_index=evidence_index,
        source_id=_optional_str(evidence.get("source_id")),
        original_region=original_region,
        reason=reason,
    )


def _warning_message(event: RegionSanitizationEvent) -> str:
    context = (
        f"element={event.element_temporary_id!r} evidence_index={event.evidence_index} "
        f"source_id={event.source_id!r}"
    )
    return f"{event.action}: {context}; reason={event.reason or 'ok'}."


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
