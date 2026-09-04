import json

from app.services.region_sanitizer import (
    REGION_ALREADY_NORMALIZED,
    REGION_DROPPED_INVALID,
    REGION_NORMALIZED,
    REGION_NORMALIZED_X2Y2,
    REGION_SOURCE_DIMENSIONS_MISSING,
    SourceRegionFrame,
    sanitize_enrichment_regions_json,
)


def test_region_sanitizer_preserves_already_normalized_region() -> None:
    sanitized_text, events = sanitize_enrichment_regions_json(
        _payload_json({"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4})
    )

    payload = json.loads(sanitized_text)

    assert payload["elements"][0]["evidence"][0]["region"] == {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4,
    }
    assert events[0].action == REGION_ALREADY_NORMALIZED


def test_region_sanitizer_normalizes_absolute_region_with_source_dimensions() -> None:
    sanitized_text, events = sanitize_enrichment_regions_json(
        _payload_json({"x": 50, "y": 100, "width": 200, "height": 300}),
        source_frames={"source-1": SourceRegionFrame(width=1000, height=1000)},
    )

    region = json.loads(sanitized_text)["elements"][0]["evidence"][0]["region"]

    assert region == {"x": 0.05, "y": 0.1, "width": 0.2, "height": 0.3}
    assert events[0].action == REGION_NORMALIZED


def test_region_sanitizer_normalizes_x2y2_only_when_deterministic() -> None:
    sanitized_text, events = sanitize_enrichment_regions_json(
        _payload_json({"x": 48, "y": 210, "width": 885, "height": 580}),
        source_frames={"source-1": SourceRegionFrame(width=900, height=600)},
    )

    region = json.loads(sanitized_text)["elements"][0]["evidence"][0]["region"]

    assert region == {
        "x": 48 / 900,
        "y": 210 / 600,
        "width": (885 - 48) / 900,
        "height": (580 - 210) / 600,
    }
    assert events[0].action == REGION_NORMALIZED_X2Y2


def test_region_sanitizer_drops_absolute_region_without_dimensions() -> None:
    sanitized_text, events = sanitize_enrichment_regions_json(
        _payload_json({"x": 58, "y": 435, "width": 82, "height": 242})
    )

    payload = json.loads(sanitized_text)

    assert payload["elements"][0]["evidence"][0]["region"] is None
    assert events[0].action == REGION_SOURCE_DIMENSIONS_MISSING
    assert payload["elements"][0]["evidence"][0]["text"] == "Detalle visual"
    assert payload["elements"][0]["evidence"][0]["source_id"] == "source-1"
    assert payload["warnings"][0].startswith(REGION_SOURCE_DIMENSIONS_MISSING)


def test_region_sanitizer_drops_invalid_normalized_bounds() -> None:
    sanitized_text, events = sanitize_enrichment_regions_json(
        _payload_json({"x": 0.8, "y": 0.2, "width": 0.3, "height": 0.4}),
        source_frames={"source-1": SourceRegionFrame(width=1000, height=1000)},
    )

    payload = json.loads(sanitized_text)

    assert payload["elements"][0]["evidence"][0]["region"] is None
    assert events[0].action == REGION_DROPPED_INVALID


def _payload_json(region: dict) -> str:
    return json.dumps(
        {
            "elements": [
                {
                    "temporary_id": "item-1",
                    "evidence": [
                        {
                            "source_id": "source-1",
                            "type": "visual",
                            "text": "Detalle visual",
                            "region": region,
                        }
                    ],
                }
            ]
        }
    )
