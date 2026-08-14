import json
import sys
from pathlib import Path
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.providers.gemini_extraction import (  # noqa: E402
    GeminiExtractionDebugCapture,
    GeminiExtractionProvider,
)

PDF_PATH = ROOT_DIR / "manual_tests" / "20260804 EL ESPIGAL_01 Planteamiento CASA 1-2.pdf"
RESULTS_DIR = ROOT_DIR / "manual_tests" / "results"
RAW_OUTPUT_PATH = RESULTS_DIR / "espigal_gemini_raw.json"
GEMINI_OUTPUT_PATH = RESULTS_DIR / "espigal_gemini_extraction.json"
OUTPUT_PATH = RESULTS_DIR / "espigal_requirement_extraction.json"


def main() -> int:
    started_at = perf_counter()

    try:
        provider = GeminiExtractionProvider()
        debug_capture = GeminiExtractionDebugCapture()
        extraction = provider.extract_from_files(
            [PDF_PATH],
            project_id="manual-espigal",
            requirement_id="manual-espigal-001",
            debug_capture=debug_capture,
        )
        elapsed_seconds = perf_counter() - started_at

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if debug_capture.raw_response_text is None:
            raise ValueError("No se capturo response.text crudo de Gemini.")
        if debug_capture.gemini_extraction is None:
            raise ValueError("No se capturo GeminiExtraction validado.")

        RAW_OUTPUT_PATH.write_text(debug_capture.raw_response_text, encoding="utf-8")
        GEMINI_OUTPUT_PATH.write_text(
            debug_capture.gemini_extraction.model_dump_json(indent=2),
            encoding="utf-8",
        )
        OUTPUT_PATH.write_text(
            extraction.model_dump_json(indent=2),
            encoding="utf-8",
        )

        _print_summary(extraction, elapsed_seconds)
        _print_pipeline_counts(extraction, debug_capture)
        _print_elements(extraction.elements)
        print(f"\nraw_json: {RAW_OUTPUT_PATH}")
        print(f"gemini_extraction_json: {GEMINI_OUTPUT_PATH}")
        print(f"\noutput_json: {OUTPUT_PATH}")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _print_summary(extraction, elapsed_seconds: float) -> None:
    metadata = extraction.extraction_metadata
    token_usage = metadata.token_usage

    print("EL ESPIGAL - AI2 REAL EXTRACTION")
    print(f"model: {metadata.model}")
    print(f"sources: {len(extraction.sources)}")
    print(f"elements: {len(extraction.elements)}")
    print(f"conflicts: {len(extraction.conflicts)}")
    print(f"warnings: {len(extraction.warnings)}")
    print(f"elapsed_seconds: {elapsed_seconds:.2f}")
    print(f"input_tokens: {token_usage.input_tokens if token_usage else None}")
    print(f"output_tokens: {token_usage.output_tokens if token_usage else None}")
    print(f"total_tokens: {token_usage.total_tokens if token_usage else None}")


def _print_pipeline_counts(extraction, debug_capture: GeminiExtractionDebugCapture) -> None:
    raw_payload = json.loads(debug_capture.raw_response_text or "{}")
    raw_elements = raw_payload.get("elements")
    raw_element_count = len(raw_elements) if isinstance(raw_elements, list) else 0
    gemini_extraction = debug_capture.gemini_extraction

    print("\nPIPELINE COUNTS")
    print(f"raw_elements: {raw_element_count}")
    print(
        "validated_gemini_elements: "
        f"{len(gemini_extraction.elements) if gemini_extraction else 0}"
    )
    print(f"final_requirement_elements: {len(extraction.elements)}")


def _print_elements(elements) -> None:
    print("\nELEMENTS")
    for element in elements:
        print(f"\n- id: {element.id}")
        print(f"  reference: {_traceable(element.reference)}")
        print(f"  name: {_traceable(element.name)}")
        print(f"  category: {_normalized(element.category)}")
        print(f"  quantity: {_traceable(element.quantity)}")
        print(f"  measurements: {_measurements(element.measurements)}")
        print(f"  configuration: {_configuration(element.configuration)}")
        print(f"  glass: {_glass(element.glass)}")
        print(f"  profiles: {_profiles(element.profiles)}")


def _traceable(value) -> str:
    if value is None:
        return "None"
    return f"value={value.value!r}, status={value.status}, confidence={value.confidence}"


def _normalized(value) -> str:
    if value is None:
        return "None"
    return (
        f"raw={value.raw!r}, normalized={value.normalized!r}, "
        f"status={value.status}, confidence={value.confidence}"
    )


def _measurements(measurements) -> str:
    if not measurements:
        return "[]"
    return "; ".join(
        (
            f"{measurement.type}: value={measurement.value!r} {measurement.unit or ''}, "
            f"raw_label={measurement.raw_label!r}, status={measurement.status}, "
            f"confidence={measurement.confidence}"
        )
        for measurement in measurements
    )


def _configuration(configuration) -> str:
    if configuration is None:
        return "None"
    return (
        f"raw={configuration.raw_description!r}, type={configuration.normalized_type!r}, "
        f"status={configuration.status}, confidence={configuration.confidence}"
    )


def _glass(glass_items) -> str:
    if not glass_items:
        return "[]"
    return "; ".join(
        (
            f"type={_normalized(glass.type)}, thickness={_measurement(glass.thickness)}, "
            f"color={_normalized(glass.color)}, status={glass.status}, "
            f"confidence={glass.confidence}"
        )
        for glass in glass_items
    )


def _profiles(profiles) -> str:
    if not profiles:
        return "[]"
    return "; ".join(
        (
            f"code={_traceable(profile.code)}, name={_traceable(profile.name)}, "
            f"raw={profile.raw_description!r}, role={_normalized(profile.role)}, "
            f"status={profile.status}, confidence={profile.confidence}"
        )
        for profile in profiles
    )


def _measurement(measurement) -> str:
    if measurement is None:
        return "None"
    return (
        f"value={measurement.value!r} {measurement.unit or ''}, "
        f"raw_label={measurement.raw_label!r}, status={measurement.status}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
