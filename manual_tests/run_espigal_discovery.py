import json
import sys
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.providers.gemini_extraction import (  # noqa: E402
    GeminiDiscoveryDebugCapture,
    GeminiExtractionProvider,
)

PDF_PATH = ROOT_DIR / "manual_tests" / "20260804 EL ESPIGAL_01 Planteamiento CASA 1-2.pdf"
RESULTS_DIR = ROOT_DIR / "manual_tests" / "results"
RAW_OUTPUT_PATH = RESULTS_DIR / "espigal_discovery_raw.json"
DISCOVERY_OUTPUT_PATH = RESULTS_DIR / "espigal_discovery.json"


def main() -> int:
    started_at = perf_counter()

    try:
        provider = GeminiExtractionProvider()
        debug_capture = GeminiDiscoveryDebugCapture()
        discovery = provider.discover_elements_from_files(
            [PDF_PATH],
            debug_capture=debug_capture,
        )
        elapsed_seconds = perf_counter() - started_at

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if debug_capture.raw_response_text is None:
            raise ValueError("No se capturo response.text crudo de Gemini.")

        RAW_OUTPUT_PATH.write_text(debug_capture.raw_response_text, encoding="utf-8")
        DISCOVERY_OUTPUT_PATH.write_text(
            discovery.model_dump_json(indent=2),
            encoding="utf-8",
        )

        _print_summary(discovery, debug_capture, elapsed_seconds)
        _print_json("DISCOVERY JSON", discovery)
        print(f"\nraw_json: {RAW_OUTPUT_PATH}")
        print(f"discovery_json: {DISCOVERY_OUTPUT_PATH}")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _print_summary(
    discovery,
    debug_capture: GeminiDiscoveryDebugCapture,
    elapsed_seconds: float,
) -> None:
    token_usage = debug_capture.token_usage

    print("EL ESPIGAL - DISCOVERY")
    print(f"model: {debug_capture.model}")
    print(f"elements_discovered: {len(discovery.elements)}")
    print(f"elapsed_seconds: {elapsed_seconds:.2f}")
    print(f"input_tokens: {token_usage.input_tokens if token_usage else None}")
    print(f"output_tokens: {token_usage.output_tokens if token_usage else None}")
    print(f"total_tokens: {token_usage.total_tokens if token_usage else None}")


def _print_json(title: str, model: BaseModel) -> None:
    print(f"\n{title}")
    print(
        json.dumps(
            model.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
