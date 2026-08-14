import json
import sys
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.providers.gemini_extraction import (  # noqa: E402
    GeminiExtractionProvider,
    GeminiFullPipelineDebugCapture,
)

PDF_PATH = ROOT_DIR / "manual_tests" / "20260804 EL ESPIGAL_01 Planteamiento CASA 1-2.pdf"
RESULTS_DIR = ROOT_DIR / "manual_tests" / "results"
BATCHES_DIR = RESULTS_DIR / "espigal_batches"
DISCOVERY_OUTPUT_PATH = RESULTS_DIR / "espigal_full_discovery.json"
ENRICHMENT_OUTPUT_PATH = RESULTS_DIR / "espigal_full_enrichment.json"
EXTRACTION_OUTPUT_PATH = RESULTS_DIR / "espigal_full_requirement_extraction.json"
REPORT_OUTPUT_PATH = RESULTS_DIR / "espigal_full_pipeline_report.md"
SCOPE_OUTPUT_PATH = RESULTS_DIR / "espigal_full_scope.json"


def main() -> int:
    started_at = perf_counter()

    try:
        provider = GeminiExtractionProvider()
        debug_capture = GeminiFullPipelineDebugCapture()
        extraction = provider.extract_with_discovery_from_files(
            [PDF_PATH],
            project_id="manual-espigal",
            requirement_id="manual-espigal-001",
            debug_capture=debug_capture,
        )

        _write_outputs(extraction, debug_capture)
        elapsed_seconds = perf_counter() - started_at
        _write_report(extraction, debug_capture, elapsed_seconds)
        elapsed_seconds = perf_counter() - started_at

        _print_summary(extraction, debug_capture)
        print(f"\nreport:\n{REPORT_OUTPUT_PATH}")
        print(f"TOTAL ELAPSED SECONDS: {elapsed_seconds:.2f}")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _write_outputs(extraction, debug_capture: GeminiFullPipelineDebugCapture) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    if debug_capture.discovery is not None:
        DISCOVERY_OUTPUT_PATH.write_text(
            debug_capture.discovery.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if debug_capture.enrichment is not None:
        ENRICHMENT_OUTPUT_PATH.write_text(
            debug_capture.enrichment.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if debug_capture.scope is not None:
        SCOPE_OUTPUT_PATH.write_text(
            debug_capture.scope.model_dump_json(indent=2),
            encoding="utf-8",
        )

    EXTRACTION_OUTPUT_PATH.write_text(
        extraction.model_dump_json(indent=2),
        encoding="utf-8",
    )

    enrichment_debug = debug_capture.enrichment_debug
    if enrichment_debug is not None:
        for index, raw_response in enumerate(enrichment_debug.raw_responses or [], start=1):
            if raw_response is not None:
                (BATCHES_DIR / f"batch_{index:03d}_raw.json").write_text(
                    raw_response,
                    encoding="utf-8",
                )


def _write_report(
    extraction,
    debug_capture: GeminiFullPipelineDebugCapture,
    elapsed_seconds: float,
) -> None:
    REPORT_OUTPUT_PATH.write_text(
        _build_report(extraction, debug_capture, elapsed_seconds),
        encoding="utf-8",
    )


def _build_report(
    extraction,
    debug_capture: GeminiFullPipelineDebugCapture,
    elapsed_seconds: float,
) -> str:
    discovery = debug_capture.discovery
    scope = debug_capture.scope
    enrichment = debug_capture.enrichment
    discovery_usage, enrichment_usage, final_usage = _usage_tuple(extraction, debug_capture)
    scope_usage = debug_capture.scope_debug.token_usage if debug_capture.scope_debug else None
    scope_counts = _scope_counts(scope)
    traceability_counts = _traceability_counts(extraction, discovery, scope, enrichment)
    warnings = enrichment.warnings if enrichment is not None else []
    if scope is not None:
        warnings = scope.warnings + warnings
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "None"

    return f"""# EL ESPIGAL - FULL AI2 PIPELINE REPORT

## Summary

- model: {debug_capture.model}
- discovered_elements: {len(discovery.elements) if discovery else 0}
- in_scope_full: {scope_counts["in_scope_full"]}
- in_scope_partial: {scope_counts["in_scope_partial"]}
- uncertain: {scope_counts["uncertain"]}
- out_of_scope: {scope_counts["out_of_scope"]}
- enriched_elements: {len(enrichment.elements) if enrichment else 0}
- final_elements: {len(extraction.elements)}
- batches: {debug_capture.batch_count}
- batch_size: {debug_capture.batch_size}

## Timing

- elapsed_seconds: {elapsed_seconds:.2f}

## Tokens

- discovery_input: {discovery_usage.input_tokens if discovery_usage else None}
- discovery_output: {discovery_usage.output_tokens if discovery_usage else None}
- discovery_total: {discovery_usage.total_tokens if discovery_usage else None}
- scope_input: {scope_usage.input_tokens if scope_usage else None}
- scope_output: {scope_usage.output_tokens if scope_usage else None}
- scope_total: {scope_usage.total_tokens if scope_usage else None}
- enrichment_input: {enrichment_usage.input_tokens if enrichment_usage else None}
- enrichment_output: {enrichment_usage.output_tokens if enrichment_usage else None}
- enrichment_total: {enrichment_usage.total_tokens if enrichment_usage else None}
- grand_total: {final_usage.total_tokens if final_usage else None}

## Warnings

{warning_lines}

## Traceability

- sources: {traceability_counts["sources"]}
- final_evidence: {traceability_counts["final_evidence"]}
- discoveries_with_source_ids: {traceability_counts["discoveries_with_source_ids"]}
- scope_with_source_ids: {traceability_counts["scope_with_source_ids"]}
- enrichment_structured_evidence: {traceability_counts["enrichment_structured_evidence"]}

## Discovery JSON

```json
{_json_block(discovery)}
```

## Scope JSON

```json
{_json_block(scope)}
```

## Enrichment JSON

```json
{_json_block(enrichment)}
```

## Final Requirement Extraction JSON

```json
{_json_block(extraction)}
```
"""


def _print_summary(
    extraction,
    debug_capture: GeminiFullPipelineDebugCapture,
) -> None:
    discovery = debug_capture.discovery
    scope = debug_capture.scope
    enrichment = debug_capture.enrichment
    discovery_usage, enrichment_usage, final_usage = _usage_tuple(extraction, debug_capture)
    scope_usage = debug_capture.scope_debug.token_usage if debug_capture.scope_debug else None
    scope_counts = _scope_counts(scope)

    print("EL ESPIGAL - FULL AI2 PIPELINE")
    print(f"model: {debug_capture.model}")
    print(f"discovered_elements: {len(discovery.elements) if discovery else 0}")
    print(f"in_scope_full: {scope_counts['in_scope_full']}")
    print(f"in_scope_partial: {scope_counts['in_scope_partial']}")
    print(f"uncertain: {scope_counts['uncertain']}")
    print(f"out_of_scope: {scope_counts['out_of_scope']}")
    print(f"enriched_elements: {len(enrichment.elements) if enrichment else 0}")
    print(f"final_elements: {len(extraction.elements)}")
    print(f"batches: {debug_capture.batch_count}")
    print(f"batch_size: {debug_capture.batch_size}")

    print("\nTOKENS")
    print(f"discovery_input: {discovery_usage.input_tokens if discovery_usage else None}")
    print(f"discovery_output: {discovery_usage.output_tokens if discovery_usage else None}")
    print(f"discovery_total: {discovery_usage.total_tokens if discovery_usage else None}")
    print(f"scope_input: {scope_usage.input_tokens if scope_usage else None}")
    print(f"scope_output: {scope_usage.output_tokens if scope_usage else None}")
    print(f"scope_total: {scope_usage.total_tokens if scope_usage else None}")
    print(f"enrichment_input: {enrichment_usage.input_tokens if enrichment_usage else None}")
    print(f"enrichment_output: {enrichment_usage.output_tokens if enrichment_usage else None}")
    print(f"enrichment_total: {enrichment_usage.total_tokens if enrichment_usage else None}")
    print(f"grand_total: {final_usage.total_tokens if final_usage else None}")

    print("\nWARNINGS")
    warnings = []
    if scope is not None:
        warnings.extend(scope.warnings)
    if enrichment is not None:
        warnings.extend(enrichment.warnings)
    if not warnings:
        print("None")
    for warning in warnings:
        print(f"- {warning}")


def _usage_tuple(extraction, debug_capture: GeminiFullPipelineDebugCapture):
    discovery_usage = (
        debug_capture.discovery_debug.token_usage
        if debug_capture.discovery_debug is not None
        else None
    )
    enrichment = debug_capture.enrichment
    enrichment_usage = enrichment.usage if enrichment is not None else None
    final_usage = extraction.extraction_metadata.token_usage
    return discovery_usage, enrichment_usage, final_usage


def _scope_counts(scope) -> dict[str, int]:
    counts = {
        "in_scope_full": 0,
        "in_scope_partial": 0,
        "uncertain": 0,
        "out_of_scope": 0,
    }
    if scope is None:
        return counts
    for item in scope.elements:
        counts[item.scope.value] += 1
    return counts


def _traceability_counts(extraction, discovery, scope, enrichment) -> dict[str, int]:
    return {
        "sources": len(extraction.sources),
        "final_evidence": len(extraction.evidence),
        "discoveries_with_source_ids": sum(
            1 for element in (discovery.elements if discovery else []) if element.source_ids
        ),
        "scope_with_source_ids": sum(
            1 for element in (scope.elements if scope else []) if element.evidence_source_ids
        ),
        "enrichment_structured_evidence": sum(
            len(element.evidence) for element in (enrichment.elements if enrichment else [])
        ),
    }


def _json_block(model: BaseModel | None) -> str:
    if model is None:
        return "null"
    return json.dumps(
        model.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
