"""Experimental localized technical reader.

Plan:
  python manual_tests/corpus_v1/read_localized_elements.py plan
    --localization <replay_report.json> --prepared <source_preparation_report.json>

Run one candidate with paid model call:
  python manual_tests/corpus_v1/read_localized_elements.py run
    --plan <plan.json> --candidate <id> --max-calls 1 --allow-paid-calls

Replay a saved technical response without network:
  python manual_tests/corpus_v1/read_localized_elements.py replay
    --plan <plan.json> --candidate <id> --response <response_envelope.json>
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.document_localization_v1 import read_json, write_json  # noqa: E402
from app.services.localized_technical_prompt import (  # noqa: E402
    LOCALIZED_TECHNICAL_SYSTEM_INSTRUCTION,
)
from app.services.localized_technical_reader import (  # noqa: E402
    LocalizedTechnicalProvider,
    LocalizedTechnicalReaderError,
    build_localized_technical_plan,
    localized_technical_api_schema,
    replay_localized_technical_response,
    request_for_candidate,
    run_localized_technical_plan,
    select_candidate_jobs,
)


def new_output(out: Path | None, kind: str, *, create: bool) -> Path:
    output = out or BASE / "results" / "localized_technical" / (kind + "-" + uuid.uuid4().hex)
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise LocalizedTechnicalReaderError("OUTPUT_ALREADY_EXISTS")
    if create:
        output.mkdir(parents=True, exist_ok=False)
    return output


def write_plan(
    localization: Path,
    out: Path | None = None,
    *,
    prepared: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    plan = build_localized_technical_plan(localization, preparation_report_path=prepared)
    output = new_output(out, "plan", create=True)
    write_json(output / "localized_technical_plan.json", plan)
    report = {key: value for key, value in plan.items() if key != "candidates"}
    report["plan_file"] = str((output / "localized_technical_plan.json").resolve())
    report["candidate_ids"] = [candidate["candidate_id"] for candidate in plan["candidates"]]
    write_json(output / "localized_technical_plan_report.json", report)
    return report, output


def load_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path, 10 * 1024 * 1024)
    if not isinstance(plan, dict) or plan.get("status") != "LOCALIZED_TECHNICAL_READING_PLAN_READY":
        raise LocalizedTechnicalReaderError("INVALID_LOCALIZED_TECHNICAL_PLAN")
    current = build_localized_technical_plan(
        Path(plan["localization_report_path"]),
        preparation_report_path=Path(plan["preparation_report_path"]),
    )
    for key, value in current.items():
        if plan.get(key) != value:
            raise LocalizedTechnicalReaderError("LOCALIZED_TECHNICAL_PLAN_CHANGED")
    return plan


class GeminiLocalizedTechnicalClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types

        if not api_key or not model:
            raise ValueError("Missing API key or configured model")
        self.model = model
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def generate_localized_technical(
        self,
        prompt: str,
        images: list[tuple[str, str, bytes]],
    ) -> dict[str, Any]:
        types = self._types
        parts = []
        for label, mime_type, data in images:
            parts.append(types.Part.from_text(text=label))
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        parts.append(types.Part.from_text(text=prompt))
        response = self._client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=LOCALIZED_TECHNICAL_SYSTEM_INSTRUCTION,
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=localized_technical_api_schema(),
            ),
        )
        return {
            "text": getattr(response, "text", None),
            "finish_reason": _finish_reason(response),
            "requested_model": self.model,
            "usage": _usage(response),
        }

    def close(self) -> None:
        self._client.close()


def live_client() -> LocalizedTechnicalProvider:
    from app.core.settings import get_settings

    settings = get_settings()
    return GeminiLocalizedTechnicalClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )


def _finish_reason(response: object) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    finish = getattr(finish, "value", finish)
    return str(finish) if finish is not None else None


def _usage(response: object) -> dict[str, Any] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    return {
        field: getattr(usage, field, None)
        for field in ("prompt_token_count", "candidates_token_count", "total_token_count")
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    plan_parser = subs.add_parser("plan")
    plan_parser.add_argument("--localization", type=Path, required=True)
    plan_parser.add_argument("--prepared", type=Path)
    plan_parser.add_argument("--out", type=Path)
    run_parser = subs.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--candidate")
    run_parser.add_argument("--max-calls", type=int, required=True)
    run_parser.add_argument("--allow-paid-calls", action="store_true")
    run_parser.add_argument("--out", type=Path)
    replay_parser = subs.add_parser("replay")
    replay_parser.add_argument("--plan", type=Path, required=True)
    replay_parser.add_argument("--candidate", required=True)
    replay_parser.add_argument("--response", type=Path, required=True)
    replay_parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            report, output = write_plan(args.localization, args.out, prepared=args.prepared)
            print(f"STATUS={report['status']}")
            print(f"CANDIDATES={report['candidate_count']}")
            print(f"NETWORK_CALLS={report['network_calls']}")
            print(f"PLAN={output / 'localized_technical_plan.json'}")
            return 0
        if args.command == "run":
            if not args.allow_paid_calls:
                raise LocalizedTechnicalReaderError("PAID_CALLS_NOT_AUTHORIZED")
            plan = load_plan(args.plan)
            jobs = select_candidate_jobs(plan, args.candidate)
            for job in jobs:
                request_for_candidate(plan, job)
            output = new_output(args.out, "run", create=False)
            report = run_localized_technical_plan(
                plan,
                output,
                live_client(),
                max_calls=args.max_calls,
                candidate_id=args.candidate,
            )
            print(f"STATUS={report['status']}")
            print(f"REQUESTED_CANDIDATES={report['requested_candidates']}")
            print(f"NETWORK_CALLS_ATTEMPTED={report['network_calls_attempted']}")
            print(f"REPORT={output / 'localized_technical_reading_report.json'}")
            return 0
        plan = load_plan(args.plan)
        output = new_output(args.out, "replay", create=False)
        result = replay_localized_technical_response(
            plan,
            args.candidate,
            args.response,
            output,
        )
        print(f"STATUS={result.status}")
        print("NETWORK_CALLS=0")
        print(f"REPORT={output / 'localized_technical_reading.json'}")
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, LocalizedTechnicalReaderError) else type(exc).__name__
        print(f"LOCALIZED_TECHNICAL_ERROR={reason}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
