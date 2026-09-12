"""Opt-in Gemini page localizer. Never instantiated by the existing extractor.

No retries/tools; one generate_content call per supplied page request. Local plan and
replay modes never import this SDK, read credentials or instantiate this class.
"""
from __future__ import annotations

from typing import Any

from app.models.document_localization_v1 import PageLocalizationProposal
from app.services.localization_prompt_v1 import SYSTEM_INSTRUCTION


class GeminiLocalizationClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types

        if not api_key or not model:
            raise ValueError("Missing API key or configured model")
        self.model = model
        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=180_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def generate(self, prompt: str, images: list[tuple[str, str, bytes]]) -> dict[str, Any]:
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
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0,
                max_output_tokens=16384,
                response_mime_type="application/json",
                response_json_schema=PageLocalizationProposal.model_json_schema(),
            ),
        )
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish = getattr(finish, "value", finish)
        usage = getattr(response, "usage_metadata", None)
        return {
            "text": getattr(response, "text", None),
            "finish_reason": str(finish) if finish is not None else None,
            "requested_model": self.model,
            "response_model_version": getattr(response, "model_version", None),
            "usage": ({
                field: getattr(usage, field, None)
                for field in ("prompt_token_count", "candidates_token_count",
                              "thoughts_token_count", "total_token_count")
            } if usage is not None else None),
        }

    def close(self) -> None:
        self._client.close()
