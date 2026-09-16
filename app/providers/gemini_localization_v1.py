"""Opt-in Gemini page localizer. Never instantiated by the existing extractor.

No retries/tools; one generate_content call per supplied page request. Local plan and
replay modes never import this SDK, read credentials or instantiate this class.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any
from urllib.parse import quote, quote_plus

from app.models.document_localization_v1 import (
    ImageFrameLocalizationProposal,
    PageLocalizationProposal,
)
from app.services.localization_prompt_v1 import (
    IMAGE_FRAME_LOCALIZATION_RULES,
    SYSTEM_INSTRUCTION,
)

# Generation and validation have different jobs. Keep the strict Pydantic model
# unchanged; send a smaller structural schema to the API. This is a controlled
# compatibility candidate, NOT proof of the cause of any generic HTTP 400.
_API_SCHEMA_VERSION = "page-localization-api-v1.1"
_SCHEMA_LOCAL_ONLY = frozenset({"pattern", "minLength", "maxLength", "title", "default"})
_SCHEMA_SUPPORTED = frozenset(
    {
        "$defs", "$ref", "type", "properties", "items", "required", "enum",
        "anyOf", "additionalProperties", "description", "minimum", "maximum",
        "minItems", "maxItems",
    }
)


def _compact_api_schema(
    node: dict[str, Any],
    definitions: dict[str, Any],
    resolving: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Project this known, nonrecursive schema; do not silently drop new keywords.

    All property names, types, required fields, enums and nullability are kept.
    Patterns/string limits and collection limits remain enforced locally. Exact,
    small array arity (the four coordinates) is also preserved in the API schema.
    This helper never edits the schema returned by Pydantic.
    """
    if not isinstance(node, dict):
        raise ValueError("LOCALIZATION_API_SCHEMA_EXPECTED_OBJECT")
    node = copy.deepcopy(node)
    if "const" in node:
        const = node.pop("const")
        if not isinstance(const, str) or node.get("type") != "string":
            raise ValueError("LOCALIZATION_API_SCHEMA_UNSUPPORTED_CONST")
        if "enum" in node:
            values = node["enum"]
            if not isinstance(values, list) or const not in values:
                raise ValueError("LOCALIZATION_API_SCHEMA_CONST_ENUM_MISMATCH")
        node["enum"] = [const]
    unknown = set(node) - _SCHEMA_SUPPORTED - _SCHEMA_LOCAL_ONLY
    if unknown:
        raise ValueError("LOCALIZATION_API_SCHEMA_UNREVIEWED_KEYWORD")
    if "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise ValueError("LOCALIZATION_API_SCHEMA_EXTERNAL_REFERENCE")
        if set(node) - {"$ref", "title", "description"}:
            raise ValueError("LOCALIZATION_API_SCHEMA_REFERENCE_SIBLINGS")
        name = ref[len("#/$defs/"):]
        if name not in definitions or name in resolving or len(resolving) >= 8:
            raise ValueError("LOCALIZATION_API_SCHEMA_UNRESOLVABLE_REFERENCE")
        result = _compact_api_schema(definitions[name], definitions, (*resolving, name))
        if "description" in node:
            result["description"] = node["description"]
        return result

    result = {}
    for key, value in node.items():
        if key in _SCHEMA_LOCAL_ONLY or key in {"$defs", "minItems", "maxItems"}:
            continue
        if key == "properties":
            result[key] = {
                name: _compact_api_schema(child, definitions, resolving)
                for name, child in value.items()
            }
        elif key == "items":
            result[key] = _compact_api_schema(value, definitions, resolving)
        elif key == "anyOf":
            result[key] = [
                _compact_api_schema(child, definitions, resolving) for child in value
            ]
        elif key == "additionalProperties" and isinstance(value, dict):
            result[key] = _compact_api_schema(value, definitions, resolving)
        else:
            result[key] = copy.deepcopy(value)

    # A box still has exactly 4 coordinates. Do not copy high collection bounds
    # (e.g. 300 regions x nested lists) into constrained generation.
    minimum, maximum = node.get("minItems"), node.get("maxItems")
    if (
        node.get("type") == "array"
        and type(minimum) is int
        and type(maximum) is int
        and minimum == maximum
        and 0 < minimum <= 8
    ):
        result["minItems"] = minimum
        result["maxItems"] = maximum
    return result


def build_localization_api_schema() -> dict[str, Any]:
    """A fresh generation schema. Local acceptance still uses PageLocalizationProposal."""
    schema = PageLocalizationProposal.model_json_schema()
    return _compact_api_schema(schema, schema.get("$defs", {}))


def build_image_frame_localization_api_schema() -> dict[str, Any]:
    """Generation schema for the opt-in image-frame contract."""
    schema = ImageFrameLocalizationProposal.model_json_schema()
    return _compact_api_schema(schema, schema.get("$defs", {}))


def localization_api_schema_metadata() -> dict[str, Any]:
    """Safe request provenance: no credentials, images or document content."""
    return _localization_api_schema_metadata(
        version=_API_SCHEMA_VERSION,
        api_schema=build_localization_api_schema(),
        local_schema=PageLocalizationProposal.model_json_schema(),
        local_validator="PageLocalizationProposal (unchanged)",
    )


def image_frame_localization_api_schema_metadata() -> dict[str, Any]:
    """Safe request provenance for the opt-in image-frame contract."""
    metadata = _localization_api_schema_metadata(
        version="page-localization-image-frames-api-v1.0",
        api_schema=build_image_frame_localization_api_schema(),
        local_schema=ImageFrameLocalizationProposal.model_json_schema(),
        local_validator="ImageFrameLocalizationProposal",
    )
    metadata["local_validation_policy"] = "conditional-localization-notes-v1"
    return metadata


def _localization_api_schema_metadata(
    *,
    version: str,
    api_schema: dict[str, Any],
    local_schema: dict[str, Any],
    local_validator: str,
) -> dict[str, Any]:
    """Safe request provenance: no credentials, images or document content."""
    def digest(schema: dict[str, Any]) -> str:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    return {
        "version": version,
        "api_schema_sha256": digest(api_schema),
        "local_validation_schema_sha256": digest(local_schema),
        "hash_encoding": "UTF-8 JSON; sorted keys; compact separators; ensure_ascii=False",
        "local_validator": local_validator,
        "local_only_rules": [
            "string_patterns", "string_lengths", "collection_limits",
            "coordinate_ranges_and_order", "unique_ids", "typed_region_links",
        ],
        "purpose": "GENERATION_STRUCTURE_ONLY_NOT_SEMANTIC_APPROVAL",
    }


_API_STATUSES = frozenset(
    {
        "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND",
        "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED",
        "INTERNAL", "UNAVAILABLE", "DATA_LOSS", "UNAUTHENTICATED",
    }
)


def _redacted_api_message(value: object, api_key: str) -> str | None:
    """Keep only a bounded API message, never the full exception/body/headers.

    Redaction is defensive, not a general guarantee that server text contains no
    business data. Original requests and credentials must not be shared for diagnosis.
    """
    if not isinstance(value, str):
        return None
    if len(value) > 32_768:
        return "API_MESSAGE_OMITTED_TOO_LONG"
    # Remove the actual configured key (including common escaped forms) BEFORE
    # truncation, so a key cut at the output limit cannot leak its prefix.
    if api_key:
        forms = {
            api_key,
            quote(api_key, safe=""),
            quote_plus(api_key, safe=""),
            json.dumps(api_key)[1:-1],
        }
        for secret in sorted(forms, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"AIza[0-9A-Za-z_-]+", "[REDACTED]", value)
    value = re.sub(r"ya29\.[0-9A-Za-z._-]+", "[REDACTED]", value)
    value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[^\s,;\"']+", "[REDACTED_AUTH]", value)
    value = re.sub(
        r"(?i)(?:[\"']?(?:api[_-]?key|x-goog-api-key|authorization|"
        r"access[_-]?token|refresh[_-]?token|password|secret)[\"']?\s*[:=]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&}]+)",
        "[REDACTED_CREDENTIAL]", value,
    )
    value = re.sub(r"https?://[^\s<>\"']+", "[URL_REDACTED]", value, flags=re.I)
    value = re.sub(r"[A-Za-z0-9_+/=-]{128,}", "[LONG_VALUE_REDACTED]", value)
    # No terminal escape/control sequences in printed server text.
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
    value = " ".join(value.split())
    return value if len(value) <= 1600 else value[:1580] + " [TRUNCATED]"


class GeminiLocalizationAPIError(RuntimeError):
    """A deliberately small diagnostic, not a serialized SDK exception."""

    def __init__(self, error: Exception, *, api_key: str) -> None:
        code = getattr(error, "code", None)
        status = getattr(error, "status", None)
        self.original_error_type = (
            type(error).__name__
            if type(error).__name__ in {"ClientError", "ServerError", "APIError"}
            else "APIError"
        )
        self.http_code = code if type(code) is int and 100 <= code <= 599 else None
        self.api_status = status if isinstance(status, str) and status in _API_STATUSES else None
        self.message_redacted = _redacted_api_message(getattr(error, "message", None), api_key)
        # Neither str(error), repr(error), details, response nor request is copied.
        super().__init__("GEMINI_API_ERROR")

    def diagnostic(self) -> dict[str, Any]:
        return {
            "error_type": self.original_error_type,
            "http_code": self.http_code,
            "status": self.api_status,
            "message_redacted": self.message_redacted,
            "raw_error_saved": False,
        }


class GeminiLocalizationClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import errors, types

        if not api_key or not model:
            raise ValueError("Missing API key or configured model")
        self.model = model
        self._types = types
        self._api_error_type = errors.APIError
        self._api_key_for_redaction = api_key
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=180_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def generate(self, prompt: str, images: list[tuple[str, str, bytes]]) -> dict[str, Any]:
        return self._generate_with_schema(
            prompt,
            images,
            system_instruction=SYSTEM_INSTRUCTION,
            response_json_schema=build_localization_api_schema(),
            api_schema=localization_api_schema_metadata(),
        )

    def generate_image_frames(
        self,
        prompt: str,
        images: list[tuple[str, str, bytes]],
    ) -> dict[str, Any]:
        return self._generate_with_schema(
            prompt,
            images,
            system_instruction=IMAGE_FRAME_LOCALIZATION_RULES,
            response_json_schema=build_image_frame_localization_api_schema(),
            api_schema=image_frame_localization_api_schema_metadata(),
        )

    def _generate_with_schema(
        self,
        prompt: str,
        images: list[tuple[str, str, bytes]],
        *,
        system_instruction: str,
        response_json_schema: dict[str, Any],
        api_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        types = self._types
        parts = []
        for label, mime_type, data in images:
            parts.append(types.Part.from_text(text=label))
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        parts.append(types.Part.from_text(text=prompt))
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    temperature=0,
                    max_output_tokens=16384,
                    response_mime_type="application/json",
                    response_json_schema=response_json_schema,
                ),
            )
        except self._api_error_type as exc:
            raise GeminiLocalizationAPIError(
                exc, api_key=self._api_key_for_redaction,
            ) from None
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish = getattr(finish, "value", finish)
        usage = getattr(response, "usage_metadata", None)
        return {
            "text": getattr(response, "text", None),
            "finish_reason": str(finish) if finish is not None else None,
            "requested_model": self.model,
            "api_schema": api_schema,
            "response_model_version": getattr(response, "model_version", None),
            "usage": ({
                field: getattr(usage, field, None)
                for field in ("prompt_token_count", "candidates_token_count",
                              "thoughts_token_count", "total_token_count")
            } if usage is not None else None),
        }

    def close(self) -> None:
        try:
            self._client.close()
        finally:
            self._api_key_for_redaction = ""
