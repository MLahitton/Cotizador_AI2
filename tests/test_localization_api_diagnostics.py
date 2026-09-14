"""Offline checks for API failure reporting. Never calls Google or changes extraction."""
from __future__ import annotations

import json
import socket
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from app.providers.gemini_localization_v1 import (
    GeminiLocalizationAPIError,
    GeminiLocalizationClient,
    _redacted_api_message,
    build_localization_api_schema,
)
from manual_tests.corpus_v1 import localize_sources as runner


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("No network allowed in diagnostic tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


class APIError(Exception):
    def __init__(self, code=400, status="INVALID_ARGUMENT", message="Invalid request"):
        self.code, self.status, self.message = code, status, message

    def __str__(self):
        raise AssertionError("Full SDK exception must not be formatted")

    @property
    def details(self):
        raise AssertionError("SDK details must not be copied")

    @property
    def response(self):
        raise AssertionError("SDK response must not be copied")


class ClientError(APIError):
    pass


def fake_provider(outcome):
    """Run the real provider method with in-process types and a fake transport."""
    calls = []
    def generate_content(**kwargs):
        calls.append(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    provider = GeminiLocalizationClient.__new__(GeminiLocalizationClient)
    provider.model = "configured-model-unchanged"
    provider._api_key_for_redaction = "test-private-credential"
    provider._api_error_type = APIError
    provider._types = SimpleNamespace(
        Part=SimpleNamespace(
            from_text=lambda **kw: SimpleNamespace(**kw),
            from_bytes=lambda **kw: SimpleNamespace(**kw),
        ),
        GenerateContentConfig=lambda **kw: SimpleNamespace(**kw),
        AutomaticFunctionCallingConfig=lambda **kw: SimpleNamespace(**kw),
    )
    provider._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content), close=lambda: None,
    )
    return provider, calls


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (400, "INVALID_ARGUMENT"),
        (401, "UNAUTHENTICATED"),
        (403, "PERMISSION_DENIED"),
        (404, "NOT_FOUND"),
        (429, "RESOURCE_EXHAUSTED"),
        (503, "UNAVAILABLE"),
    ],
)
def test_http_code_and_status_are_preserved_without_details(code, status):
    wrapped = GeminiLocalizationAPIError(
        ClientError(code, status, "A useful diagnostic"),
        api_key="key",
    )
    assert wrapped.diagnostic() == {
        "error_type": "ClientError", "http_code": code, "status": status,
        "message_redacted": "A useful diagnostic", "raw_error_saved": False,
    }
    assert str(wrapped) == "GEMINI_API_ERROR"


@pytest.mark.parametrize(
    "message",
    [
        "Invalid credential test-private-credential",
        "api_key='test-private-credential'",
        'x-goog-api-key: "different-private-credential"',
        "Authorization: Bearer different-private-credential",
        "Request https://example.test/?key=different-private-credential failed",
        "AIza01234567890123456789012345678901234",
    ],
)
def test_credentials_are_not_in_report(message):
    wrapped = GeminiLocalizationAPIError(
        ClientError(message=message),
        api_key="test-private-credential",
    )
    serialized = json.dumps(wrapped.diagnostic())
    assert "test-private-credential" not in serialized
    assert "different-private-credential" not in serialized
    assert "AIza0123" not in serialized
    assert "https://example" not in serialized


def test_encoded_key_is_removed_before_output_truncation():
    key = "private value/+?="
    message = "A " * 788 + " " + quote(key, safe="") + " " + key + " suffix " * 100
    redacted = _redacted_api_message(message, key)
    assert key not in redacted and quote(key, safe="") not in redacted
    assert len(redacted) <= 1600


def test_message_is_optional_and_bounded():
    assert _redacted_api_message({"do_not_serialize": "secret"}, "key") is None
    assert _redacted_api_message("x" * 33000, "key") == "API_MESSAGE_OMITTED_TOO_LONG"
    assert _redacted_api_message("\x1b[31mInvalid\r\nrequest\x00", "key") == "Invalid request"


def test_opaque_or_invalid_fields_not_stringified():
    wrapped = GeminiLocalizationAPIError(ClientError(True, "arbitrary secret", None), api_key="key")
    assert wrapped.diagnostic()["http_code"] is None
    assert wrapped.diagnostic()["status"] is None
    assert wrapped.diagnostic()["message_redacted"] is None


def test_useful_schema_message_remains_readable():
    message = 'generationConfig.responseJsonSchema: unsupported constraint at properties.regions'
    wrapped = GeminiLocalizationAPIError(ClientError(message=message), api_key="private")
    assert wrapped.diagnostic()["message_redacted"] == message


def test_provider_keeps_config_and_disables_unused_afc():
    outcome = SimpleNamespace(text="{}", candidates=[SimpleNamespace(finish_reason="STOP")],
                              model_version="mock", usage_metadata=None)
    provider, calls = fake_provider(outcome)
    response = provider.generate("prompt", [("full page", "image/png", b"fake")])
    assert response["text"] == "{}" and response["finish_reason"] == "STOP"
    assert len(calls) == 1
    sent = calls[0]
    assert sent["model"] == "configured-model-unchanged"
    assert sent["config"].response_json_schema == build_localization_api_schema()
    assert sent["config"].max_output_tokens == 16384
    assert sent["config"].temperature == 0
    assert sent["config"].automatic_function_calling.disable is True
    assert sent["contents"][1].data == b"fake"


def test_provider_wraps_api_error_once_without_retry():
    provider, calls = fake_provider(ClientError(message="Invalid test-private-credential"))
    with pytest.raises(GeminiLocalizationAPIError) as caught:
        provider.generate("prompt", [("page", "image/png", b"fake")])
    assert len(calls) == 1
    assert caught.value.http_code == 400
    assert "test-private-credential" not in json.dumps(caught.value.diagnostic())
    provider.close()
    assert provider._api_key_for_redaction == ""


def test_other_exceptions_are_not_relabelled_as_api_failures():
    provider, calls = fake_provider(ValueError("local error"))
    with pytest.raises(ValueError, match="local error"):
        provider.generate("prompt", [])
    assert len(calls) == 1


def test_runner_persists_diagnostic_and_stops_remaining_jobs(tmp_path, monkeypatch, capsys):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = {
        "preparation_report_path": str(tmp_path / "prepared" / "report.json"),
        "preparation_report_sha256": "a" * 64, "source_archive_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "response_schema_sha256": "d" * 64,
        "jobs": [
            {"job_id": f"job-{index}", "project_id": "synthetic", "document_id": f"doc-{index}",
             "corpus_document_id": f"TEST{index}", "page_number": 1, "view_index": 1}
            for index in range(3)
        ],
    }
    monkeypatch.setattr(runner, "load_verified_plan", lambda path: plan)
    monkeypatch.setattr(
        runner,
        "request_for_job",
        lambda p, j: ("prompt", [("page", "image/png", b"fake")], {}),
    )
    provider, calls = fake_provider(ClientError(message="Bad schema, key=test-private-credential"))
    result, output = runner.execute_plan(
        plan_path, allow_paid_calls=True, max_calls=3,
        out=tmp_path / "run", client_factory=lambda: provider,
    )
    assert len(calls) == 1 and result["network_calls_attempted"] == 1
    assert result["responses_received"] == 0
    assert result["status"] == "LOCALIZATION_HAS_ERRORS"
    assert [r["status"] for r in result["jobs"]] == [
        "FAILED",
        "NOT_RUN_AFTER_FAILURE",
        "NOT_RUN_AFTER_FAILURE",
    ]
    assert result["jobs"][0]["reason"] == "GEMINI_API_ERROR"
    assert result["jobs"][0]["error_type"] == "ClientError"
    assert result["jobs"][0]["api_error"]["http_code"] == 400
    assert not list(output.rglob("response_envelope.json"))
    stdout = capsys.readouterr().out
    assert "API_HTTP_CODE=400" in stdout and "API_STATUS=INVALID_ARGUMENT" in stdout
    assert "Bad schema" in stdout and "test-private-credential" not in stdout
    for path in output.rglob("*.json"):
        assert "test-private-credential" not in path.read_text(encoding="utf-8")
