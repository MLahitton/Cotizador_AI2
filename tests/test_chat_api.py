from fastapi.testclient import TestClient
from google.genai.errors import APIError

from app.api.chat import get_chat_responder
from app.main import app
from app.models.chat import ChatRespondRequest, ChatRespondResponse
from app.services.chat_responder import CHAT_SYSTEM_PROMPT, ChatResponder, ChatResponderError


class _FakeChatResponder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    def respond(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return ChatRespondResponse(message="Respuesta basada en el contexto recibido.")


def test_chat_respond_accepts_requirement_context() -> None:
    fake = _FakeChatResponder()

    response = _post_with_fake(fake, _payload(scope="REQUIREMENT"))

    assert response.status_code == 200
    assert response.json()["message"] == "Respuesta basada en el contexto recibido."
    assert len(fake.calls) == 1
    assert fake.calls[0].scope == "REQUIREMENT"
    assert fake.calls[0].context["requirement"]["requirementId"] == "req-1"


def test_chat_respond_accepts_item_context() -> None:
    fake = _FakeChatResponder()

    response = _post_with_fake(fake, _payload(scope="ITEM", item_id="item-1"))

    assert response.status_code == 200
    assert fake.calls[0].scope == "ITEM"
    assert fake.calls[0].context["item"]["itemId"] == "item-1"


def test_chat_respond_empty_message_rejected() -> None:
    response = _post_with_fake(_FakeChatResponder(), {**_payload(), "userMessage": "   "})

    assert response.status_code == 422


def test_chat_respond_provider_failure_returns_502_without_sensitive_details() -> None:
    response = _post_with_fake(
        _FakeChatResponder(error=RuntimeError("GEMINI_API_KEY=secret-value")),
        _payload(),
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Error del proveedor de IA."
    assert "secret-value" not in response.text


def test_chat_respond_api_error_returns_502_without_sensitive_details() -> None:
    response = _post_with_fake(
        _FakeChatResponder(error=APIError(403, {"message": "GEMINI_API_KEY=secret-value"})),
        _payload(),
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Error del proveedor de IA."
    assert "secret-value" not in response.text


def test_chat_respond_invalid_llm_json_returns_controlled_502() -> None:
    response = _post_with_fake(
        _FakeChatResponder(error=ChatResponderError("Gemini devolvio JSON invalido para chat.")),
        _payload(),
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Respuesta de IA invalida."


def test_chat_responder_uses_structured_output_schema_and_system_instruction() -> None:
    provider = _FakeGeminiProvider(
        _FakeGeminiResponse(parsed=ChatRespondResponse(message="Chat funcionando."))
    )

    response = ChatResponder(provider).respond(ChatRespondRequest.model_validate(_payload()))

    assert response.message == "Chat funcionando."
    call = provider._client.models.calls[0]
    assert call["contents"].startswith("INPUT_JSON:")
    assert "Cotizador Steel & Glass read-only chat." not in call["contents"]
    assert call["config"].system_instruction.startswith("Cotizador Steel & Glass")
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is ChatRespondResponse
    assert call["config"].temperature == 0.2



def test_chat_system_prompt_hides_internal_codes_in_normal_answers() -> None:
    assert "Never expose internal rule codes" in CHAT_SYSTEM_PROMPT
    assert "unless the user explicitly asks" in CHAT_SYSTEM_PROMPT
    assert "SYSTEM_SLIDING_DOOR_NAPOLES" in CHAT_SYSTEM_PROMPT
    assert "SLIDING_DOOR" in CHAT_SYSTEM_PROMPT
    assert "PRE_SELECTION" in CHAT_SYSTEM_PROMPT
    assert "POST_DESIGN" in CHAT_SYSTEM_PROMPT
    assert "candidate scores" in CHAT_SYSTEM_PROMPT
    assert "normal answers" in CHAT_SYSTEM_PROMPT


def test_chat_system_prompt_requires_human_supported_explanations() -> None:
    assert "translate internal technical evidence into clear" in CHAT_SYSTEM_PROMPT
    assert "using only factors present in Backend context" in CHAT_SYSTEM_PROMPT
    assert "Do not invent dimensions or units" in CHAT_SYSTEM_PROMPT
    assert "Do not make a richer explanation than the context supports" in CHAT_SYSTEM_PROMPT
    assert "wind load" in CHAT_SYSTEM_PROMPT
    assert "Always return structured JSON matching ChatRespondResponse" in CHAT_SYSTEM_PROMPT


def test_chat_system_prompt_allows_explicit_debug_codes() -> None:
    assert "If the user explicitly asks for internal technical details" in CHAT_SYSTEM_PROMPT
    assert "you may reveal those values" in CHAT_SYSTEM_PROMPT
    assert "Label them" in CHAT_SYSTEM_PROMPT
    assert "internal technical information" in CHAT_SYSTEM_PROMPT


def test_chat_responder_sends_item_context_without_exposing_prompt_in_contents() -> None:
    provider = _FakeGeminiProvider(
        _FakeGeminiResponse(
            parsed=ChatRespondResponse(
                message=(
                    "Se escogio Napoles porque el elemento funciona como puerta corrediza, "
                    "tiene unas dimensiones de 3,20 m x 2,90 m y la configuracion "
                    "registrada es compatible con este tipo de sistema."
                )
            )
        )
    )
    payload = _item_system_reason_payload("Por que escogiste este sistema?")

    response = ChatResponder(provider).respond(ChatRespondRequest.model_validate(payload))

    assert response.message.startswith("Se escogio Napoles")
    assert "SYSTEM_SLIDING_DOOR_NAPOLES" not in response.message
    assert "SLIDING_DOOR" not in response.message
    assert "3,20 m" in response.message
    call = provider._client.models.calls[0]
    assert "SYSTEM_SLIDING_DOOR_NAPOLES" in call["contents"]
    assert "SLIDING_DOOR" in call["contents"]
    assert "Cotizador Steel & Glass read-only chat." not in call["contents"]
    assert call["config"].system_instruction is CHAT_SYSTEM_PROMPT
    assert call["config"].response_schema is ChatRespondResponse


def test_chat_responder_can_return_internal_codes_when_user_explicitly_asks_debug() -> None:
    provider = _FakeGeminiProvider(
        _FakeGeminiResponse(
            parsed=ChatRespondResponse(
                message=(
                    "Informacion tecnica interna: la regla registrada fue "
                    "SYSTEM_SLIDING_DOOR_NAPOLES y el tipo funcional interno fue SLIDING_DOOR."
                )
            )
        )
    )
    payload = _item_system_reason_payload("Muestrame la regla interna exacta que usaste.")

    response = ChatResponder(provider).respond(ChatRespondRequest.model_validate(payload))

    assert "Informacion tecnica interna" in response.message
    assert "SYSTEM_SLIDING_DOOR_NAPOLES" in response.message
    assert "SLIDING_DOOR" in response.message
def test_chat_responder_accepts_valid_parsed_dict() -> None:
    provider = _FakeGeminiProvider(_FakeGeminiResponse(parsed={"message": "Desde parsed."}))

    response = ChatResponder(provider).respond(ChatRespondRequest.model_validate(_payload()))

    assert response.message == "Desde parsed."


def test_chat_responder_falls_back_to_valid_text_json() -> None:
    provider = _FakeGeminiProvider(_FakeGeminiResponse(text='{"message":"Desde text."}'))

    response = ChatResponder(provider).respond(ChatRespondRequest.model_validate(_payload()))

    assert response.message == "Desde text."


def test_chat_responder_without_parsed_or_text_raises_error() -> None:
    provider = _FakeGeminiProvider(_FakeGeminiResponse())

    try:
        ChatResponder(provider).respond(ChatRespondRequest.model_validate(_payload()))
    except ChatResponderError as exc:
        assert "no devolvio texto" in str(exc)
    else:
        raise AssertionError("Expected ChatResponderError")


def test_chat_responder_rejects_invalid_text_schema() -> None:
    provider = _FakeGeminiProvider(_FakeGeminiResponse(text='{"answer":"No message key"}'))

    try:
        ChatResponder(provider).respond(ChatRespondRequest.model_validate(_payload()))
    except ChatResponderError as exc:
        assert "JSON invalido" in str(exc)
    else:
        raise AssertionError("Expected ChatResponderError")


def test_chat_respond_accepts_backend_item_selection_context() -> None:
    fake = _FakeChatResponder()
    payload = {
        "scope": "ITEM",
        "userMessage": "Por que escogiste este sistema?",
        "conversation": [
            {"role": "user", "content": "Necesito revisar la seleccion."},
            {"role": "assistant", "content": "Reviso el contexto disponible."},
        ],
        "context": {
            "item": {
                "reference": "PV-01",
                "dimensions": {"widthMm": 2500, "heightMm": 3740, "areaM2": 9.35},
                "Suggested": {"system": "3831", "glass": "Templado 6 mm"},
                "Selected": {"system": "3831", "glass": "Templado 6 mm"},
                "reviewReasons": ["Sistema solicitado coincide con evidencia."],
                "resolutionReasons": ["Seleccion conservadora con base en contexto."],
            }
        },
    }

    response = _post_with_fake(fake, payload)

    assert response.status_code == 200
    assert response.json()["message"] == "Respuesta basada en el contexto recibido."
    assert fake.calls[0].context["item"]["Selected"]["system"] == "3831"


def test_chat_openapi_exposes_endpoint() -> None:
    client = TestClient(app)

    openapi = client.get("/openapi.json").json()

    assert "/chat/respond" in openapi["paths"]
    operation = openapi["paths"]["/chat/respond"]["post"]
    assert "application/json" in operation["requestBody"]["content"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ChatRespondResponse"
    )


def _post_with_fake(fake: _FakeChatResponder, payload: dict):
    app.dependency_overrides[get_chat_responder] = lambda: fake
    try:
        client = TestClient(app)
        return client.post("/chat/respond", json=payload)
    finally:
        app.dependency_overrides.clear()


def _payload(scope: str = "REQUIREMENT", item_id: str | None = None) -> dict:
    return {
        "scope": scope,
        "userMessage": "Que falta para cotizar?",
        "conversation": [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Te ayudo con la propuesta."},
        ],
        "context": {
            "requirement": {"requirementId": "req-1", "status": "Processed"},
            "item": None if item_id is None else {"itemId": item_id, "reference": "V-01"},
            "instructions": ["READ_ONLY_CHAT", "USE_ONLY_CONTEXT", "DO_NOT_MUTATE_SELECTION"],
        },
    }



def _item_system_reason_payload(user_message: str) -> dict:
    return {
        "scope": "ITEM",
        "userMessage": user_message,
        "conversation": [],
        "context": {
            "suggestedSystemName": "PUERTA CORREDIZA LINEA PREMIUM TIPO EUROPEO VENECIA NAPOLES",
            "selectedSystemName": "PUERTA CORREDIZA LINEA PREMIUM TIPO EUROPEO VENECIA NAPOLES",
            "functionalType": "SLIDING_DOOR",
            "widthMm": 3200,
            "heightMm": 2900,
            "resolutionReasons": ["SYSTEM_SLIDING_DOOR_NAPOLES"],
        },
    }

class _FakeGeminiProvider:
    def __init__(self, response) -> None:
        self.model = "gemini-test"
        self._client = _FakeGeminiClient(response)


class _FakeGeminiClient:
    def __init__(self, response) -> None:
        self.models = _FakeGeminiModels(response)


class _FakeGeminiModels:
    def __init__(self, response) -> None:
        self._response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeGeminiResponse:
    def __init__(self, parsed=None, text=None) -> None:
        self.parsed = parsed
        self.text = text
