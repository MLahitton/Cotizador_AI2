from fastapi.testclient import TestClient

from app.api.chat import get_chat_action_interpreter
from app.main import app
from app.models.chat_actions import ChatActionInterpretRequest
from app.services.chat_action_interpreter import ChatActionInterpreter


def test_interprets_change_system_with_reference() -> None:
    intent = _interpret("cambia V-9 a S50")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.scope == "REQUIREMENT"
    assert intent.targetReference == "V-9"
    assert intent.requestedValue == "S50"


def test_interprets_contextual_item_scope() -> None:
    intent = _interpret(
        "cambialo a K50",
        scope="ITEM",
        context={"technicalProposalItemId": "item-1"},
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.scope == "ITEM"
    assert intent.targetReference is None
    assert intent.requestedValue == "K50"


def test_interprets_finish() -> None:
    intent = _interpret("pon el acabado en inox")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.requestedValue == "inox"


def test_interprets_finish_ambiguous() -> None:
    intent = _interpret("quiero otro acabado")

    assert intent.isAction is False
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.requiresClarification is True


def test_interprets_glass() -> None:
    intent = _interpret("sube el vidrio a 8 mm")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedValue == "8 mm"


def test_interprets_quantity() -> None:
    intent = _interpret("pon 3 unidades")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_QUANTITY"
    assert intent.requestedQuantity == 3


def test_interprets_dimensions_in_meters() -> None:
    intent = _interpret("cambialo a 1.80 x 2.40")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_DIMENSIONS"
    assert intent.requestedWidthMm == 1800
    assert intent.requestedHeightMm == 2400
    assert intent.requestedQuantity is None


def test_interprets_dimensions_in_centimeters() -> None:
    intent = _interpret("pon 90 cm x 210 cm")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_DIMENSIONS"
    assert intent.requestedWidthMm == 900
    assert intent.requestedHeightMm == 2100


def test_interprets_exclude_item() -> None:
    intent = _interpret("no cotices este item", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "EXCLUDE_ITEM"
    assert intent.scope == "ITEM"


def test_interprets_include_item() -> None:
    intent = _interpret("vuelve a incluirlo", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "INCLUDE_ITEM"
    assert intent.scope == "ITEM"


def test_interprets_commercial_line_requirement() -> None:
    intent = _interpret("quiero toda la propuesta en premium")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_COMMERCIAL_LINE"
    assert intent.scope == "REQUIREMENT"
    assert intent.requestedValue == "premium"


def test_informational_chat_is_not_action() -> None:
    intent = _interpret("que sistema tiene este item?", scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"


def test_ambiguous_change_requires_clarification() -> None:
    intent = _interpret("cambialo", scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.requiresClarification is True


def test_chat_action_endpoint_returns_intent() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/chat/actions/interpret",
            json=_payload("cambia V-9 a S50"),
        )

    assert response.status_code == 200
    assert response.json()["actionType"] == "CHANGE_SYSTEM"
    assert response.json()["targetReference"] == "V-9"
    assert response.json()["requestedValue"] == "S50"


def test_chat_action_endpoint_uses_dependency_override() -> None:
    class FakeInterpreter:
        def __init__(self) -> None:
            self.calls = []

        def interpret(self, request):
            self.calls.append(request)
            return ChatActionInterpreter().interpret(request)

    fake = FakeInterpreter()
    app.dependency_overrides[get_chat_action_interpreter] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat/actions/interpret",
                json=_payload("pon 3 unidades"),
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake.calls[0].userMessage == "pon 3 unidades"
    assert response.json()["requestedQuantity"] == 3


def test_chat_action_openapi_exposes_endpoint() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()

    assert "/chat/actions/interpret" in openapi["paths"]
    operation = openapi["paths"]["/chat/actions/interpret"]["post"]
    assert "application/json" in operation["requestBody"]["content"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("/ChatActionIntent")


def _interpret(
    message: str,
    *,
    scope: str = "REQUIREMENT",
    context: dict | None = None,
):
    return ChatActionInterpreter().interpret(
        ChatActionInterpretRequest.model_validate(
            _payload(message, scope=scope, context=context)
        )
    )


def _payload(
    message: str,
    *,
    scope: str = "REQUIREMENT",
    context: dict | None = None,
) -> dict:
    return {
        "scope": scope,
        "userMessage": message,
        "conversation": [],
        "context": context or {"requirementId": "req-1"},
    }
