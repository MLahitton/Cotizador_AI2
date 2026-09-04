import pytest
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


@pytest.mark.parametrize(
    ("message", "expected_value"),
    [
        ("Cambia V-01 a Fermo", "Fermo"),
        ("Cambia V-01 a Venecia Fermo", "Venecia Fermo"),
        ("Podrias por favor cambiar V-01 a Venecia Fermo?", "Venecia Fermo"),
        ("ponlo en Monza", "Monza"),
        ("usa Primavera Lago", "Primavera Lago"),
        ("cambialo por Siena", "Siena"),
        ("pon este en Napoles", "Napoles"),
        ("cambia v-01 a venecia fermo", "venecia fermo"),
    ],
)
def test_interprets_natural_system_requested_values(
    message: str,
    expected_value: str,
) -> None:
    intent = _interpret(message, scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedValue == expected_value
    assert intent.classificationReason == "SYSTEM_VALUE_FROM_MUTATION_PHRASE"


@pytest.mark.parametrize(
    "message",
    [
        "que es Venecia Fermo?",
        "que sistema tiene V-01?",
        "Venecia Fermo sirve para esta ventana?",
    ],
)
def test_natural_system_informational_messages_are_not_actions(message: str) -> None:
    intent = _interpret(message, scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"


def test_change_reference_without_system_value_requires_clarification() -> None:
    intent = _interpret("cambia V-01", scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.targetReference == "V-01"
    assert intent.requiresClarification is True


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


@pytest.mark.parametrize(
    "message",
    [
        "Dime que items les falta algun dato para pasar a sacar el precio",
        "Que items no estan listos para cotizar",
        "Cuales estan bloqueados",
        "Por que este item no tiene precio",
        "Que sistema tiene V-9",
        "Explicame que le falta a este item",
        "Muestrame los items incompletos",
        "Indicame cuales requieren revision",
        "Que items estan excluidos",
        "Cuales no tienen precio",
    ],
)
def test_informational_queries_are_not_actions(message: str) -> None:
    intent = _interpret(message, scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.classificationReason == "INFORMATIONAL_GUARD"


@pytest.mark.parametrize(
    ("message", "expected_action"),
    [
        ("Cambia V-9 a S50", "CHANGE_SYSTEM"),
        ("Usa K50 en este item", "CHANGE_SYSTEM"),
        ("Sube el vidrio a 8 mm", "CHANGE_GLASS"),
        ("Pon templado de 10 mm", "CHANGE_GLASS"),
        ("Ponlo en inox", "CHANGE_FINISH"),
        ("Cambia el acabado a negro mate", "CHANGE_FINISH"),
        ("Pon 3 unidades", "CHANGE_QUANTITY"),
        ("Cambia la cantidad a 2", "CHANGE_QUANTITY"),
        ("Cambialo a 1.80 x 2.40", "CHANGE_DIMENSIONS"),
        ("Pon ancho 1800 y alto 2400", "CHANGE_DIMENSIONS"),
        ("No cotices este item", "EXCLUDE_ITEM"),
        ("Quita V-9 de la cotizacion", "EXCLUDE_ITEM"),
        ("Vuelve a incluirlo", "INCLUDE_ITEM"),
        ("Agrega nuevamente este item a la cotizacion", "INCLUDE_ITEM"),
        ("Quiero toda la propuesta en premium", "CHANGE_COMMERCIAL_LINE"),
        ("Cambia toda la propuesta a signature", "CHANGE_COMMERCIAL_LINE"),
    ],
)
def test_action_table_keeps_mutations_actionable(
    message: str,
    expected_action: str,
) -> None:
    intent = _interpret(message, scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == expected_action


@pytest.mark.parametrize("message", ["Cambialo", "hazlo diferente", "pon otro"])
def test_ambiguous_table_requires_clarification(message: str) -> None:
    intent = _interpret(message, scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.requiresClarification is True


def test_mixed_informational_and_mutation_keeps_explicit_action() -> None:
    intent = _interpret("Dime que sistema tiene V-9 y cambialo a S50")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-9"
    assert intent.requestedValue == "S50"


def test_pending_system_requested_value_follow_up_completes_action() -> None:
    intent = _interpret(
        "Que sea a CUERPO PROYECTANTE LINEA PREMIUM TIPO EUROPEO VENECIA FERMO",
        context=_pending_context(
            action_type="CHANGE_SYSTEM",
            target_reference="V-01",
            requested_value="venecia fermo",
        ),
    )

    assert intent.isAction is True
    assert intent.isFollowUpToPendingAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.scope == "REQUIREMENT"
    assert intent.targetReference == "V-01"
    assert (
        intent.requestedValue
        == "CUERPO PROYECTANTE LINEA PREMIUM TIPO EUROPEO VENECIA FERMO"
    )
    assert intent.requiresClarification is False
    assert intent.classificationReason == "PENDING_ACTION_FOLLOWUP"


def test_pending_system_short_follow_up_preserves_target() -> None:
    intent = _interpret(
        "el proyectante fermo",
        context=_pending_context(action_type="CHANGE_SYSTEM", target_reference="V-01"),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-01"
    assert intent.requestedValue == "proyectante fermo"


def test_pending_finish_follow_up_completes_value() -> None:
    intent = _interpret(
        "que sea inox",
        context=_pending_context(action_type="CHANGE_FINISH", target_reference="V-01"),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.targetReference == "V-01"
    assert intent.requestedValue == "inox"


def test_pending_glass_follow_up_completes_value() -> None:
    intent = _interpret(
        "8 mm",
        context=_pending_context(action_type="CHANGE_GLASS", target_reference="V-01"),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedValue == "8 mm"


def test_pending_quantity_follow_up_completes_quantity() -> None:
    intent = _interpret(
        "3 unidades",
        context=_pending_context(action_type="CHANGE_QUANTITY", target_reference="V-01"),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_QUANTITY"
    assert intent.requestedQuantity == 3


def test_pending_dimensions_follow_up_completes_dimensions() -> None:
    intent = _interpret(
        "1.80 x 2.40",
        context=_pending_context(action_type="CHANGE_DIMENSIONS", target_reference="V-01"),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_DIMENSIONS"
    assert intent.requestedWidthMm == 1800
    assert intent.requestedHeightMm == 2400


def test_pending_item_scope_follow_up_preserves_item_scope() -> None:
    intent = _interpret(
        "que sea inox",
        scope="ITEM",
        context=_pending_context(
            action_type="CHANGE_FINISH",
            scope="ITEM",
            target_reference=None,
            target_item_id="item-1",
        ),
    )

    assert intent.isAction is True
    assert intent.scope == "ITEM"
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.requestedValue == "inox"


def test_pending_action_unrelated_informational_message_is_preserved() -> None:
    intent = _interpret(
        "Cuanto cuesta V-3?",
        context=_pending_context(action_type="CHANGE_SYSTEM", target_reference="V-01"),
    )

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.isFollowUpToPendingAction is False
    assert intent.classificationReason == "INFORMATIONAL_GUARD"


def test_pending_action_new_explicit_target_overrides_pending() -> None:
    intent = _interpret(
        "Cambia V-03 a S50",
        context=_pending_context(action_type="CHANGE_SYSTEM", target_reference="V-01"),
    )

    assert intent.isAction is True
    assert intent.isFollowUpToPendingAction is False
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-03"
    assert intent.requestedValue == "S50"


def test_pending_action_option_reference_resolves_available_option() -> None:
    intent = _interpret(
        "la segunda",
        context=_pending_context(
            action_type="CHANGE_SYSTEM",
            target_reference="V-01",
            available_options=[
                {"displayName": "Venecia Siena"},
                {"code": "FERMO-01", "displayName": "Venecia Fermo"},
            ],
        ),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedValue == "FERMO-01"
    assert intent.classificationReason == "PENDING_ACTION_OPTION_RESOLVED"


def test_pending_action_ambiguous_option_without_options_requires_clarification() -> None:
    intent = _interpret(
        "esa",
        context=_pending_context(action_type="CHANGE_SYSTEM", target_reference="V-01"),
    )

    assert intent.isAction is False
    assert intent.isFollowUpToPendingAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requiresClarification is True
    assert intent.classificationReason == "PENDING_ACTION_FOLLOWUP_AMBIGUOUS"


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


def _pending_context(
    *,
    action_type: str,
    scope: str = "REQUIREMENT",
    target_reference: str | None = "V-01",
    target_item_id: str | None = None,
    requested_value: str | None = None,
    available_options: list[dict] | None = None,
) -> dict:
    return {
        "pendingAction": {
            "scope": scope,
            "actionType": action_type,
            "targetTechnicalProposalItemId": target_item_id,
            "targetReference": target_reference,
            "requestedValue": requested_value,
            "clarificationExpected": "requestedValue",
            "clarificationReason": "SYSTEM_AMBIGUOUS",
            "availableOptions": available_options or [],
        }
    }
