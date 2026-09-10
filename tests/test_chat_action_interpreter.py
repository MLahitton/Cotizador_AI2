import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_action_interpreter
from app.main import app
from app.models.chat_actions import ChatActionInterpretRequest
from app.services.chat_action_interpreter import ChatActionInterpreter


@pytest.mark.parametrize(
    "message",
    [
        "Confirma la seleccion",
        "Confirma esta propuesta",
        "Quiero confirmar la propuesta",
        "Dejala confirmada",
        "Puedes confirmar la seleccion?",
        "Me confirmas la propuesta?",
    ],
)
def test_interprets_confirm_selection_as_proposal_action(message: str) -> None:
    intent = _interpret(message, scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CONFIRM_SELECTION"
    assert intent.scope == "REQUIREMENT"
    assert intent.targetReference is None
    assert intent.targetReferences == []
    assert intent.classificationReason == "CONFIRM_SELECTION_MUTATION"


@pytest.mark.parametrize(
    "message",
    [
        "La propuesta ya esta confirmada?",
        "Esta lista para confirmar?",
        "Que falta para confirmar?",
    ],
)
def test_confirmation_status_questions_remain_read_only(message: str) -> None:
    intent = _interpret(message)

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.classificationReason == "INFORMATIONAL_GUARD"

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


@pytest.mark.parametrize(
    "message",
    [
        "Cambia V-4 a Monza",
        "Ponle Monza a V-4",
        "Quiero Monza en V-4",
        "A V-4 dejale el sistema Monza",
        "Ese V-4 mejor con Monza",
        "Me cambias V-4 por Monza?",
    ],
)
def test_interprets_natural_system_phrasing_with_target(message: str) -> None:
    intent = _interpret(message)

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-4"
    assert intent.targetReferences == ["V-4"]
    assert intent.requestedValue == "Monza"


@pytest.mark.parametrize(
    "message",
    [
        "Que sistema tiene V-4?",
        "Monza seria compatible con V-4?",
        "Que opciones tengo para V-4?",
    ],
)
def test_natural_read_only_system_queries_do_not_mutate(message: str) -> None:
    intent = _interpret(message)

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.targetReference == "V-4"
    assert intent.classificationReason == "INFORMATIONAL_GUARD"


@pytest.mark.parametrize(
    "message",
    [
        "Pon 2 unidades en V-4",
        "Deja dos unidades en V-4",
        "Cantidad 2 para V-4",
        "V-4 dejalo en 2",
    ],
)
def test_interprets_natural_quantity_without_reference_number_leak(message: str) -> None:
    intent = _interpret(message)

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_QUANTITY"
    assert intent.targetReference == "V-4"
    assert intent.targetReferences == ["V-4"]
    assert intent.requestedQuantity == 2


def test_interprets_dimensions_without_false_targets() -> None:
    intent = _interpret("cambia V-4 a 1.8 x 2.4")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_DIMENSIONS"
    assert intent.targetReference == "V-4"
    assert intent.targetReferences == ["V-4"]
    assert intent.requestedWidthMm == 1800
    assert intent.requestedHeightMm == 2400


@pytest.mark.parametrize(
    "message",
    [
        "Pon 2 unidades en V-4",
        "cambia V-4 a 1.8 x 2.4",
    ],
)
def test_does_not_extract_false_targets_from_commands_or_dimensions(message: str) -> None:
    intent = _interpret(message)

    assert "PON-2" not in intent.targetReferences
    assert "A-1" not in intent.targetReferences
    assert "X-2" not in intent.targetReferences


def test_partial_dimension_requires_clarification_instead_of_system_change() -> None:
    intent = _interpret("dejalo en 1800 mm de ancho", scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "CHANGE_DIMENSIONS"
    assert intent.requestedWidthMm == 1800
    assert intent.requestedHeightMm is None
    assert intent.requiresClarification is True
    assert intent.classificationReason == "PARTIAL_DIMENSION_MUTATION"


@pytest.mark.parametrize("target", ["A-01", "TAG-01", "HOJA-02"])
def test_interprets_supported_alphanumeric_target_prefixes(target: str) -> None:
    intent = _interpret(f"Cambia {target} a Monza")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == target
    assert intent.targetReferences == [target]
    assert intent.requestedValue == "Monza"


def test_interprets_homogeneous_system_batch_with_shared_value() -> None:
    intent = _interpret("Pon V-4 y V-5 en Monza")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-4"
    assert intent.targetReferences == ["V-4", "V-5"]
    assert intent.requestedValue == "Monza"

@pytest.mark.parametrize(
    ("message", "expected_value"),
    [
        ("Pon TEMP_5 en V-4", "TEMP_5"),
        ("Cambia V-4 a TEMP_5", "TEMP_5"),
        ("Dejale TEMP_6 a V-4", "TEMP_6"),
    ],
)
def test_interprets_glass_codes_before_system(message: str, expected_value: str) -> None:
    intent = _interpret(message)

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.targetReference == "V-4"
    assert intent.targetReferences == ["V-4"]
    assert intent.requestedValue == expected_value


@pytest.mark.parametrize(
    ("message", "expected_value"),
    [
        ("Me puedes poner templado 5 en V-4?", "templado 5"),
        ("Me cambias el vidrio de V-4 a templado 5?", "templado 5"),
        ("Puedes ponerle TEMP_5 a V-4?", "TEMP_5"),
    ],
)
def test_interprets_action_questions_for_glass_as_mutations(
    message: str,
    expected_value: str,
) -> None:
    intent = _interpret(message)

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.targetReference == "V-4"
    assert intent.targetReferences == ["V-4"]
    assert intent.requestedValue == expected_value
    assert intent.requiresClarification is False


@pytest.mark.parametrize(
    "message",
    [
        "Templado 5 esta bien para V-4?",
        "Templado 5 seria compatible con V-4?",
        "Que vidrio tiene V-4?",
    ],
)
def test_glass_read_only_questions_do_not_mutate(message: str) -> None:
    intent = _interpret(message)

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.classificationReason == "INFORMATIONAL_GUARD"


@pytest.mark.parametrize(
    "message",
    [
        "Copia la configuracion de V-4 a V-5",
        "Copia V-4 en V-5",
        "Duplica la configuracion de V-4 a V-5",
        "Haz que V-5 quede igual que V-4",
    ],
)
def test_copy_configuration_fails_safe_without_supported_action(message: str) -> None:
    intent = _interpret(message)

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.targetReferences == ["V-4", "V-5"] or intent.targetReferences == ["V-5", "V-4"]
    assert intent.requiresClarification is True
    assert intent.classificationReason == "COPY_CONFIGURATION_UNSUPPORTED"
    assert "Copiar configuracion" in (intent.clarificationReason or "")

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


def test_interprets_exclude_multiple_targets() -> None:
    intent = _interpret("Excluye C-4b y C-4c por favor")

    assert intent.isAction is True
    assert intent.actionType == "EXCLUDE_ITEM"
    assert intent.targetReference == "C-4b"
    assert intent.targetReferences == ["C-4b", "C-4c"]
    assert intent.targetCount == 2
    assert intent.classificationReason == "MULTI_TARGET_ACTION"


def test_interprets_include_multiple_targets() -> None:
    intent = _interpret("Vuelve a incluir PV-1 y PV-2")

    assert intent.isAction is True
    assert intent.actionType == "INCLUDE_ITEM"
    assert intent.targetReference == "PV-1"
    assert intent.targetReferences == ["PV-1", "PV-2"]
    assert intent.targetCount == 2


def test_interprets_system_multiple_targets_shared_value() -> None:
    intent = _interpret("Pon V-9 y V-10 en S50")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-9"
    assert intent.targetReferences == ["V-9", "V-10"]
    assert intent.targetCount == 2
    assert intent.requestedValue == "S50"
    assert intent.classificationReason == "MULTI_TARGET_ACTION"


def test_interprets_glass_multiple_targets_shared_attributes() -> None:
    intent = _interpret("Cambia V-1, V-2 y V-3 a vidrio templado de 8 mm")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.targetReferences == ["V-1", "V-2", "V-3"]
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 8


def test_interprets_finish_multiple_targets_shared_attributes() -> None:
    intent = _interpret("Pon acabado negro mate en C-1, C-2a y C-2b")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.targetReferences == ["C-1", "C-2a", "C-2b"]
    assert intent.requestedValue == "negro mate"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.finish is not None
    assert intent.requestedAttributes.finish.color == "BLACK"
    assert intent.requestedAttributes.finish.texture == "MATTE"


def test_multi_target_deduplicates_repeated_references() -> None:
    intent = _interpret("Excluye C-4b, C-4b y C-4c")

    assert intent.isAction is True
    assert intent.targetReferences == ["C-4b", "C-4c"]
    assert intent.targetCount == 2


def test_multi_target_accepts_semicolon_separator() -> None:
    intent = _interpret("Excluye C-4b, C-4c; C-4a")

    assert intent.isAction is True
    assert intent.targetReferences == ["C-4b", "C-4c", "C-4a"]
    assert intent.targetCount == 3


def test_interprets_commercial_line_requirement() -> None:
    intent = _interpret("quiero toda la propuesta en premium")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_COMMERCIAL_LINE"
    assert intent.scope == "REQUIREMENT"
    assert intent.requestedValue == "premium"


def test_interprets_tempered_glass_attributes() -> None:
    intent = _interpret("Ponle vidrio templado de 6 mm", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedValue == "vidrio templado de 6 mm"
    assert intent.targetReference is None
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 6


def test_interprets_laminated_glass_attributes() -> None:
    intent = _interpret("Quiero laminado 4+4", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.composition == "LAMINATED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 4
    assert intent.requestedAttributes.glass.innerThicknessMm == 4


def test_interprets_tempered_black_glass_attributes() -> None:
    intent = _interpret("pon templado de 10 mm negro", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 10
    assert intent.requestedAttributes.glass.color == "BLACK"


def test_interprets_monolithic_tempered_glass_attributes() -> None:
    intent = _interpret("Cambia a vidrio monolÃ­tico templado de 6 mm", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.family == "MONOLITHIC"
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 6
    assert intent.classificationReason == "GLASS_FAMILY_EXTRACTED"


def test_interprets_laminated_family_and_pair_thickness() -> None:
    intent = _interpret("Pon laminado 4+4", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.family == "LAMINATED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 4
    assert intent.requestedAttributes.glass.innerThicknessMm == 4


def test_interprets_laminated_tempered_family_and_composition() -> None:
    intent = _interpret("Pon laminado templado 6+6", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.family == "LAMINATED"
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 6
    assert intent.requestedAttributes.glass.innerThicknessMm == 6


def test_interprets_igu_family_and_chamber_thickness() -> None:
    intent = _interpret("Pon doble vidrio con cÃ¡mara de 12 mm", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.family == "IGU"
    assert intent.requestedAttributes.glass.chamberThicknessMm == 12


def test_interprets_sliding_door_system_attributes() -> None:
    intent = _interpret("Cambia PV-5 a puerta corrediza Venecia Monaco")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "PV-5"
    assert intent.requestedValue == "puerta corrediza Venecia Monaco"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.system is not None
    assert intent.requestedAttributes.system.functionalType == "SLIDING_DOOR"
    assert intent.requestedAttributes.system.operation == "SLIDING"
    assert intent.requestedAttributes.system.commercialName == "VENECIA MONACO"
    assert intent.requestedAttributes.system.family == "VENECIA MONACO"


def test_interprets_sliding_window_system_attributes() -> None:
    intent = _interpret("Pon una ventana corrediza Monza")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.system is not None
    assert intent.requestedAttributes.system.functionalType == "SLIDING_WINDOW"
    assert intent.requestedAttributes.system.operation == "SLIDING"
    assert intent.requestedAttributes.system.commercialName == "MONZA"


def test_interprets_swing_door_system_attributes() -> None:
    intent = _interpret("Usa puerta batiente 3890")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.system is not None
    assert intent.requestedAttributes.system.functionalType == "SWING_DOOR"
    assert intent.requestedAttributes.system.operation == "CASEMENT"
    assert intent.requestedAttributes.system.commercialName == "3890"


def test_interprets_fixed_system_attributes() -> None:
    intent = _interpret("Pon un fijo Venecia Fermo")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.system is not None
    assert intent.requestedAttributes.system.functionalType == "FIXED"
    assert intent.requestedAttributes.system.operation == "FIXED"
    assert intent.requestedAttributes.system.commercialName == "VENECIA FERMO"


def test_commercial_name_without_function_does_not_invent_function() -> None:
    intent = _interpret("Cambia a Venecia Monaco")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.system is not None
    assert intent.requestedAttributes.system.functionalType is None
    assert intent.requestedAttributes.system.operation is None
    assert intent.requestedAttributes.system.commercialName == "VENECIA MONACO"


def test_interprets_finish_attributes_black_matte() -> None:
    intent = _interpret("Ponlo negro mate", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.finish is not None
    assert intent.requestedAttributes.finish.color == "BLACK"
    assert intent.requestedAttributes.finish.texture == "MATTE"


def test_interprets_finish_attributes_inox() -> None:
    intent = _interpret("Quiero acabado inox", scope="ITEM")

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.finish is not None
    assert intent.requestedAttributes.finish.material == "STAINLESS_STEEL"
    assert intent.requestedAttributes.finish.normalizedType == "STAINLESS_STEEL"


def test_glass_informational_message_is_not_action() -> None:
    intent = _interpret("Que es un vidrio templado de 6 mm?", scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"


def test_glass_family_informational_message_is_not_action() -> None:
    intent = _interpret("QuÃ© diferencia hay entre monolÃ­tico y laminado?", scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"


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


def test_multi_target_informational_query_is_not_action() -> None:
    intent = _interpret("Â¿QuÃ© sistema tienen V-9 y V-10?", scope="ITEM")

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.targetReferences == ["V-9", "V-10"]
    assert intent.classificationReason == "INFORMATIONAL_GUARD"


def test_heterogeneous_system_batch_requires_clarification() -> None:
    intent = _interpret("Cambia V-1 a S50 y V-2 a K50")

    assert intent.isAction is False
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReferences == ["V-1", "V-2"]
    assert intent.requiresClarification is True
    assert intent.classificationReason == "HETEROGENEOUS_BATCH_REQUIRES_CLARIFICATION"


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


def test_pending_action_copy_same_change_to_explicit_target() -> None:
    intent = _interpret(
        "Ahora haz lo mismo con V-4",
        context=_pending_context(
            action_type="CHANGE_SYSTEM",
            target_reference="V-3",
            requested_value="Siena",
        ),
    )

    assert intent.isAction is True
    assert intent.isFollowUpToPendingAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-4"
    assert intent.targetReferences == ["V-4"]
    assert intent.requestedValue == "Siena"
    assert intent.classificationReason == "PENDING_ACTION_COPY_TO_TARGET"


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


def test_pending_glass_follow_up_populates_attributes() -> None:
    intent = _interpret(
        "templado de 6 mm",
        context=_pending_context(action_type="CHANGE_GLASS", target_reference="V-01"),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedValue == "templado de 6 mm"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 6


def test_pending_glass_follow_up_enriches_family_without_losing_attributes() -> None:
    intent = _interpret(
        "monolÃ­tico",
        context=_pending_context(
            action_type="CHANGE_GLASS",
            target_reference="PV-1",
            requested_value="templado de 6 mm",
            requested_attributes={
                "glass": {
                    "composition": "TEMPERED",
                    "outerThicknessMm": 6,
                }
            },
        ),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedValue == "monolÃ­tico"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.family == "MONOLITHIC"
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 6
    assert intent.classificationReason == "PENDING_GLASS_ATTRIBUTES_ENRICHED"


def test_pending_glass_follow_up_enriches_monolithic_composition_phrase() -> None:
    intent = _interpret(
        "Que sea Composicion Monolitico templado 6 MM INC",
        context=_pending_context(
            action_type="CHANGE_GLASS",
            target_reference="PV-1",
            requested_value="templado de 6 mm",
            requested_attributes={
                "glass": {
                    "composition": "TEMPERED",
                    "outerThicknessMm": 6,
                }
            },
        ),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedValue == "Composicion Monolitico templado 6 MM INC"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.family == "MONOLITHIC"
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 6
    assert intent.requiresClarification is False


def test_pending_glass_follow_up_explicit_laminated_overrides_monolithic() -> None:
    intent = _interpret(
        "mejor laminado 4+4",
        context=_pending_context(
            action_type="CHANGE_GLASS",
            target_reference="PV-1",
            requested_value="monolitico",
            requested_attributes={
                "glass": {
                    "family": "MONOLITHIC",
                }
            },
        ),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.family == "LAMINATED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 4
    assert intent.requestedAttributes.glass.innerThicknessMm == 4


def test_pending_system_follow_up_populates_attributes() -> None:
    intent = _interpret(
        "puerta corrediza Venecia Monaco",
        context=_pending_context(action_type="CHANGE_SYSTEM", target_reference="PV-5"),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.system is not None
    assert intent.requestedAttributes.system.functionalType == "SLIDING_DOOR"
    assert intent.requestedAttributes.system.commercialName == "VENECIA MONACO"


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


def test_pending_glass_target_reference_follow_up_preserves_value_and_attributes() -> None:
    requested_attributes = {
        "glass": {
            "composition": "TEMPERED",
            "outerThicknessMm": 6,
        }
    }

    intent = _interpret(
        "PV-1",
        context=_pending_context(
            action_type="CHANGE_GLASS",
            target_reference=None,
            requested_value="COMPOSICION MONOLITICO TEMPLADO 6 MM INC.",
            requested_attributes=requested_attributes,
            clarification_expected="targetReference",
            available_options=[
                {"reference": "PV-1"},
                {"reference": "PV-2"},
                {"reference": "PV-3a"},
            ],
        ),
    )

    assert intent.isAction is True
    assert intent.isFollowUpToPendingAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.targetReference == "PV-1"
    assert intent.requestedValue == "COMPOSICION MONOLITICO TEMPLADO 6 MM INC."
    assert intent.requestedAttributes is not None
    assert intent.requestedAttributes.glass is not None
    assert intent.requestedAttributes.glass.composition == "TEMPERED"
    assert intent.requestedAttributes.glass.outerThicknessMm == 6
    assert intent.requiresClarification is False
    assert intent.classificationReason == "PENDING_ACTION_TARGET_RESOLVED"


@pytest.mark.parametrize(
    ("message", "expected_target"),
    [
        ("V-3", "V-3"),
        ("V1", "V-1"),
        ("pv-1", "PV-1"),
        ("C-2a", "C-2a"),
    ],
)
def test_pending_target_reference_accepts_reference_formats(
    message: str,
    expected_target: str,
) -> None:
    intent = _interpret(
        message,
        context=_pending_context(
            action_type="CHANGE_SYSTEM",
            target_reference=None,
            requested_value="S50",
            clarification_expected="targetReference",
        ),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == expected_target
    assert intent.requestedValue == "S50"


def test_pending_target_reference_resolves_ordinal_available_option() -> None:
    intent = _interpret(
        "la segunda",
        context=_pending_context(
            action_type="CHANGE_FINISH",
            target_reference=None,
            requested_value="negro mate",
            clarification_expected="targetReference",
            available_options=[
                {"reference": "PV-1"},
                {"reference": "PV-2"},
                {"reference": "PV-3a"},
            ],
        ),
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_FINISH"
    assert intent.targetReference == "PV-2"
    assert intent.requestedValue == "negro mate"
    assert intent.classificationReason == "PENDING_ACTION_TARGET_RESOLVED"


def test_pending_target_reference_rejects_unavailable_target_option() -> None:
    intent = _interpret(
        "PV-99",
        context=_pending_context(
            action_type="CHANGE_GLASS",
            target_reference=None,
            requested_value="templado de 6 mm",
            clarification_expected="targetReference",
            available_options=[
                {"reference": "PV-1"},
                {"reference": "PV-2"},
            ],
        ),
    )

    assert intent.isAction is False
    assert intent.isFollowUpToPendingAction is True
    assert intent.actionType == "CHANGE_GLASS"
    assert intent.targetReference is None
    assert intent.requiresClarification is True
    assert intent.classificationReason == "PENDING_ACTION_TARGET_AMBIGUOUS"


def test_pending_target_reference_informational_message_does_not_complete() -> None:
    intent = _interpret(
        "Cuanto cuesta V-3?",
        context=_pending_context(
            action_type="CHANGE_GLASS",
            target_reference=None,
            requested_value="templado de 6 mm",
            clarification_expected="targetReference",
        ),
    )

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.isFollowUpToPendingAction is False
    assert intent.classificationReason == "INFORMATIONAL_GUARD"


def test_pending_target_reference_new_explicit_action_overrides_pending() -> None:
    intent = _interpret(
        "Cambia V-4 a S50",
        context=_pending_context(
            action_type="CHANGE_GLASS",
            target_reference=None,
            requested_value="templado de 6 mm",
            clarification_expected="targetReference",
        ),
    )

    assert intent.isAction is True
    assert intent.isFollowUpToPendingAction is False
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReference == "V-4"
    assert intent.requestedValue == "S50"


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


def test_contextual_homogeneous_system_change_uses_previous_user_message() -> None:
    intent = _interpret_with_conversation(
        "haz esos cambios",
        [
            {"role": "user", "content": "Pon V-01 y V-02 en MONACO"},
            {"role": "assistant", "content": "Entendi que quieres V-01 y V-02 en MONACO."},
        ],
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.scope == "REQUIREMENT"
    assert intent.targetReference == "V-01"
    assert intent.targetReferences == ["V-01", "V-02"]
    assert intent.requestedValue == "MONACO"
    assert intent.requiresClarification is False
    assert intent.classificationReason == "CONTEXTUAL_SYSTEM_CHANGE_RESOLVED"


def test_contextual_system_change_with_verbal_confirmation_materializes() -> None:
    intent = _interpret_with_conversation(
        "si, cambia los sistemas",
        [
            {"role": "user", "content": "Pon V-01 y V-02 en MONACO"},
            {"role": "assistant", "content": "Entendi que quieres V-01 y V-02 en MONACO."},
        ],
    )

    assert intent.isAction is True
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.targetReferences == ["V-01", "V-02"]
    assert intent.requestedValue == "MONACO"


def test_contextual_read_only_question_stays_informational() -> None:
    intent = _interpret_with_conversation(
        "tienes claros los cambios?",
        [{"role": "user", "content": "Pon V-01 y V-02 en MONACO"}],
    )

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.classificationReason == "INFORMATIONAL_GUARD"


def test_isolated_yes_without_pending_plan_does_not_reconstruct_old_action() -> None:
    intent = _interpret_with_conversation(
        "si",
        [{"role": "user", "content": "Pon V-01 y V-02 en MONACO"}],
    )

    assert intent.isAction is False
    assert intent.actionType == "UNKNOWN"
    assert intent.classificationReason == "NO_MUTATION_EVIDENCE"


def test_contextual_conflicting_change_sets_require_clarification() -> None:
    intent = _interpret_with_conversation(
        "haz ese cambio",
        [
            {"role": "user", "content": "Pon V-01 en MONACO"},
            {"role": "assistant", "content": "Entendido."},
            {"role": "user", "content": "Pon V-01 en NAPOLES"},
        ],
    )

    assert intent.isAction is False
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requiresClarification is True
    assert intent.classificationReason == "CONTEXTUAL_ACTION_CONFLICTING_CHANGE_SETS"


def test_contextual_uses_those_options_does_not_become_literal_system_value() -> None:
    intent = _interpret_with_conversation("usa esas opciones", [])

    assert intent.isAction is False
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requestedValue is None
    assert intent.requiresClarification is True
    assert intent.classificationReason == "CONTEXTUAL_ACTION_NOT_RESOLVED"


def test_contextual_same_reference_different_occurrence_requires_clarification() -> None:
    intent = _interpret_with_conversation(
        "haz esos cambios",
        [{"role": "user", "content": "Pon V-01 Nivel 1 en MONACO"}],
        context={
            "technicalProposal": {
                "items": [
                    {"itemId": "item-1", "sequence": 1, "reference": "V-01", "occurrenceContext": "Nivel 1"},
                    {"itemId": "item-2", "sequence": 2, "reference": "V-01", "occurrenceContext": "Nivel 2"},
                ]
            }
        },
    )

    assert intent.isAction is False
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requiresClarification is True
    assert intent.classificationReason == "CONTEXTUAL_TARGET_REFERENCE_AMBIGUOUS"


def test_contextual_heterogeneous_system_values_fail_safe() -> None:
    intent = _interpret_with_conversation(
        "si, cambia los sistemas",
        [
            {"role": "user", "content": "V-01 y V-02 en MONACO y V-03 en SIENA"},
            {"role": "assistant", "content": "Entendi V-01 y V-02 en MONACO y V-03 en SIENA."},
        ],
    )

    assert intent.isAction is False
    assert intent.actionType == "CHANGE_SYSTEM"
    assert intent.requiresClarification is True
    assert intent.classificationReason == "CONTEXTUAL_HETEROGENEOUS_VALUES_UNSUPPORTED"


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


def _interpret_with_conversation(
    message: str,
    conversation: list[dict[str, str]],
    *,
    scope: str = "REQUIREMENT",
    context: dict | None = None,
):
    payload = _payload(message, scope=scope, context=context)
    payload["conversation"] = conversation
    return ChatActionInterpreter().interpret(ChatActionInterpretRequest.model_validate(payload))


def _pending_context(
    *,
    action_type: str,
    scope: str = "REQUIREMENT",
    target_reference: str | None = "V-01",
    target_item_id: str | None = None,
    requested_value: str | None = None,
    requested_attributes: dict | None = None,
    clarification_expected: str = "requestedValue",
    available_options: list[dict] | None = None,
) -> dict:
    return {
        "pendingAction": {
            "scope": scope,
            "actionType": action_type,
            "targetTechnicalProposalItemId": target_item_id,
            "targetReference": target_reference,
            "requestedValue": requested_value,
            "requestedAttributes": requested_attributes,
            "clarificationExpected": clarification_expected,
            "clarificationReason": "SYSTEM_AMBIGUOUS",
            "availableOptions": available_options or [],
        }
    }
