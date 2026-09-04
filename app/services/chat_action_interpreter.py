import re

from app.models.chat_actions import ChatActionIntent, ChatActionInterpretRequest


class ChatActionInterpreter:
    def interpret(self, request: ChatActionInterpretRequest) -> ChatActionIntent:
        message = request.userMessage.strip()
        text = _normalize_text(message)
        scope = _scope(request)
        target_reference = _target_reference(message)

        if _is_informational(text):
            return _intent(
                request,
                action_type="UNKNOWN",
                scope=scope,
                target_reference=target_reference,
                is_action=False,
                confidence=0.9,
            )

        dimensions = _dimensions_mm(text)
        if dimensions is not None and _has_dimension_intent(text):
            return _intent(
                request,
                action_type="CHANGE_DIMENSIONS",
                scope=scope,
                target_reference=target_reference,
                requested_width_mm=dimensions[0],
                requested_height_mm=dimensions[1],
                confidence=0.92,
            )

        quantity = _quantity(text)
        if quantity is not None and _has_quantity_intent(text):
            return _intent(
                request,
                action_type="CHANGE_QUANTITY",
                scope=scope,
                target_reference=target_reference,
                requested_quantity=quantity,
                confidence=0.94,
            )
        if _has_quantity_intent(text):
            return _intent(
                request,
                action_type="CHANGE_QUANTITY",
                scope=scope,
                target_reference=target_reference,
                requires_clarification=True,
                clarification_reason="Falta una cantidad entera positiva.",
                confidence=0.65,
            )

        if _has_exclude_intent(text):
            return _intent(
                request,
                action_type="EXCLUDE_ITEM",
                scope=scope,
                target_reference=target_reference,
                confidence=0.95,
            )

        if _has_include_intent(text):
            return _intent(
                request,
                action_type="INCLUDE_ITEM",
                scope=scope,
                target_reference=target_reference,
                confidence=0.95,
            )

        commercial_line = _commercial_line_value(message, text)
        if commercial_line is not None or _has_commercial_line_intent(text):
            return _intent(
                request,
                action_type="CHANGE_COMMERCIAL_LINE",
                scope="REQUIREMENT" if _global_scope_requested(text) else scope,
                target_reference=target_reference,
                requested_value=commercial_line,
                requires_clarification=commercial_line is None,
                clarification_reason=(
                    "Falta la linea comercial solicitada."
                    if commercial_line is None
                    else None
                ),
                confidence=0.88 if commercial_line is not None else 0.68,
            )

        glass = _glass_value(message, text)
        if glass is not None or _has_glass_intent(text):
            return _intent(
                request,
                action_type="CHANGE_GLASS",
                scope=scope,
                target_reference=target_reference,
                requested_value=glass,
                requires_clarification=glass is None,
                clarification_reason=(
                    "Falta el vidrio solicitado."
                    if glass is None
                    else None
                ),
                confidence=0.9 if glass is not None else 0.68,
            )

        finish = _finish_value(message, text)
        if finish is not None or _has_finish_intent(text):
            return _intent(
                request,
                action_type="CHANGE_FINISH",
                scope=scope,
                target_reference=target_reference,
                requested_value=finish,
                requires_clarification=finish is None,
                clarification_reason=(
                    "Falta el acabado solicitado."
                    if finish is None
                    else None
                ),
                confidence=0.9 if finish is not None else 0.68,
            )

        system = _system_value(message, text)
        if system is not None or _has_system_intent(text):
            return _intent(
                request,
                action_type="CHANGE_SYSTEM",
                scope=scope,
                target_reference=target_reference,
                requested_value=system,
                requires_clarification=system is None,
                clarification_reason=(
                    "Falta el sistema solicitado."
                    if system is None
                    else None
                ),
                confidence=0.92 if system is not None else 0.68,
            )

        if _has_generic_change_intent(text):
            return _intent(
                request,
                action_type="UNKNOWN",
                scope=scope,
                target_reference=target_reference,
                requires_clarification=True,
                clarification_reason="Falta especificar que cambio desea realizar.",
                confidence=0.55,
            )

        return _intent(
            request,
            action_type="UNKNOWN",
            scope=scope,
            target_reference=target_reference,
            is_action=False,
            confidence=0.72,
        )


def _intent(
    request: ChatActionInterpretRequest,
    *,
    action_type: str,
    scope: str,
    target_reference: str | None = None,
    requested_value: str | None = None,
    requested_quantity: int | None = None,
    requested_width_mm: int | None = None,
    requested_height_mm: int | None = None,
    requires_clarification: bool = False,
    clarification_reason: str | None = None,
    is_action: bool = True,
    confidence: float,
) -> ChatActionIntent:
    executable = is_action and action_type != "UNKNOWN"
    return ChatActionIntent(
        isAction=executable and not requires_clarification,
        actionType=action_type,
        scope=scope,
        targetReference=target_reference,
        requestedValue=requested_value,
        requestedQuantity=requested_quantity,
        requestedWidthMm=requested_width_mm,
        requestedHeightMm=requested_height_mm,
        confidence=confidence,
        requiresClarification=requires_clarification,
        clarificationReason=clarification_reason,
        rawUserMessage=request.userMessage,
    )


def _scope(request: ChatActionInterpretRequest) -> str:
    if request.scope == "ITEM":
        return "ITEM"
    context = request.context
    item_context = context.get("item") if isinstance(context, dict) else None
    if isinstance(item_context, dict) and (
        item_context.get("technicalProposalItemId") or item_context.get("itemId")
    ):
        return "ITEM"
    if isinstance(context, dict) and context.get("technicalProposalItemId"):
        return "ITEM"
    return "REQUIREMENT"


def _target_reference(message: str) -> str | None:
    match = re.search(r"\b([A-Za-z]{1,4})\s*[-_ ]\s*(\d{1,3})\b", message)
    if not match:
        return None
    return f"{match.group(1).upper()}-{match.group(2)}"


def _normalize_text(value: str) -> str:
    text = value.casefold()
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _is_informational(text: str) -> bool:
    return (
        "?" in text
        or text.startswith(("que ", "cual ", "cuanto ", "por que ", "porque "))
    ) and not _has_operational_verb(text)


def _has_operational_verb(text: str) -> bool:
    return any(word in text for word in ("cambia", "pon", "quita", "exclu", "inclu", "sube"))


def _has_generic_change_intent(text: str) -> bool:
    return any(word in text for word in ("cambia", "cambialo", "ponlo", "ponle", "quiero otro"))


def _has_system_intent(text: str) -> bool:
    return "sistema" in text or bool(_system_code_match(text))


def _system_value(message: str, text: str) -> str | None:
    match = _system_code_match(message)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(?:a|en|por|usa|usar|pon|ponlo en|cambia a)\s+([A-Za-z]+\d*)\b", message)
    if match and _has_system_intent(text):
        return match.group(1).strip()
    return None


def _system_code_match(text: str) -> re.Match[str] | None:
    return re.search(r"\b([KS]\d{2,3}|[A-Za-z]*\d{4}|Monumental|Monza)\b", text, re.IGNORECASE)


def _has_glass_intent(text: str) -> bool:
    return any(word in text for word in ("vidrio", "templado", "laminado", "cristal"))


def _glass_value(message: str, text: str) -> str | None:
    if not _has_glass_intent(text):
        return None
    thickness = re.search(r"\b(\d+(?:[.,]\d+)?)\s*mm\b", message, re.IGNORECASE)
    if thickness:
        return f"{thickness.group(1).replace(',', '.')} mm"
    composition = re.search(r"\b(\d+(?:[.,]\d+)?\s*\+\s*\d+(?:[.,]\d+)?)\b", message)
    if composition:
        return composition.group(1).replace(",", ".").strip()
    match = re.search(
        r"\b(?:vidrio|cristal|templado|laminado)\s+(?:de\s+)?(.+)$",
        message,
        re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()
    return None


def _has_finish_intent(text: str) -> bool:
    return any(
        word in text
        for word in ("acabado", "color", "inox", "negro", "blanco", "gris", "champana")
    )


def _finish_value(message: str, text: str) -> str | None:
    if "otro acabado" in text or "otro color" in text:
        return None
    match = re.search(r"\b(?:acabado|color)\s+(?:en\s+)?(.+)$", message, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        return value if value else None
    for word in ("inox", "negro mate", "negro", "blanco", "gris", "champaña", "champana"):
        if word in text:
            return _slice_original(message, word) or word
    return None


def _has_quantity_intent(text: str) -> bool:
    return any(word in text for word in ("cantidad", "unidades", "unidad", "estos"))


def _quantity(text: str) -> int | None:
    match = re.search(r"\b(\d+)\s*(?:unidades|unidad|und|de estos)?\b", text)
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _has_dimension_intent(text: str) -> bool:
    return any(word in text for word in (" x ", " por ", "ancho", "alto", "medida", "dimens"))


def _dimensions_mm(text: str) -> tuple[int, int] | None:
    number = r"(\d+(?:[.,]\d+)?)"
    match = re.search(
        rf"{number}\s*(mm|cm|m)?\s*(?:x|por)\s*{number}\s*(mm|cm|m)?",
        text,
    )
    if match:
        first_unit = match.group(2) or match.group(4) or _default_unit(match.group(1))
        second_unit = match.group(4) or first_unit
        return (
            _to_mm(match.group(1), first_unit),
            _to_mm(match.group(3), second_unit),
        )
    match = re.search(
        rf"ancho\s*{number}\s*(mm|cm|m)?\D+alto\s*{number}\s*(mm|cm|m)?",
        text,
    )
    if not match:
        return None
    first_unit = match.group(2) or match.group(4) or _default_unit(match.group(1))
    second_unit = match.group(4) or first_unit
    return (
        _to_mm(match.group(1), first_unit),
        _to_mm(match.group(3), second_unit),
    )


def _to_mm(value: str, unit: str) -> int:
    number = float(value.replace(",", "."))
    if unit == "m":
        return round(number * 1000)
    if unit == "cm":
        return round(number * 10)
    return round(number)


def _default_unit(value: str) -> str:
    return "m" if "." in value or "," in value else "mm"


def _has_exclude_intent(text: str) -> bool:
    return any(phrase in text for phrase in ("no cotices", "quita", "excluye", "saca"))


def _has_include_intent(text: str) -> bool:
    return any(phrase in text for phrase in ("vuelve a incluir", "incluye", "agrega", "cotizalo"))


def _has_commercial_line_intent(text: str) -> bool:
    return any(
        word in text
        for word in ("premium", "essential", "essentials", "signature", "linea")
    )


def _commercial_line_value(message: str, text: str) -> str | None:
    for word in ("premium", "essential", "essentials", "signature"):
        if word in text:
            return _slice_original(message, word) or word
    match = re.search(r"\blinea\s+(.+)$", message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _global_scope_requested(text: str) -> bool:
    return any(phrase in text for phrase in ("toda la propuesta", "todos", "todas"))


def _slice_original(message: str, normalized_word: str) -> str | None:
    normalized_message = _normalize_text(message)
    index = normalized_message.find(_normalize_text(normalized_word))
    if index < 0:
        return None
    return message[index : index + len(normalized_word)].strip()
