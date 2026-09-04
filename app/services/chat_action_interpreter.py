import re
import unicodedata

from app.models.chat_actions import (
    ChatActionIntent,
    ChatActionInterpretRequest,
    ChatFinishRequestedAttributes,
    ChatGlassRequestedAttributes,
    ChatRequestedAttributes,
    ChatSystemRequestedAttributes,
)

_AMBIGUOUS = object()


class ChatActionInterpreter:
    def interpret(self, request: ChatActionInterpretRequest) -> ChatActionIntent:
        message = request.userMessage.strip()
        text = _normalize_text(message)
        scope = _scope(request)
        target_reference = _target_reference(message)
        has_mutation = _has_mutation_intent(text)
        pending_action = _pending_action(request)

        if (
            pending_action is not None
            and (target_reference is not None or _looks_like_pending_follow_up(text))
            and not (
                has_mutation
                and target_reference is not None
                and pending_action.get("clarificationExpected") == "targetReference"
            )
            and not _is_new_action_for_different_target(
                pending_action,
                target_reference,
                has_mutation,
            )
        ):
            follow_up = _pending_action_follow_up(
                request,
                pending_action,
                message,
                text,
            )
            if follow_up is not None:
                return follow_up

        if _is_informational(text) and not has_mutation:
            return _intent(
                request,
                action_type="UNKNOWN",
                scope=scope,
                target_reference=target_reference,
                is_action=False,
                confidence=0.9,
                classification_reason="INFORMATIONAL_GUARD",
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
                classification_reason="EXPLICIT_DIMENSION_MUTATION",
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
                classification_reason="EXPLICIT_QUANTITY_MUTATION",
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
                classification_reason="INCOMPLETE_QUANTITY_MUTATION",
            )

        if _has_exclude_intent(text) and has_mutation:
            return _intent(
                request,
                action_type="EXCLUDE_ITEM",
                scope=scope,
                target_reference=target_reference,
                confidence=0.95,
                classification_reason="EXPLICIT_EXCLUDE_MUTATION",
            )

        if _has_include_intent(text) and has_mutation:
            return _intent(
                request,
                action_type="INCLUDE_ITEM",
                scope=scope,
                target_reference=target_reference,
                confidence=0.95,
                classification_reason="EXPLICIT_INCLUDE_MUTATION",
            )

        commercial_line = _commercial_line_value(message, text)
        if commercial_line is not None or _has_commercial_line_intent(text):
            return _intent(
                request,
                action_type="CHANGE_COMMERCIAL_LINE",
                scope="REQUIREMENT" if _global_scope_requested(text) else scope,
                target_reference=target_reference,
                requested_value=commercial_line,
                requested_attributes=_attributes_for_action(
                    "CHANGE_COMMERCIAL_LINE",
                    commercial_line,
                    text,
                ),
                requires_clarification=commercial_line is None,
                clarification_reason=(
                    "Falta la linea comercial solicitada."
                    if commercial_line is None
                    else None
                ),
                confidence=0.88 if commercial_line is not None else 0.68,
                classification_reason="COMMERCIAL_LINE_MUTATION",
            )

        glass = _glass_value(message, text)
        if glass is not None or _has_glass_intent(text):
            requested_attributes = _attributes_for_action("CHANGE_GLASS", glass, text)
            return _intent(
                request,
                action_type="CHANGE_GLASS",
                scope=scope,
                target_reference=target_reference,
                requested_value=glass,
                requested_attributes=requested_attributes,
                requires_clarification=glass is None,
                clarification_reason=(
                    "Falta el vidrio solicitado."
                    if glass is None
                    else None
                ),
                confidence=0.9 if glass is not None else 0.68,
                classification_reason=(
                    "GLASS_FAMILY_EXTRACTED"
                    if _has_glass_family(requested_attributes)
                    else "GLASS_MUTATION"
                ),
            )

        finish = _finish_value(message, text)
        if finish is not None or _has_finish_intent(text):
            return _intent(
                request,
                action_type="CHANGE_FINISH",
                scope=scope,
                target_reference=target_reference,
                requested_value=finish,
                requested_attributes=_attributes_for_action("CHANGE_FINISH", finish, text),
                requires_clarification=finish is None,
                clarification_reason=(
                    "Falta el acabado solicitado."
                    if finish is None
                    else None
                ),
                confidence=0.9 if finish is not None else 0.68,
                classification_reason="FINISH_MUTATION",
            )

        system = _system_value(message, text)
        if system is not None or _has_system_intent(text):
            return _intent(
                request,
                action_type="CHANGE_SYSTEM",
                scope=scope,
                target_reference=target_reference,
                requested_value=system,
                requested_attributes=_attributes_for_action("CHANGE_SYSTEM", system, text),
                requires_clarification=system is None,
                clarification_reason=(
                    "Falta el sistema solicitado."
                    if system is None
                    else None
                ),
                confidence=0.92 if system is not None else 0.68,
                classification_reason=(
                    "SYSTEM_VALUE_FROM_MUTATION_PHRASE"
                    if system is not None
                    else "SYSTEM_MUTATION"
                ),
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
                classification_reason="INCOMPLETE_MUTATION",
            )

        return _intent(
            request,
            action_type="UNKNOWN",
            scope=scope,
            target_reference=target_reference,
            is_action=False,
            confidence=0.72,
            classification_reason="NO_MUTATION_EVIDENCE",
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
    requested_attributes: ChatRequestedAttributes | None = None,
    requires_clarification: bool = False,
    clarification_reason: str | None = None,
    is_action: bool = True,
    confidence: float,
    classification_reason: str,
    is_follow_up: bool = False,
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
        requestedAttributes=requested_attributes,
        confidence=confidence,
        requiresClarification=requires_clarification,
        clarificationReason=clarification_reason,
        classificationReason=classification_reason,
        isFollowUpToPendingAction=is_follow_up,
        rawUserMessage=request.userMessage,
    )


def _pending_action(request: ChatActionInterpretRequest) -> dict | None:
    context = request.context
    if not isinstance(context, dict):
        return None
    pending_action = context.get("pendingAction")
    return pending_action if isinstance(pending_action, dict) else None


def _pending_action_follow_up(
    request: ChatActionInterpretRequest,
    pending_action: dict,
    message: str,
    text: str,
) -> ChatActionIntent | None:
    action_type = pending_action.get("actionType")
    if _is_informational(text) and not _is_pending_value_phrase(text):
        return None
    if action_type not in {
        "CHANGE_SYSTEM",
        "CHANGE_FINISH",
        "CHANGE_GLASS",
        "CHANGE_QUANTITY",
        "CHANGE_DIMENSIONS",
        "EXCLUDE_ITEM",
        "INCLUDE_ITEM",
    }:
        return None

    scope = (
        pending_action.get("scope")
        if pending_action.get("scope") in {"ITEM", "REQUIREMENT"}
        else request.scope
    )
    target_reference = _optional_str(pending_action.get("targetReference"))
    clarification_expected = pending_action.get("clarificationExpected")

    if clarification_expected == "targetReference":
        resolved_target = _target_follow_up_reference(
            message,
            pending_action.get("availableOptions"),
        )
        if resolved_target is None or resolved_target is _AMBIGUOUS:
            return _pending_clarification(
                request,
                action_type,
                scope,
                target_reference,
                "Falta identificar el item objetivo para completar la accion pendiente.",
                "PENDING_ACTION_TARGET_AMBIGUOUS",
            )
        return _intent(
            request,
            action_type=action_type,
            scope=scope,
            target_reference=resolved_target,
            requested_value=_optional_str(pending_action.get("requestedValue")),
            requested_quantity=_optional_int(pending_action.get("requestedQuantity")),
            requested_width_mm=_optional_int(pending_action.get("requestedWidthMm")),
            requested_height_mm=_optional_int(pending_action.get("requestedHeightMm")),
            requested_attributes=_pending_requested_attributes(pending_action),
            confidence=0.9,
            classification_reason="PENDING_ACTION_TARGET_RESOLVED",
            is_follow_up=True,
        )

    option_value = _available_option_value(message, pending_action.get("availableOptions"))
    if option_value is _AMBIGUOUS:
        return _pending_clarification(
            request,
            action_type,
            scope,
            target_reference,
            "No se pudo resolver la opcion solicitada.",
            "PENDING_ACTION_FOLLOWUP_AMBIGUOUS",
        )

    if action_type in {"CHANGE_SYSTEM", "CHANGE_FINISH", "CHANGE_GLASS"}:
        value = (
            option_value
            if isinstance(option_value, str)
            else _follow_up_requested_value(message)
        )
        if _is_ambiguous_follow_up(text) or value is None:
            return _pending_clarification(
                request,
                action_type,
                scope,
                target_reference,
                "Falta el valor solicitado para completar la accion pendiente.",
                "PENDING_ACTION_FOLLOWUP_AMBIGUOUS",
            )
        requested_attributes = _attributes_for_action(action_type, value, text)
        classification_reason = (
            "PENDING_ACTION_OPTION_RESOLVED"
            if option_value is not None
            else "PENDING_ACTION_FOLLOWUP"
        )
        if action_type == "CHANGE_GLASS":
            requested_attributes = _merge_requested_attributes(
                _pending_requested_attributes(pending_action),
                requested_attributes,
            )
            classification_reason = "PENDING_GLASS_ATTRIBUTES_ENRICHED"
        return _intent(
            request,
            action_type=action_type,
            scope=scope,
            target_reference=target_reference,
            requested_value=value,
            requested_attributes=requested_attributes,
            confidence=0.88,
            classification_reason=classification_reason,
            is_follow_up=True,
        )

    if action_type == "CHANGE_QUANTITY":
        quantity = _quantity(text) or _standalone_positive_int(text)
        if quantity is None:
            return _pending_clarification(
                request,
                action_type,
                scope,
                target_reference,
                "Falta una cantidad entera positiva.",
                "PENDING_ACTION_FOLLOWUP_AMBIGUOUS",
            )
        return _intent(
            request,
            action_type=action_type,
            scope=scope,
            target_reference=target_reference,
            requested_quantity=quantity,
            confidence=0.9,
            classification_reason="PENDING_ACTION_FOLLOWUP",
            is_follow_up=True,
        )

    dimensions = _dimensions_mm(text)
    if dimensions is None:
        return _pending_clarification(
            request,
            action_type,
            scope,
            target_reference,
            "Faltan dimensiones claras para completar la accion pendiente.",
            "PENDING_ACTION_FOLLOWUP_AMBIGUOUS",
        )
    return _intent(
        request,
        action_type=action_type,
        scope=scope,
        target_reference=target_reference,
        requested_width_mm=dimensions[0],
        requested_height_mm=dimensions[1],
        confidence=0.9,
        classification_reason="PENDING_ACTION_FOLLOWUP",
        is_follow_up=True,
    )


def _pending_clarification(
    request: ChatActionInterpretRequest,
    action_type: str,
    scope: str,
    target_reference: str | None,
    reason: str,
    classification_reason: str,
) -> ChatActionIntent:
    return _intent(
        request,
        action_type=action_type,
        scope=scope,
        target_reference=target_reference,
        requires_clarification=True,
        clarification_reason=reason,
        confidence=0.52,
        classification_reason=classification_reason,
        is_follow_up=True,
    )


def _target_follow_up_reference(message: str, options: object) -> str | object | None:
    option_target = _available_option_target_reference(message, options)
    if option_target is not None:
        return option_target
    explicit_target = _target_reference(message)
    if explicit_target is None:
        return _AMBIGUOUS if _is_option_reference(_normalize_text(message)) else None
    if _has_available_target_options(options):
        return (
            explicit_target
            if _target_is_available(explicit_target, options)
            else _AMBIGUOUS
        )
    return explicit_target


def _pending_requested_attributes(pending_action: dict) -> ChatRequestedAttributes | None:
    attributes = pending_action.get("requestedAttributes")
    if attributes is None:
        return None
    if isinstance(attributes, ChatRequestedAttributes):
        return attributes
    if isinstance(attributes, dict):
        return ChatRequestedAttributes.model_validate(attributes)
    return None


def _merge_requested_attributes(
    previous: ChatRequestedAttributes | None,
    current: ChatRequestedAttributes | None,
) -> ChatRequestedAttributes | None:
    if previous is None:
        return current
    if current is None:
        return previous
    return ChatRequestedAttributes(
        glass=_merge_glass_attributes(previous.glass, current.glass),
        system=current.system or previous.system,
        finish=current.finish or previous.finish,
    )


def _merge_glass_attributes(
    previous: ChatGlassRequestedAttributes | None,
    current: ChatGlassRequestedAttributes | None,
) -> ChatGlassRequestedAttributes | None:
    if previous is None:
        return current
    if current is None:
        return previous
    return ChatGlassRequestedAttributes(
        family=current.family or previous.family,
        composition=current.composition or previous.composition,
        treatment=current.treatment or previous.treatment,
        outerThicknessMm=current.outerThicknessMm or previous.outerThicknessMm,
        innerThicknessMm=current.innerThicknessMm or previous.innerThicknessMm,
        pvbThicknessMm=current.pvbThicknessMm or previous.pvbThicknessMm,
        chamberThicknessMm=current.chamberThicknessMm or previous.chamberThicknessMm,
        color=current.color or previous.color,
        pattern=current.pattern or previous.pattern,
    )


def _has_glass_family(attributes: ChatRequestedAttributes | None) -> bool:
    return (
        attributes is not None
        and attributes.glass is not None
        and attributes.glass.family is not None
    )


def _is_new_action_for_different_target(
    pending_action: dict,
    target_reference: str | None,
    has_mutation: bool,
) -> bool:
    pending_reference = _optional_str(pending_action.get("targetReference"))
    return (
        has_mutation
        and target_reference is not None
        and pending_reference is not None
        and _normalize_reference(target_reference) != _normalize_reference(pending_reference)
    )


def _looks_like_pending_follow_up(text: str) -> bool:
    stripped = text.strip()
    return (
        _dimensions_mm(text) is not None
        or _standalone_positive_int(text) is not None
        or _is_option_reference(text)
        or _is_ambiguous_follow_up(text)
        or stripped.startswith(
            (
                "que sea ",
                "que sean ",
                "usa ",
                "mejor ",
                "prefiero ",
                "el ",
                "la ",
                "templado",
                "laminado",
                "monolitico",
                "monolithic",
                "doble vidrio",
                "dvh",
                "pvb",
                "puerta ",
                "ventana ",
                "fijo ",
                "sistema ",
            )
        )
        or bool(re.fullmatch(r"\d+(?:[.,]\d+)?\s*mm", stripped))
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
    match = re.search(
        r"\b([A-Za-z]{1,4})\s*[-_ ]?\s*(\d{1,3})([A-Za-z]?)\b",
        message,
    )
    if not match:
        return None
    prefix = match.group(1).upper()
    if prefix in {"DE", "K", "S"}:
        return None
    suffix = match.group(3).lower()
    return f"{prefix}-{match.group(2)}{suffix}"


def _normalize_reference(value: str) -> str:
    return value.strip().casefold().replace("_", "-")


def _normalize_text(value: str) -> str:
    text = value.casefold()
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
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
    return "?" in text or text.startswith(
        (
            "que ",
            "cual ",
            "cuales ",
            "como ",
            "por que ",
            "porque ",
            "cuanto ",
            "cuantos ",
            "dime ",
            "indicame ",
            "muestrame ",
            "explicame ",
            "quiero saber ",
            "necesito saber ",
        )
    )


def _has_mutation_intent(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (
            re.compile(r"\b(cambia|cambialo|cambiar|modifica|reemplaza|actualiza)\b"),
            re.compile(r"\b(pon|ponlo|ponle|usa|usar|ajusta|sube|baja)\b"),
            re.compile(r"\b(quita|quitalo|excluye|excluyelo|saca|sacalo|elimina)\b"),
            re.compile(r"\b(incluye|incluyelo|incluir|incluirlo|agrega|agregalo)\b"),
            re.compile(r"\bvuelve\s+a\s+incluir\b"),
            re.compile(r"\bno\s+cotices\b"),
        )
    )


def _has_generic_change_intent(text: str) -> bool:
    return any(
        word in text
        for word in (
            "cambia",
            "cambialo",
            "hazlo diferente",
            "pon otro",
            "ponlo",
            "ponle",
            "quiero otro",
        )
    )


def _has_system_intent(text: str) -> bool:
    return (
        "sistema" in text
        or "puerta corrediza" in text
        or "ventana corrediza" in text
        or "puerta batiente" in text
        or "fijo" in text
        or bool(_system_code_match(text))
    )


def _system_value(message: str, text: str) -> str | None:
    match = re.search(
        r"\b(?:cambia|cambialo|cambiar|modifica|reemplaza|actualiza|pon|ponlo|usa)"
        r"\b(?:\s+(?:este|esta|un|una|item|elemento|[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}))*"
        r"\s+(?:a|por|con|en)\s+(.+)$",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        return _clean_requested_value(match.group(1))
    match = re.search(
        r"\b(?:pon|ponlo|usa)\s+((?:(?:un|una)\s+)?"
        r"(?:puerta|ventana|fijo|sistema)\b.+)$",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        return _clean_requested_value(match.group(1))
    match = re.search(r"\b(?:usa|usar)\s+(.+)$", message, flags=re.IGNORECASE)
    if match:
        return _clean_requested_value(match.group(1))
    match = _system_code_match(message)
    if match:
        return match.group(1).upper()
    return None


def _system_code_match(text: str) -> re.Match[str] | None:
    return re.search(r"\b([KS]\d{2,3}|[A-Za-z]*\d{4})\b", text, re.IGNORECASE)


def _has_glass_intent(text: str) -> bool:
    return any(
        word in text
        for word in (
            "vidrio",
            "templado",
            "laminado",
            "monolitico",
            "monolithic",
            "cristal",
            "dvh",
            "igu",
        )
    )


def _glass_value(message: str, text: str) -> str | None:
    if not _has_glass_intent(text):
        return None
    if "templado" in text or "laminado" in text:
        match = re.search(
            r"\b(?:vidrio\s+)?(?:templado|laminado)\b.*$",
            message,
            re.IGNORECASE,
        )
        if match:
            return _clean_requested_value(match.group(0))
    if _glass_family(text) is not None:
        match = re.search(
            r"\b(?:vidrio|cristal|monolitico|monolítico|laminado|doble vidrio|dvh|igu)\b.*$",
            message,
            re.IGNORECASE,
        )
        if match:
            return _clean_requested_value(match.group(0))
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


def _standalone_positive_int(text: str) -> int | None:
    match = re.fullmatch(r"\s*(?:que\s+sean\s+)?(\d+)\s*(?:unidades|unidad|und)?\s*", text)
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


def _clean_requested_value(value: str) -> str | None:
    cleaned = value.strip()
    cleaned = re.sub(r"[?.!,;:]+$", "", cleaned).strip()
    cleaned = re.sub(
        r"\b(?:por\s+favor|gracias|si\s+puedes|quiero\s+que|me\s+gustaria|podrias)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def _follow_up_requested_value(message: str) -> str | None:
    cleaned = _clean_requested_value(message)
    if cleaned is None:
        return None
    normalized = _normalize_text(cleaned)
    prefixes = (
        "que sea a ",
        "que sea el ",
        "que sea la ",
        "que sea ",
        "que sean ",
        "usa el de ",
        "usa la de ",
        "usa ",
        "mejor ",
        "prefiero ",
        "el de ",
        "la de ",
        "la opcion de ",
        "opcion ",
        "el ",
        "la ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return cleaned[len(prefix) :].strip() or None
    return cleaned


def _is_ambiguous_follow_up(text: str) -> bool:
    return text.strip() in {"esa", "ese", "la que dijiste", "el que dijiste", "esa misma"}


def _is_pending_value_phrase(text: str) -> bool:
    return text.strip().startswith(("que sea ", "que sean "))


def _available_option_value(message: str, options: object) -> str | object | None:
    if not isinstance(options, list) or not options:
        return _AMBIGUOUS if _is_option_reference(_normalize_text(message)) else None
    index = _option_index(_normalize_text(message))
    if index is None:
        return None
    if index < 0 or index >= len(options):
        return _AMBIGUOUS
    option = options[index]
    if not isinstance(option, dict):
        return _AMBIGUOUS
    value = _first_option_value(option)
    return value if value is not None else _AMBIGUOUS


def _available_option_target_reference(message: str, options: object) -> str | object | None:
    if not isinstance(options, list) or not options:
        return _AMBIGUOUS if _is_option_reference(_normalize_text(message)) else None
    index = _option_index(_normalize_text(message))
    if index is not None:
        if index < 0 or index >= len(options):
            return _AMBIGUOUS
        return _option_target_reference(options[index]) or _AMBIGUOUS

    explicit_target = _target_reference(message)
    if explicit_target is None:
        return None
    return explicit_target if _target_is_available(explicit_target, options) else _AMBIGUOUS


def _target_is_available(target_reference: str, options: object) -> bool:
    if not isinstance(options, list):
        return False
    normalized_target = _normalize_reference(target_reference)
    return any(
        _normalize_reference(option_target) == normalized_target
        for option in options
        if (option_target := _option_target_reference(option)) is not None
    )


def _has_available_target_options(options: object) -> bool:
    return isinstance(options, list) and any(
        _option_target_reference(option) is not None for option in options
    )


def _option_target_reference(option: object) -> str | None:
    if isinstance(option, str):
        return _target_reference(option)
    if not isinstance(option, dict):
        return None
    for key in ("reference", "targetReference", "displayName", "name", "value", "id"):
        value = option.get(key)
        if isinstance(value, str):
            target = _target_reference(value)
            if target is not None:
                return target
    return None


def _is_option_reference(text: str) -> bool:
    stripped = text.strip()
    return stripped in {
        "la primera",
        "primera",
        "la segunda",
        "segunda",
        "la tercera",
        "tercera",
        "esa",
        "ese",
        "la que dijiste",
        "el que dijiste",
    } or re.fullmatch(r"(?:la\s+)?opcion\s+\d+", stripped) is not None


def _option_index(text: str) -> int | None:
    stripped = text.strip()
    if stripped in {"la primera", "primera"}:
        return 0
    if stripped in {"la segunda", "segunda"}:
        return 1
    if stripped in {"la tercera", "tercera"}:
        return 2
    match = re.fullmatch(r"(?:la\s+)?opcion\s+(\d+)", stripped)
    if match:
        return int(match.group(1)) - 1
    return None


def _first_option_value(option: dict) -> str | None:
    for key in ("code", "displayName", "name", "value", "requestedValue", "id"):
        value = option.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _attributes_for_action(
    action_type: str,
    requested_value: str | None,
    text: str,
) -> ChatRequestedAttributes | None:
    if requested_value is None:
        return None
    if action_type == "CHANGE_GLASS":
        return ChatRequestedAttributes(glass=_glass_attributes(requested_value, text))
    if action_type == "CHANGE_SYSTEM":
        return ChatRequestedAttributes(system=_system_attributes(requested_value))
    if action_type == "CHANGE_FINISH":
        return ChatRequestedAttributes(finish=_finish_attributes(requested_value, text))
    if action_type == "CHANGE_COMMERCIAL_LINE":
        return ChatRequestedAttributes(
            system=ChatSystemRequestedAttributes(commercialLine=requested_value.upper())
        )
    return None


def _glass_attributes(
    requested_value: str,
    text: str,
) -> ChatGlassRequestedAttributes:
    normalized_value = _normalize_text(requested_value)
    normalized_context = _normalize_text(f"{requested_value} {text}")
    family = _glass_family(normalized_context)
    composition = None
    if "templado" in normalized_value or "templado" in text:
        composition = "TEMPERED"
    elif family == "LAMINATED" or "laminado" in normalized_value or "laminado" in text:
        composition = "LAMINATED"

    outer_thickness = None
    inner_thickness = None
    chamber_thickness = None
    composition_match = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*\+\s*(\d+(?:[.,]\d+)?)\b",
        requested_value,
    )
    if composition_match:
        outer_thickness = _number_value(composition_match.group(1))
        inner_thickness = _number_value(composition_match.group(2))
    else:
        thickness_match = re.search(
            r"\b(\d+(?:[.,]\d+)?)\s*(?:mm)?\b",
            requested_value,
            re.IGNORECASE,
        )
        if thickness_match:
            outer_thickness = _number_value(thickness_match.group(1))

    chamber_match = re.search(
        r"\b(?:camara)\s+de\s+(\d+(?:[.,]\d+)?)\s*mm\b",
        normalized_context,
        re.IGNORECASE,
    )
    if chamber_match:
        chamber_thickness = _number_value(chamber_match.group(1))
        if family == "IGU" and outer_thickness == chamber_thickness:
            outer_thickness = None

    return ChatGlassRequestedAttributes(
        family=family,
        composition=composition,
        outerThicknessMm=outer_thickness,
        innerThicknessMm=inner_thickness,
        chamberThicknessMm=chamber_thickness,
        color=_color_attribute(normalized_value),
    )


def _glass_family(text: str) -> str | None:
    if any(
        signal in text
        for signal in (
            "doble vidrio",
            "dvh",
            "camara",
            "insulado",
            "insulated",
            "igu",
        )
    ):
        return "IGU"
    if (
        "laminado" in text
        or "laminated" in text
        or "pvb" in text
        or re.search(r"\b\d+(?:[.,]\d+)?\s*\+\s*\d+(?:[.,]\d+)?\b", text)
    ):
        return "LAMINATED"
    if any(
        signal in text
        for signal in (
            "monolitico",
            "monolithic",
            "una sola hoja",
            "vidrio monolitico",
            "composicion monolitica",
        )
    ):
        return "MONOLITHIC"
    return None


def _system_attributes(requested_value: str) -> ChatSystemRequestedAttributes:
    normalized = _normalize_text(requested_value)
    functional_type = None
    operation = None
    if "puerta corrediza" in normalized:
        functional_type = "SLIDING_DOOR"
        operation = "SLIDING"
    elif "ventana corrediza" in normalized:
        functional_type = "SLIDING_WINDOW"
        operation = "SLIDING"
    elif "puerta batiente" in normalized:
        functional_type = "SWING_DOOR"
        operation = "CASEMENT"
    elif re.search(r"\bfijo\b", normalized):
        functional_type = "FIXED"
        operation = "FIXED"
    elif "plegable" in normalized:
        operation = "FOLDING"

    commercial_name = _system_commercial_name(requested_value)
    return ChatSystemRequestedAttributes(
        functionalType=functional_type,
        operation=operation,
        commercialName=commercial_name,
        family=commercial_name,
    )


def _system_commercial_name(requested_value: str) -> str | None:
    cleaned = re.sub(
        r"\b(?:puerta|ventana|corrediza|batiente|fijo|sistema|plegable|una|un)\b",
        " ",
        requested_value,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split())
    return cleaned.upper() if cleaned else None


def _finish_attributes(
    requested_value: str,
    text: str,
) -> ChatFinishRequestedAttributes:
    normalized = _normalize_text(f"{requested_value} {text}")
    return ChatFinishRequestedAttributes(
        color=_color_attribute(normalized),
        texture="MATTE" if "mate" in normalized else None,
        process="ANODIZED" if "anodizado" in normalized else None,
        material=(
            "STAINLESS_STEEL"
            if "inox" in normalized or "acero inoxidable" in normalized
            else None
        ),
        normalizedType=(
            "STAINLESS_STEEL"
            if "inox" in normalized or "acero inoxidable" in normalized
            else None
        ),
    )


def _color_attribute(text: str) -> str | None:
    if "negro" in text:
        return "BLACK"
    if "blanco" in text:
        return "WHITE"
    if "gris" in text:
        return "GRAY"
    if "champana" in text:
        return "CHAMPAGNE"
    return None


def _number_value(value: str) -> int | float:
    number = float(value.replace(",", "."))
    return int(number) if number.is_integer() else number
