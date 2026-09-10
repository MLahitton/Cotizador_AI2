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
_VALID_TARGET_PREFIXES = {"V", "PV", "C", "P", "F", "D", "M", "A", "TAG", "HOJA"}
_TARGET_REFERENCE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_VALID_TARGET_PREFIXES, key=len, reverse=True))
    + r")\s*([-_ ]?)\s*(\d{1,3})([A-Za-z]?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
}
_GLASS_CODE_PATTERN = re.compile(r"\b(?:TEMP_\d+(?:[.,]\d+)?|LAM_\d+(?:[.,]\d+)?_\d+(?:[.,]\d+)?)\b", re.IGNORECASE)


class ChatActionInterpreter:
    def interpret(self, request: ChatActionInterpretRequest) -> ChatActionIntent:
        message = request.userMessage.strip()
        text = _normalize_text(message)
        scope = _scope(request)
        dimension_text = _strip_dimension_expressions(text)
        target_references = _target_references(dimension_text)
        target_reference = target_references[0] if target_references else None
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

        if _has_confirm_selection_action_intent(text):
            return _intent(
                request,
                action_type="CONFIRM_SELECTION",
                scope="REQUIREMENT",
                target_reference=None,
                target_references=[],
                confidence=0.91,
                classification_reason="CONFIRM_SELECTION_MUTATION",
            )
        if _is_informational(text) and not has_mutation:
            return _intent(
                request,
                action_type="UNKNOWN",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                is_action=False,
                confidence=0.9,
                classification_reason="INFORMATIONAL_GUARD",
            )

        contextual_intent = _contextual_action_intent(request, text, scope)
        if contextual_intent is not None:
            return contextual_intent

        if _has_copy_configuration_intent(text):
            return _intent(
                request,
                action_type="UNKNOWN",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                is_action=False,
                requires_clarification=True,
                clarification_reason="Copiar configuracion entre items aun no esta disponible en el chat.",
                confidence=0.82,
                classification_reason="COPY_CONFIGURATION_UNSUPPORTED",
            )

        if _is_heterogeneous_batch(message, text, target_references):
            return _intent(
                request,
                action_type="CHANGE_SYSTEM",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                requires_clarification=True,
                clarification_reason=(
                    "La instruccion contiene valores distintos por item; "
                    "AI2 requiere una accion homogenea por lote."
                ),
                confidence=0.62,
                classification_reason="HETEROGENEOUS_BATCH_REQUIRES_CLARIFICATION",
            )

        dimensions = _dimensions_mm(text)
        if dimensions is not None and _has_dimension_intent(text):
            return _intent(
                request,
                action_type="CHANGE_DIMENSIONS",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                requested_width_mm=dimensions[0],
                requested_height_mm=dimensions[1],
                confidence=0.92,
                classification_reason=_action_reason(
                    "EXPLICIT_DIMENSION_MUTATION",
                    target_references,
                ),
            )
        partial_dimension = _partial_dimension_mm(text)
        if partial_dimension is not None and _has_dimension_intent(text):
            return _intent(
                request,
                action_type="CHANGE_DIMENSIONS",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                requested_width_mm=partial_dimension[0],
                requested_height_mm=partial_dimension[1],
                requires_clarification=True,
                clarification_reason="Falta la otra dimension para preparar el cambio de medidas.",
                confidence=0.68,
                classification_reason="PARTIAL_DIMENSION_MUTATION",
            )

        quantity = _quantity(text)
        if quantity is not None and _has_quantity_intent(text):
            return _intent(
                request,
                action_type="CHANGE_QUANTITY",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                requested_quantity=quantity,
                confidence=0.94,
                classification_reason=_action_reason(
                    "EXPLICIT_QUANTITY_MUTATION",
                    target_references,
                ),
            )
        if _has_quantity_intent(text):
            return _intent(
                request,
                action_type="CHANGE_QUANTITY",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
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
                target_references=target_references,
                confidence=0.95,
                classification_reason=_action_reason(
                    "EXPLICIT_EXCLUDE_MUTATION",
                    target_references,
                ),
            )

        if _has_include_intent(text) and has_mutation:
            return _intent(
                request,
                action_type="INCLUDE_ITEM",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                confidence=0.95,
                classification_reason=_action_reason(
                    "EXPLICIT_INCLUDE_MUTATION",
                    target_references,
                ),
            )

        commercial_line = _commercial_line_value(message, text)
        if commercial_line is not None or _has_commercial_line_intent(text):
            return _intent(
                request,
                action_type="CHANGE_COMMERCIAL_LINE",
                scope="REQUIREMENT" if _global_scope_requested(text) else scope,
                target_reference=target_reference,
                target_references=target_references,
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
            classification_reason = (
                "GLASS_FAMILY_EXTRACTED"
                if _has_glass_family(requested_attributes)
                else "GLASS_MUTATION"
            )
            return _intent(
                request,
                action_type="CHANGE_GLASS",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                requested_value=glass,
                requested_attributes=requested_attributes,
                requires_clarification=glass is None,
                clarification_reason=(
                    "Falta el vidrio solicitado."
                    if glass is None
                    else None
                ),
                confidence=0.9 if glass is not None else 0.68,
                classification_reason=_action_reason(
                    classification_reason,
                    target_references,
                ),
            )

        finish = _finish_value(message, text)
        if finish is not None or _has_finish_intent(text):
            return _intent(
                request,
                action_type="CHANGE_FINISH",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                requested_value=finish,
                requested_attributes=_attributes_for_action("CHANGE_FINISH", finish, text),
                requires_clarification=finish is None,
                clarification_reason=(
                    "Falta el acabado solicitado."
                    if finish is None
                    else None
                ),
                confidence=0.9 if finish is not None else 0.68,
                classification_reason=_action_reason("FINISH_MUTATION", target_references),
            )

        system = _system_value(message, text)
        if system is not None or _has_system_intent(text):
            return _intent(
                request,
                action_type="CHANGE_SYSTEM",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
                requested_value=system,
                requested_attributes=_attributes_for_action("CHANGE_SYSTEM", system, text),
                requires_clarification=system is None,
                clarification_reason=(
                    "Falta el sistema solicitado."
                    if system is None
                    else None
                ),
                confidence=0.92 if system is not None else 0.68,
                classification_reason=_action_reason(
                    (
                        "SYSTEM_VALUE_FROM_MUTATION_PHRASE"
                        if system is not None
                        else "SYSTEM_MUTATION"
                    ),
                    target_references,
                ),
            )

        if _has_generic_change_intent(text):
            return _intent(
                request,
                action_type="UNKNOWN",
                scope=scope,
                target_reference=target_reference,
                target_references=target_references,
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
            target_references=target_references,
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
    target_references: list[str] | None = None,
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
    references = target_references or ([target_reference] if target_reference else [])
    return ChatActionIntent(
        isAction=executable and not requires_clarification,
        actionType=action_type,
        scope=scope,
        targetReference=target_reference,
        targetReferences=references,
        targetCount=len(references),
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


def _contextual_action_intent(
    request: ChatActionInterpretRequest,
    text: str,
    scope: str,
) -> ChatActionIntent | None:
    if not _is_contextual_action_reference(text):
        return None

    resolution = _latest_contextual_system_change(request)
    if resolution is None:
        return _intent(
            request,
            action_type=_contextual_fallback_action_type(text),
            scope=scope,
            target_reference=None,
            target_references=[],
            is_action=False,
            requires_clarification=True,
            clarification_reason="No tengo un cambio conversado suficientemente claro para prepararlo.",
            confidence=0.55,
            classification_reason="CONTEXTUAL_ACTION_NOT_RESOLVED",
        )

    if resolution["state"] == "AMBIGUOUS":
        return _intent(
            request,
            action_type="CHANGE_SYSTEM",
            scope=scope,
            target_reference=None,
            target_references=[],
            is_action=False,
            requires_clarification=True,
            clarification_reason="Hay mas de un cambio conversado posible. Indica los elementos y el sistema que quieres aplicar.",
            confidence=0.58,
            classification_reason=resolution["reason"],
        )

    target_references = resolution["target_references"]
    requested_value = resolution["requested_value"]
    return _intent(
        request,
        action_type="CHANGE_SYSTEM",
        scope="REQUIREMENT" if len(target_references) > 1 else scope,
        target_reference=target_references[0] if target_references else None,
        target_references=target_references,
        requested_value=requested_value,
        requested_attributes=_attributes_for_action("CHANGE_SYSTEM", requested_value, text),
        confidence=0.86,
        classification_reason="CONTEXTUAL_SYSTEM_CHANGE_RESOLVED",
    )


def _is_contextual_action_reference(text: str) -> bool:
    stripped = text.strip().strip("?!. ,")
    if stripped == "si":
        return False

    contextual_phrases = (
        "esos cambios",
        "ese cambio",
        "lo que hablamos",
        "esas opciones",
        "esa opcion",
        "como dijimos",
        "los anteriores",
        "valores anteriores",
        "cambios anteriores",
        "aplicalos",
        "aplica eso",
        "hazlo",
        "haz esos",
        "haz ese",
    )
    if any(phrase in stripped for phrase in contextual_phrases):
        return bool(
            _has_mutation_intent(stripped)
            or stripped.startswith(("si ", "si,", "haz", "aplica", "usa"))
            or "opciones" in stripped
        )

    return bool(
        re.search(
            r"\bsi\b.*\b(cambia|aplica|haz|usa)\b.*\b(sistemas|cambios|opciones|hablamos|dijimos)\b",
            stripped,
        )
    )


def _latest_contextual_system_change(request: ChatActionInterpretRequest) -> dict | None:
    candidates = _contextual_system_change_candidates(request)
    if not candidates:
        return None

    latest = candidates[0]
    if latest["state"] == "AMBIGUOUS":
        return latest

    for previous in candidates[1:3]:
        if previous["state"] == "AMBIGUOUS":
            return previous
        if _contextual_candidates_conflict(latest, previous):
            return {
                "state": "AMBIGUOUS",
                "reason": "CONTEXTUAL_ACTION_CONFLICTING_CHANGE_SETS",
            }

    if _references_are_ambiguous_in_context(request.context, latest["target_references"]):
        return {
            "state": "AMBIGUOUS",
            "reason": "CONTEXTUAL_TARGET_REFERENCE_AMBIGUOUS",
        }

    return latest


def _contextual_system_change_candidates(
    request: ChatActionInterpretRequest,
) -> list[dict]:
    current_message = _normalize_text(request.userMessage).strip()
    candidates: list[dict] = []

    for message in reversed(request.conversation):
        if message.role != "user":
            continue

        normalized = _normalize_text(message.content).strip()
        if normalized == current_message or _is_contextual_action_reference(normalized):
            continue

        assignments = _system_assignments(message.content)
        if not assignments:
            continue

        values = {assignment["requested_value"].casefold() for assignment in assignments}
        if len(values) > 1:
            candidates.append(
                {
                    "state": "AMBIGUOUS",
                    "reason": "CONTEXTUAL_HETEROGENEOUS_VALUES_UNSUPPORTED",
                }
            )
            continue

        target_references = _dedupe_references(
            reference
            for assignment in assignments
            for reference in assignment["target_references"]
        )
        if not target_references:
            continue

        candidates.append(
            {
                "state": "VALID",
                "target_references": target_references,
                "requested_value": assignments[0]["requested_value"],
            }
        )

    return candidates


def _contextual_fallback_action_type(text: str) -> str:
    return "CHANGE_SYSTEM" if any(
        phrase in text
        for phrase in ("sistema", "sistemas", "opcion", "opciones", "cambio", "cambios")
    ) else "UNKNOWN"


def _system_assignments(message: str) -> list[dict]:
    normalized_message = _normalize_text(message)
    if _has_dimension_intent(normalized_message) or _has_quantity_intent(normalized_message):
        return []

    reference = r"[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?"
    target_group = rf"{reference}(?:\s*(?:,|;|y|e)\s*{reference})*"
    pattern = re.compile(
        rf"(?P<targets>{target_group})\s+(?:a|en|con|por)\s+(?P<value>.+?)"
        rf"(?=(?:\s*(?:,|;|y|e)\s*{reference}\s+(?:a|en|con|por)\s+)|$)",
        re.IGNORECASE,
    )
    assignments: list[dict] = []
    for match in pattern.finditer(message):
        target_references = _target_references(match.group("targets"))
        value = _clean_requested_value(match.group("value"))
        if target_references and value:
            assignments.append(
                {
                    "target_references": target_references,
                    "requested_value": value,
                }
            )

    if assignments:
        return assignments

    target_references = _target_references(message)
    value = _system_value(message, _normalize_text(message))
    if value is None and target_references and _has_mutation_intent(_normalize_text(message)):
        fallback = re.search(r"\b(?:a|en|con|por)\s+(.+)$", message, flags=re.IGNORECASE)
        if fallback:
            value = _clean_requested_value(fallback.group(1))
    if target_references and value:
        return [{"target_references": target_references, "requested_value": value}]

    return []


def _contextual_candidates_conflict(current: dict, previous: dict) -> bool:
    if current["state"] != "VALID" or previous["state"] != "VALID":
        return False

    current_targets = {_normalize_reference(value) for value in current["target_references"]}
    previous_targets = {_normalize_reference(value) for value in previous["target_references"]}
    if not current_targets.intersection(previous_targets):
        return False

    return current["requested_value"].casefold() != previous["requested_value"].casefold()


def _references_are_ambiguous_in_context(context: object, references: list[str]) -> bool:
    items = _context_items(context)
    if not items:
        return False

    counts: dict[str, int] = {}
    for item in items:
        reference = item.get("reference") if isinstance(item, dict) else None
        if isinstance(reference, str) and reference.strip():
            normalized = _normalize_reference(reference)
            counts[normalized] = counts.get(normalized, 0) + 1

    return any(counts.get(_normalize_reference(reference), 0) > 1 for reference in references)


def _context_items(context: object) -> list[dict]:
    if not isinstance(context, dict):
        return []

    source = context.get("originalContext") if isinstance(context.get("originalContext"), dict) else context
    proposal = source.get("technicalProposal") if isinstance(source, dict) else None
    items = proposal.get("items") if isinstance(proposal, dict) else None
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]

    item = source.get("item") if isinstance(source, dict) else None
    return [item] if isinstance(item, dict) else []


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
    message_target = _target_reference(message)

    if "lo mismo" in text and message_target is not None:
        requested_value = _optional_str(pending_action.get("requestedValue"))
        if action_type in {"CHANGE_SYSTEM", "CHANGE_FINISH", "CHANGE_GLASS"} and requested_value is not None:
            return _intent(
                request,
                action_type=action_type,
                scope=scope,
                target_reference=message_target,
                target_references=[message_target],
                requested_value=requested_value,
                requested_attributes=_pending_requested_attributes(pending_action),
                confidence=0.86,
                classification_reason="PENDING_ACTION_COPY_TO_TARGET",
                is_follow_up=True,
            )

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
                _pending_target_references(pending_action),
                "Falta identificar el item objetivo para completar la accion pendiente.",
                "PENDING_ACTION_TARGET_AMBIGUOUS",
            )
        return _intent(
            request,
            action_type=action_type,
            scope=scope,
            target_reference=resolved_target,
            target_references=[resolved_target],
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
            _pending_target_references(pending_action),
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
                _pending_target_references(pending_action),
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
            target_references=_pending_target_references(pending_action),
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
                _pending_target_references(pending_action),
                "Falta una cantidad entera positiva.",
                "PENDING_ACTION_FOLLOWUP_AMBIGUOUS",
            )
        return _intent(
            request,
            action_type=action_type,
            scope=scope,
            target_reference=target_reference,
            target_references=_pending_target_references(pending_action),
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
            _pending_target_references(pending_action),
            "Faltan dimensiones claras para completar la accion pendiente.",
            "PENDING_ACTION_FOLLOWUP_AMBIGUOUS",
        )
    return _intent(
        request,
        action_type=action_type,
        scope=scope,
        target_reference=target_reference,
        target_references=_pending_target_references(pending_action),
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
    target_references: list[str] | None,
    reason: str,
    classification_reason: str,
) -> ChatActionIntent:
    return _intent(
        request,
        action_type=action_type,
        scope=scope,
        target_reference=target_reference,
        target_references=target_references,
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


def _pending_target_references(pending_action: dict) -> list[str]:
    references = pending_action.get("targetReferences")
    if isinstance(references, list):
        return _dedupe_references(
            reference
            for reference in references
            if isinstance(reference, str)
        )
    target_reference = _optional_str(pending_action.get("targetReference"))
    return [target_reference] if target_reference else []


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


def _action_reason(default_reason: str, target_references: list[str]) -> str:
    return "MULTI_TARGET_ACTION" if len(target_references) > 1 else default_reason


def _is_heterogeneous_batch(
    message: str,
    text: str,
    target_references: list[str],
) -> bool:
    if len(target_references) < 2 or not _has_system_intent(text):
        return False
    assignments = re.findall(
        r"\b[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?\s+"
        r"(?:a|en|con)\s+([KS]\d{2,3}|[A-Za-z]*\d{4})\b",
        message,
        flags=re.IGNORECASE,
    )
    return len({assignment.upper() for assignment in assignments}) > 1


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
    match = _target_reference_match(_strip_dimension_expressions(message))
    if not match:
        return None
    return _reference_from_match(match)


def _target_references(message: str) -> list[str]:
    cleaned = _strip_dimension_expressions(message)
    return _dedupe_references(
        reference
        for match in re.finditer(_TARGET_REFERENCE_PATTERN, cleaned)
        if (reference := _reference_from_match(match)) is not None
    )


def _target_reference_match(message: str) -> re.Match[str] | None:
    return re.search(_TARGET_REFERENCE_PATTERN, message)


def _reference_from_match(match: re.Match[str]) -> str | None:
    prefix = match.group(1).upper()
    if prefix not in _VALID_TARGET_PREFIXES:
        return None
    separator = match.group(2)
    if prefix not in {"V", "PV", "C"} and separator not in {"-", "_"}:
        return None
    suffix = match.group(4).lower()
    return f"{prefix}-{match.group(3)}{suffix}"


def _normalize_reference(value: str) -> str:
    return value.strip().casefold().replace("_", "-")


def _strip_dimension_expressions(value: str) -> str:
    number = r"\d+(?:[.,]\d+)?"
    text = re.sub(
        rf"\b{number}\s*(?:mm|cm|m)?\s*(?:x|por)\s*{number}\s*(?:mm|cm|m)?\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    text = re.sub(rf"\b(?:ancho|alto)\s*{number}\s*(?:mm|cm|m)?\b", " ", text)
    return text


def _dedupe_references(references: object) -> list[str]:
    deduped = []
    seen = set()
    for reference in references:
        if not isinstance(reference, str) or not reference.strip():
            continue
        normalized = _normalize_reference(reference)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(reference)
    return deduped


def _normalize_text(value: str) -> str:
    text = value.casefold()
    replacements = {
        "ã¡": "a",
        "ã©": "e",
        "ã­": "i",
        "ã³": "o",
        "ãº": "u",
        "ã±": "n",
        "Ã¡": "a",
        "Ã©": "e",
        "Ã­": "i",
        "Ã³": "o",
        "Ãº": "u",
        "Ã±": "n",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return text


def _has_confirm_selection_action_intent(text: str) -> bool:
    stripped = text.strip().strip("?!. ,")
    stripped = stripped.removeprefix("¿").removeprefix("¡").strip()
    if any(
        phrase in stripped
        for phrase in (
            "ya esta confirmada",
            "esta confirmada",
            "esta lista para confirmar",
            "lista para confirmar",
            "que falta para confirmar",
            "que le falta para confirmar",
            "que hace falta para confirmar",
        )
    ):
        return False
    if re.search(r"\b(confirma|confirmar|confirmas|confirmame)\b", stripped) is not None:
        return "seleccion" in stripped or "propuesta" in stripped
    return bool(re.search(r"\b(dejala|dejarla)\s+confirmada\b", stripped))


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
            re.compile(r"\b(cambia|cambias|cambialo|cambiar|modifica|reemplaza|actualiza)\b"),
            re.compile(r"\b(pon|poner|ponlo|ponle|ponerle|usa|usar|ajusta|sube|baja|deja|dejale|dejalo)\b"),
            re.compile(r"\b(quita|quitalo|excluye|excluyelo|saca|sacalo|elimina)\b"),
            re.compile(r"\b(incluye|incluyelo|incluir|incluirlo|agrega|agregalo)\b"),
            re.compile(r"\bvuelve\s+a\s+incluir\b"),
            re.compile(r"\bno\s+cotices\b"),
            re.compile(r"\bquiero\s+.+\s+en\s+"),
            re.compile(r"\bme\s+cambias\b"),
            re.compile(r"\bmejor\s+con\b"),
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


def _has_copy_configuration_intent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(copia|copiar|duplica|duplicar)\b.*\b(configuracion|config)\b"
            r"|\bcopia\s+[a-z]+\s*[-_ ]?\s*\d{1,3}[a-z]?\s+(?:en|a)\s+[a-z]+\s*[-_ ]?\s*\d{1,3}[a-z]?\b"
            r"|\bhaz\s+que\s+[a-z]+\s*[-_ ]?\s*\d{1,3}[a-z]?\s+quede\s+igual\s+que\s+[a-z]+\s*[-_ ]?\s*\d{1,3}[a-z]?\b",
            text,
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
    if _has_dimension_intent(text) or _has_quantity_intent(text):
        return None
    patterns = [
        r"\b(?:pon|ponlo|ponle|cambia|cambialo|cambiar|usa)\s+[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?(?:\s*(?:,|;|y|e)\s*[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?)+\s+(?:a|en|con)\s+(.+)$",
        r"\bme\s+cambias\s+[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?\s+(?:a|por|con|en)\s+(.+?)[?!.]*$",
        r"\b(?:cambia|cambias|cambialo|cambiar|modifica|reemplaza|actualiza|pon|ponlo|ponle|usa)"
        r"\b(?:\s+(?:este|esta|un|una|item|elemento|[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?))*"
        r"\s+(?:a|por|con|en)\s+(.+)$",
        r"\b(?:pon|ponlo|ponle|quiero)\s+(.+?)\s+(?:a|en|para)\s+[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?\b",
        r"\b[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?\s+(?:dejale|dejalo|deja|ponle|ponlo)\s+(?:el\s+sistema\s+)?(.+)$",
        r"\b[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?\s+y\s+[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?\s+(?:a|en|con)\s+(.+)$",
        r"\bmejor\s+con\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
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
    return _glass_code_match(text) is not None or any(
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
    code = _glass_code_match(message)
    if code:
        return code.group(0).upper()
    if not _has_glass_intent(text):
        return None
    if "templado" in text or "laminado" in text:
        match = re.search(
            r"\b(?:vidrio\s+)?(?:templado|laminado)\b.*$",
            message,
            re.IGNORECASE,
        )
        if match:
            return _clean_requested_value(_strip_trailing_target_clause(match.group(0)))
    if _glass_family(text) is not None:
        match = re.search(
            r"\b(?:vidrio|cristal|monolitico|monolÃ­tico|laminado|doble vidrio|dvh|igu)\b.*$",
            message,
            re.IGNORECASE,
        )
        if match:
            return _clean_requested_value(_strip_trailing_target_clause(match.group(0)))
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
        return _strip_trailing_target_clause(match.group(0)).strip()
    return None


def _glass_code_match(text: str) -> re.Match[str] | None:
    return re.search(_GLASS_CODE_PATTERN, text)


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
        value = _strip_trailing_target_clause(match.group(1))
        return value if value else None
    for word in ("inox", "negro mate", "negro", "blanco", "gris", "champaÃ±a", "champana"):
        if word in text:
            return _slice_original(message, word) or word
    return None


def _has_quantity_intent(text: str) -> bool:
    return any(word in text for word in ("cantidad", "unidades", "unidad", "estos")) or bool(
        re.search(r"\bdejalo\s+en\s+\d+\b", text)
    )


def _quantity(text: str) -> int | None:
    cleaned = _strip_target_references(text)
    match = re.search(r"\b(?:cantidad\s*(?:a|en|para)?\s*)?(\d+)\s*(?:unidades|unidad|und|de estos)?\b", cleaned)
    if match:
        value = int(match.group(1))
        return value if value > 0 else None
    word_match = re.search(
        r"\b(?:cantidad\s*(?:a|en|para)?\s*)?(" + "|".join(_NUMBER_WORDS) + r")\s*(?:unidades|unidad|und|de estos)?\b",
        cleaned,
    )
    if word_match:
        return _NUMBER_WORDS[word_match.group(1)]
    return None


def _standalone_positive_int(text: str) -> int | None:
    match = re.fullmatch(r"\s*(?:que\s+sean\s+)?(\d+)\s*(?:unidades|unidad|und)?\s*", text)
    if match:
        value = int(match.group(1))
        return value if value > 0 else None
    word_match = re.fullmatch(
        r"\s*(?:que\s+sean\s+)?(" + "|".join(_NUMBER_WORDS) + r")\s*(?:unidades|unidad|und)?\s*",
        text,
    )
    return _NUMBER_WORDS[word_match.group(1)] if word_match else None


def _has_dimension_intent(text: str) -> bool:
    return (
        any(word in text for word in (" x ", "ancho", "alto", "medida", "dimens"))
        or re.search(r"\d+(?:[.,]\d+)?\s*(?:mm|cm|m)?\s+por\s+\d", text) is not None
    )


def _strip_target_references(value: str) -> str:
    def replace_if_valid(match: re.Match[str]) -> str:
        return " " if _reference_from_match(match) is not None else match.group(0)

    return re.sub(_TARGET_REFERENCE_PATTERN, replace_if_valid, value)


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


def _partial_dimension_mm(text: str) -> tuple[int | None, int | None] | None:
    number = r"(\d+(?:[.,]\d+)?)"
    match = re.search(rf"ancho\s*{number}\s*(mm|cm|m)?", text)
    if match:
        return (_to_mm(match.group(1), match.group(2) or _default_unit(match.group(1))), None)
    match = re.search(rf"{number}\s*(mm|cm|m)?\s+de\s+ancho", text)
    if match:
        return (_to_mm(match.group(1), match.group(2) or _default_unit(match.group(1))), None)
    match = re.search(rf"alto\s*{number}\s*(mm|cm|m)?", text)
    if match:
        return (None, _to_mm(match.group(1), match.group(2) or _default_unit(match.group(1))))
    match = re.search(rf"{number}\s*(mm|cm|m)?\s+de\s+alto", text)
    if match:
        return (None, _to_mm(match.group(1), match.group(2) or _default_unit(match.group(1))))
    return None


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


def _strip_trailing_target_clause(value: str) -> str:
    return re.sub(
        r"\s+en\s+[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?"
        r"(?:\s*(?:,|;|/|y|e|ademas|tambien)\s*"
        r"[A-Za-z]{1,4}\s*[-_ ]?\s*\d{1,3}[A-Za-z]?)*\s*[?.!,;:]*\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


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
    glass_code = _glass_code_match(normalized_value)
    if glass_code and normalized_value.startswith("temp_"):
        composition = "TEMPERED"
    elif "templado" in normalized_value or "templado" in text:
        composition = "TEMPERED"
    elif glass_code and normalized_value.startswith("lam_"):
        composition = "LAMINATED"
        family = family or "LAMINATED"
    elif family == "LAMINATED" or "laminado" in normalized_value or "laminado" in text:
        composition = "LAMINATED"

    outer_thickness = None
    inner_thickness = None
    chamber_thickness = None
    composition_match = re.search(
        r"\b(\d+(?:[.,]\d+)?)\s*\+\s*(\d+(?:[.,]\d+)?)\b",
        requested_value.replace("_", "+")
        if normalized_value.startswith("lam_")
        else requested_value,
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
