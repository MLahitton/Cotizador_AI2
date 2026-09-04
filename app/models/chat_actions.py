from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.chat import ChatConversationMessage, ChatScope

ChatActionType = Literal[
    "CHANGE_SYSTEM",
    "CHANGE_GLASS",
    "CHANGE_FINISH",
    "CHANGE_QUANTITY",
    "CHANGE_DIMENSIONS",
    "EXCLUDE_ITEM",
    "INCLUDE_ITEM",
    "CHANGE_COMMERCIAL_LINE",
    "UNKNOWN",
]


class ChatActionInterpretRequest(BaseModel):
    scope: ChatScope
    userMessage: str = Field(min_length=1, max_length=4000)
    conversation: list[ChatConversationMessage] = Field(default_factory=list, max_length=40)
    context: dict[str, Any] = Field(default_factory=dict)


class ChatGlassRequestedAttributes(BaseModel):
    family: str | None = None
    composition: str | None = None
    treatment: str | None = None
    outerThicknessMm: float | None = Field(default=None, gt=0)
    innerThicknessMm: float | None = Field(default=None, gt=0)
    pvbThicknessMm: float | None = Field(default=None, gt=0)
    chamberThicknessMm: float | None = Field(default=None, gt=0)
    color: str | None = None
    pattern: str | None = None


class ChatSystemRequestedAttributes(BaseModel):
    functionalType: str | None = None
    operation: str | None = None
    commercialName: str | None = None
    family: str | None = None
    variant: str | None = None
    commercialLine: str | None = None


class ChatFinishRequestedAttributes(BaseModel):
    color: str | None = None
    texture: str | None = None
    process: str | None = None
    material: str | None = None
    normalizedType: str | None = None


class ChatRequestedAttributes(BaseModel):
    glass: ChatGlassRequestedAttributes | None = None
    system: ChatSystemRequestedAttributes | None = None
    finish: ChatFinishRequestedAttributes | None = None


class ChatActionIntent(BaseModel):
    isAction: bool
    actionType: ChatActionType
    scope: ChatScope
    targetReference: str | None = None
    requestedValue: str | None = None
    requestedQuantity: int | None = Field(default=None, gt=0)
    requestedWidthMm: int | None = Field(default=None, gt=0)
    requestedHeightMm: int | None = Field(default=None, gt=0)
    requestedAttributes: ChatRequestedAttributes | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    requiresClarification: bool
    clarificationReason: str | None = None
    classificationReason: str | None = None
    isFollowUpToPendingAction: bool = False
    rawUserMessage: str
