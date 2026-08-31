import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from google.genai.errors import APIError

from app.models.chat import ChatRespondRequest, ChatRespondResponse
from app.services.chat_responder import ChatResponder, ChatResponderError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_responder() -> ChatResponder:
    return ChatResponder()


@router.post("/respond", response_model=ChatRespondResponse)
def respond_to_chat(
    request: ChatRespondRequest,
    responder: Annotated[ChatResponder, Depends(get_chat_responder)],
) -> ChatRespondResponse:
    if not request.userMessage.strip():
        raise HTTPException(status_code=422, detail="Mensaje vacio.")
    try:
        return responder.respond(request)
    except ChatResponderError as exc:
        logger.exception("CHAT_RESPOND_INVALID_AI_RESPONSE")
        raise HTTPException(status_code=502, detail="Respuesta de IA invalida.") from exc
    except APIError as exc:
        logger.exception("CHAT_RESPOND_PROVIDER_API_ERROR")
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
    except RuntimeError as exc:
        logger.exception("CHAT_RESPOND_PROVIDER_RUNTIME_ERROR")
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
    except Exception as exc:
        logger.exception("CHAT_RESPOND_UNEXPECTED_ERROR")
        raise HTTPException(status_code=502, detail="Error del proveedor de IA.") from exc
