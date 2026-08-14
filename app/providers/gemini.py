from google import genai

from app.core.settings import get_settings


class GeminiProvider:
    def __init__(self) -> None:
        settings = get_settings()

        self._client = genai.Client(
            api_key=settings.gemini_api_key,
        )
        self._model = settings.gemini_model

    @property
    def model(self) -> str:
        return self._model