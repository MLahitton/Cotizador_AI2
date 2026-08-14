from app.providers.gemini import GeminiProvider


def test_gemini_provider_exposes_configured_model(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")

    provider = GeminiProvider()

    assert provider.model == "gemini-3.6-flash"