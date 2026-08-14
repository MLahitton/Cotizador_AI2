import os

import pytest

from app.models.requirement_extraction import RequirementExtraction
from app.providers.gemini_extraction import GeminiExtractionProvider


def test_gemini_provider_extracts_current_v1_flow_from_text() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY no configurada.")

    provider = GeminiExtractionProvider()

    extraction = provider.extract_from_text(
        """
        Extrae este requerimiento tecnico:

        Proyecto: Demo Steel and Glass.
        Elemento: V1.
        Es una ventana de 1.80 m de ancho por 2.40 m de alto.
        El documento no especifica el tipo exacto de ventana.
        No calcules precios.
        """
    )

    token_usage = extraction.extraction_metadata.token_usage
    print("\nGEMINI USAGE")
    print(f"model: {extraction.extraction_metadata.model}")
    print(f"input_tokens: {token_usage.input_tokens if token_usage else None}")
    print(f"output_tokens: {token_usage.output_tokens if token_usage else None}")
    print(f"total_tokens: {token_usage.total_tokens if token_usage else None}")

    assert isinstance(extraction, RequirementExtraction)
    assert extraction.extraction_metadata.model_provider == "google"
    assert len(extraction.elements) >= 1
    assert token_usage is not None
    assert token_usage.total_tokens is not None
    assert token_usage.total_tokens > 0
