from app.models.common import (
    ExtractionStatus,
    NormalizedValue,
    TraceableValue,
)
from app.models.element import Element
from app.models.evidence import Source
from app.models.requirement import ExtractionMetadata, Requirement
from app.models.requirement_extraction import RequirementExtraction


def test_requirement_extraction_accepts_partial_element() -> None:
    extraction = RequirementExtraction(
        requirement=Requirement(
            project_id="project-test",
            requirement_id="requirement-test",
        ),
        sources=[
            Source(
                id="source-1",
                file_name="example.pdf",
                media_type="application/pdf",
            )
        ],
        elements=[
            Element(
                id="element-1",
                name=TraceableValue(
                    value="Elemento detectado",
                    status=ExtractionStatus.EXPLICIT,
                    confidence=0.99,
                ),
                category=NormalizedValue(
                    normalized=None,
                    raw=None,
                    status=ExtractionStatus.UNKNOWN,
                    confidence=None,
                ),
                confidence=0.90,
            )
        ],
        extraction_metadata=ExtractionMetadata(
            model_provider="google",
            model="gemini-3.6-flash",
            source_count=1,
            element_count=1,
        ),
    )

    assert extraction.elements[0].id == "element-1"
    assert extraction.elements[0].category is not None
    assert extraction.elements[0].category.status == ExtractionStatus.UNKNOWN
    assert extraction.elements[0].category.normalized is None