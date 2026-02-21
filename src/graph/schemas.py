"""
Pydantic schemas for structured LLM outputs.
Used by grading, sufficiency, and hallucination nodes
to get reliable boolean results instead of parsing raw strings.
"""
from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    """Result of a document relevance or sufficiency grading."""
    score: bool = Field(
        description="Set to true if the document is relevant or sufficient, false otherwise."
    )


class HallucinationResult(BaseModel):
    """Result of a hallucination or answer-resolution check."""
    score: bool = Field(
        description="Set to true if the generation is grounded in facts / resolves the question, false otherwise."
    )
