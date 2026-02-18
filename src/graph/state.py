from typing import TypedDict, List


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    documents: List[str]          # The graded/filtered context
    raw_documents: List[str]      # The original unfiltered retrieved docs
    generation: str               # The LLM's answer
    relevance_score: float        # Score from the grader
    hallucination_status: bool    # True if answer is supported by docs
    sufficiency_status: bool      # True if docs are sufficient to answer
    generation_tier: str          # "local", "powerful", or "gemini"
    retry_count: int              # Number of generation retries
    user_id: str                  # User context
