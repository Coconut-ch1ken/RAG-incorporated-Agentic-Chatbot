from typing import TypedDict, List

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    documents: List[str]  # The retrieved context
    generation: str       # The LLM's answer
    relevance_score: float # Score from the grader
    hallucination_status: bool # True if answer is supported by docs
    user_id: str          # Added to track user context
