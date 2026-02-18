"""
Hallucination Check Node — verifies the generated answer is grounded
in the retrieved documents and actually addresses the question.
Uses local Ollama for privacy.
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState
from src.config import settings


def _parse_yes_no(response: str) -> bool:
    """Robustly parse a yes/no response from an LLM.

    Returns True if the response contains 'yes', False otherwise.
    Handles verbose models that bury the answer in reasoning.
    """
    cleaned = response.strip().lower()
    # Check for explicit no first (handles "no, ..." responses)
    if cleaned.startswith("no"):
        return False
    # Look for "yes" anywhere in the response
    return "yes" in cleaned


class HallucinationNode:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    def __call__(self, state: GraphState) -> GraphState:
        print("---CHECK HALLUCINATIONS---")
        documents = state["documents"]
        generation = state["generation"]
        question = state["question"]

        # --- Phase 1: Groundedness Check ---
        hallucination_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader. You must respond with ONLY the word 'yes' or 'no', "
                "nothing else.\n\n"
                "Assess whether the following LLM generation is grounded in "
                "(i.e., supported by) the provided facts.\n"
                "'yes' = the generation is supported by the facts.\n"
                "'no' = the generation contains claims not in the facts.",
            ),
            (
                "human",
                "Facts:\n{documents}\n\n"
                "Generation: {generation}\n\n"
                "Is the generation grounded in the facts? Respond with only 'yes' or 'no':",
            ),
        ])

        chain = hallucination_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "documents": "\n\n".join(documents),
            "generation": generation,
        })
        print(f"    Hallucination check raw response: '{result.strip()}'")

        if not _parse_yes_no(result):
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
            return {"hallucination_status": False}

        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        # --- Phase 2: Question Resolution Check ---
        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader. You must respond with ONLY the word 'yes' or 'no', "
                "nothing else.\n\n"
                "Assess whether the following answer addresses the user's question.\n"
                "'yes' = the answer resolves the question.\n"
                "'no' = the answer does not resolve the question.",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Answer: {generation}\n\n"
                "Does the answer address the question? Respond with only 'yes' or 'no':",
            ),
        ])

        chain2 = answer_prompt | self.llm | StrOutputParser()
        result2 = chain2.invoke({
            "question": question,
            "generation": generation,
        })
        print(f"    Answer check raw response: '{result2.strip()}'")

        if _parse_yes_no(result2):
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return {"hallucination_status": True}
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return {"hallucination_status": False}
