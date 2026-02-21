"""
Hallucination Check Node — verifies the generated answer is grounded
in the retrieved documents and actually addresses the question.
Uses Pydantic structured output for reliable boolean results.
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState
from src.graph.schemas import HallucinationResult
from src.config import settings


class HallucinationNode:
    def __init__(self):
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self.structured_llm = llm.with_structured_output(HallucinationResult)

    def __call__(self, state: GraphState) -> GraphState:
        print("---CHECK HALLUCINATIONS---")
        documents = state["documents"]
        generation = state["generation"]
        question = state["question"]

        # --- Phase 1: Groundedness Check ---
        hallucination_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader assessing whether an LLM generation is "
                "grounded in (i.e., supported by) the provided facts.\n"
                "Grade as grounded if the generation is supported by the facts. "
                "Grade as not grounded if the generation contains claims not in "
                "the facts.",
            ),
            (
                "human",
                "Facts:\n{documents}\n\n"
                "Generation: {generation}\n\n"
                "Is the generation grounded in the facts?",
            ),
        ])

        chain = hallucination_prompt | self.structured_llm
        result: HallucinationResult = chain.invoke({
            "documents": "\n\n".join(documents),
            "generation": generation,
        })
        print(f"    Hallucination check result: grounded={result.score}")

        if not result.score:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
            return {"hallucination_status": False}

        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        # --- Phase 2: Question Resolution Check ---
        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader assessing whether an answer addresses "
                "the user's question.\n"
                "Grade as resolving if the answer addresses the question. "
                "Grade as not resolving if the answer does not address the question.",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Answer: {generation}\n\n"
                "Does the answer address the question?",
            ),
        ])

        chain2 = answer_prompt | self.structured_llm
        result2: HallucinationResult = chain2.invoke({
            "question": question,
            "generation": generation,
        })
        print(f"    Answer check result: resolves={result2.score}")

        if result2.score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return {"hallucination_status": True}
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return {"hallucination_status": False}
