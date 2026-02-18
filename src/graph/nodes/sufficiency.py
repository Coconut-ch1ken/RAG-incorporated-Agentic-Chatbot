"""
Sufficiency Check Node — determines whether the retrieved documents
contain enough information to fully answer the user's question.
Routes to local LLM (sufficient) or powerful LLM (insufficient).
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState
from src.config import settings


def _parse_yes_no(response: str) -> bool:
    """Robustly parse a yes/no response from an LLM."""
    cleaned = response.strip().lower()
    if cleaned.startswith("no"):
        return False
    return "yes" in cleaned


class SufficiencyNode:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    def __call__(self, state: GraphState) -> GraphState:
        print("---CHECK SUFFICIENCY---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            print("---DECISION: NO DOCUMENTS → INSUFFICIENT---")
            return {"sufficiency_status": False}

        context = "\n\n".join(documents)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader. You must respond with ONLY the word 'yes' or 'no', "
                "nothing else.\n\n"
                "Assess whether the provided documents contain ENOUGH information "
                "to FULLY answer the user's question.\n"
                "'yes' = the documents contain all key facts needed.\n"
                "'no' = the documents are partial or missing critical info.",
            ),
            (
                "human",
                "Documents:\n{context}\n\n"
                "Question: {question}\n\n"
                "Can you fully answer this question using ONLY these documents? "
                "Respond with only 'yes' or 'no':",
            ),
        ])

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"context": context, "question": question})

        is_sufficient = _parse_yes_no(result)

        if is_sufficient:
            print("---DECISION: DOCUMENTS SUFFICIENT → LOCAL LLM---")
        else:
            print("---DECISION: DOCUMENTS INSUFFICIENT → POWERFUL LLM---")

        return {"sufficiency_status": is_sufficient}
