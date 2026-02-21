"""
Sufficiency Check Node — determines whether the retrieved documents
contain enough information to fully answer the user's question.
Routes to local LLM (sufficient) or powerful LLM (insufficient).
Uses Pydantic structured output for reliable boolean results.
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState
from src.graph.schemas import GradeResult
from src.config import settings


class SufficiencyNode:
    def __init__(self):
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self.structured_llm = llm.with_structured_output(GradeResult)

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
                "You are a grader assessing whether the provided documents "
                "contain ENOUGH information to FULLY answer the user's question.\n"
                "Grade as sufficient if the documents contain all key facts needed. "
                "Grade as insufficient if the documents are partial or missing "
                "critical info.",
            ),
            (
                "human",
                "Documents:\n{context}\n\n"
                "Question: {question}\n\n"
                "Can you fully answer this question using ONLY these documents?",
            ),
        ])

        chain = prompt | self.structured_llm
        result: GradeResult = chain.invoke({"context": context, "question": question})

        is_sufficient = result.score

        if is_sufficient:
            print("---DECISION: DOCUMENTS SUFFICIENT → LOCAL LLM---")
        else:
            print("---DECISION: DOCUMENTS INSUFFICIENT → POWERFUL LLM---")

        return {"sufficiency_status": is_sufficient}
