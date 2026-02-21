"""
Grade Node — filters retrieved documents for relevance using local Ollama
with Pydantic structured output.
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState
from src.graph.schemas import GradeResult
from src.config import settings


class GradeNode:
    def __init__(self):
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self.structured_llm = llm.with_structured_output(GradeResult)

    def __call__(self, state: GraphState) -> GraphState:
        print("---CHECK RELEVANCE---")
        question = state["question"]
        documents = state["documents"]

        grade_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader assessing whether a retrieved document is "
                "relevant to the user's question.\n"
                "If the document contains keywords or meaning related to the "
                "question, grade it as relevant.",
            ),
            (
                "human",
                "Document:\n{document}\n\n"
                "Question: {question}\n\n"
                "Is this document relevant to the question?",
            ),
        ])

        chain = grade_prompt | self.structured_llm

        filtered_docs = []
        for doc in documents:
            result: GradeResult = chain.invoke({"question": question, "document": doc})
            if result.score:
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs}
