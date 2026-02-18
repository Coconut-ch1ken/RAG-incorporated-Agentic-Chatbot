"""
Grade Node — filters retrieved documents for relevance using local Ollama.
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


class GradeNode:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    def __call__(self, state: GraphState) -> GraphState:
        print("---CHECK RELEVANCE---")
        question = state["question"]
        documents = state["documents"]

        grade_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a grader. You must respond with ONLY the word 'yes' or 'no', "
                "nothing else.\n\n"
                "Assess whether a retrieved document is relevant to the user's question.\n"
                "If the document contains keywords or meaning related to the question, "
                "answer 'yes'. Otherwise answer 'no'.",
            ),
            (
                "human",
                "Document:\n{document}\n\n"
                "Question: {question}\n\n"
                "Is this document relevant? Respond with only 'yes' or 'no':",
            ),
        ])

        chain = grade_prompt | self.llm | StrOutputParser()

        filtered_docs = []
        for doc in documents:
            result = chain.invoke({"question": question, "document": doc})
            if _parse_yes_no(result):
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs}
