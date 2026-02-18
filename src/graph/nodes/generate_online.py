"""
Online Generate Node — uses Google Gemini API for richer answers.
Invoked when the sufficiency check determines local docs are insufficient.
Sends personal context + question to Gemini.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState


class OnlineGenerateNode:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
        )

    def __call__(self, state: GraphState) -> GraphState:
        print("---GENERATE (GEMINI — docs insufficient)---")
        question = state["question"]
        documents = state["documents"]

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a personal assistant. The user has a question that "
                "their local documents can only partially answer. Below is the "
                "relevant personal context retrieved from their knowledge base. "
                "Use this context AND your broader knowledge to provide a "
                "comprehensive, helpful answer. Always ground your answer in "
                "the provided context where possible.",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "My personal context:\n{context}\n\n"
                "Please provide a thorough answer:",
            ),
        ])

        context = "\n\n".join(documents)
        print(f"\n📝 Prompt sent to Gemini:\n"
              f"---\n"
              f"Personal context:\n{context}\n\n"
              f"Question: {question}\n"
              f"---\n")

        rag_chain = prompt | self.llm | StrOutputParser()

        generation = rag_chain.invoke({
            "context": context,
            "question": question,
        })
        return {"generation": generation, "generation_tier": "gemini"}
