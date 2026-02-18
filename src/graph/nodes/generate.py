"""
Local Generate Node — uses Ollama for fast, private local generation,
then calls Gemini to enrich the answer with supplementary information.
"""
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState
from src.config import settings


class GenerateNode:
    def __init__(self):
        self.local_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
        )

    def __call__(self, state: GraphState) -> GraphState:
        print("---GENERATE (LOCAL)---")
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join(documents)

        # Step 1: Local LLM generates core answer from personal docs
        local_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a personal assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise.",
            ),
            (
                "human",
                "Question: {question}\n\nContext: {context}\n\nAnswer:",
            ),
        ])

        local_chain = local_prompt | self.local_llm | StrOutputParser()
        local_answer = local_chain.invoke({
            "context": context,
            "question": question,
        })
        print(f"    Local answer: {local_answer[:100]}...")

        # Step 2: Gemini enriches with supplementary info
        print("---ENRICH (GEMINI)---")
        enrich_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a personal assistant. A local AI has already answered "
                "the user's question using their personal data. Your job is to "
                "enrich this answer with useful supplementary information from "
                "your broader knowledge. Combine the local answer with your "
                "additions into one cohesive response. Keep the personal data "
                "as the primary source of truth — do not contradict it.",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Personal context:\n{context}\n\n"
                "Local AI answer: {local_answer}\n\n"
                "Please provide an enriched, comprehensive answer:",
            ),
        ])

        enrich_chain = enrich_prompt | self.gemini_llm | StrOutputParser()
        enriched = enrich_chain.invoke({
            "question": question,
            "context": context,
            "local_answer": local_answer,
        })

        print(f"\n📝 Prompt sent to Gemini (enrichment):\n"
              f"---\n"
              f"Personal context:\n{context}\n\n"
              f"Local AI answer: {local_answer}\n\n"
              f"Question: {question}\n"
              f"---\n")

        return {
            "generation": enriched,
            "generation_tier": "local+gemini",
            "retry_count": state.get("retry_count", 0) + 1,
        }
