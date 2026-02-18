"""
Gemini Fallback Node — uses Google Gemini API for answering questions
when no relevant local documents pass the grading filter.
Still includes the raw (unfiltered) retrieved documents as personal
background context, so Gemini has some knowledge of the user.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState


class GeminiFallbackNode:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
        )

    def __call__(self, state: GraphState) -> GraphState:
        print("---GENERATE (GEMINI FALLBACK — no graded docs)---")
        question = state["question"]

        # Use raw (unfiltered) docs as background context about the user,
        # even though none passed the relevance grading filter.
        raw_docs = state.get("raw_documents", [])

        if raw_docs:
            print(f"    Including {len(raw_docs)} raw documents as personal background")
            context_block = (
                "The following is background information about the user from their "
                "personal knowledge base. None of these were deemed directly relevant "
                "to the question by the grading system, but they may provide useful "
                "personal context:\n\n" + "\n\n".join(raw_docs)
            )

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a helpful personal assistant. The user asked a question "
                    "but no documents in their knowledge base were graded as directly "
                    "relevant. However, you are given some background information about "
                    "the user. Use this background AND your general knowledge to answer. "
                    "Be helpful, concise, and honest.",
                ),
                (
                    "human",
                    "{context}\n\n"
                    "Question: {question}",
                ),
            ])

            chain = prompt | self.llm | StrOutputParser()
            print(f"\n📝 Prompt sent to Gemini (fallback):\n"
                  f"---\n"
                  f"{context_block}\n\n"
                  f"Question: {question}\n"
                  f"---\n")
            generation = chain.invoke({"context": context_block, "question": question})
        else:
            print("    No raw documents available, answering with general knowledge")
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a helpful personal assistant. The user asked a question "
                    "but no information was found in their personal knowledge base. "
                    "Answer using your general knowledge. Be helpful and concise.",
                ),
                (
                    "human",
                    "{question}",
                ),
            ])

            chain = prompt | self.llm | StrOutputParser()
            generation = chain.invoke({"question": question})

        return {
            "generation": generation,
            "generation_tier": "gemini",
            "hallucination_status": True,  # Skip hallucination check
        }
