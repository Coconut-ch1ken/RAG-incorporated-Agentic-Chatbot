from src.graph.state import GraphState
from src.database.vector_store import VectorStore

class RetrieveNode:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def __call__(self, state: GraphState) -> GraphState:
        print("---RETRIEVE---")
        question = state["question"]
        user_id = state["user_id"]

        # Scope retrieval to user_id
        retriever = self.vector_store.as_retriever(user_id=user_id)
        documents = retriever.invoke(question)
        
        # Extract page_content and source metadata
        doc_texts = [doc.page_content for doc in documents]
        sources = list(dict.fromkeys(
            doc.metadata.get("source", "unknown") for doc in documents
        ))

        return {"documents": doc_texts, "raw_documents": doc_texts, "sources": sources}
