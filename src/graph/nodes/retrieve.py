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
        documents = retriever.get_relevant_documents(question)
        
        # Extract page_content
        doc_texts = [doc.page_content for doc in documents]
        
        return {"documents": doc_texts}
