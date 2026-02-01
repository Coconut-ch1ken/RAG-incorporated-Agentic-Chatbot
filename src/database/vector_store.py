import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, collection_name: str = "rag_store"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.collection_name = collection_name
        
        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add texts + metadata to the vector store."""
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def as_retriever(self, user_id: str):
        """Return a retriever scoped to the specific user."""
        return self.vectorstore.as_retriever(
            search_kwargs={'filter': {'user_id': user_id}}
        )
