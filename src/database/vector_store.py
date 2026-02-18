import chromadb
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict, Any
from datetime import datetime
from src.config import settings


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self.embedding_function = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
        self.collection_name = settings.chroma_collection_name

        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add texts + metadata to the vector store.

        Automatically enriches metadata with ingestion timestamp.
        """
        now = datetime.now().isoformat()
        for meta in metadatas:
            meta.setdefault("ingested_at", now)

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def as_retriever(self, user_id: str, **kwargs):
        """Return a retriever scoped to the specific user."""
        search_kwargs = {"filter": {"user_id": user_id}}
        search_kwargs.update(kwargs)
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
