"""
Centralized configuration for the Personal Assistant RAG Chatbot.
Loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings, loaded from .env file."""

    # --- Directory to scan for personal data ---
    watch_directory: str = Field(
        default="./data",
        description="Path to the local directory containing personal data files.",
    )

    # --- Ollama (Local LLM) ---
    ollama_model: str = Field(
        default="llama3.1",
        description="Ollama model name for local generation and grading.",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model name for local embeddings.",
    )
    ollama_fallback_model: str = Field(
        default="gemma3:12b",
        description="Ollama model name for the powerful fallback generation tier.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama server.",
    )

    # --- Google Gemini (Online LLM Fallback) ---
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini online fallback.",
    )

    # --- ChromaDB ---
    chroma_db_path: str = Field(
        default="./chroma_db",
        description="Path to the ChromaDB persistent storage directory.",
    )
    chroma_collection_name: str = Field(
        default="personal_assistant",
        description="Name of the ChromaDB collection.",
    )

    # --- SQLite Metadata ---
    metadata_db_path: str = Field(
        default="./file_metadata.db",
        description="Path to the SQLite metadata database file.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()
