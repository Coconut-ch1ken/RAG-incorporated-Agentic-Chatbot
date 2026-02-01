import pandas as pd
from typing import List, Dict
from src.database.vector_store import VectorStore
from src.database.metadata_store import MetadataStore

class CSVIngestor:
    def __init__(self, vector_store: VectorStore, metadata_store: MetadataStore):
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def ingest(self, file_path: str, user_id: str):
        """
        Ingests a CSV file.
        1. Reads CSV.
        2. Serializes each row to 'Column: Value' format.
        3. Embeds and stores in ChromaDB.
        4. Updates SQLite metadata.
        """
        df = pd.read_csv(file_path)
        
        # Serialize rows
        documents = []
        metadatas = []
        
        filename = file_path.split("/")[-1]

        for _, row in df.iterrows():
            # Row-to-Context Serialization
            # e.g. "Name: Alice \n Role: Manager"
            context_str = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            documents.append(context_str)
            
            # Metadata for filtration
            metadatas.append({
                "user_id": user_id,
                "source": filename
            })

        # Add to Vector DB
        if documents:
            self.vector_store.add_documents(texts=documents, metadatas=metadatas)
        
        # Update Metadata DB
        self.metadata_store.add_file(user_id, filename)
        print(f"Successfully ingested {len(documents)} rows for user {user_id}.")
