"""
CSV Processor — Row-to-Context serialization for CSV files.
Preserves the original semantic approach from the project.
"""
import pandas as pd
from typing import List, Dict, Any, Tuple


def process_csv_file(file_path: str, user_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Read a CSV file and serialize each row to 'Column: Value' format.
    Returns (documents, metadatas) ready for vector store ingestion.
    """
    df = pd.read_csv(file_path)

    documents = []
    metadatas = []
    filename = file_path.split("/")[-1]

    for _, row in df.iterrows():
        # Row-to-Context Serialization
        # e.g. "Name: Alice \n Role: Manager"
        context_str = "\n".join(
            [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        )
        if not context_str.strip():
            continue

        documents.append(context_str)
        metadatas.append({
            "user_id": user_id,
            "source": filename,
            "source_type": "csv",
            "file_path": file_path,
        })

    return documents, metadatas
