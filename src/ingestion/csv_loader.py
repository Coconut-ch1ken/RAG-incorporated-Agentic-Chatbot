"""
CSV Processor — Table-level serialization for CSV files.
Concatenates all rows into a single document so the LLM can see the
complete table context when answering queries.
"""
import pandas as pd
from typing import List, Dict, Any, Tuple


def process_csv_file(file_path: str, user_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Read a CSV file and serialize the entire table into a single document.
    Each row is rendered as 'Column: Value' pairs, separated by blank lines.
    A header preamble lists all column names for additional context.

    Returns (documents, metadatas) ready for vector store ingestion.
    """
    df = pd.read_csv(file_path)
    filename = file_path.split("/")[-1]

    if df.empty:
        return [], []

    # Build a preamble that describes the table structure
    columns_list = ", ".join(df.columns.tolist())
    preamble = f"Data from {filename} (columns: {columns_list}):\n"

    # Serialize every row into 'Column: Value' lines
    row_blocks = []
    for _, row in df.iterrows():
        context_str = "\n".join(
            [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        )
        if context_str.strip():
            row_blocks.append(context_str)

    if not row_blocks:
        return [], []

    # Join all rows into one document separated by blank lines
    full_document = preamble + "\n\n".join(row_blocks)

    documents = [full_document]
    metadatas = [{
        "user_id": user_id,
        "source": filename,
        "source_type": "csv",
        "file_path": file_path,
    }]

    return documents, metadatas
