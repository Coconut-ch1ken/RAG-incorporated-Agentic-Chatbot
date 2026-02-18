"""
Processor for text-based files: .txt, .md, .pdf
Chunks text using LangChain's RecursiveCharacterTextSplitter.
"""
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def process_text_file(file_path: str, user_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read a .txt or .md file, chunk it, and return (documents, metadatas)."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    return _chunk_and_prepare(raw_text, file_path, user_id, source_type="text")


def process_pdf_file(file_path: str, user_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extract text from a PDF, chunk it, and return (documents, metadatas)."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("  ⚠️  PyPDF2 not installed. Skipping PDF: " + file_path)
        return [], []

    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    raw_text = "\n\n".join(pages_text)
    if not raw_text.strip():
        print(f"  ⚠️  No text extracted from PDF: {file_path}")
        return [], []

    return _chunk_and_prepare(raw_text, file_path, user_id, source_type="pdf")


def _chunk_and_prepare(
    raw_text: str,
    file_path: str,
    user_id: str,
    source_type: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Split raw text into chunks and prepare metadata for each."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

    filename = file_path.split("/")[-1]
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        documents.append(chunk)
        metadatas.append({
            "user_id": user_id,
            "source": filename,
            "source_type": source_type,
            "file_path": file_path,
            "chunk_index": i,
        })

    return documents, metadatas
