"""
Directory Scanner — walks a local directory, detects file types,
and routes each file to the appropriate processor for ingestion.
"""
import os
from typing import Dict, Callable, Tuple, List, Any

from src.database.vector_store import VectorStore
from src.database.metadata_store import MetadataStore
from src.ingestion.text_processor import process_text_file, process_pdf_file
from src.ingestion.csv_loader import process_csv_file


# Map file extensions to their processor functions
# Each processor returns (documents: List[str], metadatas: List[Dict])
EXTENSION_MAP: Dict[str, Callable] = {
    ".txt": process_text_file,
    ".md": process_text_file,
    ".pdf": process_pdf_file,
    ".csv": process_csv_file,
    "": process_text_file,  # Extensionless files treated as plain text
}

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())


class DirectoryScanner:
    def __init__(self, vector_store: VectorStore, metadata_store: MetadataStore):
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def scan(self, directory: str, user_id: str) -> Dict[str, int]:
        """
        Walk a directory, process all supported files, and ingest them.

        Returns a summary dict: {"ingested": N, "skipped": M, "errors": E}
        """
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            print(f"Error: '{directory}' is not a valid directory.")
            return {"ingested": 0, "skipped": 0, "errors": 1}

        stats = {"ingested": 0, "skipped": 0, "errors": 0}

        for root, _dirs, files in os.walk(directory):
            for filename in sorted(files):
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()

                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                # --- Change detection: skip unchanged files ---
                try:
                    current_hash = MetadataStore.compute_file_hash(file_path)
                except OSError as e:
                    print(f"  ⚠️  Cannot read {file_path}: {e}")
                    stats["errors"] += 1
                    continue

                if not self.metadata_store.check_file_changed(file_path, current_hash):
                    print(f"  ⏩ Skipping (unchanged): {file_path}")
                    stats["skipped"] += 1
                    continue

                # --- Process the file ---
                processor = EXTENSION_MAP[ext]
                try:
                    print(f"  📄 Processing: {file_path}")
                    documents, metadatas = processor(file_path, user_id)

                    if documents:
                        self.vector_store.add_documents(
                            texts=documents, metadatas=metadatas
                        )
                        self.metadata_store.add_file(
                            user_id=user_id,
                            filename=filename,
                            file_path=file_path,
                            file_hash=current_hash,
                            source_type=ext.lstrip("."),
                        )
                        print(f"    ✅ Ingested {len(documents)} chunks")
                        stats["ingested"] += 1
                    else:
                        print(f"    ⚠️  No content extracted from {file_path}")
                        stats["skipped"] += 1

                except Exception as e:
                    print(f"    ❌ Error processing {file_path}: {e}")
                    stats["errors"] += 1

        return stats
