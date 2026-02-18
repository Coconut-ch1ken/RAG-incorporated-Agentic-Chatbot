import sqlite3
import hashlib
from typing import List, Optional
from src.config import settings


class MetadataStore:
    def __init__(self):
        self.db_path = settings.metadata_db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT,
                file_hash TEXT,
                source_type TEXT DEFAULT 'unknown',
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def add_file(self, user_id: str, filename: str, file_path: str = "",
                 file_hash: str = "", source_type: str = "unknown"):
        """Record a new file upload."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO uploads (user_id, filename, file_path, file_hash, source_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, file_path, file_hash, source_type),
        )
        conn.commit()
        conn.close()

    def get_user_files(self, user_id: str) -> List[str]:
        """Get list of filenames uploaded by a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filename FROM uploads WHERE user_id = ?", (user_id,)
        )
        files = [row[0] for row in cursor.fetchall()]
        conn.close()
        return files

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get the stored hash for a file path. Returns None if not found."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_hash FROM uploads WHERE file_path = ? "
            "ORDER BY upload_timestamp DESC LIMIT 1",
            (file_path,),
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def check_file_changed(self, file_path: str, current_hash: str) -> bool:
        """Return True if the file is new or has changed since last ingestion."""
        stored_hash = self.get_file_hash(file_path)
        return stored_hash != current_hash

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute the SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
