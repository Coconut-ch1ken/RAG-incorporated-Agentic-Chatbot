import sqlite3
from typing import List, Tuple
from datetime import datetime

class MetadataStore:
    def __init__(self, db_path: str = "file_metadata.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_file(self, user_id: str, filename: str):
        """Record a new file upload."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO uploads (user_id, filename) VALUES (?, ?)',
            (user_id, filename)
        )
        conn.commit()
        conn.close()

    def get_user_files(self, user_id: str) -> List[str]:
        """Get list of filenames uploaded by a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT filename FROM uploads WHERE user_id = ?',
            (user_id,)
        )
        files = [row[0] for row in cursor.fetchall()]
        conn.close()
        return files
