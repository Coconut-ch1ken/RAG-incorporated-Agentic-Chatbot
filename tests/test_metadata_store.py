"""
数据存储层测试 — 元数据存储
验证 SQLite 的 CRUD 操作、文件哈希和变更检测。
"""
import os
from unittest.mock import patch
from src.database.metadata_store import MetadataStore


class TestMetadataStore:
    """测试 MetadataStore 类。"""

    def _make_store(self, tmp_path):
        """创建使用临时数据库路径的 MetadataStore。"""
        db_path = str(tmp_path / "test_metadata.db")
        with patch("src.database.metadata_store.settings") as mock_settings:
            mock_settings.metadata_db_path = db_path
            return MetadataStore()

    def test_add_and_get_files(self, tmp_path):
        """验证添加文件记录并按用户检索。"""
        store = self._make_store(tmp_path)
        store.add_file("user1", "data.txt", "/path/data.txt", "abc123", "text")
        store.add_file("user1", "info.csv", "/path/info.csv", "def456", "csv")
        store.add_file("user2", "other.txt", "/path/other.txt", "ghi789", "text")

        user1_files = store.get_user_files("user1")
        assert len(user1_files) == 2
        assert "data.txt" in user1_files
        assert "info.csv" in user1_files

        user2_files = store.get_user_files("user2")
        assert len(user2_files) == 1

    def test_file_hash_storage(self, tmp_path):
        """验证文件哈希的存储和检索。"""
        store = self._make_store(tmp_path)
        store.add_file("user1", "data.txt", "/path/data.txt", "hash_abc", "text")

        stored_hash = store.get_file_hash("/path/data.txt")
        assert stored_hash == "hash_abc"

    def test_get_file_hash_not_found(self, tmp_path):
        """验证未找到文件时返回 None。"""
        store = self._make_store(tmp_path)
        assert store.get_file_hash("/nonexistent/path.txt") is None

    def test_check_file_changed_new_file(self, tmp_path):
        """验证新文件被标记为已变更。"""
        store = self._make_store(tmp_path)
        assert store.check_file_changed("/new/file.txt", "some_hash") is True

    def test_check_file_unchanged(self, tmp_path):
        """验证未变更的文件被正确跳过。"""
        store = self._make_store(tmp_path)
        store.add_file("user1", "data.txt", "/path/data.txt", "same_hash", "text")
        assert store.check_file_changed("/path/data.txt", "same_hash") is False

    def test_check_file_changed_modified(self, tmp_path):
        """验证修改过的文件被标记为已变更。"""
        store = self._make_store(tmp_path)
        store.add_file("user1", "data.txt", "/path/data.txt", "old_hash", "text")
        assert store.check_file_changed("/path/data.txt", "new_hash") is True

    def test_compute_file_hash_deterministic(self, sample_txt_file):
        """验证 SHA-256 哈希的确定性。"""
        hash1 = MetadataStore.compute_file_hash(sample_txt_file)
        hash2 = MetadataStore.compute_file_hash(sample_txt_file)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex 长度
