"""
目录扫描器测试 — 验证文件发现、扩展名过滤和变更检测。
"""
import os
from unittest.mock import MagicMock, patch
from src.ingestion.directory_scanner import DirectoryScanner


class TestDirectoryScanner:
    """测试 DirectoryScanner 类。"""

    def _make_scanner(self):
        """创建一个使用 mock 依赖的 DirectoryScanner。"""
        mock_vector_store = MagicMock()
        mock_metadata_store = MagicMock()
        # 默认: 所有文件都是 "新的"（触发摄入）
        mock_metadata_store.check_file_changed.return_value = True
        mock_metadata_store.compute_file_hash = MagicMock(return_value="fake_hash")
        return DirectoryScanner(mock_vector_store, mock_metadata_store)

    def test_scan_ingests_supported_files(self, sample_data_dir):
        """验证 .txt 和 .csv 文件都被摄入。"""
        scanner = self._make_scanner()
        stats = scanner.scan(sample_data_dir, user_id="test_user")
        # sample_data_dir 包含 test_data.txt, test_data.csv, photo.jpg
        assert stats["ingested"] == 2  # .txt 和 .csv
        assert stats["errors"] == 0

    def test_scan_skips_unsupported_extensions(self, sample_data_dir):
        """验证 .jpg 文件被忽略。"""
        scanner = self._make_scanner()
        stats = scanner.scan(sample_data_dir, user_id="test_user")
        # 应该只摄入 .txt 和 .csv，跳过 .jpg
        assert stats["ingested"] == 2

    def test_scan_skips_unchanged_files(self, sample_data_dir):
        """验证未变更的文件被跳过。"""
        scanner = self._make_scanner()
        # 模拟: 文件未变更
        scanner.metadata_store.check_file_changed.return_value = False
        stats = scanner.scan(sample_data_dir, user_id="test_user")
        assert stats["ingested"] == 0
        assert stats["skipped"] == 2  # .txt 和 .csv 都被跳过

    def test_scan_invalid_directory(self):
        """验证不存在的目录返回错误。"""
        scanner = self._make_scanner()
        stats = scanner.scan("/nonexistent/directory", user_id="test_user")
        assert stats["errors"] == 1
        assert stats["ingested"] == 0

    def test_vector_store_called_on_ingest(self, sample_data_dir):
        """验证摄入时调用了 vector_store.add_documents。"""
        scanner = self._make_scanner()
        scanner.scan(sample_data_dir, user_id="test_user")
        assert scanner.vector_store.add_documents.call_count == 2

    def test_metadata_store_records_file(self, sample_data_dir):
        """验证摄入时调用了 metadata_store.add_file。"""
        scanner = self._make_scanner()
        scanner.scan(sample_data_dir, user_id="test_user")
        assert scanner.metadata_store.add_file.call_count == 2
