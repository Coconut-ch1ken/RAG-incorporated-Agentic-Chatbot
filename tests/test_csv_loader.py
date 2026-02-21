"""
数据摄入层测试 — CSV 加载器
验证 CSV 行到上下文的序列化、NaN 处理和空文件。
"""
import csv
from src.ingestion.csv_loader import process_csv_file


class TestProcessCsvFile:
    """测试 process_csv_file 函数。"""

    def test_returns_documents_and_metadatas(self, sample_csv_file):
        """验证正常 CSV 返回单个合并文档。"""
        docs, metas = process_csv_file(sample_csv_file, user_id="test_user")
        assert len(docs) == 1  # 整个表合并为一个文档
        assert len(metas) == 1
        # Both rows should be in the single document
        assert "Alice" in docs[0]
        assert "Bob" in docs[0]

    def test_row_serialization_format(self, sample_csv_file):
        """验证行被序列化为 'Column: Value' 格式。"""
        docs, _ = process_csv_file(sample_csv_file, user_id="test_user")
        assert "Name: Alice" in docs[0]
        assert "Age: 25" in docs[0]
        assert "City: Toronto" in docs[0]

    def test_metadata_fields(self, sample_csv_file):
        """验证元数据包含正确字段。"""
        _, metas = process_csv_file(sample_csv_file, user_id="test_user")
        meta = metas[0]
        assert meta["user_id"] == "test_user"
        assert meta["source_type"] == "csv"
        assert "source" in meta
        assert "file_path" in meta

    def test_empty_csv_returns_empty(self, empty_csv_file):
        """验证只有表头的 CSV 返回空列表。"""
        docs, metas = process_csv_file(empty_csv_file, user_id="test_user")
        assert docs == []
        assert metas == []

    def test_nan_values_skipped(self, tmp_path):
        """验证 NaN 值不会出现在序列化结果中。"""
        file_path = tmp_path / "nan_test.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Age", "City"])
            writer.writerow(["Alice", "", "Toronto"])
        docs, _ = process_csv_file(str(file_path), user_id="test_user")
        assert len(docs) == 1
        # The preamble lists "Age" as a column, but no "Age: <value>" row should exist
        lines = docs[0].split("\n")
        row_lines = [l for l in lines if l.startswith("Age:")]
        assert len(row_lines) == 0  # NaN values should be skipped
        assert "Name: Alice" in docs[0]
        assert "City: Toronto" in docs[0]
