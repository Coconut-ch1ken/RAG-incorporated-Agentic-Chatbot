"""
数据摄入层测试 — 文本处理器
验证 .txt 文件的分块、元数据生成和边界情况处理。
"""
from src.ingestion.text_processor import process_text_file, _chunk_and_prepare


class TestProcessTextFile:
    """测试 process_text_file 函数。"""

    def test_returns_documents_and_metadatas(self, sample_txt_file):
        """验证正常文件返回非空的文档和元数据列表。"""
        docs, metas = process_text_file(sample_txt_file, user_id="test_user")
        assert len(docs) > 0
        assert len(metas) > 0
        assert len(docs) == len(metas)

    def test_metadata_fields(self, sample_txt_file):
        """验证元数据包含所有必要字段。"""
        docs, metas = process_text_file(sample_txt_file, user_id="test_user")
        meta = metas[0]
        assert meta["user_id"] == "test_user"
        assert meta["source"] == "test_data.txt"
        assert meta["source_type"] == "text"
        assert "file_path" in meta
        assert "chunk_index" in meta

    def test_empty_file_returns_empty(self, empty_txt_file):
        """验证空文件返回空列表，不会报错。"""
        docs, metas = process_text_file(empty_txt_file, user_id="test_user")
        assert docs == []
        assert metas == []

    def test_chunk_content_preserved(self, sample_txt_file):
        """验证分块后原始内容仍然存在。"""
        docs, _ = process_text_file(sample_txt_file, user_id="test_user")
        combined = " ".join(docs)
        assert "James Yuan" in combined
        assert "Computer Science" in combined


class TestChunkAndPrepare:
    """测试内部分块函数。"""

    def test_small_text_single_chunk(self):
        """短文本应生成单个分块。"""
        docs, metas = _chunk_and_prepare(
            "Hello world.", "/fake/path.txt", "user1", "text"
        )
        assert len(docs) == 1
        assert docs[0] == "Hello world."

    def test_chunk_index_sequential(self):
        """分块索引应按顺序递增。"""
        long_text = "This is a test sentence. " * 100
        docs, metas = _chunk_and_prepare(
            long_text, "/fake/long.txt", "user1", "text"
        )
        indices = [m["chunk_index"] for m in metas]
        assert indices == sorted(indices)
