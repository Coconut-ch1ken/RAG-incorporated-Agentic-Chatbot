"""
共享测试 Fixtures — 为所有测试提供临时目录、样本文件和 mock 配置。
"""
import os
import csv
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """提供一个临时目录用于测试。"""
    return tmp_path


@pytest.fixture
def sample_txt_file(tmp_path):
    """创建一个包含示例文本的临时 .txt 文件。"""
    file_path = tmp_path / "test_data.txt"
    file_path.write_text(
        "My name is James Yuan. I study Computer Science at the University of Waterloo.\n"
        "I enjoy building AI projects and working with RAG systems.\n"
        "My favourite programming language is Python.\n",
        encoding="utf-8",
    )
    return str(file_path)


@pytest.fixture
def sample_csv_file(tmp_path):
    """创建一个包含示例数据的临时 .csv 文件。"""
    file_path = tmp_path / "test_data.csv"
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Age", "City"])
        writer.writerow(["Alice", "25", "Toronto"])
        writer.writerow(["Bob", "30", "Vancouver"])
    return str(file_path)


@pytest.fixture
def empty_txt_file(tmp_path):
    """创建一个空的 .txt 文件。"""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return str(file_path)


@pytest.fixture
def empty_csv_file(tmp_path):
    """创建一个只有表头的空 .csv 文件。"""
    file_path = tmp_path / "empty.csv"
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Age", "City"])
    return str(file_path)


@pytest.fixture
def sample_data_dir(tmp_path, sample_txt_file, sample_csv_file):
    """创建一个包含多种文件类型的数据目录用于扫描测试。"""
    # 已有 .txt 和 .csv 文件在 tmp_path 中
    # 添加一个不支持的文件类型
    jpg_file = tmp_path / "photo.jpg"
    jpg_file.write_bytes(b"\xff\xd8\xff\xe0fake_jpg_data")
    return str(tmp_path)
