"""
图节点测试 — Grade、Sufficiency、Hallucination
所有 LLM 调用均 mock，验证节点的路由逻辑。
Uses Pydantic structured output mocks (GradeResult / HallucinationResult).
"""
from unittest.mock import patch, MagicMock
from src.graph.nodes.grade import GradeNode
from src.graph.nodes.sufficiency import SufficiencyNode
from src.graph.nodes.hallucination import HallucinationNode
from src.graph.schemas import GradeResult, HallucinationResult


# ========== Grade Node 测试 ==========

class TestGradeNode:
    """测试文档相关性评分节点。"""

    def _make_node_with_mock(self, score: bool):
        """创建一个使用 mock structured LLM 的 GradeNode。"""
        with patch("src.graph.nodes.grade.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = GradeNode()
        return node

    def test_grade_keeps_relevant(self):
        """mock LLM 返回 score=True → 文档应保留。"""
        with patch("src.graph.nodes.grade.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = GradeNode()

            # Mock the chain to return GradeResult(score=True)
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = GradeResult(score=True)
            with patch("src.graph.nodes.grade.ChatPromptTemplate") as MockPrompt:
                MockPrompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

                state = {
                    "question": "What is my name?",
                    "documents": ["My name is James Yuan."],
                }
                result = node(state)
                assert len(result["documents"]) == 1

    def test_grade_filters_irrelevant(self):
        """mock LLM 返回 score=False → 文档应被过滤。"""
        with patch("src.graph.nodes.grade.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = GradeNode()

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = GradeResult(score=False)
            with patch("src.graph.nodes.grade.ChatPromptTemplate") as MockPrompt:
                MockPrompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

                state = {
                    "question": "What is the weather?",
                    "documents": ["My name is James Yuan."],
                }
                result = node(state)
                assert len(result["documents"]) == 0


# ========== Sufficiency Node 测试 ==========

class TestSufficiencyNode:
    """测试文档充分性检查节点。"""

    def test_empty_docs_insufficient(self):
        """无文档 → 自动判定不充分。"""
        with patch("src.graph.nodes.sufficiency.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = SufficiencyNode()
        state = {"question": "What is my name?", "documents": []}
        result = node(state)
        assert result["sufficiency_status"] is False

    def test_sufficient_response(self):
        """mock LLM 返回 score=True → sufficiency_status=True。"""
        with patch("src.graph.nodes.sufficiency.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = SufficiencyNode()

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = GradeResult(score=True)
            with patch("src.graph.nodes.sufficiency.ChatPromptTemplate") as MockPrompt:
                MockPrompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

                state = {
                    "question": "What is my name?",
                    "documents": ["My name is James Yuan."],
                }
                result = node(state)
                assert result["sufficiency_status"] is True

    def test_insufficient_response(self):
        """mock LLM 返回 score=False → sufficiency_status=False。"""
        with patch("src.graph.nodes.sufficiency.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = SufficiencyNode()

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = GradeResult(score=False)
            with patch("src.graph.nodes.sufficiency.ChatPromptTemplate") as MockPrompt:
                MockPrompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

                state = {
                    "question": "What is quantum computing?",
                    "documents": ["My name is James Yuan."],
                }
                result = node(state)
                assert result["sufficiency_status"] is False


# ========== Hallucination Node 测试 ==========

class TestHallucinationNode:
    """测试幻觉检查节点。"""

    def test_grounded_and_addresses_question(self):
        """两项检查均通过 → hallucination_status=True。"""
        with patch("src.graph.nodes.hallucination.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = HallucinationNode()

            mock_chain = MagicMock()
            # 第一次调用: 接地检查 → True，第二次调用: 问题解决检查 → True
            mock_chain.invoke.side_effect = [
                HallucinationResult(score=True),
                HallucinationResult(score=True),
            ]
            with patch("src.graph.nodes.hallucination.ChatPromptTemplate") as MockPrompt:
                MockPrompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

                state = {
                    "question": "What is my name?",
                    "documents": ["My name is James Yuan."],
                    "generation": "Your name is James Yuan.",
                }
                result = node(state)
                assert result["hallucination_status"] is True

    def test_not_grounded(self):
        """接地检查失败 → hallucination_status=False。"""
        with patch("src.graph.nodes.hallucination.ChatOllama") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_instance
            MockLLM.return_value = mock_instance
            node = HallucinationNode()

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = HallucinationResult(score=False)
            with patch("src.graph.nodes.hallucination.ChatPromptTemplate") as MockPrompt:
                MockPrompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

                state = {
                    "question": "What is my name?",
                    "documents": ["I like pizza."],
                    "generation": "Your name is John Smith.",
                }
                result = node(state)
                assert result["hallucination_status"] is False
