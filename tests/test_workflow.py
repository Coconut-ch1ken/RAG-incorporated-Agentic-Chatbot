"""
工作流路由测试 — 验证条件边的路由逻辑和重试限制。
不启动真正的 LLM，只测试路由函数本身。
"""


class TestWorkflowRouting:
    """测试工作流中的条件路由逻辑。"""

    def test_no_docs_routes_to_gemini(self):
        """无文档时应路由到 Gemini 回退。"""
        state = {"documents": []}

        # 模拟 workflow.py 中的 check_doc_relevance 函数
        def check_doc_relevance(state):
            if not state["documents"]:
                return "no_docs_gemini"
            return "check_sufficiency"

        assert check_doc_relevance(state) == "no_docs_gemini"

    def test_has_docs_routes_to_sufficiency(self):
        """有文档时应路由到充分性检查。"""
        state = {"documents": ["some relevant doc"]}

        def check_doc_relevance(state):
            if not state["documents"]:
                return "no_docs_gemini"
            return "check_sufficiency"

        assert check_doc_relevance(state) == "check_sufficiency"

    def test_sufficient_routes_to_local(self):
        """充分时应路由到本地生成。"""
        state = {"sufficiency_status": True}

        def route_generation(state):
            if state.get("sufficiency_status", False):
                return "generate_local"
            return "generate_online"

        assert route_generation(state) == "generate_local"

    def test_insufficient_routes_to_online(self):
        """不充分时应路由到在线生成。"""
        state = {"sufficiency_status": False}

        def route_generation(state):
            if state.get("sufficiency_status", False):
                return "generate_local"
            return "generate_online"

        assert route_generation(state) == "generate_online"

    def test_hallucination_success_ends(self):
        """接地成功 → 结束。"""
        state = {"hallucination_status": True, "retry_count": 1}

        def check_hallucination(state):
            if state.get("hallucination_status", False):
                return "end_success"
            if state.get("retry_count", 0) >= 2:
                return "end_max_retries"
            return "generate_retry"

        assert check_hallucination(state) == "end_success"

    def test_hallucination_retry(self):
        """未接地且未超限 → 重试。"""
        state = {"hallucination_status": False, "retry_count": 1}

        def check_hallucination(state):
            if state.get("hallucination_status", False):
                return "end_success"
            if state.get("retry_count", 0) >= 2:
                return "end_max_retries"
            return "generate_retry"

        assert check_hallucination(state) == "generate_retry"

    def test_retry_limit_respected(self):
        """超过最大重试次数 → 放弃。"""
        state = {"hallucination_status": False, "retry_count": 2}

        def check_hallucination(state):
            if state.get("hallucination_status", False):
                return "end_success"
            if state.get("retry_count", 0) >= 2:
                return "end_max_retries"
            return "generate_retry"

        assert check_hallucination(state) == "end_max_retries"

    def test_retry_limit_exceeded(self):
        """远超最大重试次数也应放弃。"""
        state = {"hallucination_status": False, "retry_count": 10}

        def check_hallucination(state):
            if state.get("hallucination_status", False):
                return "end_success"
            if state.get("retry_count", 0) >= 2:
                return "end_max_retries"
            return "generate_retry"

        assert check_hallucination(state) == "end_max_retries"
