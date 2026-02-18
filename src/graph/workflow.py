"""
RAG Agent Workflow — LangGraph state machine with two-tier generation.

Flow:
  Retrieve → Grade
    → (no relevant docs) → Gemini Fallback → END
    → (has relevant docs) → Sufficiency Check
      → (sufficient)   → Generate Local  → Hallucination Check → END / retry
      → (insufficient) → Generate Online → Hallucination Check → END / retry
"""
from langgraph.graph import END, StateGraph
from src.graph.state import GraphState
from src.graph.nodes.retrieve import RetrieveNode
from src.graph.nodes.grade import GradeNode
from src.graph.nodes.sufficiency import SufficiencyNode
from src.graph.nodes.generate import GenerateNode
from src.graph.nodes.generate_online import OnlineGenerateNode
from src.graph.nodes.hallucination import HallucinationNode
from src.graph.nodes.gemini_fallback import GeminiFallbackNode
from src.database.vector_store import VectorStore

MAX_RETRIES = 2


class RagAgent:
    def __init__(self):
        # Initialize Dependencies
        self.vector_store = VectorStore()

        # Initialize Nodes
        self.retrieve_node = RetrieveNode(self.vector_store)
        self.grade_node = GradeNode()
        self.sufficiency_node = SufficiencyNode()
        self.generate_node = GenerateNode()
        self.online_generate_node = OnlineGenerateNode()
        self.hallucination_node = HallucinationNode()
        self.gemini_fallback_node = GeminiFallbackNode()

        # Build Graph
        self.workflow = StateGraph(GraphState)

        # Add Nodes
        self.workflow.add_node("retrieve", self.retrieve_node)
        self.workflow.add_node("grade_documents", self.grade_node)
        self.workflow.add_node("sufficiency_check", self.sufficiency_node)
        self.workflow.add_node("generate_local", self.generate_node)
        self.workflow.add_node("generate_online", self.online_generate_node)
        self.workflow.add_node("hallucination_check", self.hallucination_node)
        self.workflow.add_node("gemini_fallback", self.gemini_fallback_node)

        # --- Edges ---
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")

        # After grading: check if any relevant docs remain
        def check_doc_relevance(state):
            if not state["documents"]:
                return "no_docs_gemini"
            return "check_sufficiency"

        self.workflow.add_conditional_edges(
            "grade_documents",
            check_doc_relevance,
            {
                "check_sufficiency": "sufficiency_check",
                "no_docs_gemini": "gemini_fallback",
            },
        )

        # Gemini fallback goes straight to END (no hallucination check needed)
        self.workflow.add_edge("gemini_fallback", END)

        # After sufficiency check: route to local or online generation
        def route_generation(state):
            if state.get("sufficiency_status", False):
                return "generate_local"
            return "generate_online"

        self.workflow.add_conditional_edges(
            "sufficiency_check",
            route_generation,
            {
                "generate_local": "generate_local",
                "generate_online": "generate_online",
            },
        )

        # Both generation paths lead to hallucination check
        self.workflow.add_edge("generate_local", "hallucination_check")
        self.workflow.add_edge("generate_online", "hallucination_check")

        # After hallucination check: success, retry, or give up
        def check_hallucination(state):
            if state.get("hallucination_status", False):
                return "end_success"
            if state.get("retry_count", 0) >= MAX_RETRIES:
                print(f"---MAX RETRIES ({MAX_RETRIES}) REACHED, RETURNING BEST ANSWER---")
                return "end_max_retries"
            return "generate_retry"

        self.workflow.add_conditional_edges(
            "hallucination_check",
            check_hallucination,
            {
                "end_success": END,
                "end_max_retries": END,
                "generate_retry": "generate_local",
            },
        )

        self.app = self.workflow.compile()

    def run(self, question: str, user_id: str):
        inputs = {"question": question, "user_id": user_id, "retry_count": 0}
        try:
            return self.app.invoke(inputs, config={"recursion_limit": 10})
        except Exception as e:
            if "recursion" in str(e).lower():
                print(f"---RECURSION LIMIT REACHED, STOPPING---")
                return {
                    "generation": "I wasn't able to produce a fully verified answer. "
                                  "Please try rephrasing your question.",
                    "hallucination_status": False,
                    "generation_tier": "local",
                }
            raise
