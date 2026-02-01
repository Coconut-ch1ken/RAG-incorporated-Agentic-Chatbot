from langgraph.graph import END, StateGraph
from src.graph.state import GraphState
from src.graph.nodes.retrieve import RetrieveNode
from src.graph.nodes.grade import GradeNode
from src.graph.nodes.generate import GenerateNode
from src.graph.nodes.hallucination import HallucinationNode
from src.database.vector_store import VectorStore

class RagAgent:
    def __init__(self):
        # Initialize Dependencies
        self.vector_store = VectorStore()
        
        # Initialize Nodes
        self.retrieve_node = RetrieveNode(self.vector_store)
        self.grade_node = GradeNode()
        self.generate_node = GenerateNode()
        self.hallucination_node = HallucinationNode()
        
        # Build Graph
        self.workflow = StateGraph(GraphState)
        
        # Add Nodes
        self.workflow.add_node("retrieve", self.retrieve_node)
        self.workflow.add_node("grade_documents", self.grade_node)
        self.workflow.add_node("generate", self.generate_node)
        self.workflow.add_node("hallucination_check", self.hallucination_node)
        
        # Add Edges
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional Edge after Grading
        def check_doc_relevance(state):
            if not state["documents"]:
                return "end_no_data"
            return "generate"

        self.workflow.add_conditional_edges(
            "grade_documents",
            check_doc_relevance,
            {
                "generate": "generate",
                "end_no_data": END # In a real app, this might go to web search
            }
        )
        
        self.workflow.add_edge("generate", "hallucination_check")
        
        # Conditional Edge after Hallucination Check
        def check_hallucination(state):
            if state["hallucination_status"]:
                return "end_success"
            return "generate_retry"

        self.workflow.add_conditional_edges(
            "hallucination_check",
            check_hallucination,
            {
                "end_success": END,
                "generate_retry": "generate" # Functionally a retry loop
            }
        )
        
        self.app = self.workflow.compile()

    def run(self, question: str, user_id: str):
        inputs = {"question": question, "user_id": user_id}
        return self.app.invoke(inputs)
