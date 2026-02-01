from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from src.graph.state import GraphState

class HallucinationGrade(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class AnswerGrade(BaseModel):
    """Binary score to check if the question is answered."""
    binary_score: str = Field(description="Answer resolves the question, 'yes' or 'no'")

class HallucinationNode:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    def __call__(self, state: GraphState) -> GraphState:
        print("---CHECK HALLUCINATIONS---")
        documents = state["documents"]
        generation = state["generation"]
        question = state["question"]

        structured_llm_hallucination = self.llm.with_structured_output(HallucinationGrade)
        
        # 1. Check Hallucinations (Groundedness)
        system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'yes' means that the answer is completely supported by the facts."""
        
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_hallucination),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        
        hallucination_grader = hallucination_prompt | structured_llm_hallucination
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # 2. Check if it answers the question
            structured_llm_answer = self.llm.with_structured_output(AnswerGrade)
            system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question."""
            
            answer_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_answer),
                    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
                ]
            )
            answer_grader = answer_prompt | structured_llm_answer
            score = answer_grader.invoke({"question": question, "generation": generation})
            if score.binary_score == "yes":
                 print("---DECISION: GENERATION ADDRESSES QUESTION---")
                 return {"hallucination_status": True}
            else:
                 print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                 return {"hallucination_status": False, "generation": "The answer was not addressed by the generation."} # Could trigger retry
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
            return {"hallucination_status": False}
