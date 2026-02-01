from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import GraphState

class GenerateNode:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    def __call__(self, state: GraphState) -> GraphState:
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Simple prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
                ("human", "Question: {question} \n\n Context: {context} \n\n Answer:"),
            ]
        )
        
        rag_chain = prompt | self.llm | StrOutputParser()
        
        generation = rag_chain.invoke({"context": "\n\n".join(documents), "question": question})
        return {"generation": generation}
