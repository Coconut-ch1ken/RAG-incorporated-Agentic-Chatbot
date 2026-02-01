# RAG Incorporated Agentic Chatbot

A production-grade, hallucination-proof RAG (Retrieval-Augmented Generation) chatbot built with **LangGraph**, **ChromaDB**, and **Python**.

## Features

- **Agentic Workflow**: Uses `LangGraph` to orchestrate a "Corrective RAG" flow.
- **Hallucination Proof**: Implements a dedicated "Hallucination Check" node that grades generated answers against retrieved documents.
- **Modules**: `chromadb` (requires Python < 3.13), `langchain`, `sqlite3`

> [!WARNING]
> This project requires **Python 3.10 - 3.12**.
> Python 3.13+ (and 3.14) are **NOT** supported due to dependency limitations in `onnxruntime`.
- **CSV Ingestion**: Intelligent "Row-to-Context" serialization preserves the semantic meaning of tabular data.
- **Modular Architecture**: Clean separation of concerns between database, ingestion, and agent logic.

## Architecture

The system is built on a graph-based state machine:
1.  **Retrieve**: Fetches relevant documents from ChromaDB.
2.  **Grade**: Filters out irrelevant documents using an LLM.
3.  **Generate**: Synthesizes an answer using the filtered context.
4.  **Verify**: Checks if the answer is grounded in the documents and addresses the user's question.

## Project Structure

```
RAG-incorporated-Agentic-Chatbot/
├── main.py              # CLI Entry Point
├── data/                # Data storage (CSVs)
├── src/
│   ├── database/        # Vector Store (Chroma) & Metadata Store (SQLite)
│   ├── ingestion/       # CSV processing logic
│   └── graph/           # LangGraph nodes and workflow definition
├── requirements.txt     # Python dependencies
└── .env                 # Configuration
```

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your-openai-api-key
    ```

3.  **Run the Application**:
    ```bash
    python main.py
    ```

4.  **Interact**:
    - **Ingest Data**: Type `/ingest data/sample.csv` to load the provided sample file.
    - **Chat**: Ask questions like "Who is working on Project Apollo?"

## Technologies

- **LangChain / LangGraph**: Orchestration
- **ChromaDB**: Vector Database
- **SQLite**: Metadata Management
- **Pandas**: Data Manipulation
- **OpenAI**: LLM & Embeddings
