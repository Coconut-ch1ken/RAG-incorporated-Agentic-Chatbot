# Agent Guidelines for RAG-Agentic-Chatbot

This document provides instructions for AI agents (and human developers) working on the `rag-agentic-chatbot` repository.

## Project Overview
This project is a hallucination-proof RAG chatbot utilizing **LangChain**, **LangGraph**, and **Google Gemini Pro**. It ingests CSV data, stores it in **ChromaDB**, and processes queries through a graph-based workflow.

## 1. Build, Run, and Test Commands

The project uses **Poetry** for dependency management.

### Setup
Ensure you have Poetry installed.
```bash
poetry install
```

### Running the Application
The entry point is `main.py` in the root directory.
```bash
poetry run python main.py
```
*Note: Ensure `.env` is configured with `GOOGLE_API_KEY` before running.*

### Testing
There are currently no existing tests. Agents are expected to **add tests** for any new functionality using `pytest`.

*Recommended workflow for adding/running tests:*
1.  Add `pytest` to dev dependencies if missing:
    ```bash
    poetry add --group dev pytest
    ```
2.  Run all tests:
    ```bash
    poetry run pytest
    ```
3.  Run a single test file:
    ```bash
    poetry run pytest tests/path/to/test_file.py
    ```
4.  Run a specific test case:
    ```bash
    poetry run pytest tests/path/to/test_file.py::test_function_name
    ```

### Linting & Formatting
The project follows standard Python conventions. Agents should use `ruff` for linting/formatting if available, or adhere to standard PEP 8.

*Recommended commands:*
```bash
# Add ruff if missing
poetry add --group dev ruff

# Run linter
poetry run ruff check .

# Format code
poetry run ruff format .
```

## 2. Code Style & Conventions

### General
-   **Language**: Python 3.10+
-   **Indentation**: 4 spaces.
-   **Line Length**: 88 characters (standard Black/Ruff compatible).

### Imports
-   Use **absolute imports** starting from `src`.
    -   *Good*: `from src.graph.state import GraphState`
    -   *Bad*: `from ..graph.state import GraphState`
-   Group imports: Standard library, Third-party (LangChain, etc.), Local (`src...`).

### Type Hinting
-   **Mandatory** for function signatures, especially in `src/graph` nodes and strict interfaces.
-   Use `typing` module or standard types (e.g., `List`, `Dict`, `Optional`).

```python
# Example
def retrieve(self, state: GraphState) -> dict:
    ...
```

### Naming Conventions
-   **Variables/Functions**: `snake_case` (e.g., `user_input`, `retrieve_documents`).
-   **Classes**: `PascalCase` (e.g., `RagAgent`, `VectorStore`).
-   **Constants**: `UPPER_CASE` (e.g., `USER_ID`, `EMBEDDING_MODEL`).
-   **Files**: `snake_case.py`.

### Error Handling
-   Use explicit `try/except` blocks where external failures (API calls, I/O) are possible.
-   Log errors gracefully (currently using `print` in `main.py`, but consider `logging` for deeper modules).

### Architecture Structure
-   `src/database/`: Persistence layers (ChromaDB, Metadata).
-   `src/ingestion/`: Data loaders (CSV, etc.).
-   `src/graph/`: LangGraph definitions (Nodes, State, Workflow).
    -   Nodes should implement `__call__` or be callable.
    -   State is typed via `TypedDict` (see `src/graph/state.py`).

## 3. Agentic Workflow Rules

1.  **Exploration**: Before making changes, explore existing nodes in `src/graph/nodes/` to understand the state flow.
2.  **Dependencies**: If adding a new library, use `poetry add <library>`. Do not manually edit `pyproject.toml` or `requirements.txt` unless necessary.
3.  **No Hallucinations**: When modifying the RAG logic, ensure the "hallucination grader" or verification steps in the graph are preserved or enhanced.
4.  **Verification**: After generating code, **always** attempt to run it. If writing a new node, create a simple test script to verify its input/output transformation.

## 4. Cursor/Copilot Rules
*No specific .cursorrules or .github/copilot-instructions.md found. Adhere to the guidelines above.*
