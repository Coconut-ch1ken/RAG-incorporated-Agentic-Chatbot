# Critical Issue: Python Version Incompatibility

**Problem**: You are running **Python 3.14.2**.
**Cause**: The core AI libraries (`chromadb`, `onnxruntime`, `numpy`) do not yet support Python 3.14. This causes the "ResolutionImpossible" or "No matching distribution" errors.

## Solution

You **must** use Python 3.10, 3.11, or 3.12.

### Steps to Fix

1.  **Delete the current venv**:
    ```bash
    rm -rf .venv
    ```

2.  **Create a new venv with a supported Python version**:
    *(Assuming you have python3.11 or python3.12 installed)*
    ```bash
    python3.11 -m venv .venv
    # OR
    python3.12 -m venv .venv
    ```

3.  **Activate and Install**:
    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Run**:
    ```bash
    python main.py
    ```
