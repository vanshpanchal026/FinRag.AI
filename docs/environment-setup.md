# Environment Setup Guide

## FinRAG-Eval Local Environment

This guide details how to set up the Python development environment for the **FinRAG-Eval** project on Windows 11 / Linux / macOS.

### Prerequisites
- **Python**: Version 3.11+ (recommended CPython 3.11)
- **Package Manager**: `uv` (recommended for ultra-fast installs) or standard `venv` + `pip`
- **Git**

---

### Step 1: Create and Activate Virtual Environment

Using `uv` (Recommended):
```powershell
# Create Python 3.11 virtual environment
uv venv --python 3.11

# Activate virtual environment
.venv\Scripts\activate      # Windows PowerShell
# or: source .venv/bin/activate  # Linux/macOS
```

Or using standard Python:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

### Step 2: Install Dependencies

```powershell
# Using uv:
uv pip install -r requirements.txt

# Or using pip:
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Provide your `GROQ_API_KEY` or custom OpenAI-compatible endpoint credentials in `.env`.

---

### Step 4: Run Tests

Verify Phase 1 ingestion and smart chunker:
```powershell
pytest tests/test_chunker.py -v
```
