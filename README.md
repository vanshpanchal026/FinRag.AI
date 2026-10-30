# FinRAG-Eval

**Zero-Hallucination Financial Report & Concall Q&A Engine for Indian Equities (NSE/BSE)**

FinRAG-Eval is a 100% free, local-first Multi-Agent Retrieval-Augmented Generation (RAG) system engineered specifically for institutional equity research analysts and traders. It ingests complex financial PDFs (Quarterly Financial Results, Earnings Call Transcripts, and Investor Presentations), locks financial tables and speaker turns intact, provides exact inline source citations `[Source: File | Page: X | Chunk: Y]`, and audits every answer via a dual-layer evaluation engine (Deterministic Regex + LLM-as-a-Judge).

> 📖 **Detailed Architecture & Phase Walkthrough:** See [Walkthrough.md](Walkthrough.md) for the complete phase-by-phase implementation narrative, verification matrix, and execution guide.

---

## 🏛️ System Architecture

```
[Flow Stage 1: Document Ingestion Pipeline]
User Uploads PDF (e.g. BSE/NSE Quarterly Report, Concall Transcript)
        │
        ▼
PDF Parser (PyMuPDF4LLM + pdfplumber Fallback) ──→ Converts to Markdown with exact 1-indexed page_no
        │
        ▼
Smart Financial Chunker
   ├── Rule 1 (Table Preservation): Locks Markdown tables (|---|) + up to 2 header/unit lines (<=1500 tokens)
   └── Rule 2 (Speaker-Turn Splitter): Binds [Analyst Question + Management Response] (<=700 tokens)
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
Dense Indexer (ChromaDB)                  Sparse Indexer (BM25Okapi)
BAAI/bge-small-en-v1.5 (CPU)              Financial Tokenizer (data/bm25_index/bm25_store.pkl)

[Flow Stage 2: Hybrid Retrieval, Synthesis & Evaluation Pipeline]
User Query / Trader 1-Click Preset
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
Dense Search (Top 10 Semantic Chunks)     Sparse Search (Top 10 Keyword Chunks)
        │                                         │
        └────────────────────┬────────────────────┘
                             ▼
              Reciprocal Rank Fusion (RRF, k=60)
                             │
                             ▼
         Local CPU Reranker (FlashRank ms-marco-MiniLM)
                             │
                             ▼
               Top 4 Golden Chunks (<50ms latency)
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
[Main Financial Analyst Agent]    [Two-Tier Evaluation Sub-Agent]
(Groq / llama-3.3-70b-versatile)  (Regex Check + llama-3.1-8b-instant)
Generates cited response with     Tier 1: Verbatim numeric match
mandatory inline citations        Tier 2: Faithfulness (0-10), Citations (0-10)
               │                           │
               └─────────────┬─────────────┘
                             ▼
                Final Structured Payload to UI
    (Cited Answer + Live Eval Scorecard + Expandable Raw Sources)
```

---

## 🛠️ Technology Stack (100% Local CPU Execution)

- **Language & Runtime:** Python 3.11+
- **PDF-to-Markdown Parser:** `pymupdf4llm` (v2.0+) with `pdfplumber` (v0.11+) fallback
- **Token Counter & Splitter:** `tiktoken` (`cl100k_base`)
- **Dense Embedding Model:** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (384 dimensions, local CPU)
- **Vector Database:** `ChromaDB` (Persistent local client with cosine distance space)
- **Sparse Keyword Search:** `rank-bm25` (`BM25Okapi` with financial tokenization and noise stopword filtering)
- **Cross-Encoder Reranker:** `FlashRank` (`ms-marco-MiniLM-L-12-v2` ONNX runtime on CPU)
- **LLM Synthesis & Auditor:** Unified client supporting `Groq Cloud` (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) or any custom OpenAI-compatible endpoint
- **API Framework:** `FastAPI` + `Uvicorn`
- **Trader Workbench UI:** `Streamlit`

---

## 📁 Repository Structure

```
FinRAG-Eval/
│
├── docs/                           # Architecture Specifications & Prompts
│   ├── Prd.md                     # Product Requirements Document
│   ├── DesignDocument.md          # Architectural Design & Schemas
│   ├── Techstack.md               # Technology Stack Specifications
│   ├── DetectionLogic.md          # Regex rules, prompts & eval logic
│   ├── environment-setup.md       # Setup instructions
│   └── implementation-plan.md     # Roadmap & completion tracking
│
├── data/
│   ├── raw_pdfs/                  # Raw uploaded PDF documents
│   ├── processed_md/              # Intermediate parsed Markdown files
│   ├── chroma_db/                 # Persistent local ChromaDB vector store
│   └── bm25_index/                # Serialized BM25Okapi index (bm25_store.pkl)
│
├── src/
│   ├── config.py                  # Environment settings, model IDs, paths
│   ├── ingestion/
│   │   ├── pdf_parser.py          # PyMuPDF4LLM + pdfplumber fallback parser
│   │   ├── smart_chunker.py       # Table preservation & concall speaker-turn chunker
│   │   └── indexer.py             # Dual indexing pipeline (ChromaDB + BM25)
│   ├── retrieval/
│   │   ├── hybrid_retriever.py    # Dense + Sparse search with Reciprocal Rank Fusion
│   │   └── reranker.py            # FlashRank CPU cross-encoder reranking
│   ├── agents/
│   │   ├── llm_client.py          # Unified OpenAI/Groq client with backoff retry
│   │   ├── analyst_agent.py       # Main analyst agent enforcing inline citations
│   │   └── eval_subagent.py       # Tier-1 regex check + Tier-2 LLM Judge
│   ├── presets/
│   │   └── trader_workflows.py    # Red Flags, Guidance vs Actuals, Growth Triggers
│   └── api/
│       └── server.py              # FastAPI endpoints (/api/ingest, /api/query, /api/preset)
│
├── ui/
│   └── app.py                     # Streamlit Trader Workbench UI
│
├── tests/
│   ├── test_chunker.py            # Unit tests for parser and smart chunker
│   ├── test_retrieval.py          # Unit tests for Chroma, BM25, RRF, and FlashRank
│   ├── test_agents.py             # Unit tests for Tier-1/2 evals and trader presets
│   └── eval_dataset.json          # Benchmark dataset with 5 Indian equity Q&A pairs
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python**: Version 3.11+
- **uv** (recommended) or standard `venv` + `pip`

### 2. Create Virtual Environment & Install Dependencies

Using `uv`:
```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Or using standard `pip`:
```powershell
python -m venv .venv
.\.venv\Scripts\activate

# Install CPU-only PyTorch first to avoid heavy CUDA downloads and missing DLL issues
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Provide your Groq API key or OpenAI-compatible endpoint in `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## 🚀 Running the System

### Option A: Interactive Streamlit Workbench (Recommended)
Launch the graphical trader workbench:
```powershell
streamlit run ui/app.py
```
**Features in the UI:**
1. **Sidebar Ingestion:** Upload any BSE/NSE filing (PDF), assign Company Ticker and Quarter, and ingest in 1 click.
2. **Top Action Bar:** 1-Click presets for **Scan Concall Red Flags**, **Guidance vs Actuals**, and **Growth Triggers & Capex**.
3. **Live Faithfulness Scorecard:** Visual badges for Deterministic Numeric Match (`PASS`/`FAIL`), Faithfulness (`X/10`), Citation Accuracy (`X/10`), and Relevance (`X/10`).
4. **Verify Raw Sources Accordion:** Expandable viewer displaying all Top 4 Golden Chunks with Chunk IDs, rerank scores, and raw Markdown tables.

### Option B: FastAPI Backend Server
Launch the RESTful API for custom dashboards:
```powershell
uvicorn src.api.server:app --reload --port 8000
```
Access interactive Swagger API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📡 REST API Reference

### 1. Ingest PDF Document
`POST /api/ingest` (multipart/form-data)
- `file`: PDF binary
- `company`: `TATAELXSI`
- `quarter`: `Q2_FY27`
- `doc_type`: `Financial_Results`

### 2. Query Reports
`POST /api/query` (application/json)
```json
{
  "query": "What was the Revenue from Operations and EBITDA margin in Q2 FY27?",
  "company": "TATAELXSI",
  "quarter": "Q2_FY27",
  "top_k": 10,
  "top_n": 4
}
```

### 3. Trigger Trader Preset
`POST /api/preset/{preset_name}` (application/json)
- Presets: `red_flag_scanner`, `guidance_vs_actuals`, `growth_triggers_capex`
```json
{
  "company": "TATAELXSI",
  "quarter": "Q2_FY27"
}
```

---

## 🧪 Running the Test Suite

Run all unit tests across the entire ingestion, retrieval, agent, and evaluation pipeline:
```powershell
pytest -v
```
All 20 tests pass with 100% verification covering:
- Table locking and header/unit line preservation (<=1500 tokens).
- Concall speaker-turn Q&A binding (<=700 tokens).
- Exact financial keyword search via BM25Okapi.
- Metadata isolation by quarter (`quarter="Q2_FY27"`).
- FlashRank sub-second CPU cross-encoder reranking.
- Tier-1 Deterministic Regex fake number interception (`numeric_match_pass = False`).
- Pydantic `EvalReport` schema compliance.
