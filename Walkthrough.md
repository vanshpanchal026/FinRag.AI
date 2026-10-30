# FinRAG-Eval: End-to-End Project Walkthrough & Implementation Guide

**Zero-Hallucination Financial Report & Concall Q&A Engine for Indian Equities (NSE/BSE)**  
*Repository:* [https://github.com/vanshpanchal026/FinRag.AI](https://github.com/vanshpanchal026/FinRag.AI)

---

## 1. Executive Summary & Core Objective

Institutional equity research analysts and traders analyze quarterly earnings reports, investor presentations, and earnings call transcripts (50–150 pages each). Standard RAG pipelines and LLMs routinely fail in this domain because:
1. **Naive Chunking Slices Tables:** Multi-column financial tables get cut mid-row, stripping column headers and causing models to hallucinate numbers or units.
2. **Severed Speaker Turns:** Concall Q&A exchanges get severed, divorcing an analyst's question from management's immediate answer.
3. **No Mathematical Audit:** Traditional RAG lacks verification, allowing numerical hallucinations (e.g., confusing `1,250.5 Cr` with `1,205.5 Cr`) to slip into trader reports undetected.

**FinRAG-Eval** solves this by establishing a **100% local CPU, zero-cost, source-driven pipeline** featuring table-preserving chunking, dual dense-sparse indexing, FlashRank cross-encoder reranking, and a **Two-Tier Evaluation Sub-Agent** combining zero-latency deterministic regex math checks with LLM-as-a-Judge verification.

---

## 2. End-to-End System Architecture

```
[FLOW 1: DOCUMENT INGESTION PIPELINE]
User Uploads Filing PDF (Quarterly Results, Concall, Presentation)
   │
   ▼
[PDF Parser] (src/ingestion/pdf_parser.py)
   ├── Primary: PyMuPDF4LLM extracting page-by-page Markdown with 1-indexed page_no
   └── Secondary Fallback: pdfplumber table reconstruction on broken layouts
   │
   ▼
[Smart Financial Chunker] (src/ingestion/smart_chunker.py)
   ├── Rule 1 (Tables): Detects Markdown tables (|---|), captures 2 preceding unit lines,
   │                     and locks the entire table intact (contains_table=True, <=1500 tokens)
   └── Rule 2 (Concall): Uses lookahead regex to bind [Analyst Question + Management Response]
                         (<=700 tokens) or splits recursively with 80 tokens overlap
   │
   ├──────────────────────────────────────────┐
   ▼                                          ▼
[ChromaDB Local Vector Store]      [BM25Okapi Serialized Index]
Embedding: BAAI/bge-small-en-v1.5  Tokenizer: Normalized financial tokens (roce, capex, 4.4%)
Cosine distance space (CPU)        Serialized store: data/bm25_index/bm25_store.pkl


[FLOW 2: RETRIEVAL, SYNTHESIS & AUDIT PIPELINE]
User Query / 1-Click Trader Preset
   │
   ├──────────────────────────────────────────┐
   ▼                                          ▼
[Dense Semantic Search]            [Sparse Keyword Search]
Top 10 chunks from ChromaDB        Top 10 exact matches from BM25
   │                                          │
   └────────────────────┬─────────────────────┘
                        ▼
    [Reciprocal Rank Fusion (RRF, k=60)]
    Combines dense and sparse rankings; deduplicates by chunk_id
                        │
                        ▼
    [FlashRank Local CPU Cross-Encoder Reranker]
    Model: ms-marco-MiniLM-L-12-v2 (<50ms latency on CPU)
                        │
                        ▼
              Top 4 Golden Chunks
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
[Main Financial Analyst Agent]   [Two-Tier Evaluation Sub-Agent]
(llama-3.3-70b-versatile)        (Regex + llama-3.1-8b-instant)
- Zero extrapolation             - Tier 1: Deterministic regex extracting all numbers,
- Mandatory inline citations:      stripping citation tags, and comparing against raw chunks.
  [Source: File | Page | Chunk]  - Tier 2: Scores Faithfulness (0-10), Citations (0-10),
- Markdown table output            and Relevance (0-10). Flags hallucinations.
         │                             │
         └──────────────┬──────────────┘
                        ▼
      [Unified Payload to Streamlit / FastAPI]
      (Cited Response + 4-Badge Scorecard + Raw Retrieved Sources Accordion)
```

---

## 3. What Was Built (Phase-by-Phase Breakdown)

### Phase 1: Foundation & Table-Aware Ingestion Engine
- **Configuration (`src/config.py`):**
  - Built with `pydantic-settings` to load `.env` variables with strong validation.
  - Configured model identifiers, token parameters (`CHUNK_SIZE_TOKENS=600`, `CHUNK_OVERLAP_TOKENS=80`, `MAX_TABLE_TOKENS=1500`, `MAX_CONCALL_TOKENS=700`), and automatic on-startup directory creation for `data/raw_pdfs`, `data/processed_md`, `data/chroma_db`, and `data/bm25_index`.
- **PDF-to-Markdown Parser (`src/ingestion/pdf_parser.py`):**
  - Implemented `parse_pdf` using `pymupdf4llm.to_markdown(..., page_chunks=True)` maintaining exact 1-indexed `page_no`.
  - Added `fallback_extract_page_with_pdfplumber` and `format_table_to_markdown` to convert raw tabular cells into markdown tables when standard parsing fails.
  - Automatically writes intermediate markdown files to `data/processed_md/` with explicit page boundary dividers for debugging.
- **Smart Financial Chunker (`src/ingestion/smart_chunker.py`):**
  - **Rule 1 (Table Preservation):** Regex scans for `|---|---|`, captures up to 2 preceding unit/title lines (e.g., `(Rs. in Crores)`), and locks the entire table as an atomic chunk (`contains_table: True`) up to 1500 tokens without splitting rows.
  - **Rule 2 (Speaker-Turn Grouping):** Lookahead regex `(?=(?:Moderator|Management|[A-Z][a-z]+ [A-Z][a-z]+)\s*:)` groups `[Analyst Question + Immediate Management Response]` together (up to 700 tokens). Non-concall text splits recursively via `tiktoken` (`cl100k_base`) with 80 tokens overlap.
  - Generates deterministic chunk IDs: `{company}_{quarter}_p{page_no}_c{index:02d}` (e.g., `TATAELXSI_Q2_FY27_p14_c02`).

### Phase 2: Dual Indexing & Hybrid Retrieval Engine
- **Dual Indexer (`src/ingestion/indexer.py`):**
  - **Dense Index:** Uses `BAAI/bge-small-en-v1.5` on local CPU (384 dimensions) with cosine distance space (`{"hnsw:space": "cosine"}`) inside a persistent ChromaDB client. Uses idempotent `.upsert()` to eliminate duplicate chunks.
  - **Sparse Index:** Uses `BM25Okapi` with financial tokenization (`tokenize_for_bm25`) and stopword noise filtering, serialized to `data/bm25_index/bm25_store.pkl`.
- **Hybrid Retriever (`src/retrieval/hybrid_retriever.py`):**
  - Retrieves Top 10 dense chunks and Top 10 sparse keyword chunks with optional metadata filtering (`company`, `quarter`, `doc_type`).
  - Merges rankings via Reciprocal Rank Fusion ($k=60$):
    $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- **Cross-Encoder Reranker (`src/retrieval/reranker.py`):**
  - Leverages FlashRank (`ms-marco-MiniLM-L-12-v2`) on CPU ONNX Runtime. Reranks candidate chunks in ~25ms (<50ms budget) and returns the Top 4 Golden Chunks with `rerank_score` attached.

### Phase 3: Multi-Agent Layer & Two-Tier Evaluation Sub-Agent
- **Unified LLM Client (`src/agents/llm_client.py`):**
  - Unified wrapper for Groq Cloud API and custom OpenAI-compatible endpoints with automatic exponential backoff retry for HTTP 429 and 503 errors.
  - Implements `generate_json()` with Pydantic model validation.
- **Main Financial Analyst Agent (`src/agents/analyst_agent.py`):**
  - Formats Golden Chunks with explicit chunk boundary headers:
    ```
    === CHUNK ID: {chunk_id} | SOURCE: {source_file} | PAGE: {page_no} ===
    {text}
    ```
  - Enforces strict system instructions: zero external extrapolation, mandatory inline `[Source: {source_file} | Page: {page_no} | Chunk: {chunk_id}]` tags after every number or claim, and clean Markdown tables for Q-o-Q / Y-o-Y comparisons.
- **Two-Tier Evaluation Sub-Agent (`src/agents/eval_subagent.py`):**
  - **Tier 1 (Zero-Latency Deterministic Regex Audit):** Strips `[Source: ...]` tags from the generated answer, extracts all numerical figures and percentages via Regex (`r'(?<![a-zA-Z0-9_])\d+(?:,\d+)*(?:\.\d+)?%?(?![a-zA-Z0-9_])'`), normalizes comma formatting, and cross-verifies every number against the raw text of the Top 4 Golden Chunks. Outputs `numeric_match_pass` (`True`/`False`) and lists any `unverified_numbers`.
  - **Tier 2 (LLM-as-a-Judge Verification):** Passes the query, Golden Chunks, generated response, and Tier-1 numeric report to `llama-3.1-8b-instant` to output a strict Pydantic-validated JSON scorecard (`faithfulness_score`, `citation_accuracy_score`, `answer_relevance_score`, `hallucination_flag`, and `audit_reasoning`).
- **1-Click Trader Preset Workflows (`src/presets/trader_workflows.py`):**
  - **Concall Red-Flag Scanner:** Injects risk/headwind booster keywords into BM25 (`margin pressure`, `delay`, `working capital stretch`, `subdued`, `headwinds`) and generates a categorized severity table with exact citations.
  - **Guidance vs. Actuals Tracker:** Cross-filters `q_prev` outlook chunks against `q_curr` actual result chunks to output a Beat / Met / Missed comparison table.
  - **Growth Triggers & Capex Extractor:** Summarizes order book visibility, capacity expansion, and new commissioning timelines with page citations.

### Phase 4: FastAPI Server, Streamlit Trader Workbench & Benchmark Suite
- **FastAPI Backend Server (`src/api/server.py`):**
  - Exposes asynchronous REST endpoints (`/api/ingest`, `/api/query`, `/api/preset/{preset_name}`, `/api/health`) ready to plug into a custom React dashboard or external algorithmic trading workflow.
- **Streamlit Trader Workbench (`ui/app.py`):**
  - **Sidebar:** PDF filing uploader with metadata inputs (Company Ticker, Quarter, Doc Type) and real-time step progress (`Pages Parsed -> Tables Detected -> Chunks Indexed`), plus active query metadata filters.
  - **Main Panel:** Interactive Q&A chat and 3 One-Click Trader Preset buttons.
  - **Live 4-Badge Eval Scorecard:** Displays Numeric Match (`PASS`/`FAIL`), Faithfulness (`X/10`), Citation Accuracy (`X/10`), and Answer Relevance (`X/10`) right below every answer.
  - **Verify Raw Retrieved Sources Accordion:** Expandable `st.expander` showing the exact Top 4 Golden Chunks, their FlashRank confidence scores, page numbers, `📊 TABLE INTACT` badge, and raw Markdown tables.
- **Automated Test & Benchmark Suite (`tests/`):**
  - Complete unit and integration tests (`test_chunker.py`, `test_retrieval.py`, `test_agents.py`) alongside `eval_dataset.json` containing 5 realistic Indian equity benchmark test cases for regression testing.

---

## 4. Quickstart & Execution Commands

### 1. Environment Setup (Windows 11 CPU-Optimized)

```powershell
git clone https://github.com/vanshpanchal026/FinRag.AI.git
cd FinRag.AI
python -m venv .venv
.\.venv\Scripts\activate

# Install CPU-only PyTorch first to avoid heavy CUDA downloads and missing DLL issues
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and configure your credentials:
```powershell
Copy-Item .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY="gsk_your_api_key_here"
MAIN_AGENT_MODEL="llama-3.3-70b-versatile"
EVAL_AGENT_MODEL="llama-3.1-8b-instant"
```

### 3. Run Verification Tests

Run the full 20-test test suite across all 4 phases:
```powershell
pytest tests/ -v
```

### 4. Launch the Application

Launch the Streamlit Trader Workbench:
```powershell
streamlit run ui/app.py
```

Or launch the FastAPI Backend Server (Port 8000):
```powershell
uvicorn src.api.server:app --reload --port 8000
```
Interactive API docs available at: `http://localhost:8000/docs`

---

## 5. Test Suite & Verification Matrix

Running `pytest -v` across the entire codebase yields **20 out of 20 unit tests passing (100% pass rate)**:

| Test File | Test Case | Functionality Verified | Status |
|:---|:---|:---|:---:|
| `test_chunker.py` | `test_table_preservation_and_header_capture` | Multi-row P&L table stays in 1 chunk (`contains_table: True`); unit lines captured | **PASSED** |
| `test_chunker.py` | `test_concall_speaker_turn_grouping` | Concall Analyst Question + Management Answer bound together (<=700 tokens) | **PASSED** |
| `test_chunker.py` | `test_chunk_metadata_schema_conformance` | Schema keys (`company`, `doc_type`, `quarter`, `page_no`, `chunk_id`, etc.) | **PASSED** |
| `test_chunker.py` | `test_recursive_commentary_splitting_with_overlap` | Commentary split at 600 tokens with 80 tokens semantic overlap | **PASSED** |
| `test_chunker.py` | `test_multi_table_page_segmentation` | Multiple tables on 1 page isolated into separate atomic chunks | **PASSED** |
| `test_chunker.py` | `test_empty_and_whitespace_page` | Whitespace handling without crashes | **PASSED** |
| `test_chunker.py` | `test_chunk_document_multi_page` | Document-wide aggregation and 1-indexed page numbering | **PASSED** |
| `test_chunker.py` | `test_pdfplumber_fallback_formatting` | Fallback 2D table converter and broken format detector | **PASSED** |
| `test_chunker.py` | `test_parse_pdf_end_to_end` | Synthetic PDF creation, page extraction, and debug markdown export | **PASSED** |
| `test_retrieval.py` | `test_exact_financial_keyword_retrieval` | Exact terms (`ROCE 36.5%`, `Capex 200 Cr`) hit BM25 + Dense | **PASSED** |
| `test_retrieval.py` | `test_metadata_filtering_isolates_quarter` | Pre-filter `quarter="Q2_FY27"` strictly isolates target chunks | **PASSED** |
| `test_retrieval.py` | `test_flashrank_reranking_speed_and_order` | FlashRank CPU reranking in ~25ms (<1.0s budget) sorted by score | **PASSED** |
| `test_retrieval.py` | `test_reciprocal_rank_fusion_scoring` | RRF properly scores dual dense/sparse hits higher | **PASSED** |
| `test_retrieval.py` | `test_duplicate_upsert_idempotency` | Re-indexing same chunks prevents duplicates | **PASSED** |
| `test_agents.py` | `test_tier1_deterministic_numeric_pass` | Grounded answer with true figures passes Tier-1 check | **PASSED** |
| `test_agents.py` | `test_tier1_injected_fake_number_caught` | Injected fake number (`999.99 Cr`) immediately caught (`numeric_match_pass=False`) | **PASSED** |
| `test_agents.py` | `test_citation_exclusion_filter_prevents_false_positives` | Stripping `[Source: ...]` prevents page/chunk IDs from triggering false alarms | **PASSED** |
| `test_agents.py` | `test_analyst_agent_golden_chunk_formatting` | Explicit Golden Chunk boundary headers | **PASSED** |
| `test_agents.py` | `test_eval_subagent_schema_conformance` | Output strictly matches DesignDocument Section 3B schema | **PASSED** |
| `test_agents.py` | `test_trader_presets_mock_execution` | Red Flags, Guidance vs Actuals, and Growth Triggers workflows | **PASSED** |

---

## 6. Security & GitHub Release

- **Secret Scanning:** All codebase files were checked against API tokens, credentials, and private keys prior to commit. Result: **0 secrets found**.
- **Clean Git Tracking:** Git repository configured on branch `main` with `.gitignore` strictly omitting `.env`, `data/`, virtual environments, and compiler caches.
- **Upstream Repository:** Pushed to [https://github.com/vanshpanchal026/FinRag.AI.git](https://github.com/vanshpanchal026/FinRag.AI.git).
