PROJECT TITLE: FinRAG-Eval (Zero-Hallucination Financial Report & Concall Q&A Engine)
DOCUMENT: Technology Stack & Resource Specifications
VERSION: 1.0.0
TARGET HARDWARE: Windows 11 PC | AMD Ryzen 5 5600GT | 16GB DDR4 RAM (CPU-First Execution)
COST CONSTRAINT: 100% Free & Open-Source / Free-Tier APIs Only 

1. CORE PROGRAMMING & RUNTIME

* Language: Python 3.11+
* Package & Virtual Env Manager: uv (ultra-fast pip replacement) or standard venv
* API Framework: FastAPI + Uvicorn (Asynchronous REST endpoints for ingestion and querying)

1. DOCUMENT PARSING & SMART CHUNKING LAYER

* Primary PDF-to-Markdown Parser: pymupdf4llm (v0.0.17+)
Rationale: Blazingly fast on CPU, extracts clean Markdown tables without requiring heavy GPU OCR models, and preserves exact page numbers.
* Secondary Table Parser (Fallback for Complex Multi-Column PDFs): pdfplumber (v0.11+)
Rationale: Deterministic bounding-box table extraction if standard Markdown conversion misses nested balance sheet rows.
* Token Counter & Splitter: tiktoken (cl100k_base) + Custom Regex Table/Speaker-Turn Chunker.

1. EMBEDDING & HYBRID RETRIEVAL LAYER (100% LOCAL CPU)

* Dense Embedding Model: BAAI/bge-small-en-v1.5 (via sentence-transformers / HuggingFace)
Specs: 384 dimensions, ~130MB RAM footprint, runs at ~150 chunks/sec on Ryzen 5 5600GT CPU.
* Local Vector Database: ChromaDB (PersistentClient mode)
Rationale: Serverless, stores embeddings + rich JSON metadata locally on disk in SQLite/Parquet format, zero cloud cost, supports pre-filtering by company/quarter.
* Sparse Keyword Retriever: rank-bm25 (BM25Okapi algorithm)
Rationale: Crucial for exact financial ticker, accounting metric (EBITDA, ROCE, PAT, Working Capital), and clause matching where dense embeddings struggle.
* Reranking Engine (Cross-Encoder): FlashRank (Model: ms-marco-MiniLM-L-12-v2)
Rationale: Runs strictly on ONNX runtime CPU with <200ms latency for 20 chunks; zero PyTorch GPU overhead.

1. MULTI-AGENT LLM INFERENCE LAYER

* Primary Inference Provider: Groq Cloud API (and/or Custom OpenAI-Compatible Endpoint)
* Unified SDK Client: openai Python SDK (configured with custom base_url for Groq/OpenAI-compatible providers)
* Main Analyst Agent Model: llama-3.3-70b-versatile (via Groq)
Role: Complex financial synthesis, table interpretation, Q-o-Q comparison, and strict inline citation formatting.
* Eval & Verification Sub-Agent Model: llama-3.1-8b-instant (via Groq)
Role: Ultra-low latency LLM-as-a-Judge grading (Faithfulness, Citation Accuracy, Completeness) and structured JSON output generation.

1. EVALUATION & GUARDRAILS LAYER

* Tier-1 Deterministic Evals: Python built-in re (Regular Expressions) + rapidfuzz
Role: Zero-latency extraction of numbers, percentages, and currency figures from LLM answers to cross-verify against raw retrieved chunks.
* Tier-2 Semantic Evals: Custom Lightweight Eval Sub-Agent (Structured JSON Schema validation via Pydantic v2).
Rationale: Avoids the heavy token bloat and rate-limit bottlenecks of full RAGAS pipelines while keeping exact RAG Triad visibility.

1. FRONTEND & WORKFLOW INTEGRATION

* Rapid Prototyping UI: Streamlit (v1.40+)
Features: Split-screen layout (Chat + Citation Viewer on Left, Live Eval Scorecard & Raw Markdown Chunks on Right).
* Production UI Integration (Optional v1.1): React + TailwindCSS Dashboard connecting to FastAPI backend via REST.
* Codebase Context Bundler: npx repomix (For packaging the repository into an AI-readable XML context file during iterative development).