SYSTEM DESIGN DOCUMENT: FinRAG-Eval Architecture
VERSION: 1.0.0
PATTERN: Source-Driven Modular RAG + Multi-Agent Evaluation 

1. HIGH-LEVEL ARCHITECTURE & DATA FLOW

[Flow Stage 1: Ingestion Pipeline (Offline / On-Upload)]
User Uploads PDF (e.g., TATAELXSI_Q2FY27_Concall.pdf)
|
v
PDF Parser (PyMuPDF4LLM / Docling) ---> Converts PDF pages to Markdown (preserving tables & page numbers)
|
v
Smart Financial Chunker
|-- Rule 1 (Table Detector): Keeps Markdown tables intact with preceding header.
|-- Rule 2 (Text/Concall Splitter): Recursive split (600 tokens, 80 overlap) respecting speaker turns.
|-- Metadata Injector: Attaches {company, quarter, doc_type, page_no, chunk_id, file_name}.
|
+---> [Local Embedding Model: BAAI/bge-small-en-v1.5] ---> Stores Vectors + Metadata in ChromaDB (Persistent Local)
|
+---> [Tokenized Corpus Builder] -----------------------> Saves Sparse Index in Local BM25 Pickled Store 

[Flow Stage 2: Retrieval, Synthesis & Evaluation Pipeline (Online / Per Query)]
User Query (e.g., "What is the order book status and margin guidance for H2?")
|
v
Query Router / Filter Extractor (Extracts metadata filters if mentioned, e.g., quarter="Q2")
|
+---> Dense Search (ChromaDB) ---------> Top 10 Semantic Chunks 

|                                                |
+---> Sparse Search (BM25 Keyword) ----> Top 10 Keyword Chunks
|
v
Reciprocal Rank Fusion (RRF) & Deduplication
|
v
Local CPU Reranker (FlashRank - ms-marco-MiniLM)
|
v
Top 4 Golden Chunks (with full metadata)
|
+----------------------------+----------------------------+ 

               |                                                         |
               v                                                         v
[MAIN AGENT: Financial Analyst]                           [SUB-AGENT: Eval & Audit Engine]
(Groq: llama-3.3-70b-versatile)                           (Regex Check + Fast LLM Judge)
Generates answer with strict                              1. Extracts all numbers from Answer
[Source: File, Page X] tags.                              2. Verifies numbers exist in Golden Chunks
               |                                          3. Scores: Faithfulness, Citation, Relevance
               +----------------------------+----------------------------+
                                            |
                                            v
                            Final JSON Payload to UI / Dashboard
                            (Answer + Clickable Citations + Eval Scorecard + Raw Chunks)

1. FOLDER & REPOSITORY STRUCTURE

finrag-eval/
│
├── docs/                           # Source-Driven Development Docs
│   ├── Prd.md
│   ├── DesignDocument.md
│   ├── Techstack.md
│   ├── DetectionLogic.md           # Prompts & Regex rules for Evals & Red Flags
│   ├── environment-setup.md
│   └── implementation-plan.md
│
├── data/
│   ├── raw_pdfs/                   # Uploaded BSE/NSE reports & concalls
│   ├── processed_md/               # Intermediate Markdown files for debugging
│   ├── chroma_db/                  # Persistent local vector storage
│   └── bm25_index/                 # Serialized BM25 sparse indices
│
├── src/
│   ├── **init**.py
│   ├── config.py                   # API keys, chunk sizes, model names, paths
│   │
│   ├── ingestion/
│   │   ├── pdf_parser.py           # PyMuPDF4LLM wrapper with page boundary tracking
│   │   ├── smart_chunker.py        # Table-aware & speaker-turn chunking logic
│   │   └── indexer.py              # ChromaDB + BM25 dual indexing pipeline
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # Combines ChromaDB + BM25 using RRF
│   │   └── reranker.py             # FlashRank local CPU reranking
│   │
│   ├── agents/
│   │   ├── llm_client.py           # Unified wrapper for Groq / OpenAI-compatible API
│   │   ├── analyst_agent.py        # Main RAG synthesis agent with citation enforcement
│   │   └── eval_subagent.py        # Deterministic Regex number checker + LLM-as-a-Judge
│   │
│   ├── presets/
│   │   └── trader_workflows.py     # Pre-built prompts (Red Flags, Guidance vs Actuals)
│   │
│   └── api/
│       └── server.py               # FastAPI backend endpoints (/ingest, /query, /eval-history)
│
├── ui/                             # Frontend (Streamlit for v1 / React Dashboard integration)
│   └── app.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── eval_dataset.json           # Golden Q&A benchmark pairs for testing pipeline changes
│
├── .env.example
├── requirements.txt
└── README.md 

1. DATA SCHEMAS & INTERFACES

A. Chunk Object Schema (Stored in Vector DB & BM25):
{
"chunk_id": "TATAELXSI_Q2FY27_p14_c02",
"text": "### Segment Revenue\n| Segment | Q2 FY27 | Q1 FY27 | QoQ % |\n|---|---|---|---|\n| Transportation | 512 Cr | 490 Cr | +4.4% |",
"metadata": {
"company": "TATAELXSI",
"doc_type": "Investor_Presentation",
"quarter": "Q2_FY27",
"page_no": 14,
"contains_table": true,
"source_file": "TataElxsi_Q2_Presentation.pdf"
}
} 

B. Eval Sub-Agent Output Schema:
{
"query_id": "uuid-1234",
"deterministic_checks": {
"numbers_in_answer": ["512", "490", "4.4%"],
"unverified_numbers": [],
"numeric_match_pass": true
},
"llm_judge_scores": {
"faithfulness_score": 10,
"citation_accuracy_score": 10,
"answer_relevance_score": 9,
"reasoning": "All figures (512 Cr, 4.4% QoQ) match Chunk TATAELXSI_Q2FY27_p14_c02 on Page 14."
},
"hallucination_flag": false
} 

1. CORE ALGORITHM DETAILS

4.1 Table-Preserving Chunking Logic (smart_chunker.py): 

1. Parse PDF page-by-page using PyMuPDF4LLM.
2. Regex scan each page for Markdown table blocks: (|.*|\n|[-:\s|]+|\n(?:|.*|\n?)+)
3. If a table is found: 

  * Extract the preceding 2 lines (usually the table title/unit like "Rs. in Crores").
  * Bind [Header + Full Table] as a standalone chunk regardless of token length (up to 1500 tokens).
4. For remaining non-table text: 

  * Split by paragraphs/speaker names ("Moderator:", "Management:") with max_tokens=600 and overlap=80.

4.2 Two-Tier Evaluation Logic (eval_subagent.py):
Tier 1 (Zero-Latency Deterministic Check): 

* Extract all numerical tokens (integers, decimals, percentages) from the Main Agent's response using Regex: r'\b\d+(?:.\d+)?%?\b'
* Check if each extracted number exists verbatim inside the concatenated text of the Top 4 Golden Chunks.
* If any number is missing from the source chunks (unless explicitly marked as calculated by the agent), flag numeric_match_pass = False.

Tier 2 (LLM-as-a-Judge Verification): 

* Pass {User_Query, Golden_Chunks, Generated_Answer, Tier1_Report} to the Fast Eval Model (e.g., llama-3.1-8b-instant).
* System Prompt forces structured JSON output grading Faithfulness, Citation Accuracy, and Completeness.