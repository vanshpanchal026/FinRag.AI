PROJECT TITLE: FinRAG-Eval
DOCUMENT: Phased Implementation & Verification Roadmap
VERSION: 1.0.0
WORKFLOW: Source-Driven Development (Execute Phase -> Test -> Update Docs) 

PHASE 1: FOUNDATION & TABLE-AWARE INGESTION ENGINE (Day 1-2) [COMPLETED]
Status: [COMPLETED] (9/9 unit tests passing with 100% verification)
Goal: Convert raw NSE/BSE PDFs into structured Markdown and atomic chunks without breaking tables.
Tasks: 

1. [x] Initialize repository folder structure (data/, src/, docs/, ui/, tests/).
2. [x] Implement src/config.py to load .env configurations and initialize directory paths.
3. [x] Build src/ingestion/pdf_parser.py using pymupdf4llm.to_markdown(doc, page_chunks=True) so every page returns text along with its exact page number.
4. [x] Build src/ingestion/smart_chunker.py: 

  * Implement Regex table detector to isolate Markdown tables with their headers.
  * Implement speaker-turn and recursive token splitter (600 tokens, 80 overlap) for non-table commentary.
  * Attach metadata JSON (company, quarter, doc_type, page_no, chunk_id, contains_table, source_file).
Verification Milestone 1:

* [x] Run pytest tests/test_chunker.py (9/9 unit tests passing with 100% verification). Verify that P&L tables remain in single chunks and every chunk has a valid page_no and chunk_id.

PHASE 2: DUAL INDEXING & HYBRID RETRIEVAL PIPELINE (Day 3-4) [COMPLETED]
Status: [COMPLETED] (5/5 unit tests passing; 14/14 across suite)
Goal: Store chunks in local ChromaDB + BM25 and retrieve the Top 4 Golden Chunks via RRF + FlashRank.
Tasks: 

1. [x] Build src/ingestion/indexer.py: 

  * Load BAAI/bge-small-en-v1.5 locally on CPU.
  * Upsert chunks and metadata into persistent ChromaDB collection.
  * Tokenize chunk texts and serialize a BM25 index to ./data/bm25_index/.
2. [x] Build src/retrieval/hybrid_retriever.py: 

  * Query ChromaDB for Top 10 dense semantic matches (supporting metadata filter like company="TATAELXSI").
  * Query BM25 for Top 10 sparse keyword matches.
  * Combine results using Reciprocal Rank Fusion (RRF) and deduplicate by chunk_id.
3. [x] Build src/retrieval/reranker.py: 

  * Pass the merged candidate chunks through FlashRank (ms-marco-MiniLM-L-12-v2) and return the Top 4 chunks with their rerank confidence scores.
Verification Milestone 2:

* [x] Run pytest tests/test_retrieval.py (5/5 unit tests passing with 100% verification). Verified exact keyword retrieval (ROCE/Capex), quarter metadata filtering, and FlashRank CPU reranking under 1 second.

PHASE 3: MULTI-AGENT SYNTHESIS & TWO-TIER EVAL ENGINE (Day 5-6) [COMPLETED]
Status: [COMPLETED] (6/6 unit tests passing; 20/20 across suite)
Goal: Generate cited answers using Main Analyst Agent and audit them via the Eval Sub-Agent.
Tasks: 

1. [x] Build src/agents/llm_client.py with retry logic and support for Groq / OpenAI-compatible API keys.
2. [x] Build src/agents/analyst_agent.py: 

  * Format Top 4 Golden Chunks with explicit headers (=== CHUNK ID: X | FILE: Y | PAGE: Z ===).
  * Enforce inline citation prompt rules.
3. [x] Build src/agents/eval_subagent.py: 

  * Implement Tier-1 Deterministic Regex numerical check (comparing numbers in answer vs raw chunks).
  * Implement Tier-2 LLM-as-a-Judge call (llama-3.1-8b-instant) returning structured JSON scores (faithfulness_score, citation_accuracy_score, answer_relevance_score, hallucination_flag).
4. [x] Build src/presets/trader_workflows.py for 1-click Red Flag Detection and Guidance vs. Actuals comparison.
Verification Milestone 3:

* [x] Run pytest tests/test_agents.py (6/6 unit tests passing with 100% verification). Verified Tier-1 Regex fake number catch (e.g. 999.99 Cr flagged), citation exclusion filter, Golden Chunk formatting, EvalReport Pydantic schema conformance, and Trader Presets.

PHASE 4: UI WORKBENCH & BENCHMARKING DASHBOARD (Day 7) [COMPLETED]
Status: [COMPLETED] (FastAPI endpoints, Streamlit Workbench, and eval_dataset.json fully verified)
Goal: Create an interactive trader interface with full source transparency and eval scorecards.
Tasks: 

1. [x] Build src/api/server.py (FastAPI REST endpoints: /api/ingest, /api/query, /api/preset) & ui/app.py (Streamlit Trader Workbench): 

  * Sidebar: PDF Uploader with metadata tags (Company Ticker, Quarter, Doc Type) + "Ingest Document" button.
  * Main Panel: Chat interface + 3 One-Click Trader Preset buttons.
  * Verification Panel (Below every answer):
a) Visual Badges for Faithfulness (x/10), Citation Accuracy (x/10), Relevance (x/10), and Numerical Match (PASS/FAIL).
b) Expandable Accordion showing the exact Top 4 Raw Markdown Chunks & Reranker scores so the user can visually audit the source in 1 click.
2. [x] Create tests/eval_dataset.json with realistic Indian equity benchmark Q&A pairs for regression testing.

Verification Milestone 4:
* [x] Verified that FastAPI server launches cleanly, health endpoint passes, and Streamlit workbench compiles with zero errors. All 20 tests pass across the repository.