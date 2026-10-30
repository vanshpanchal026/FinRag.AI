PROJECT TITLE: FinRAG-Eval (Zero-Hallucination Financial Report & Concall Q&A Engine)
VERSION: 1.0.0
AUTHOR: Research & Trading Workflow
STATUS: Architecture & Prototyping Phase 

1. EXECUTIVE SUMMARY & OBJECTIVE
Traders aur investors ko har quarter dozens of companies ki Annual Reports, Investor Presentations, aur Concall Transcripts (50-150 pages each) analyze karni padti hain. Normal LLMs financial numbers hallucinate karte hain aur tables ko tod dete hain.
Objective: Ek aisa 100% free, local-first Multi-Agent RAG system banana jo financial PDFs ko ingest kare, tables aur speaker conversations ko intact rakhe, har jawab ke saath exact source citation (File, Page, Section) de, aur ek built-in Evaluation Sub-Agent ke zariye answer ki faithfulness (zero-hallucination) ko score kare.
2. TARGET USER & CORE USE CASES
Target User: Equity Trader / Investor (Indian Stock Market focus - NSE/BSE companies).
Core Use Cases:

* Multi-Report Q&A: Company ki Q1 aur Q2 reports upload karke cross-document sawal poochna (e.g., "EBITDA margin Q1 ke mukable Q2 mein kyu gira?").
* Concall Tone & Red Flag Detection: Management commentary mein hidden risks, delays, ya margin pressures ko exact page citations ke saath nikalna.
* Guidance vs. Reality Check: Pichle quarter ki management guidance ko current quarter ke actual delivered numbers se compare karna.
* Source Audit: Answer ke har claim aur number ko raw retrieved chunks se verify karna.

1. FUNCTIONAL REQUIREMENTS (FR)
FR1 - Document Ingestion & Table Preservation:

* System PDF files (Quarterly Results, Concall Transcripts, Investor Presentations) accept karega.
* PDF ke andar maujood financial tables (P&L, Balance Sheet, Segment Revenue) ko bina structure kharab kiye Markdown format mein extract karega.

FR2 - Context-Aware Smart Chunking: 

* Tables ko beech se split nahi kiya jayega; poori table + uska header ek single chunk banega.
* Concall transcripts mein Analyst Question + Management Answer ko ek logical chunk mein bind kiya jayega.
* Har chunk ke saath strict metadata attach hoga: company_ticker, doc_type, quarter, page_number, chunk_id, aur source_file.

FR3 - Hybrid Retrieval & Reranking: 

* System exact financial keywords (jaise ROCE, Capex, Order Book, Working Capital) ke liye Sparse Search (BM25) aur semantic meaning ke liye Dense Vector Search (Cosine Similarity) dono chalayega.
* Dono retrievers ke Top 10-10 chunks ko merge karke ek local Cross-Encoder Reranker ke through Top 4 most relevant chunks filter kiye jayenge.

FR4 - Grounded Generation with Inline Citations: 

* Main Analyst Agent sirf retrieved chunks ke base par answer generate karega.
* Har sentence ya financial figure ke aage inline citation mandatory hoga: [Source: FileName.pdf | Page: X | Chunk: Y].
* Agar answer retrieved context mein nahi hai, toh system strictly bolega: "Insufficient data in uploaded financial reports."

FR5 - Automated Evals Layer (Multi-Agent Verification): 

* Har generated answer user ko dikhane ke saath-saath ek Eval Sub-Agent ko jayega.
* Eval Sub-Agent 3 cheezein check karke score (0-10) dega:
a) Numerical Groundedness: Answer ke saare numbers/percentages raw chunk mein exact match karte hain ya nahi.
b) Citation Accuracy: Cite kiya gaya page number aur chunk ID valid hai ya nahi.
c) Answer Relevance: User ke pooche gaye sawal ke saare parts cover hue ya nahi.

FR6 - Trader One-Click Presets: 

* UI mein 3 preset buttons honge: "Extract Growth Triggers & Capex", "Find Concall Red Flags", aur "Q-o-Q Guidance vs Actuals".

1. NON-FUNCTIONAL REQUIREMENTS (NFR) & CONSTRAINTS

* Zero-Cost Stack: Koi bhi paid vector DB ya paid embedding API use nahi hogi. Embeddings aur Reranking local CPU (AMD Ryzen 5 5600GT, 16GB RAM) par chalengi.
* High-Speed Inference: LLM generation aur evaluation ke liye Groq API ya high-quota OpenAI-compatible API use hogi taaki response latency 3-4 seconds se kam rahe.
* Explainability: UI mein har answer ke neeche "View Raw Retrieved Chunks & Eval Scores" ka expandable section hoga.

1. OUT OF SCOPE (FOR V1.0)

* Live stock price ticker ya OHLC charting integration.
* Scanned image-only PDFs ke liye heavy OCR (v1.0 focuses on digital PDFs from BSE/NSE/Screener).