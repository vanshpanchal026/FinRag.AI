PROJECT TITLE: FinRAG-Eval
DOCUMENT: Prompts, Regex Rules, Red-Flag Detection & Multi-Agent Eval Logic
VERSION: 1.0.0 

1. SMART CHUNKING DETECTION RULES (src/ingestion/smart_chunker.py)

1.1 Markdown Table Detection Regex
Pattern:
r'((?:#+ .*\n+)?(?:|.*|\n|[-:\s|]+|\n(?:|.*|\n?)+))'
Logic: 

* Scans each page's Markdown string for contiguous pipe-delimited table rows preceded by an optional heading.
* Captures 2 lines immediately above the table (to retain units like "Rs. in Crores" or "Standalone Financial Results").
* Locks the entire block as an atomic chunk (contains_table = True) so rows and column headers are never separated.

1.2 Concall Speaker-Turn Detection Regex
Pattern:
r'(?=(?:Moderator|Management|[A-Z][a-z]+ [A-Z][a-z]+)\s*:)'
Logic: 

* Identifies speaker transitions in earnings call transcripts.
* Groups [Analyst Question + Immediate Management Response] into a single chunk (up to 700 tokens) so the question context is never lost from the management's answer.

1. MAIN FINANCIAL ANALYST AGENT SYSTEM PROMPT (src/agents/analyst_agent.py)

SYSTEM PROMPT:
"You are an elite Institutional Equity Research Analyst specializing in Indian stock market filings (NSE/BSE Quarterly Results, Concall Transcripts, and Investor Presentations).
Your sole objective is to answer the user's query with 100% numerical precision based ONLY on the provided Retrieved Context Chunks. 

STRICT OPERATING RULES: 

1. ZERO EXTRAPOLATION: Never assume, extrapolate, or bring in external knowledge not present in the chunks. If a requested metric or guidance is missing from the chunks, state verbatim: 'Data not available in the uploaded financial reports.'
2. MANDATORY INLINE CITATIONS: Every single sentence, financial figure, ratio, or management claim MUST end with an exact citation tag in this format:
[Source: {source_file} | Page: {page_no} | Chunk: {chunk_id}]
3. NUMERICAL INTEGRITY: Copy numbers, percentages, and units (Cr, Lakhs, Mn, bps, YoY, QoQ) exactly as written in the source chunk. If you perform a basic calculation (e.g., difference between Q1 and Q2), explicitly show the formula alongside both cited source numbers.
4. TABLE FORMATTING: Whenever comparing numbers across quarters or segments, present the output in a clean Markdown table alongside citations."

1. DETERMINISTIC TIER-1 EVAL LOGIC (src/agents/eval_subagent.py)

3.1 Numerical Token Extractor Regex
Pattern:
r'(?<![a-zA-Z0-9_])\d+(?:,\d+)*(?:.\d+)?%?(?![a-zA-Z0-9_])'
Exclusion Filter: 

* Strip out page numbers and chunk IDs inside [Source: ...] brackets before extracting numbers from the generated answer.

3.2 Verification Algorithm: 

1. answer_Clean = Remove all [Source: ...] substrings from Generated_Answer.
2. extracted_nums = Find all regex matches in answer_Clean.
3. raw_corpus = Concatenate text of all Top 4 Golden Chunks (removing commas from numbers in both sets for normalized matching, e.g., "1,250.5" -> "1250.5").
4. unverified_nums = [num for num in extracted_nums if num not in raw_corpus]
5. If len(unverified_nums) == 0:
numeric_match_pass = True
Else:
numeric_match_pass = False (Triggers high-alert warning in UI and passes unverified_nums to Tier-2 Judge).

1. TIER-2 LLM-AS-A-JUDGE SYSTEM PROMPT (src/agents/eval_subagent.py)

SYSTEM PROMPT:
"You are a strict Financial Compliance & RAG Evaluation Auditor.
You will receive: 

1. USER_QUERY
2. RETRIEVED_CHUNKS (with Chunk IDs and Page Numbers)
3. GENERATED_ANSWER
4. DETERMINISTIC_NUMERIC_REPORT (List of verified and unverified numbers)

Evaluate the GENERATED_ANSWER strictly across three criteria on a scale of 0 to 10: 

* faithfulness_score (0-10): Are all claims, reasons, and numbers 100% grounded in RETRIEVED_CHUNKS? Deduct 5 points immediately for any hallucinated number or fabricated management statement.
* citation_accuracy_score (0-10): Do the [Source: File | Page: X | Chunk: Y] tags accurately point to the specific chunk containing that fact?
* answer_relevance_score (0-10): Did the answer directly and completely address all parts of the USER_QUERY?

Return ONLY a valid JSON object matching this exact schema:
{
"faithfulness_score": int,
"citation_accuracy_score": int,
"answer_relevance_score": int,
"hallucination_flag": bool,
"unverified_claims": ["list of any unsupported statements or numbers"],
"audit_reasoning": "1-2 crisp sentences explaining the scores."
}" 

1. TRADER PRESET DETECTION LOGIC (src/presets/trader_workflows.py)

5.1 Concall Red-Flag & Tone Scanner
Keyword Booster List (Injected into BM25 Query):
["headwinds", "margin pressure", "delay", "slowdown", "working capital stretch", "attrition", "pricing pressure", "regulatory", "uncertain", "challenging", "subdued", "inventory buildup", "debt", "pledge", "auditor", "contingent liability"]
Prompt Instruction: 

* Scans retrieved management commentary for defensive tone, guidance cuts, or delayed capex timelines, outputting a categorized table: [Risk Category | Management Quote Summary | Severity (High/Med/Low) | Exact Citation].

5.2 Guidance vs. Actuals Tracker
Logic: 

* Filters Vector DB metadata across two distinct quarters (e.g., quarter="Q1_FY27" and quarter="Q2_FY27").
* Retrieves "Outlook/Guidance" chunks from Q1 and "Financial Highlights/Performance" chunks from Q2.
* Outputs a side-by-side comparison table: [Metric | Q1 Guidance Given | Q2 Actual Delivered | Status (Beat / Met / Missed) | Citations].