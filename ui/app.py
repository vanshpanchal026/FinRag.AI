"""Streamlit Trader Workbench for FinRAG-Eval.

Provides an institutional-grade research interface for equity analysts and traders:
- Sidebar: Document upload, smart ingestion, and metadata pre-filtering.
- Top Action Bar: 1-Click Trader Presets (Red Flags, Guidance vs Actuals, Growth Triggers).
- Main Chat & Audit Interface: Cited answers, 4-badge Live Eval Scorecard, and
  expandable "Verify Raw Retrieved Sources" inspection accordion.

Sources:
- docs/Prd.md FR6 & Section 2
- docs/DesignDocument.md Section 1 & 3
"""

import os
from pathlib import Path
import re
import sys
import tempfile

# Ensure project root is in sys.path for direct 'streamlit run ui/app.py' execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from src.agents.analyst_agent import AnalystAgent
from src.agents.eval_subagent import EvalSubAgent
from src.config import settings
from src.ingestion.indexer import get_chroma_collection, ingest_chunks, reset_indices
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.smart_chunker import chunk_document
from src.presets.trader_workflows import (
    run_growth_triggers_capex,
    run_guidance_vs_actuals,
    run_red_flag_scanner,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# Page configuration
st.set_page_config(
    page_title="FinRAG-Eval | Zero-Hallucination Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Financial Terminal aesthetics
st.markdown(
    """
    <style>
    .metric-card-pass {
        background-color: #0e2f1f;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        color: #10b981;
    }
    .metric-card-fail {
        background-color: #3b1111;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        color: #ef4444;
    }
    .metric-card-score {
        background-color: #1a202c;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        color: #e2e8f0;
    }
    .source-tag {
        background-color: #2b3b55;
        color: #63b3ed;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85em;
    }
    .table-badge {
        background-color: #2d3748;
        color: #f6ad55;
        border: 1px solid #dd6b20;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    """Cache core retrieval and synthesis engines."""
    settings.ensure_directories()
    return {
        "retriever": HybridRetriever(),
        "reranker": Reranker(),
        "analyst": AnalystAgent(),
        "evaluator": EvalSubAgent(),
    }


pipeline = load_pipeline()

# Initialize session state for chat and results
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==============================================================================
# SIDEBAR: DOCUMENT INGESTION & METADATA FILTERS
# ==============================================================================
with st.sidebar:
    st.title("📈 FinRAG-Eval")
    st.caption("Zero-Hallucination Financial Report & Concall Q&A Engine")
    st.markdown("---")

    st.subheader("📥 Ingest Document")
    uploaded_file = st.file_uploader(
        "Upload BSE/NSE Filing (PDF)", type=["pdf"], key="pdf_uploader"
    )

    company_input = st.text_input(
        "Company Ticker", value="TATAELXSI", placeholder="e.g., TATAELXSI, RELIANCE"
    )
    quarter_input = st.text_input(
        "Reporting Quarter", value="Q2_FY27", placeholder="e.g., Q2_FY27, Q1_FY27"
    )
    doc_type_input = st.selectbox(
        "Document Type",
        [
            "Financial_Results",
            "Concall_Transcript",
            "Investor_Presentation",
            "Annual_Report",
        ],
    )

    if st.button("🚀 Ingest & Index PDF", use_container_width=True):
        if uploaded_file is None:
            st.error("Please select a PDF file to upload.")
        elif not company_input.strip() or not quarter_input.strip():
            st.error("Please specify both Company Ticker and Quarter.")
        else:
            with st.status("Ingesting and processing financial filing...", expanded=True) as status:
                st.write("1️⃣ Saving and parsing PDF page-by-page...")
                save_dest = settings.raw_pdfs / uploaded_file.name
                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = parse_pdf(save_dest, save_debug_md=True)
                st.write(f"✅ Extracted **{len(pages)}** pages into Markdown.")

                st.write("2️⃣ Executing Smart Chunking (Table & Concall preservation)...")
                chunks = chunk_document(
                    pages=pages,
                    company=company_input.strip().upper(),
                    quarter=quarter_input.strip(),
                    doc_type=doc_type_input,
                    source_file=uploaded_file.name,
                )
                table_chunks_count = sum(
                    1 for c in chunks if c["metadata"].get("contains_table", False)
                )
                st.write(
                    f"✅ Created **{len(chunks)}** atomic chunks (**{table_chunks_count}** tables locked intact)."
                )

                st.write("3️⃣ Indexing into ChromaDB (Dense) & BM25Okapi (Sparse)...")
                stats = ingest_chunks(chunks=chunks)
                st.write(
                    f"✅ Indexed! Total Collection Size: **{stats['total_chroma_count']}** chunks."
                )

                status.update(
                    label=f"Successfully indexed {uploaded_file.name}!",
                    state="complete",
                    expanded=False,
                )
            st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

    st.markdown("---")
    st.subheader("🎯 Active Metadata Filters")
    filter_company = st.text_input("Filter Company", value="", placeholder="All companies")
    filter_quarter = st.text_input("Filter Quarter", value="", placeholder="All quarters")
    filter_doc_type = st.selectbox(
        "Filter Doc Type",
        ["All", "Financial_Results", "Concall_Transcript", "Investor_Presentation", "Annual_Report"],
    )

    # Database stats
    collection = get_chroma_collection()
    doc_count = collection.count()
    st.caption(f"💾 Local Database: **{doc_count}** active chunks indexed")

    if st.button("⚠️ Reset All Indices", help="Clears local vector and BM25 stores"):
        reset_indices()
        st.warning("All indices reset.")
        st.rerun()

# ==============================================================================
# TOP ACTION BAR: 1-CLICK TRADER PRESETS
# ==============================================================================
st.title("📊 Trader Research Workbench")
st.markdown("Query cross-quarter filings with mathematical grounding, inline source citations, and dual-layer faithfulness audit.")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    if st.button("🔍 Scan Concall Red Flags", use_container_width=True):
        co = filter_company.strip().upper() or company_input.strip().upper()
        q = filter_quarter.strip() or quarter_input.strip()
        with st.spinner(f"Scanning {co} {q} commentary for headwinds & risks..."):
            result = run_red_flag_scanner(
                company=co,
                quarter=q,
                retriever=pipeline["retriever"],
                reranker=pipeline["reranker"],
                analyst_agent=pipeline["analyst"],
                eval_subagent=pipeline["evaluator"],
            )
            st.session_state.last_result = result
            st.session_state.history.append((result["query"], result["answer"], result))

with col_p2:
    if st.button("📊 Guidance vs Actuals", use_container_width=True):
        co = filter_company.strip().upper() or company_input.strip().upper()
        with st.spinner(f"Comparing Q1 Guidance vs Q2 Delivered for {co}..."):
            result = run_guidance_vs_actuals(
                company=co,
                q_prev="Q1_FY27",
                q_curr="Q2_FY27",
                retriever=pipeline["retriever"],
                reranker=pipeline["reranker"],
                analyst_agent=pipeline["analyst"],
                eval_subagent=pipeline["evaluator"],
            )
            st.session_state.last_result = result
            st.session_state.history.append((result["query"], result["answer"], result))

with col_p3:
    if st.button("💡 Growth Triggers & Capex", use_container_width=True):
        co = filter_company.strip().upper() or company_input.strip().upper()
        q = filter_quarter.strip() or quarter_input.strip()
        with st.spinner(f"Extracting capex and deal pipelines for {co} {q}..."):
            result = run_growth_triggers_capex(
                company=co,
                quarter=q,
                retriever=pipeline["retriever"],
                reranker=pipeline["reranker"],
                analyst_agent=pipeline["analyst"],
                eval_subagent=pipeline["evaluator"],
            )
            st.session_state.last_result = result
            st.session_state.history.append((result["query"], result["answer"], result))

st.markdown("---")

# ==============================================================================
# MAIN QUERY INTERFACE
# ==============================================================================
user_query = st.chat_input("Ask a question (e.g., 'What was the EBITDA margin and revenue in Q2?')...")

if user_query:
    active_filter = {}
    if filter_company.strip():
        active_filter["company"] = filter_company.strip().upper()
    if filter_quarter.strip():
        active_filter["quarter"] = filter_quarter.strip()
    if filter_doc_type != "All":
        active_filter["doc_type"] = filter_doc_type

    with st.spinner("Executing Hybrid Retrieval (Dense + Sparse RRF) & FlashRank Reranking..."):
        candidates = pipeline["retriever"].retrieve(
            query=user_query,
            top_k=10,
            metadata_filter=active_filter if active_filter else None,
        )

        if not candidates:
            st.warning("Insufficient data in uploaded financial reports. No matching chunks found for the given query and filters.")
            result_payload = None
        else:
            golden_chunks = pipeline["reranker"].rerank(
                query=user_query,
                candidate_chunks=candidates,
                top_n=4,
            )

            synthesis = pipeline["analyst"].answer_query(
                query=user_query,
                golden_chunks=golden_chunks,
            )

            eval_report = pipeline["evaluator"].evaluate(
                user_query=user_query,
                golden_chunks=golden_chunks,
                generated_answer=synthesis["answer"],
            )

            result_payload = {
                "query": user_query,
                "answer": synthesis["answer"],
                "golden_chunks": golden_chunks,
                "eval_report": eval_report,
            }

            st.session_state.last_result = result_payload
            st.session_state.history.append((user_query, synthesis["answer"], result_payload))

# ==============================================================================
# DISPLAY RESULT & LIVE EVAL SCORECARD
# ==============================================================================
active_result = st.session_state.last_result

if active_result:
    answer_text = active_result["answer"]
    golden_chunks = active_result.get("golden_chunks", [])
    eval_report = active_result.get("eval_report", {})
    det_checks = eval_report.get("deterministic_checks", {})
    judge_scores = eval_report.get("llm_judge_scores", {})

    st.subheader("📝 Analyst Grounded Response")
    st.markdown(answer_text)

    st.markdown("---")
    st.subheader("🛡️ Live Faithfulness & Audit Scorecard")

    # 4 Visual Metric Badges
    c1, c2, c3, c4 = st.columns(4)

    # 1. Numeric Match Badge
    num_pass = det_checks.get("numeric_match_pass", True)
    unverified = det_checks.get("unverified_numbers", [])
    with c1:
        if num_pass:
            st.markdown(
                """
                <div class="metric-card-pass">
                    <div style="font-size: 0.8em; text-transform: uppercase;">Deterministic Check</div>
                    <div style="font-size: 1.6em; font-weight: bold;">PASS ✅</div>
                    <div style="font-size: 0.8em;">All numbers verbatim</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            unverified_str = ", ".join(unverified[:3])
            st.markdown(
                f"""
                <div class="metric-card-fail">
                    <div style="font-size: 0.8em; text-transform: uppercase;">Deterministic Check</div>
                    <div style="font-size: 1.6em; font-weight: bold;">FAIL ❌</div>
                    <div style="font-size: 0.75em;">Unverified: {unverified_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. Faithfulness Score
    faith_score = judge_scores.get("faithfulness_score", 10)
    with c2:
        st.markdown(
            f"""
            <div class="metric-card-score">
                <div style="font-size: 0.8em; text-transform: uppercase;">Faithfulness</div>
                <div style="font-size: 1.6em; font-weight: bold; color: {'#10b981' if faith_score >= 8 else '#ef4444'};">
                    {faith_score} / 10
                </div>
                <div style="font-size: 0.8em;">Zero Hallucination</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Citation Accuracy Score
    cit_score = judge_scores.get("citation_accuracy_score", 10)
    with c3:
        st.markdown(
            f"""
            <div class="metric-card-score">
                <div style="font-size: 0.8em; text-transform: uppercase;">Citation Accuracy</div>
                <div style="font-size: 1.6em; font-weight: bold; color: #38bdf8;">
                    {cit_score} / 10
                </div>
                <div style="font-size: 0.8em;">Valid Page & Chunk Tags</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Answer Relevance Score
    rel_score = judge_scores.get("answer_relevance_score", 10)
    with c4:
        st.markdown(
            f"""
            <div class="metric-card-score">
                <div style="font-size: 0.8em; text-transform: uppercase;">Relevance</div>
                <div style="font-size: 1.6em; font-weight: bold; color: #a78bfa;">
                    {rel_score} / 10
                </div>
                <div style="font-size: 0.8em;">Completeness</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Audit Reasoning Line
    reasoning_text = judge_scores.get("reasoning", "Audit complete.")
    st.caption(f"🔍 **Auditor Reasoning:** {reasoning_text}")

    # ==============================================================================
    # EXPANDABLE ACCORDION: VERIFY RAW RETRIEVED SOURCES
    # ==============================================================================
    st.markdown("###")
    with st.expander("🔍 Verify Raw Retrieved Sources (Top Golden Chunks)", expanded=False):
        st.markdown("Inspect the exact source chunks retrieved from the local database that formed the basis of the response:")

        if not golden_chunks:
            st.info("No raw chunks available.")
        else:
            for idx, c in enumerate(golden_chunks, start=1):
                cid = c.get("chunk_id", f"chunk_{idx}")
                meta = c.get("metadata", {})
                sfile = meta.get("source_file", "unknown.pdf")
                pno = meta.get("page_no", 1)
                rscore = c.get("rerank_score", meta.get("rerank_score", 0.0))
                contains_tbl = meta.get("contains_table", False)

                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    tbl_label = " 📊 TABLE INTACT" if contains_tbl else ""
                    st.markdown(
                        f"**#{idx} | `{cid}`** — `File: {sfile}` | `Page: {pno}`{tbl_label}"
                    )
                with col_h2:
                    st.markdown(f"**Rerank Score:** `{rscore:.4f}`")

                st.code(c.get("text", "").strip(), language="markdown")
                st.markdown("---")
