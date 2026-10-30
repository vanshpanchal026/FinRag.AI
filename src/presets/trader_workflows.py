"""Trader Preset Workflows module for FinRAG-Eval.

Provides 3 one-click trader analysis presets:
1. Concall Red-Flag & Tone Scanner: Injects financial risk keywords into BM25
   and generates a categorized severity table.
2. Guidance vs. Actuals Tracker: Cross-quarter analysis comparing previous quarter
   outlook with current quarter delivered numbers.
3. Growth Triggers & Capex Analyzer: Synthesizes order book, deal wins, and
   capital expenditure plans with verified citations.

Sources:
- docs/Prd.md FR6
- docs/DetectionLogic.md Section 5
"""

from typing import Any, Dict, List, Optional
from src.agents.analyst_agent import AnalystAgent
from src.agents.eval_subagent import EvalSubAgent
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# Section 5.1 Keyword Booster List
RED_FLAG_KEYWORDS = [
    "headwinds",
    "margin pressure",
    "delay",
    "slowdown",
    "working capital stretch",
    "attrition",
    "pricing pressure",
    "regulatory",
    "uncertain",
    "challenging",
    "subdued",
    "inventory buildup",
    "debt",
    "pledge",
    "auditor",
    "contingent liability",
]


def run_red_flag_scanner(
    company: str,
    quarter: str,
    retriever: Optional[HybridRetriever] = None,
    reranker: Optional[Reranker] = None,
    analyst_agent: Optional[AnalystAgent] = None,
    eval_subagent: Optional[EvalSubAgent] = None,
    top_n: int = 4,
    skip_llm_judge: bool = False,
) -> Dict[str, Any]:
    """Execute Concall Red-Flag & Tone Scanner workflow.

    Injects negative sentiment / risk keywords into retrieval query to surface
    margin pressures, guidance cuts, or delayed timelines.

    Args:
        company: Company ticker (e.g. 'TATAELXSI').
        quarter: Reporting quarter (e.g. 'Q2_FY27').
        retriever: Optional HybridRetriever instance.
        reranker: Optional Reranker instance.
        analyst_agent: Optional AnalystAgent instance.
        eval_subagent: Optional EvalSubAgent instance.
        top_n: Number of Golden Chunks to retain (default 4).
        skip_llm_judge: Whether to skip LLM judge in unit testing.

    Returns:
        Dictionary with 'query', 'answer', 'golden_chunks', and 'eval_report'.
    """
    retriever_inst = retriever or HybridRetriever()
    reranker_inst = reranker or Reranker(top_n=top_n)
    analyst_inst = analyst_agent or AnalystAgent()
    eval_inst = eval_subagent or EvalSubAgent()

    # Formulate booster query
    booster_str = " ".join(RED_FLAG_KEYWORDS[:8])
    scanner_query = (
        f"{company} {quarter} concall management risks challenges headwinds "
        f"margin pressure delays {booster_str}"
    )

    metadata_filter = {"company": company, "quarter": quarter}
    candidates = retriever_inst.retrieve(
        query=scanner_query,
        top_k=10,
        metadata_filter=metadata_filter,
    )

    golden_chunks = reranker_inst.rerank(
        query=scanner_query,
        candidate_chunks=candidates,
        top_n=top_n,
    )

    workflow_prompt = (
        f"Scan the retrieved commentary of {company} for {quarter} for defensive tone, "
        f"guidance cuts, margin compression, or delayed timelines. "
        f"Output findings in a clean Markdown table with columns: "
        f"| Risk Category | Management Quote Summary | Severity (High/Med/Low) | Exact Citation |. "
        f"If no explicit risks are disclosed in the chunks, state: 'No significant red flags identified in uploaded reports.'"
    )

    synthesis = analyst_inst.answer_query(
        query=workflow_prompt,
        golden_chunks=golden_chunks,
    )

    eval_report = eval_inst.evaluate(
        user_query=workflow_prompt,
        golden_chunks=golden_chunks,
        generated_answer=synthesis["answer"],
        skip_llm_judge=skip_llm_judge,
    )

    return {
        "workflow": "red_flag_scanner",
        "company": company,
        "quarter": quarter,
        "query": workflow_prompt,
        "answer": synthesis["answer"],
        "golden_chunks": golden_chunks,
        "eval_report": eval_report,
    }


def run_guidance_vs_actuals(
    company: str,
    q_prev: str,
    q_curr: str,
    retriever: Optional[HybridRetriever] = None,
    reranker: Optional[Reranker] = None,
    analyst_agent: Optional[AnalystAgent] = None,
    eval_subagent: Optional[EvalSubAgent] = None,
    skip_llm_judge: bool = False,
) -> Dict[str, Any]:
    """Execute Guidance vs. Actuals Tracker workflow.

    Cross-references outlook/guidance statements from q_prev with actual
    delivered financials from q_curr to construct a Beat/Met/Missed score table.

    Args:
        company: Company ticker.
        q_prev: Previous quarter identifier (e.g. 'Q1_FY27').
        q_curr: Current quarter identifier (e.g. 'Q2_FY27').
        retriever: Optional HybridRetriever instance.
        reranker: Optional Reranker instance.
        analyst_agent: Optional AnalystAgent instance.
        eval_subagent: Optional EvalSubAgent instance.
        skip_llm_judge: Whether to skip LLM judge in unit testing.

    Returns:
        Dictionary with 'query', 'answer', 'golden_chunks', and 'eval_report'.
    """
    retriever_inst = retriever or HybridRetriever()
    reranker_inst = reranker or Reranker(top_n=4)
    analyst_inst = analyst_agent or AnalystAgent()
    eval_inst = eval_subagent or EvalSubAgent()

    # Step 1: Retrieve guidance chunks from q_prev
    guidance_query = f"{company} outlook guidance target capex revenue margin expectation"
    prev_candidates = retriever_inst.retrieve(
        query=guidance_query,
        top_k=6,
        metadata_filter={"company": company, "quarter": q_prev},
    )
    prev_golden = reranker_inst.rerank(
        query=guidance_query, candidate_chunks=prev_candidates, top_n=2
    )

    # Step 2: Retrieve actual financial performance from q_curr
    actuals_query = f"{company} financial results actual delivered revenue EBITDA net profit margin"
    curr_candidates = retriever_inst.retrieve(
        query=actuals_query,
        top_k=6,
        metadata_filter={"company": company, "quarter": q_curr},
    )
    curr_golden = reranker_inst.rerank(
        query=actuals_query, candidate_chunks=curr_candidates, top_n=2
    )

    # Combine golden chunks across quarters
    combined_golden = prev_golden + curr_golden

    workflow_prompt = (
        f"Compare the management guidance given in {q_prev} against the actual performance delivered in {q_curr} for {company}. "
        f"Present the comparison in a clean Markdown table with columns: "
        f"| Metric | {q_prev} Guidance Given | {q_curr} Actual Delivered | Status (Beat / Met / Missed) | Citations |. "
        f"Cite both previous and current quarter chunks explicitly."
    )

    synthesis = analyst_inst.answer_query(
        query=workflow_prompt,
        golden_chunks=combined_golden,
    )

    eval_report = eval_inst.evaluate(
        user_query=workflow_prompt,
        golden_chunks=combined_golden,
        generated_answer=synthesis["answer"],
        skip_llm_judge=skip_llm_judge,
    )

    return {
        "workflow": "guidance_vs_actuals",
        "company": company,
        "q_prev": q_prev,
        "q_curr": q_curr,
        "query": workflow_prompt,
        "answer": synthesis["answer"],
        "golden_chunks": combined_golden,
        "eval_report": eval_report,
    }


def run_growth_triggers_capex(
    company: str,
    quarter: str,
    retriever: Optional[HybridRetriever] = None,
    reranker: Optional[Reranker] = None,
    analyst_agent: Optional[AnalystAgent] = None,
    eval_subagent: Optional[EvalSubAgent] = None,
    top_n: int = 4,
    skip_llm_judge: bool = False,
) -> Dict[str, Any]:
    """Execute Growth Triggers & Capex workflow.

    Summarizes key growth drivers, deal pipeline, and capital expenditure allocations.

    Args:
        company: Company ticker.
        quarter: Target reporting quarter.
        retriever: Optional HybridRetriever instance.
        reranker: Optional Reranker instance.
        analyst_agent: Optional AnalystAgent instance.
        eval_subagent: Optional EvalSubAgent instance.
        top_n: Number of Golden Chunks to retain (default 4).
        skip_llm_judge: Whether to skip LLM judge in unit testing.

    Returns:
        Dictionary with 'query', 'answer', 'golden_chunks', and 'eval_report'.
    """
    retriever_inst = retriever or HybridRetriever()
    reranker_inst = reranker or Reranker(top_n=top_n)
    analyst_inst = analyst_agent or AnalystAgent()
    eval_inst = eval_subagent or EvalSubAgent()

    query_str = (
        f"{company} {quarter} order book pipeline deal wins capex expansion capacity growth triggers"
    )

    candidates = retriever_inst.retrieve(
        query=query_str,
        top_k=10,
        metadata_filter={"company": company, "quarter": quarter},
    )

    golden_chunks = reranker_inst.rerank(
        query=query_str,
        candidate_chunks=candidates,
        top_n=top_n,
    )

    workflow_prompt = (
        f"Identify and summarize the major growth triggers, order book status, and capex investments "
        f"disclosed by {company} for {quarter}. Every statement must cite its source chunk."
    )

    synthesis = analyst_inst.answer_query(
        query=workflow_prompt,
        golden_chunks=golden_chunks,
    )

    eval_report = eval_inst.evaluate(
        user_query=workflow_prompt,
        golden_chunks=golden_chunks,
        generated_answer=synthesis["answer"],
        skip_llm_judge=skip_llm_judge,
    )

    return {
        "workflow": "growth_triggers_capex",
        "company": company,
        "quarter": quarter,
        "query": workflow_prompt,
        "answer": synthesis["answer"],
        "golden_chunks": golden_chunks,
        "eval_report": eval_report,
    }
