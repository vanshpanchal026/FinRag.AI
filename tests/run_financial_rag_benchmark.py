"""Automated Financial RAG Evaluation Benchmark Runner for FinRAG-Eval.

Empirically benchmarks retrieval performance across all 15 Indian capital market domains:
1. Pure Dense Vector Search (ChromaDB + BAAI/bge-small-en-v1.5)
2. Pure Sparse Keyword Search (BM25Okapi with financial tokenization)
3. Hybrid RRF Search (Dense + Sparse with Reciprocal Rank Fusion k=60)
4. Hybrid + FlashRank ONNX Cross-Encoder Reranker (ms-marco-MiniLM-L-12-v2)

Computes quantitative metrics across the 15-question evaluation set:
- Hit Rate @ 4 & @ 10
- Recall @ 4 & @ 10
- Mean Reciprocal Rank (MRR)
- Average Retrieval Latency (ms)

Demonstrates empirical recall and ranking improvements across Indian financial filings.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from src.config import settings
from src.ingestion.indexer import ingest_chunks, reset_indices
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# 15 Benchmark Chunks corresponding to the 15 evaluation queries
BENCHMARK_CHUNKS = [
    {
        "chunk_id": "TATAELXSI_Q1_FY27_p05_c01",
        "text": (
            "### Q1 FY27 Financial Highlights\n"
            "Revenue from Operations reached 1,180 Cr with EBITDA of 330.1 Cr. "
            "Operating ROCE stood at 34.2%."
        ),
        "metadata": {
            "company": "TATAELXSI",
            "quarter": "Q1_FY27",
            "doc_type": "Financial_Results",
            "page_no": 5,
            "contains_table": False,
            "source_file": "TataElxsi_Q1.pdf",
        },
    },
    {
        "chunk_id": "TATAELXSI_Q2_FY27_p12_c01",
        "text": (
            "(Rs. in Crores)\n"
            "Statement of Profit and Loss\n"
            "| Particulars | Q2 FY27 | Q1 FY27 |\n"
            "|---|---|---|\n"
            "| Revenue from Operations | 1,250.5 | 1,180.0 |\n"
            "| EBITDA | 355.5 | 330.1 |\n"
            "Operating ROCE expanded to 36.5% in Q2 FY27."
        ),
        "metadata": {
            "company": "TATAELXSI",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 12,
            "contains_table": True,
            "source_file": "TataElxsi_Q2.pdf",
        },
    },
    {
        "chunk_id": "TATAELXSI_Q2_FY27_p15_c02",
        "text": (
            "Sneha Mehta: Any update on capex guidance for FY28?\n\n"
            "Management: We revised our capex guidance to 200 Cr for Bangalore lab infrastructure "
            "and GPU clusters."
        ),
        "metadata": {
            "company": "TATAELXSI",
            "quarter": "Q2_FY27",
            "doc_type": "Concall_Transcript",
            "page_no": 15,
            "contains_table": False,
            "source_file": "TataElxsi_Q2_Concall.pdf",
        },
    },
    {
        "chunk_id": "TATAELXSI_Q2_FY27_p14_c03",
        "text": (
            "### Transportation Segment Q2\n"
            "Transportation revenue reached 512 Cr with +4.4% QoQ expansion in SDV (Software Defined Vehicles)."
        ),
        "metadata": {
            "company": "TATAELXSI",
            "quarter": "Q2_FY27",
            "doc_type": "Investor_Presentation",
            "page_no": 14,
            "contains_table": False,
            "source_file": "TataElxsi_Q2.pdf",
        },
    },
    {
        "chunk_id": "TATAELXSI_Q2_FY27_p16_c04",
        "text": (
            "### Healthcare Vertical Q2\n"
            "Healthcare division achieved revenue of 120 Cr with +4.3% QoQ growth across digital health."
        ),
        "metadata": {
            "company": "TATAELXSI",
            "quarter": "Q2_FY27",
            "doc_type": "Investor_Presentation",
            "page_no": 16,
            "contains_table": False,
            "source_file": "TataElxsi_Q2.pdf",
        },
    },
    {
        "chunk_id": "HDFCBANK_Q2_FY27_p08_c01",
        "text": (
            "### HDFC Bank Financial & Key Banking Ratios\n"
            "Net Interest Income (NII) grew 16.8% YoY. Net Interest Margin (NIM) stood at 3.45% on total assets.\n"
            "CASA deposits stood at 38.2% of total deposits. Gross NPA improved to 1.24% with Provision Coverage Ratio (PCR) at 74.3%."
        ),
        "metadata": {
            "company": "HDFCBANK",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 8,
            "contains_table": False,
            "source_file": "HDFCBank_Q2_FY27.pdf",
        },
    },
    {
        "chunk_id": "INFY_Q2_FY27_p11_c02",
        "text": (
            "### Infosys Operational & Segment Highlights\n"
            "Infosys recorded large deal Total Contract Value (TCV) of $2.4 Billion with net new contract share of 52%.\n"
            "LTM voluntary attrition moderated to 12.9% while utilization excluding trainees improved to 85.3%."
        ),
        "metadata": {
            "company": "INFY",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 11,
            "contains_table": False,
            "source_file": "Infosys_Q2_Results.pdf",
        },
    },
    {
        "chunk_id": "RELIANCE_Q2_FY27_p22_c01",
        "text": (
            "### Valuation and Capital Allocation Update\n"
            "Consolidated Net Debt stood at 1,12,500 Cr following massive 5G and green energy outlays.\n"
            "On a segment EV/EBITDA multiple basis, Oil-to-Chemicals (O2C) is valued at 7.2x and Reliance Retail at 24.5x."
        ),
        "metadata": {
            "company": "RELIANCE",
            "quarter": "Q2_FY27",
            "doc_type": "Investor_Presentation",
            "page_no": 22,
            "contains_table": False,
            "source_file": "RIL_Q2_Presentation.pdf",
        },
    },
    {
        "chunk_id": "TCS_Q2_FY27_p02_c01",
        "text": (
            "### Outcome of Board Meeting - Corporate Action\n"
            "The Board of Directors declared a second interim dividend of Rs. 10.00 per equity share of face value Re. 1 each.\n"
            "Record date for payment of interim dividend has been fixed as October 18, 2026."
        ),
        "metadata": {
            "company": "TCS",
            "quarter": "Q2_FY27",
            "doc_type": "Exchange_Announcement",
            "page_no": 2,
            "contains_table": False,
            "source_file": "TCS_Board_Outcome.pdf",
        },
    },
    {
        "chunk_id": "VEDL_Q2_FY27_p04_c02",
        "text": (
            "### Statutory Auditor's Report - Emphasis of Matter\n"
            "We draw attention to Note 7 regarding recoverability of inter-corporate deposits of 3,200 Cr extended to the ultimate parent company without collateral.\n"
            "Our opinion is not modified in respect of this matter."
        ),
        "metadata": {
            "company": "VEDL",
            "quarter": "Q2_FY27",
            "doc_type": "Auditor_Report",
            "page_no": 4,
            "contains_table": False,
            "source_file": "Vedanta_Auditor_Report.pdf",
        },
    },
    {
        "chunk_id": "ADANIENT_Q2_FY27_p03_c01",
        "text": (
            "### Shareholding Pattern - Regulation 31\n"
            "Promoter and Promoter Group shareholding was 72.6% of which encumbered/pledged shares declined to 2.4% (vs 3.8% QoQ).\n"
            "Foreign Portfolio Investors (FPI/FII) holding stood at 14.2% and Mutual Fund holding at 4.6%."
        ),
        "metadata": {
            "company": "ADANIENT",
            "quarter": "Q2_FY27",
            "doc_type": "Shareholding_Pattern",
            "page_no": 3,
            "contains_table": False,
            "source_file": "AdaniEnt_SHP_Q2.pdf",
        },
    },
    {
        "chunk_id": "HINDUNILVR_Q2_FY27_p06_c01",
        "text": (
            "### FMCG Operating Metrics & Margins\n"
            "Hindustan Unilever reported Underlying Volume Growth (UVG) of +3.0% YoY with positive rural consumption signals.\n"
            "Gross Margin expanded 180 bps YoY to 51.4% driven by benign commodity prices, while Advertising spend rose 12%."
        ),
        "metadata": {
            "company": "HINDUNILVR",
            "quarter": "Q2_FY27",
            "doc_type": "Investor_Presentation",
            "page_no": 6,
            "contains_table": False,
            "source_file": "HUL_Q2_Presentation.pdf",
        },
    },
    {
        "chunk_id": "SUNPHARMA_Q2_FY27_p01_c01",
        "text": (
            "### USFDA Regulatory Inspection Update\n"
            "The USFDA conducted a pre-approval inspection at our Halol facility from Sep 14 to Sep 22, 2026.\n"
            "The inspection closed with 3 procedural observations on Form 483 with no data integrity concerns noted."
        ),
        "metadata": {
            "company": "SUNPHARMA",
            "quarter": "Q2_FY27",
            "doc_type": "Exchange_Announcement",
            "page_no": 1,
            "contains_table": False,
            "source_file": "SunPharma_USFDA_Notice.pdf",
        },
    },
    {
        "chunk_id": "LT_Q2_FY27_p10_c02",
        "text": (
            "### Infrastructure & Order Book Performance\n"
            "L&T secured consolidated order inflows of 78,500 Cr with international orders making up 42% of total intake.\n"
            "The consolidated order book stands at a record 5,10,000 Cr providing robust execution visibility for 28 months."
        ),
        "metadata": {
            "company": "LT",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 10,
            "contains_table": False,
            "source_file": "LT_Q2_Earnings.pdf",
        },
    },
    {
        "chunk_id": "TATASTEEL_Q2_FY27_p18_c01",
        "text": (
            "### Cash Flows and Working Capital Health\n"
            "Tata Steel delivered strong operating cash flow (CFO) of 9,450 Cr for H1 FY27 driven by inventory rationalization.\n"
            "Cash Conversion Cycle (CCC) improved by 6 days to 48 days reflecting disciplined working capital control."
        ),
        "metadata": {
            "company": "TATASTEEL",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 18,
            "contains_table": False,
            "source_file": "TataSteel_H1_Results.pdf",
        },
    },
    # Hard Negative Distractor Chunks to Test Cross-Entity Disambiguation & Semantic Reranking
    {
        "chunk_id": "ICICIBANK_Q2_FY27_p09_c01",
        "text": (
            "### ICICI Bank Key Ratios\n"
            "Net Interest Income (NII) grew 14.5% YoY. Net Interest Margin (NIM) was 4.36% on average assets.\n"
            "CASA ratio stood at 41.5% and Gross NPA was 2.15% with PCR at 80.1%."
        ),
        "metadata": {
            "company": "ICICIBANK",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 9,
            "contains_table": False,
            "source_file": "ICICIBank_Q2.pdf",
        },
    },
    {
        "chunk_id": "WIPRO_Q2_FY27_p14_c02",
        "text": (
            "### Wipro Operational Highlights\n"
            "Wipro closed large deal Total Contract Value (TCV) of $1.2 Billion.\n"
            "Voluntary quarterly annualized attrition stood at 14.5% with utilization at 82.0%."
        ),
        "metadata": {
            "company": "WIPRO",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 14,
            "contains_table": False,
            "source_file": "Wipro_Q2.pdf",
        },
    },
    {
        "chunk_id": "ADANIPORTS_Q2_FY27_p25_c01",
        "text": (
            "### Adani Ports Capital Structure & Valuation\n"
            "Consolidated Net Debt was 48,000 Cr with trailing EV/EBITDA multiple of 11.2x.\n"
            "Port operating cash flows expanded 18% YoY."
        ),
        "metadata": {
            "company": "ADANIPORTS",
            "quarter": "Q2_FY27",
            "doc_type": "Investor_Presentation",
            "page_no": 25,
            "contains_table": False,
            "source_file": "APSEZ_Presentation.pdf",
        },
    },
    {
        "chunk_id": "CIPLA_Q2_FY27_p02_c01",
        "text": (
            "### Regulatory Notice - USFDA\n"
            "The USFDA inspected the Pithampur manufacturing unit, concluding with 4 observations on Form 483.\n"
            "Company is preparing a comprehensive CAPA response."
        ),
        "metadata": {
            "company": "CIPLA",
            "quarter": "Q2_FY27",
            "doc_type": "Exchange_Announcement",
            "page_no": 2,
            "contains_table": False,
            "source_file": "Cipla_USFDA_Notice.pdf",
        },
    },
    {
        "chunk_id": "BHEL_Q2_FY27_p07_c02",
        "text": (
            "### Order Intake and Execution\n"
            "BHEL booked quarterly order inflows of 28,000 Cr with thermal power equipment orders.\n"
            "The total order backlog reached 1,35,000 Cr."
        ),
        "metadata": {
            "company": "BHEL",
            "quarter": "Q2_FY27",
            "doc_type": "Financial_Results",
            "page_no": 7,
            "contains_table": False,
            "source_file": "BHEL_Q2.pdf",
        },
    },
]


def evaluate_pipeline(
    name: str,
    retrieval_fn: Any,
    eval_queries: List[Dict[str, Any]],
    k_values: List[int] = [4, 10],
) -> Dict[str, Any]:
    """Evaluate a specific retrieval pipeline across evaluation queries.

    Computes:
    - HitRate@k: Fraction of queries where at least one target chunk is retrieved in top k.
    - Recall@k: Fraction of target ground truth chunks retrieved in top k.
    - MRR: Mean Reciprocal Rank of the first target chunk.
    - Latency: Average latency per query in milliseconds.
    """
    total_queries = len(eval_queries)
    hit_counts = {k: 0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    latencies: List[float] = []

    for item in eval_queries:
        query = item["query"]
        target_chunks: Set[str] = set(item["target_chunks"])

        t0 = time.perf_counter()
        retrieved_items = retrieval_fn(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        retrieved_ids = [c["chunk_id"] for c in retrieved_items]

        # Compute MRR
        first_rank = None
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in target_chunks:
                first_rank = rank
                break
        if first_rank:
            mrr_sum += 1.0 / first_rank

        # Compute HitRate and Recall at each k
        for k in k_values:
            top_k_ids = set(retrieved_ids[:k])
            hits = top_k_ids.intersection(target_chunks)
            if hits:
                hit_counts[k] += 1
            recall_sums[k] += len(hits) / len(target_chunks)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "pipeline": name,
        "hit_rate_4": (hit_counts[4] / total_queries) * 100.0,
        "recall_4": (recall_sums[4] / total_queries) * 100.0,
        "hit_rate_10": (hit_counts[10] / total_queries) * 100.0,
        "recall_10": (recall_sums[10] / total_queries) * 100.0,
        "mrr": mrr_sum / total_queries,
        "avg_latency_ms": avg_latency,
    }


def run_benchmark(eval_dataset_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run full comparative benchmark across 4 financial RAG pipelines."""
    data_path = (
        eval_dataset_path
        or Path(__file__).resolve().parent / "eval_dataset.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    # Setup isolated temp index for evaluation
    temp_dir = tempfile.mkdtemp(prefix="finrag_eval_bench_")
    chroma_dir = Path(temp_dir) / "chroma"
    bm25_store_path = Path(temp_dir) / "bm25_store.pkl"

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection_name = "benchmark_financial_reports"

        reset_indices(
            client=client,
            collection_name=collection_name,
            bm25_store_path=bm25_store_path,
        )
        ingest_chunks(
            chunks=BENCHMARK_CHUNKS,
            client=client,
            collection_name=collection_name,
            bm25_store_path=bm25_store_path,
        )

        retriever = HybridRetriever(
            client=client,
            collection_name=collection_name,
            bm25_store_path=bm25_store_path,
            top_k=10,
        )
        reranker = Reranker(top_n=4)

        # Pipeline 1: Dense Only
        def dense_pipeline(q: str):
            return retriever.dense_search(query=q, top_k=10)

        # Pipeline 2: Sparse Only
        def sparse_pipeline(q: str):
            return retriever.sparse_search(query=q, top_k=10)

        # Pipeline 3: Hybrid Search (RRF k=60)
        def hybrid_pipeline(q: str):
            return retriever.retrieve(query=q, top_k=10, expand_acronyms=True)

        # Pipeline 4: Hybrid + Cross-Encoder Reranker
        def hybrid_rerank_pipeline(q: str):
            fused = retriever.retrieve(query=q, top_k=10, expand_acronyms=True)
            return reranker.rerank(query=q, candidate_chunks=fused, top_n=4)

        results = []
        results.append(evaluate_pipeline("1. Naive Dense (ChromaDB)", dense_pipeline, eval_queries))
        results.append(evaluate_pipeline("2. Sparse Keyword (BM25Okapi)", sparse_pipeline, eval_queries))
        results.append(evaluate_pipeline("3. Hybrid RRF (Dense+BM25)", hybrid_pipeline, eval_queries))
        results.append(evaluate_pipeline("4. Hybrid + FlashRank Reranker", hybrid_rerank_pipeline, eval_queries))

        return {
            "total_queries": len(eval_queries),
            "total_chunks_indexed": len(BENCHMARK_CHUNKS),
            "results": results,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    report = run_benchmark()
    print("\n" + "=" * 92)
    print(f"FINRAG-EVAL: QUANTITATIVE RETRIEVAL BENCHMARK REPORT ({report['total_queries']} QUERIES)")
    print("=" * 92)
    print(f"{'Pipeline':<34} | {'Recall@4':<10} | {'HitRate@4':<10} | {'Recall@10':<10} | {'MRR':<8} | {'Latency':<8}")
    print("-" * 92)
    for r in report["results"]:
        print(
            f"{r['pipeline']:<34} | "
            f"{r['recall_4']:>8.1f}% | "
            f"{r['hit_rate_4']:>8.1f}% | "
            f"{r['recall_10']:>8.1f}% | "
            f"{r['mrr']:>7.3f} | "
            f"{r['avg_latency_ms']:>6.1f}ms"
        )
    print("=" * 92 + "\n")
