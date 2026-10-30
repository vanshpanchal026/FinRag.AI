"""Unit tests for FinRAG-Eval Dual Indexer, Hybrid Retriever, and Cross-Encoder Reranker.

Verifies:
1. Exact financial keyword queries hit BM25 + Dense search properly.
2. Metadata filtering (e.g. quarter="Q2_FY27") strictly isolates target quarter chunks.
3. Reciprocal Rank Fusion (RRF) properly combines and deduplicates rankings.
4. FlashRank (ms-marco-MiniLM-L-12-v2) returns Top 4 Golden Chunks sorted by rerank_score
   in under 1 second on CPU.
5. Ingestion idempotency (no duplicate chunk insertions upon repeated indexing).

Sources:
- docs/Techstack.md Section 3
- docs/DesignDocument.md Section 1
"""

import time
import pytest
import chromadb

from src.ingestion.indexer import (
    ingest_chunks,
    reset_indices,
    tokenize_for_bm25,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


@pytest.fixture
def mock_financial_chunks():
    """8 realistic financial chunks across Q1 and Q2 with exact financial metrics."""
    return [
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
            "chunk_id": "TATAELXSI_Q1_FY27_p08_c02",
            "text": (
                "Moderator: Next question is from Rahul Sharma from Motilal Oswal.\n\n"
                "Rahul Sharma: What is the capex outlook for the next financial year?\n\n"
                "Management: Total planned capex for FY28 is 150-180 Cr for expanding facilities."
            ),
            "metadata": {
                "company": "TATAELXSI",
                "quarter": "Q1_FY27",
                "doc_type": "Concall_Transcript",
                "page_no": 8,
                "contains_table": False,
                "source_file": "TataElxsi_Q1_Concall.pdf",
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
            "chunk_id": "TATAELXSI_Q1_FY27_p06_c03",
            "text": (
                "### Transportation Segment Q1\n"
                "Transportation vertical delivered 3.8% QoQ growth driven by electric vehicle software contracts."
            ),
            "metadata": {
                "company": "TATAELXSI",
                "quarter": "Q1_FY27",
                "doc_type": "Investor_Presentation",
                "page_no": 6,
                "contains_table": False,
                "source_file": "TataElxsi_Q1.pdf",
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
            "chunk_id": "TATAELXSI_Q1_FY27_p07_c04",
            "text": (
                "### Healthcare Vertical Q1\n"
                "Healthcare and medical devices reported revenue of 115 Cr with steady European demand."
            ),
            "metadata": {
                "company": "TATAELXSI",
                "quarter": "Q1_FY27",
                "doc_type": "Investor_Presentation",
                "page_no": 7,
                "contains_table": False,
                "source_file": "TataElxsi_Q1.pdf",
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
    ]


@pytest.fixture
def setup_test_index(tmp_path, mock_financial_chunks):
    """Sets up an isolated ChromaDB and BM25 store for testing."""
    chroma_dir = tmp_path / "chroma"
    bm25_store_path = tmp_path / "bm25_store.pkl"

    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll_name = "test_financial_reports"

    # Reset any preexisting state
    reset_indices(client=client, collection_name=coll_name, bm25_store_path=bm25_store_path)

    # Ingest mock chunks
    res = ingest_chunks(
        chunks=mock_financial_chunks,
        client=client,
        collection_name=coll_name,
        bm25_store_path=bm25_store_path,
    )
    assert res["ingested_count"] == 8

    retriever = HybridRetriever(
        client=client,
        collection_name=coll_name,
        bm25_store_path=bm25_store_path,
        top_k=8,
    )
    reranker = Reranker(top_n=4)

    return {
        "retriever": retriever,
        "reranker": reranker,
        "client": client,
        "coll_name": coll_name,
        "bm25_store_path": bm25_store_path,
    }


def test_exact_financial_keyword_retrieval(setup_test_index):
    """Verify exact financial metric terms (ROCE 36.5%, Capex 200 Cr) hit BM25 and Dense search."""
    retriever = setup_test_index["retriever"]

    # Query 1: Exact metric 'ROCE 36.5%'
    query_roce = "What was the ROCE 36.5% reported in the quarter?"
    sparse_res = retriever.sparse_search(query=query_roce, top_k=5)
    assert len(sparse_res) >= 1
    # Top sparse result must be the exact Q2 chunk containing 36.5%
    assert sparse_res[0]["chunk_id"] == "TATAELXSI_Q2_FY27_p12_c01"
    assert "36.5%" in sparse_res[0]["text"]

    # Query 2: Exact Capex revision 'Capex 200 Cr'
    query_capex = "Capex guidance revised to 200 Cr"
    hybrid_res = retriever.retrieve(query=query_capex, top_k=5)
    assert len(hybrid_res) >= 1
    top_chunk_ids = [c["chunk_id"] for c in hybrid_res[:2]]
    assert "TATAELXSI_Q2_FY27_p15_c02" in top_chunk_ids


def test_metadata_filtering_isolates_quarter(setup_test_index):
    """Verify that metadata filter quarter='Q2_FY27' strictly isolates Q2 chunks."""
    retriever = setup_test_index["retriever"]

    query = "What was the revenue and EBITDA margin?"

    # Unfiltered retrieval should return chunks from both Q1 and Q2
    unfiltered = retriever.retrieve(query=query, top_k=8)
    quarters_unfiltered = {c["metadata"]["quarter"] for c in unfiltered}
    assert "Q1_FY27" in quarters_unfiltered
    assert "Q2_FY27" in quarters_unfiltered

    # Filtered retrieval strictly isolates Q2
    q2_only = retriever.retrieve(
        query=query, top_k=8, metadata_filter={"quarter": "Q2_FY27"}
    )
    assert len(q2_only) > 0
    for chunk in q2_only:
        assert chunk["metadata"]["quarter"] == "Q2_FY27", (
            f"Chunk {chunk['chunk_id']} from {chunk['metadata']['quarter']} leaked into Q2 filter!"
        )


def test_flashrank_reranking_speed_and_order(setup_test_index):
    """Verify FlashRank returns Top 4 chunks sorted by score in < 1 second on CPU."""
    retriever = setup_test_index["retriever"]
    reranker = setup_test_index["reranker"]

    query = "What was the EBITDA and revenue performance in Q2?"
    candidates = retriever.retrieve(query=query, top_k=8)
    assert len(candidates) >= 4

    # Measure CPU reranking latency
    start_time = time.perf_counter()
    golden_chunks = reranker.rerank(query=query, candidate_chunks=candidates, top_n=4)
    elapsed_time = time.perf_counter() - start_time

    # Must be under 1 second on CPU
    assert elapsed_time < 1.0, f"Reranking took {elapsed_time:.3f}s (exceeded 1.0s budget)"
    assert len(golden_chunks) == 4

    # Assert sorted in descending order of rerank_score
    scores = [c["rerank_score"] for c in golden_chunks]
    assert scores == sorted(scores, reverse=True), "Golden chunks must be sorted by rerank_score descending"

    # Assert rerank_score attached in metadata
    for gc in golden_chunks:
        assert "rerank_score" in gc
        assert "rerank_score" in gc["metadata"]
        assert isinstance(gc["rerank_score"], float)

    # Top chunk should be the Q2 financial statement table
    assert golden_chunks[0]["chunk_id"] == "TATAELXSI_Q2_FY27_p12_c01"
    assert golden_chunks[0]["metadata"]["contains_table"] is True


def test_reciprocal_rank_fusion_scoring(setup_test_index):
    """Verify RRF properly scores chunks present in both dense and sparse results higher."""
    retriever = setup_test_index["retriever"]

    query = "Transportation segment revenue and EV platform growth"
    dense_results = retriever.dense_search(query=query, top_k=5)
    sparse_results = retriever.sparse_search(query=query, top_k=5)

    fused = retriever.reciprocal_rank_fusion(
        dense_results=dense_results, sparse_results=sparse_results, top_candidates=5
    )

    assert len(fused) > 0
    # Chunks appearing in both lists should have both dense_rank and sparse_rank populated
    dual_matches = [c for c in fused if c["dense_rank"] is not None and c["sparse_rank"] is not None]
    if dual_matches:
        # A dual match has RRF score = 1/(k+dr) + 1/(k+sr), which exceeds a single match rank 1
        top_dual = dual_matches[0]
        assert top_dual["rrf_score"] > (1.0 / (retriever.rrf_k + 1.0))


def test_duplicate_upsert_idempotency(setup_test_index, mock_financial_chunks):
    """Verify re-indexing the same chunks does not cause duplicate entries."""
    client = setup_test_index["client"]
    coll_name = setup_test_index["coll_name"]
    bm25_store_path = setup_test_index["bm25_store_path"]

    # Re-ingest the exact same 8 chunks
    reingest_res = ingest_chunks(
        chunks=mock_financial_chunks,
        client=client,
        collection_name=coll_name,
        bm25_store_path=bm25_store_path,
    )

    assert reingest_res["total_chroma_count"] == 8
    assert reingest_res["total_bm25_count"] == 8
