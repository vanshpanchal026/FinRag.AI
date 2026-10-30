"""Cross-Encoder Reranker module for FinRAG-Eval.

Uses FlashRank (ms-marco-MiniLM-L-12-v2) on CPU ONNX Runtime to rerank
RRF candidate chunks against the user's query with <200ms latency.
Returns the Top 4 Golden Chunks with attached rerank scores.

Sources:
- FlashRank: https://github.com/PrithivirajDamodaran/FlashRank
- Techstack.md Section 3
- DesignDocument.md Section 1
"""

from typing import Any, Dict, List, Optional
from flashrank import Ranker, RerankRequest

from src.config import settings

# Default reranking parameters
DEFAULT_TOP_N_RERANK = 4
DEFAULT_RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

# Global cached Ranker singleton
_RANKER_INSTANCE: Optional[Ranker] = None
_CURRENT_RANKER_NAME: Optional[str] = None


def get_flashrank_ranker(model_name: Optional[str] = None) -> Ranker:
    """Return cached FlashRank Ranker instance running on CPU ONNX runtime."""
    global _RANKER_INSTANCE, _CURRENT_RANKER_NAME
    target_model = model_name or settings.reranker_model or DEFAULT_RERANKER_MODEL
    if _RANKER_INSTANCE is None or _CURRENT_RANKER_NAME != target_model:
        _RANKER_INSTANCE = Ranker(model_name=target_model)
        _CURRENT_RANKER_NAME = target_model
    return _RANKER_INSTANCE


class Reranker:
    """FlashRank CPU-First Cross-Encoder Reranker."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_n: int = DEFAULT_TOP_N_RERANK,
    ):
        self.model_name = model_name or settings.reranker_model or DEFAULT_RERANKER_MODEL
        self.top_n = top_n
        self.ranker: Ranker = get_flashrank_ranker(self.model_name)

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks against the user query.

        Args:
            query: User's financial query string.
            candidate_chunks: List of chunk dictionaries from HybridRetriever.
            top_n: Number of golden chunks to return (default 4).

        Returns:
            Top N Golden Chunks sorted by rerank_score descending, with
            rerank_score attached to chunk and chunk['metadata'].
        """
        if not candidate_chunks:
            return []

        limit = top_n or self.top_n

        # Prepare passages in format expected by FlashRank: [{'id': ..., 'text': ...}]
        passages: List[Dict[str, Any]] = []
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        for c in candidate_chunks:
            cid = str(c["chunk_id"])
            chunk_lookup[cid] = c
            passages.append(
                {
                    "id": cid,
                    "text": str(c["text"]),
                    "meta": c.get("metadata", {}),
                }
            )

        # Execute reranking via FlashRank ONNX runtime
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_req)

        golden_chunks: List[Dict[str, Any]] = []

        for item in ranked_results[:limit]:
            cid = str(item["id"])
            original_chunk = chunk_lookup.get(cid, {})

            # Copy chunk to avoid mutating source in place unexpectedly
            golden_chunk = {
                "chunk_id": cid,
                "text": item.get("text", original_chunk.get("text", "")),
                "metadata": dict(original_chunk.get("metadata", {})),
            }

            # Preserve previous scores if present
            if "rrf_score" in original_chunk:
                golden_chunk["rrf_score"] = original_chunk["rrf_score"]
            if "dense_score" in original_chunk:
                golden_chunk["dense_score"] = original_chunk["dense_score"]
            if "bm25_score" in original_chunk:
                golden_chunk["bm25_score"] = original_chunk["bm25_score"]

            score = float(item.get("score", 0.0))
            golden_chunk["rerank_score"] = score
            golden_chunk["metadata"]["rerank_score"] = score

            golden_chunks.append(golden_chunk)

        return golden_chunks
