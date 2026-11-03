"""Hybrid Retriever module for FinRAG-Eval.

Combines ChromaDB dense vector search (semantic similarity) and BM25Okapi
sparse keyword search (exact ticker and accounting metric matching) using
Reciprocal Rank Fusion (RRF). Supports metadata pre-filtering by company,
quarter, and doc_type.

Sources:
- RRF Paper: Cormack, Clarke, and Buettcher (SIGIR 2009)
- Techstack.md Section 3
- DesignDocument.md Section 1
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.indexer import (
    COLLECTION_NAME,
    get_chroma_client,
    get_chroma_collection,
    get_embedding_model,
    load_bm25_store,
    tokenize_for_bm25,
)
from src.retrieval.domain_taxonomy import expand_financial_query

# Default retrieval parameters
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60


def build_chroma_filter(
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Convert a key-value dictionary into a valid ChromaDB where clause.

    Supports single conditions: {"quarter": "Q2_FY27"}
    Supports compound conditions: {"$and": [{"company": "TATAELXSI"}, ...]}
    """
    if not metadata_filter:
        return None

    clean_items = {k: v for k, v in metadata_filter.items() if v is not None}
    if not clean_items:
        return None

    if len(clean_items) == 1:
        k, v = next(iter(clean_items.items()))
        return {k: v}

    return {"$and": [{k: v} for k, v in clean_items.items()]}


class HybridRetriever:
    """Hybrid Retriever combining Dense (ChromaDB) and Sparse (BM25) search."""

    def __init__(
        self,
        client: Optional[chromadb.PersistentClient] = None,
        collection_name: str = COLLECTION_NAME,
        bm25_store_path: Optional[Union[str, Path]] = None,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.client = client or get_chroma_client()
        self.collection_name = collection_name
        self.bm25_store_path = (
            Path(bm25_store_path)
            if bm25_store_path
            else (settings.bm25_index / "bm25_store.pkl")
        )
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.embedder: SentenceTransformer = get_embedding_model()

    def get_collection(self) -> chromadb.Collection:
        """Retrieve ChromaDB collection."""
        return get_chroma_collection(
            collection_name=self.collection_name, client=self.client
        )

    def dense_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB collection using dense embeddings with metadata filtering.

        Args:
            query: User's financial query.
            top_k: Number of dense results to retrieve (default 10).
            metadata_filter: Optional dictionary (e.g. {'quarter': 'Q2_FY27'}).

        Returns:
            List of candidate chunk dictionaries with dense_distance and dense_score.
        """
        k = top_k or self.top_k
        collection = self.get_collection()
        total_docs = collection.count()
        if total_docs == 0:
            return []

        where_clause = build_chroma_filter(metadata_filter)

        query_emb = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        try:
            results = collection.query(
                query_embeddings=query_emb,
                n_results=min(k, total_docs),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # If where filter matches 0 documents
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        dense_chunks: List[Dict[str, Any]] = []
        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = (
            results["distances"][0] if results["distances"] else [0.0] * len(ids)
        )

        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, distances), start=1
        ):
            # Cosine similarity = 1 - cosine distance
            score = max(0.0, 1.0 - float(dist))
            dense_chunks.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": dict(meta),
                    "dense_distance": float(dist),
                    "dense_score": score,
                    "dense_rank": rank,
                }
            )

        return dense_chunks

    def sparse_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        additional_tokens: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query BM25Okapi index for exact keyword matching with metadata filtering.

        Args:
            query: User's financial query.
            top_k: Number of sparse results to retrieve (default 10).
            metadata_filter: Optional dictionary (e.g. {'quarter': 'Q2_FY27'}).
            additional_tokens: Optional list of expanded financial keywords.

        Returns:
            List of candidate chunk dictionaries with bm25_score and sparse_rank.
        """
        k = top_k or self.top_k
        store = load_bm25_store(self.bm25_store_path)
        if not store or "bm25" not in store or not store["chunk_ids"]:
            return []

        bm25_model = store["bm25"]
        chunk_ids = store["chunk_ids"]
        chunks_map = store["chunks_map"]

        tokens = tokenize_for_bm25(query)
        if additional_tokens:
            extra = [t.lower() for t in additional_tokens if t and len(t) > 1]
            tokens = list(dict.fromkeys(tokens + extra))

        if not tokens:
            return []

        scores = bm25_model.get_scores(tokens)

        # Pair scores with chunk objects and apply metadata filter
        candidates: List[Dict[str, Any]] = []
        clean_filter = (
            {fk: fv for fk, fv in metadata_filter.items() if fv is not None}
            if metadata_filter
            else None
        )

        for cid, score in zip(chunk_ids, scores):
            if score <= 0:
                continue
            chunk_obj = chunks_map.get(cid)
            if not chunk_obj:
                continue

            # Check metadata filter match
            if clean_filter:
                chunk_meta = chunk_obj.get("metadata", {})
                matches = all(
                    str(chunk_meta.get(fk, "")) == str(fv)
                    for fk, fv in clean_filter.items()
                )
                if not matches:
                    continue

            candidates.append(
                {
                    "chunk_id": cid,
                    "text": chunk_obj["text"],
                    "metadata": dict(chunk_obj.get("metadata", {})),
                    "bm25_score": float(score),
                }
            )

        # Sort by BM25 score descending
        candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
        top_candidates = candidates[:k]

        for rank, c in enumerate(top_candidates, start=1):
            c["sparse_rank"] = rank

        return top_candidates

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Combine dense and sparse candidate rankings using Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_Score(d) = sum(1 / (k + rank_m(d))) for m in [Dense, Sparse]

        Args:
            dense_results: Ranked list from dense ChromaDB search.
            sparse_results: Ranked list from sparse BM25 search.
            top_candidates: Maximum merged candidates to return.

        Returns:
            Deduplicated list of chunks sorted by RRF score descending.
        """
        k = self.rrf_k
        limit = top_candidates or self.top_k
        merged: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for item in dense_results:
            cid = item["chunk_id"]
            dense_rank = item["dense_rank"]
            dense_rrf = 1.0 / (k + dense_rank)

            merged[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "metadata": dict(item["metadata"]),
                "dense_rank": dense_rank,
                "sparse_rank": None,
                "dense_score": item.get("dense_score", 0.0),
                "bm25_score": 0.0,
                "rrf_score": dense_rrf,
            }

        # Process sparse results
        for item in sparse_results:
            cid = item["chunk_id"]
            sparse_rank = item["sparse_rank"]
            sparse_rrf = 1.0 / (k + sparse_rank)

            if cid in merged:
                merged[cid]["sparse_rank"] = sparse_rank
                merged[cid]["bm25_score"] = item.get("bm25_score", 0.0)
                merged[cid]["rrf_score"] += sparse_rrf
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "text": item["text"],
                    "metadata": dict(item["metadata"]),
                    "dense_rank": None,
                    "sparse_rank": sparse_rank,
                    "dense_score": 0.0,
                    "bm25_score": item.get("bm25_score", 0.0),
                    "rrf_score": sparse_rrf,
                }

        # Sort by RRF score descending
        fused_list = sorted(
            merged.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        return fused_list[:limit]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        expand_acronyms: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute full hybrid retrieval pipeline: Dense + Sparse -> RRF.

        Args:
            query: Financial query string.
            top_k: Number of final fused candidates to return (default 10).
            metadata_filter: Optional dict to pre-filter chunks (e.g. {'quarter': 'Q2_FY27'}).
            expand_acronyms: Whether to apply financial domain query expansion.

        Returns:
            List of Top K candidate chunks with RRF scores.
        """
        k = top_k or self.top_k

        if expand_acronyms:
            expanded_q, sparse_kw, _ = expand_financial_query(query)
            dense_results = self.dense_search(
                query=expanded_q, top_k=k, metadata_filter=metadata_filter
            )
            sparse_results = self.sparse_search(
                query=query,
                top_k=k,
                metadata_filter=metadata_filter,
                additional_tokens=sparse_kw,
            )
        else:
            dense_results = self.dense_search(
                query=query, top_k=k, metadata_filter=metadata_filter
            )
            sparse_results = self.sparse_search(
                query=query, top_k=k, metadata_filter=metadata_filter
            )

        return self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_candidates=k,
        )
