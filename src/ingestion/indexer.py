"""Dual Indexer module for FinRAG-Eval (100% Local CPU).

Implements ChromaDB dense vector indexing (with BAAI/bge-small-en-v1.5)
and BM25Okapi sparse keyword indexing, saving serialized index state in
data/chroma_db/ and data/bm25_index/bm25_store.pkl.

Sources:
- ChromaDB: https://docs.trychroma.com/
- rank-bm25: https://github.com/dorianbrown/rank_bm25
- Techstack.md Section 3
- DesignDocument.md Section 1
"""

import os
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings

# Global cached singletons
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_CURRENT_MODEL_NAME: Optional[str] = None
_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None

# Collection name for financial report chunks
COLLECTION_NAME = "financial_reports"


def get_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Return cached SentenceTransformer model instance on CPU."""
    global _EMBEDDING_MODEL, _CURRENT_MODEL_NAME
    name = model_name or settings.embedding_model
    if _EMBEDDING_MODEL is None or _CURRENT_MODEL_NAME != name:
        # Avoid symlink warning on Windows if not in developer mode
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        _EMBEDDING_MODEL = SentenceTransformer(name, device="cpu")
        _CURRENT_MODEL_NAME = name
    return _EMBEDDING_MODEL


def get_chroma_client(
    persist_dir: Optional[Union[str, Path]] = None
) -> chromadb.PersistentClient:
    """Return persistent ChromaDB client pointing to specified directory."""
    global _CHROMA_CLIENT
    target_dir = Path(persist_dir) if persist_dir else settings.chroma_db
    target_dir.mkdir(parents=True, exist_ok=True)
    if _CHROMA_CLIENT is None or str(target_dir) != str(settings.chroma_db):
        client = chromadb.PersistentClient(
            path=str(target_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        if target_dir == settings.chroma_db:
            _CHROMA_CLIENT = client
        return client
    return _CHROMA_CLIENT


def get_chroma_collection(
    collection_name: str = COLLECTION_NAME,
    client: Optional[chromadb.PersistentClient] = None,
) -> chromadb.Collection:
    """Return ChromaDB collection initialized with cosine distance space."""
    c = client or get_chroma_client()
    return c.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# Standard English stop words to prevent BM25 score distortion on noise tokens
ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


def tokenize_for_bm25(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize financial text for BM25 indexing and querying.

    Preserves alphanumeric tokens, financial ratios, percentages,
    currency indicators, and accounting terms (e.g. 'q2_fy27', '+4.4%', 'ebitda'),
    while optionally removing non-informative English stop words.

    Args:
        text: Financial report text.
        remove_stopwords: If True, filters out standard English stop words.

    Returns:
        List of normalized string tokens.
    """
    if not text:
        return []
    # Match lowercased words, numbers, percentages, currency, and underscore-compounds
    pattern = re.compile(r"[a-zA-Z0-9%₹$]+(?:[.\-_][a-zA-Z0-9%₹$]+)*")
    tokens = pattern.findall(text.lower())
    clean_tokens = [t for t in tokens if len(t.strip()) > 0]
    if remove_stopwords:
        filtered = [t for t in clean_tokens if t not in ENGLISH_STOP_WORDS]
        return filtered if filtered else clean_tokens
    return clean_tokens


def ingest_chunks(
    chunks: List[Dict[str, Any]],
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    bm25_store_path: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Ingest chunk objects into both ChromaDB and BM25 index stores.

    Upserts into ChromaDB using BAAI/bge-small-en-v1.5 embeddings, avoiding duplicates.
    Builds/updates persistent BM25Okapi store in data/bm25_index/bm25_store.pkl.

    Args:
        chunks: List of chunk dictionaries conforming to Chunk Object Schema.
        client: Optional custom ChromaDB PersistentClient.
        collection_name: Target Chroma collection name.
        bm25_store_path: Path for serializing BM25 store.
        batch_size: Batch size for dense embedding computation.

    Returns:
        Dictionary summarizing ingested count and total index sizes.
    """
    if not chunks:
        return {
            "ingested_count": 0,
            "total_chroma_count": 0,
            "total_bm25_count": 0,
        }

    # Ensure directories exist
    settings.ensure_directories()
    store_file = (
        Path(bm25_store_path)
        if bm25_store_path
        else (settings.bm25_index / "bm25_store.pkl")
    )
    store_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Dense Indexing: ChromaDB
    collection = get_chroma_collection(collection_name=collection_name, client=client)
    embedder = get_embedding_model()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for c in chunks:
        chunk_id = str(c["chunk_id"])
        chunk_text = str(c["text"])
        raw_meta = dict(c.get("metadata", {}))

        # Flatten metadata ensuring primitive types accepted by ChromaDB
        flat_meta: Dict[str, Union[str, int, float, bool]] = {
            "chunk_id": chunk_id,
            "company": str(raw_meta.get("company", "")),
            "doc_type": str(raw_meta.get("doc_type", "")),
            "quarter": str(raw_meta.get("quarter", "")),
            "page_no": int(raw_meta.get("page_no", 1)),
            "contains_table": bool(raw_meta.get("contains_table", False)),
            "source_file": str(raw_meta.get("source_file", "")),
        }

        ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append(flat_meta)

    # Compute dense embeddings in batches with normalization
    embeddings = embedder.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # ChromaDB upsert avoids duplicate insertions
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 2. Sparse Indexing: BM25
    chunks_map: Dict[str, Dict[str, Any]] = {}

    if store_file.exists():
        try:
            with open(store_file, "rb") as f:
                existing_store = pickle.load(f)
                chunks_map = existing_store.get("chunks_map", {})
        except Exception:
            chunks_map = {}

    # Merge new chunks by chunk_id
    for c in chunks:
        chunks_map[c["chunk_id"]] = c

    # Rebuild tokenized corpus and BM25 index
    ordered_chunk_ids = list(chunks_map.keys())
    tokenized_corpus = [
        tokenize_for_bm25(chunks_map[cid]["text"]) for cid in ordered_chunk_ids
    ]
    bm25_model = BM25Okapi(tokenized_corpus)

    store_data = {
        "bm25": bm25_model,
        "chunk_ids": ordered_chunk_ids,
        "chunks_map": chunks_map,
    }

    with open(store_file, "wb") as f:
        pickle.dump(store_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "ingested_count": len(chunks),
        "total_chroma_count": collection.count(),
        "total_bm25_count": len(ordered_chunk_ids),
    }


def load_bm25_store(
    bm25_store_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Load serialized BM25 index, chunk IDs, and chunks map from disk."""
    store_file = (
        Path(bm25_store_path)
        if bm25_store_path
        else (settings.bm25_index / "bm25_store.pkl")
    )
    if not store_file.exists():
        return None

    try:
        with open(store_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def reset_indices(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    bm25_store_path: Optional[Union[str, Path]] = None,
) -> None:
    """Reset both ChromaDB collection and BM25 index store (used for test isolation)."""
    c = client or get_chroma_client()
    try:
        c.delete_collection(name=collection_name)
    except Exception:
        pass
    # Re-create empty collection
    c.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    store_file = (
        Path(bm25_store_path)
        if bm25_store_path
        else (settings.bm25_index / "bm25_store.pkl")
    )
    if store_file.exists():
        try:
            store_file.unlink()
        except Exception:
            pass
