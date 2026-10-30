"""FastAPI Backend Server for FinRAG-Eval.

Provides RESTful endpoints for document ingestion, hybrid querying with
two-tier faithfulness evaluation, and trader preset workflows. Can be
consumed by the Streamlit Workbench or external React dashboards.

Sources:
- docs/Techstack.md Section 1
- docs/DesignDocument.md Section 1 & 2
"""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.analyst_agent import AnalystAgent
from src.agents.eval_subagent import EvalSubAgent
from src.config import settings
from src.ingestion.indexer import get_chroma_collection, ingest_chunks
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.smart_chunker import chunk_document
from src.presets.trader_workflows import (
    run_growth_triggers_capex,
    run_guidance_vs_actuals,
    run_red_flag_scanner,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

# Initialize FastAPI application
app = FastAPI(
    title="FinRAG-Eval API",
    description="Zero-Hallucination Financial Report & Concall Q&A Engine",
    version="1.0.0",
)

# Enable CORS for local React/Streamlit clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines
retriever = HybridRetriever()
reranker = Reranker()
analyst_agent = AnalystAgent()
eval_subagent = EvalSubAgent()


class QueryRequest(BaseModel):
    """Payload for financial report Q&A query."""

    query: str = Field(..., description="Financial question to answer")
    company: Optional[str] = Field(None, description="Company filter (e.g., TATAELXSI)")
    quarter: Optional[str] = Field(None, description="Quarter filter (e.g., Q2_FY27)")
    doc_type: Optional[str] = Field(None, description="Document type filter")
    top_k: int = Field(default=10, description="Dense/Sparse retrieval candidates")
    top_n: int = Field(default=4, description="FlashRank Golden Chunks")


class PresetRequest(BaseModel):
    """Payload for 1-click Trader Preset workflows."""

    company: str = Field(..., description="Target company ticker")
    quarter: Optional[str] = Field(None, description="Quarter (required for red flags & growth triggers)")
    q_prev: Optional[str] = Field(None, description="Previous quarter (for guidance vs actuals)")
    q_curr: Optional[str] = Field(None, description="Current quarter (for guidance vs actuals)")


@app.get("/")
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint returning system status and indexed document count."""
    settings.ensure_directories()
    collection = get_chroma_collection()
    return {
        "status": "healthy",
        "engine": "FinRAG-Eval",
        "version": "1.0.0",
        "collection_name": collection.name,
        "indexed_chunks_count": collection.count(),
        "models": {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "analyst_llm": settings.main_agent_model,
            "eval_llm": settings.eval_agent_model,
        },
    }


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    company: str = Form(...),
    quarter: str = Form(...),
    doc_type: str = Form(...),
) -> Dict[str, Any]:
    """Upload PDF, parse into Markdown, smart chunk, and index into ChromaDB + BM25."""
    settings.ensure_directories()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to raw_pdfs directory
    dest_path = settings.raw_pdfs / file.filename
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {e}"
        )

    # 1. Parse PDF page-by-page
    try:
        pages = parse_pdf(dest_path, save_debug_md=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")

    # 2. Smart Chunking (Table & Concall preservation)
    try:
        chunks = chunk_document(
            pages=pages,
            company=company,
            quarter=quarter,
            doc_type=doc_type,
            source_file=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart chunking error: {e}")

    # 3. Dual Indexing (ChromaDB + BM25)
    try:
        index_stats = ingest_chunks(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {e}")

    tables_count = sum(1 for c in chunks if c["metadata"].get("contains_table", False))

    return {
        "status": "success",
        "message": f"Successfully ingested {file.filename}",
        "source_file": file.filename,
        "company": company.strip().upper(),
        "quarter": quarter.strip(),
        "doc_type": doc_type.strip(),
        "pages_count": len(pages),
        "chunks_count": len(chunks),
        "tables_count": tables_count,
        "index_stats": index_stats,
    }


@app.post("/api/query")
def query_financial_reports(req: QueryRequest) -> Dict[str, Any]:
    """Execute Hybrid Retrieval -> FlashRank -> Analyst Agent -> Two-Tier Eval."""
    metadata_filter: Dict[str, Any] = {}
    if req.company:
        metadata_filter["company"] = req.company.strip().upper()
    if req.quarter:
        metadata_filter["quarter"] = req.quarter.strip()
    if req.doc_type:
        metadata_filter["doc_type"] = req.doc_type.strip()

    # 1. Hybrid Retrieval (Dense + Sparse RRF)
    candidates = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        metadata_filter=metadata_filter if metadata_filter else None,
    )

    if not candidates:
        return {
            "query": req.query,
            "answer": "Insufficient data in uploaded financial reports. No matching chunks found for the given query and filters.",
            "golden_chunks": [],
            "eval_report": {
                "query_id": "none",
                "deterministic_checks": {
                    "numbers_in_answer": [],
                    "unverified_numbers": [],
                    "numeric_match_pass": True,
                },
                "llm_judge_scores": {
                    "faithfulness_score": 10,
                    "citation_accuracy_score": 10,
                    "answer_relevance_score": 10,
                    "reasoning": "Correctly acknowledged lack of data without hallucinating.",
                },
                "hallucination_flag": False,
            },
            "metadata_filter": metadata_filter,
        }

    # 2. FlashRank CPU Reranking
    golden_chunks = reranker.rerank(
        query=req.query,
        candidate_chunks=candidates,
        top_n=req.top_n,
    )

    # 3. Main Analyst Synthesis
    synthesis = analyst_agent.answer_query(
        query=req.query,
        golden_chunks=golden_chunks,
    )

    # 4. Two-Tier Evaluation Sub-Agent
    eval_report = eval_subagent.evaluate(
        user_query=req.query,
        golden_chunks=golden_chunks,
        generated_answer=synthesis["answer"],
    )

    return {
        "query": req.query,
        "answer": synthesis["answer"],
        "golden_chunks": golden_chunks,
        "eval_report": eval_report,
        "metadata_filter": metadata_filter,
    }


@app.post("/api/preset/{preset_name}")
def execute_preset_workflow(
    preset_name: str,
    req: PresetRequest,
) -> Dict[str, Any]:
    """Execute one of the 3 Trader Preset Workflows."""
    clean_preset = preset_name.strip().lower()

    if clean_preset == "red_flag_scanner":
        if not req.quarter:
            raise HTTPException(
                status_code=400,
                detail="Quarter parameter required for red_flag_scanner.",
            )
        return run_red_flag_scanner(
            company=req.company,
            quarter=req.quarter,
            retriever=retriever,
            reranker=reranker,
            analyst_agent=analyst_agent,
            eval_subagent=eval_subagent,
        )

    elif clean_preset == "guidance_vs_actuals":
        if not req.q_prev or not req.q_curr:
            raise HTTPException(
                status_code=400,
                detail="Both q_prev and q_curr required for guidance_vs_actuals.",
            )
        return run_guidance_vs_actuals(
            company=req.company,
            q_prev=req.q_prev,
            q_curr=req.q_curr,
            retriever=retriever,
            reranker=reranker,
            analyst_agent=analyst_agent,
            eval_subagent=eval_subagent,
        )

    elif clean_preset == "growth_triggers_capex":
        if not req.quarter:
            raise HTTPException(
                status_code=400,
                detail="Quarter parameter required for growth_triggers_capex.",
            )
        return run_growth_triggers_capex(
            company=req.company,
            quarter=req.quarter,
            retriever=retriever,
            reranker=reranker,
            analyst_agent=analyst_agent,
            eval_subagent=eval_subagent,
        )

    else:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown preset '{preset_name}'. Options: 'red_flag_scanner', 'guidance_vs_actuals', 'growth_triggers_capex'.",
        )
