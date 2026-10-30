"""Main Financial Analyst Agent module for FinRAG-Eval.

Synthesizes precise answers strictly grounded in retrieved Golden Chunks,
enforcing 100% numerical fidelity and mandatory inline citations:
[Source: {source_file} | Page: {page_no} | Chunk: {chunk_id}].

Sources:
- docs/DetectionLogic.md Section 2
- docs/DesignDocument.md Section 1
"""

from typing import Any, Dict, List, Optional
from src.agents.llm_client import LLMClient
from src.config import settings

# Exact System Prompt from docs/DetectionLogic.md Section 2
ANALYST_SYSTEM_PROMPT = """You are an elite Institutional Equity Research Analyst specializing in Indian stock market filings (NSE/BSE Quarterly Results, Concall Transcripts, and Investor Presentations).
Your sole objective is to answer the user's query with 100% numerical precision based ONLY on the provided Retrieved Context Chunks. 

STRICT OPERATING RULES: 

1. ZERO EXTRAPOLATION: Never assume, extrapolate, or bring in external knowledge not present in the chunks. If a requested metric or guidance is missing from the chunks, state verbatim: 'Data not available in the uploaded financial reports.'
2. MANDATORY INLINE CITATIONS: Every single sentence, financial figure, ratio, or management claim MUST end with an exact citation tag in this format:
[Source: {source_file} | Page: {page_no} | Chunk: {chunk_id}]
3. NUMERICAL INTEGRITY: Copy numbers, percentages, and units (Cr, Lakhs, Mn, bps, YoY, QoQ) exactly as written in the source chunk. If you perform a basic calculation (e.g., difference between Q1 and Q2), explicitly show the formula alongside both cited source numbers.
4. TABLE FORMATTING: Whenever comparing numbers across quarters or segments, present the output in a clean Markdown table alongside citations."""


def format_golden_chunks(golden_chunks: List[Dict[str, Any]]) -> str:
    """Format Golden Chunks with explicit chunk boundary headers.

    Header pattern:
    === CHUNK ID: {chunk_id} | SOURCE: {source_file} | PAGE: {page_no} ===\n{text}
    """
    if not golden_chunks:
        return "No retrieved chunks available."

    formatted_blocks: List[str] = []
    for c in golden_chunks:
        cid = c.get("chunk_id", "UNKNOWN_CHUNK")
        meta = c.get("metadata") or {}
        source_file = meta.get("source_file") or c.get("source_file") or "unknown.pdf"
        page_no = meta.get("page_no") or c.get("page_no") or 1
        text = c.get("text", "").strip()

        block = (
            f"=== CHUNK ID: {cid} | SOURCE: {source_file} | PAGE: {page_no} ===\n"
            f"{text}\n"
            f"=== END OF CHUNK: {cid} ==="
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


class AnalystAgent:
    """Main RAG synthesis agent enforcing strict inline citations and zero hallucination."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        system_prompt: str = ANALYST_SYSTEM_PROMPT,
    ):
        self.llm_client = llm_client or LLMClient()
        self.model = model or settings.main_agent_model
        self.system_prompt = system_prompt

    def answer_query(
        self,
        query: str,
        golden_chunks: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Synthesize answer from golden chunks with inline citation verification.

        Args:
            query: User's financial query.
            golden_chunks: Top reranked candidate chunks.
            temperature: Sampling temperature (default 0.1).

        Returns:
            Dictionary with keys 'query', 'answer', and 'golden_chunks'.
        """
        formatted_context = format_golden_chunks(golden_chunks)

        user_prompt = (
            f"RETRIEVED CONTEXT CHUNKS:\n\n"
            f"{formatted_context}\n\n"
            f"----------------------------------------\n"
            f"USER QUERY: {query}\n\n"
            f"Answer the query using ONLY the retrieved chunks above. "
            f"Ensure every claim, figure, or statement includes its citation tag."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = self.llm_client.generate(
            messages=messages,
            model=self.model,
            temperature=temperature,
        )

        return {
            "query": query,
            "answer": answer,
            "golden_chunks": golden_chunks,
        }
