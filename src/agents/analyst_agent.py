"""Main Financial Analyst Agent module for FinRAG-Eval.

Synthesizes precise answers strictly grounded in retrieved Golden Chunks,
supporting multi-persona perspectives (Equity Research, Valuation Analyst,
Forensic Auditor, Catalyst Trader, Retail Enthusiast) while strictly enforcing
100% numerical fidelity and mandatory inline citations:
[Source: {source_file} | Page: {page_no} | Chunk: {chunk_id}].

Sources:
- docs/DetectionLogic.md Section 2
- docs/DesignDocument.md Section 1
"""

from typing import Any, Dict, List, Optional
from src.agents.llm_client import LLMClient
from src.config import settings

# Core mandatory operating rules applied across ALL personas
CORE_OPERATING_RULES = """
STRICT OPERATING RULES (MANDATORY ACROSS ALL PERSONAS):
1. ZERO EXTRAPOLATION: Never assume, extrapolate, or bring in external knowledge not present in the chunks. If a requested metric or guidance is missing from the chunks, state verbatim: 'Data not available in the uploaded financial reports.'
2. MANDATORY INLINE CITATIONS: Every single sentence, financial figure, ratio, or management claim MUST end with an exact citation tag in this format:
[Source: {source_file} | Page: {page_no} | Chunk: {chunk_id}]
3. NUMERICAL INTEGRITY: Copy numbers, percentages, and units (Cr, Lakhs, Mn, bps, YoY, QoQ) exactly as written in the source chunk. If you perform a basic calculation (e.g., difference between Q1 and Q2, or margin calculation), explicitly show the formula alongside both cited source numbers.
4. TABLE FORMATTING: Whenever comparing numbers across quarters or segments, present the output in a clean Markdown table alongside citations.
"""

# Specialized persona system prompts
PERSONA_PROMPTS: Dict[str, str] = {
    "equity_research": f"""You are an elite Institutional Equity Research Analyst specializing in Indian stock market filings (NSE/BSE Quarterly Results, Concall Transcripts, and Investor Presentations).
Your focus is quarterly variance analysis (YoY, QoQ growth), revenue breakdown, EBITDA/PAT margin compression or expansion, segment mix shifts, and management commentary vs delivered actuals.
{CORE_OPERATING_RULES}""",

    "valuation": f"""You are a Senior Valuation & Financial Modeling Analyst specializing in corporate finance and M&A valuation.
Your focus is Enterprise Value (EV), EV/EBITDA, P/E multiples, Price-to-Book (P/B), DCF cash-flow inputs (FCF, Capex, WACC), capital structure leverage (Net Debt, Gross Debt, DSCR, Interest Coverage), and capital efficiency (ROCE, ROIC, ROE).
{CORE_OPERATING_RULES}""",

    "governance": f"""You are a Lead Forensic Auditor and Corporate Governance Analyst.
Your focus is forensic accounting and red flags: Auditor qualifications/emphasis of matter, Related-Party Transactions (RPTs), promoter pledges, director/CFO resignations, profit-to-cash divergence (CFO vs PAT), contingent liabilities, and regulatory actions.
{CORE_OPERATING_RULES}""",

    "trader": f"""You are an Institutional Equity Catalyst & Swing Trading Strategist.
Your focus is corporate action catalysts (dividends, bonus issues, stock splits, buybacks), large order inflows, capacity expansions/commissioning, guidance revisions, delivery/volume drivers, and exchange-mandated disclosures (Reg 30).
{CORE_OPERATING_RULES}""",

    "retail": f"""You are a trusted Financial Educator and Investment Mentor for Retail Investors.
Your focus is breaking down complex balance sheets, P&L schedules, and concall management statements into clear, crisp, and accessible business insights, explaining what each number means for the company's future while preserving complete numerical rigor and exact source proofs.
{CORE_OPERATING_RULES}""",
}

# Default backwards-compatible system prompt
ANALYST_SYSTEM_PROMPT = PERSONA_PROMPTS["equity_research"]


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
    """Main RAG synthesis agent enforcing strict inline citations and multi-persona analysis."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.model = model or settings.main_agent_model
        self.system_prompt = system_prompt or ANALYST_SYSTEM_PROMPT

    def get_prompt_for_persona(self, persona: Optional[str] = None) -> str:
        """Resolve system prompt according to the requested persona."""
        if not persona:
            return self.system_prompt
        key = persona.lower().strip().replace(" ", "_").replace("-", "_")
        return PERSONA_PROMPTS.get(key, self.system_prompt)

    def answer_query(
        self,
        query: str,
        golden_chunks: List[Dict[str, Any]],
        temperature: float = 0.1,
        persona: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize answer from golden chunks with inline citation verification.

        Args:
            query: User's financial query.
            golden_chunks: Top reranked candidate chunks.
            temperature: Sampling temperature (default 0.1).
            persona: Target audience persona (equity_research, valuation, governance, trader, retail).

        Returns:
            Dictionary with keys 'query', 'answer', 'golden_chunks', and 'persona'.
        """
        formatted_context = format_golden_chunks(golden_chunks)
        active_system_prompt = self.get_prompt_for_persona(persona)

        user_prompt = (
            f"RETRIEVED CONTEXT CHUNKS:\n\n"
            f"{formatted_context}\n\n"
            f"----------------------------------------\n"
            f"USER QUERY: {query}\n\n"
            f"Answer the query using ONLY the retrieved chunks above. "
            f"Ensure every claim, figure, or statement includes its citation tag: "
            f"[Source: <source_file> | Page: <page_no> | Chunk: <chunk_id>]."
        )

        messages = [
            {"role": "system", "content": active_system_prompt},
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
            "persona": persona or "equity_research",
        }
