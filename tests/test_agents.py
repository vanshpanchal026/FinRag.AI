"""Unit tests for FinRAG-Eval Multi-Agent Layer and Two-Tier Evaluation Engine.

Verifies:
1. Tier-1 Deterministic Regex catches hallucinated/unverified numbers (e.g. 999.99 Cr)
   and immediately sets numeric_match_pass = False.
2. Citation tag exclusion filter prevents page and chunk IDs from triggering false positives.
3. AnalystAgent properly formats Golden Chunks with explicit boundary headers.
4. EvalReport matches the exact Pydantic schema in docs/DesignDocument.md Section 3B.
5. Trader Presets (Red Flags, Guidance vs Actuals, Growth Triggers) execute correctly.

Sources:
- docs/DetectionLogic.md Section 2, 3, 4, 5
- docs/DesignDocument.md Section 3B, 4.2
"""

import pytest
from src.agents.analyst_agent import AnalystAgent, format_golden_chunks
from src.agents.eval_subagent import (
    EvalReport,
    EvalSubAgent,
    run_deterministic_tier1_check,
)
from src.presets.trader_workflows import (
    run_guidance_vs_actuals,
    run_growth_triggers_capex,
    run_red_flag_scanner,
)


@pytest.fixture
def sample_golden_chunks():
    """Golden Chunks reflecting Q2 financial results."""
    return [
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
    ]


def test_tier1_deterministic_numeric_pass(sample_golden_chunks):
    """Verify that a 100% grounded answer with true figures passes Tier-1 check."""
    grounded_answer = (
        "Tata Elxsi reported Revenue from Operations of 1,250.5 Cr [Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]. "
        "EBITDA reached 355.5 Cr while ROCE expanded to 36.5% [Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]. "
        "Management revised capex to 200 Cr [Source: TataElxsi_Q2_Concall.pdf | Page: 15 | Chunk: TATAELXSI_Q2_FY27_p15_c02]."
    )

    result = run_deterministic_tier1_check(
        generated_answer=grounded_answer,
        golden_chunks=sample_golden_chunks,
    )

    assert result["numeric_match_pass"] is True
    assert result["unverified_numbers"] == []
    # Verify numbers extracted from answer
    assert "1,250.5" in result["numbers_in_answer"]
    assert "355.5" in result["numbers_in_answer"]
    assert "36.5%" in result["numbers_in_answer"]
    assert "200" in result["numbers_in_answer"]


def test_tier1_injected_fake_number_caught(sample_golden_chunks):
    """Verify that an intentionally injected fake figure (e.g. 999.99 Cr) is immediately caught."""
    hallucinated_answer = (
        "Tata Elxsi reported Revenue of 1,250.5 Cr [Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]. "
        "However, net profit surged unexpectedly to 999.99 Cr [Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01] "
        "with an unverified EBITDA margin of 45.8%."
    )

    result = run_deterministic_tier1_check(
        generated_answer=hallucinated_answer,
        golden_chunks=sample_golden_chunks,
    )

    # Must fail deterministic numerical verification
    assert result["numeric_match_pass"] is False
    assert "999.99" in result["unverified_numbers"]
    assert "45.8%" in result["unverified_numbers"]


def test_citation_exclusion_filter_prevents_false_positives(sample_golden_chunks):
    """Verify that page numbers and chunk indices inside [Source: ...] brackets are stripped."""
    answer_with_citations = (
        "Operating ROCE was 36.5% [Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]."
    )

    result = run_deterministic_tier1_check(
        generated_answer=answer_with_citations,
        golden_chunks=sample_golden_chunks,
    )

    # Page number 12 and chunk index 01 must NOT appear in numbers_in_answer
    assert "12" not in result["numbers_in_answer"]
    assert "01" not in result["numbers_in_answer"]
    assert result["numbers_in_answer"] == ["36.5%"]
    assert result["numeric_match_pass"] is True


def test_analyst_agent_golden_chunk_formatting(sample_golden_chunks):
    """Verify explicit header formatting for Golden Chunks."""
    formatted = format_golden_chunks(sample_golden_chunks)

    assert "=== CHUNK ID: TATAELXSI_Q2_FY27_p12_c01 | SOURCE: TataElxsi_Q2.pdf | PAGE: 12 ===" in formatted
    assert "=== END OF CHUNK: TATAELXSI_Q2_FY27_p12_c01 ===" in formatted
    assert "=== CHUNK ID: TATAELXSI_Q2_FY27_p15_c02 | SOURCE: TataElxsi_Q2_Concall.pdf | PAGE: 15 ===" in formatted


def test_eval_subagent_schema_conformance(sample_golden_chunks):
    """Verify EvalSubAgent output strictly conforms to docs/DesignDocument.md Section 3B schema."""
    eval_subagent = EvalSubAgent()

    answer = "EBITDA was 355.5 Cr [Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]."

    report = eval_subagent.evaluate(
        user_query="What was EBITDA in Q2?",
        golden_chunks=sample_golden_chunks,
        generated_answer=answer,
        skip_llm_judge=True,
    )

    # Validate against Pydantic schema
    validated_model = EvalReport.model_validate(report)
    assert validated_model.deterministic_checks.numeric_match_pass is True
    assert validated_model.llm_judge_scores.faithfulness_score == 10
    assert validated_model.hallucination_flag is False


def test_trader_presets_mock_execution(sample_golden_chunks):
    """Verify that preset workflows construct prompts, execute retrieval, and produce reports."""
    # Mock analyst agent that returns a verified canned answer
    class MockAnalyst:
        def answer_query(self, query: str, golden_chunks):
            return {
                "query": query,
                "answer": "Revenue was 1,250.5 Cr [Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01].",
                "golden_chunks": golden_chunks,
            }

    # Mock retriever returning sample_golden_chunks
    class MockRetriever:
        def retrieve(self, query: str, top_k: int = 10, metadata_filter=None):
            return sample_golden_chunks

    class MockReranker:
        def rerank(self, query: str, candidate_chunks, top_n: int = 4):
            return candidate_chunks[:top_n]

    # Test Red Flag Scanner
    rf_res = run_red_flag_scanner(
        company="TATAELXSI",
        quarter="Q2_FY27",
        retriever=MockRetriever(),
        reranker=MockReranker(),
        analyst_agent=MockAnalyst(),
        skip_llm_judge=True,
    )
    assert rf_res["workflow"] == "red_flag_scanner"
    assert rf_res["eval_report"]["deterministic_checks"]["numeric_match_pass"] is True

    # Test Guidance vs Actuals
    gva_res = run_guidance_vs_actuals(
        company="TATAELXSI",
        q_prev="Q1_FY27",
        q_curr="Q2_FY27",
        retriever=MockRetriever(),
        reranker=MockReranker(),
        analyst_agent=MockAnalyst(),
        skip_llm_judge=True,
    )
    assert gva_res["workflow"] == "guidance_vs_actuals"
    assert gva_res["eval_report"]["deterministic_checks"]["numeric_match_pass"] is True

    # Test Growth Triggers & Capex
    gtc_res = run_growth_triggers_capex(
        company="TATAELXSI",
        quarter="Q2_FY27",
        retriever=MockRetriever(),
        reranker=MockReranker(),
        analyst_agent=MockAnalyst(),
        skip_llm_judge=True,
    )
    assert gtc_res["workflow"] == "growth_triggers_capex"
    assert gtc_res["eval_report"]["deterministic_checks"]["numeric_match_pass"] is True


def test_tier1_ast_math_recomputation_pass(sample_golden_chunks):
    """Verify that legitimate derived arithmetic (YoY growth %, bps margin compression) passes Tier-1."""
    # Chunk has 1,250.5 (Q2) and 1,180.0 (Q1). Relative growth = +5.97% (approx 5.9%)
    answer_with_derived_math = (
        "Revenue reached 1,250.5 Cr in Q2 compared to 1,180.0 Cr in Q1 (+5.9% QoQ) "
        "[Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]."
    )

    result = run_deterministic_tier1_check(
        generated_answer=answer_with_derived_math,
        golden_chunks=sample_golden_chunks,
    )

    assert result["numeric_match_pass"] is True
    assert len(result["ast_math_traces"]) >= 1
    assert result["unverified_numbers"] == []


def test_tier1_unit_conversion_pass():
    """Verify that Lakhs to Crores conversion passes Tier-1 with unit trace."""
    chunks = [
        {
            "chunk_id": "TEST_UNIT_p01_c01",
            "text": "Total reported income was 50,000 Lakhs for the period.",
            "metadata": {"source_file": "test.pdf", "page_no": 1},
        }
    ]
    answer = "Total income was 500 Cr [Source: test.pdf | Page: 1 | Chunk: TEST_UNIT_p01_c01]."

    result = run_deterministic_tier1_check(
        generated_answer=answer,
        golden_chunks=chunks,
    )

    assert result["numeric_match_pass"] is True
    assert len(result["unit_converted_matches"]) == 1
    assert result["unit_converted_matches"][0]["claimed_unit"] == "cr"
    assert result["unit_converted_matches"][0]["source_unit"] == "lakhs"
    assert result["unverified_numbers"] == []
