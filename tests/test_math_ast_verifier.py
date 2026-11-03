"""Tests for Deterministic Python AST Mathematical Recomputation and Unit Normalizer."""

import pytest
from src.agents.math_ast_verifier import (
    clean_indian_number,
    parse_and_recompute_ast,
    check_derived_arithmetic,
    verify_unit_equivalence,
    are_operands_grounded,
)


def test_clean_indian_number():
    """Verify Indian comma formatting is cleanly stripped without breaking AST."""
    assert clean_indian_number("6,58,240.00") == "658240.00"
    assert clean_indian_number("1,250.50") == "1250.50"
    assert clean_indian_number("36.5%") == "36.5"
    assert clean_indian_number("  50,000  ") == "50000"


def test_parse_and_recompute_ast_safe():
    """Verify safe AST arithmetic evaluation produces deterministic results."""
    expr = "((1250.5 - 1180.0) / 1180.0) * 100"
    result, operands = parse_and_recompute_ast(expr)
    assert round(result, 2) == 5.97
    assert operands == [1250.5, 1180.0, 1180.0, 100.0]

    expr_bps = "(36.5 - 34.2) * 100"
    result_bps, _ = parse_and_recompute_ast(expr_bps)
    assert round(result_bps, 1) == 230.0


def test_check_derived_arithmetic_growth_and_bps():
    """Verify derived QoQ growth and bps margin compression are verified against corpus numbers."""
    corpus_numbers = [1250.5, 1180.0, 355.5, 330.1]
    
    # 1. QoQ relative growth
    trace_growth = check_derived_arithmetic(
        claimed_num_str="5.97",
        corpus_numbers=corpus_numbers,
    )
    assert trace_growth is not None
    assert trace_growth.is_valid is True
    assert trace_growth.operands_grounded is True
    assert "1250.5" in trace_growth.expression
    assert "1180.0" in trace_growth.expression

    # 2. Basis points difference
    roce_corpus = [34.2, 36.5]
    trace_bps = check_derived_arithmetic(
        claimed_num_str="230",
        corpus_numbers=roce_corpus,
    )
    assert trace_bps is not None
    assert trace_bps.is_valid is True
    assert trace_bps.operator_symbol == "bps margin diff"


def test_check_derived_arithmetic_ungrounded_rejected():
    """Verify that a derived calculation with fake/unretrieved numbers is not approved."""
    corpus_numbers = [100.0, 200.0]
    # Claim 45.8% or 999.9 which cannot be derived from [100.0, 200.0]
    trace = check_derived_arithmetic(
        claimed_num_str="45.8",
        corpus_numbers=corpus_numbers,
    )
    assert trace is None


def test_verify_unit_equivalence_lakhs_to_crores():
    """Verify unit equivalence between Lakhs and Crores."""
    # Chunk has "50,000 Lakhs", answer says "500 Cr"
    raw_chunk = "The company reported total revenue of 50,000 Lakhs in Q1 FY26."
    answer_context = "Total revenue was 500 Cr in Q1 FY26."

    trace = verify_unit_equivalence(
        claimed_token="500",
        answer_context=answer_context,
        raw_chunks_text=raw_chunk,
    )
    assert trace is not None
    assert trace.is_match is True
    assert trace.claimed_unit == "cr"
    assert trace.source_unit == "lakhs"
    assert trace.normalized_claimed == trace.normalized_source


def test_verify_unit_equivalence_header_unit():
    """Verify unit equivalence when chunk contains table header '(Rs. in Lakhs)'."""
    raw_chunk = "(Rs. in Lakhs)\nParticulars | Amount\nRevenue | 125,050"
    answer_context = "Reported revenue reached 1,250.50 Crores."

    trace = verify_unit_equivalence(
        claimed_token="1,250.50",
        answer_context=answer_context,
        raw_chunks_text=raw_chunk,
    )
    assert trace is not None
    assert trace.is_match is True
    assert trace.claimed_unit in ["crores", "cr"]


def test_verify_ast_arithmetic_end_to_end():
    """Verify that verify_ast_arithmetic extracts and verifies calculations."""
    from src.agents.math_ast_verifier import verify_ast_arithmetic

    chunk_numbers = [1250.5, 1180.0, 34.2, 36.5]
    answer_text = (
        "Revenue was 1250.5 vs 1180.0 representing +5.9% QoQ growth. "
        "ROCE expanded by 230 bps."
    )
    results = verify_ast_arithmetic(answer_text, chunk_numbers)
    assert len(results) >= 2
    for r in results:
        assert r.math_valid is True
        assert r.operands_grounded is True


def test_ast_division_by_zero_and_syntax_safety():
    """Verify safe AST evaluation handles division by zero and invalid syntax gracefully."""
    with pytest.raises(ZeroDivisionError):
        parse_and_recompute_ast("100 / 0")

    with pytest.raises(SyntaxError):
        parse_and_recompute_ast("100 +* 50")

    # Non-arithmetic function calls or dangerous code must fail
    with pytest.raises(ValueError):
        parse_and_recompute_ast("__import__('os').system('ls')")

