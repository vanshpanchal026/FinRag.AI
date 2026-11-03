"""Unit tests for Lakhs vs Crores vs Millions Unit Normalizer.

Verifies:
1. Lakhs to Crores: 50,000 Lakhs == 500 Crores.
2. Crores to Lakhs: 12.5 Crores == 1,250 Lakhs.
3. Millions to Crores: 100 Million == 10 Crores.
4. Bidirectional conversion and UnitMatchResult model compliance.
"""

import pytest
from src.agents.math_ast_verifier import (
    UnitMatchResult,
    verify_unit_equivalence,
)


def test_lakhs_to_crores_conversion():
    """Verify that 50,000 Lakhs is correctly normalized as equivalent to 500 Crores."""
    chunk_text = "The company reported total revenue of 50,000 Lakhs in Q1 FY26."
    
    # Test with standard 3-arg signature
    trace = verify_unit_equivalence(
        claimed_token="500",
        answer_context="Total revenue was 500 Crores.",
        raw_chunks_text=chunk_text,
    )
    assert trace is not None
    assert trace.is_match is True
    assert trace.claimed_unit in ["crores", "cr"]
    assert trace.source_unit in ["lakhs", "lakh"]
    assert trace.normalized_claimed == 500 * 10_000_000
    assert trace.normalized_source == 50_000 * 100_000
    assert trace.normalized_claimed == trace.normalized_source

    # Test conversion to UnitMatchResult
    result: UnitMatchResult = trace.to_result()
    assert result.unit_match_pass is True
    assert result.normalized_claimed_value == result.source_chunk_value * 100_000


def test_crores_to_lakhs_conversion():
    """Verify that 12.5 Crores is correctly normalized as equivalent to 1,250 Lakhs."""
    chunk_text = "Capex allocation for the new facility is 1,250 Lakhs."
    
    trace = verify_unit_equivalence(
        claimed_token="12.5",
        answer_context="Planned capex is 12.5 Crores.",
        raw_chunks_text=chunk_text,
    )
    assert trace is not None
    assert trace.is_match is True
    assert trace.normalized_claimed == 12.5 * 10_000_000  # 125,000,000
    assert trace.normalized_source == 1_250 * 100_000     # 125,000,000
    assert trace.normalized_claimed == trace.normalized_source


def test_millions_to_crores_conversion():
    """Verify that 100 Million is correctly normalized as equivalent to 10 Crores."""
    chunk_text = "International subscription ARR reached 100 Million for the fiscal period."
    
    trace = verify_unit_equivalence(
        claimed_token="10",
        answer_context="ARR was 10 Crores.",
        raw_chunks_text=chunk_text,
    )
    assert trace is not None
    assert trace.is_match is True
    assert trace.normalized_claimed == 10 * 10_000_000   # 100,000,000
    assert trace.normalized_source == 100 * 1_000_000    # 100,000,000
    assert trace.normalized_claimed == trace.normalized_source


def test_direct_unit_token_invocation():
    """Verify 2-argument signature: verify_unit_equivalence(claimed_number, chunk_texts)."""
    chunks = [
        "Item 1: 50,000 Lakhs revenue recorded.",
        "Item 2: other disclosures.",
    ]
    trace = verify_unit_equivalence(
        claimed_token="500 Crores",
        answer_context=chunks,
    )
    assert trace is not None
    assert trace.is_match is True
    assert trace.claimed_value == 500.0
    assert trace.source_value == 50000.0
