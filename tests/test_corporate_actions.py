"""Tests for Corporate Actions Ingestion and Cross-Validation module."""

import pytest
from src.ingestion.corporate_actions import CorporateActionsManager


def test_symbol_resolution():
    """Verify ticker aliases map to standard exchange ticker."""
    mgr = CorporateActionsManager()
    assert mgr.resolve_symbol("RIL") == "RELIANCE"
    assert mgr.resolve_symbol("ril") == "RELIANCE"
    assert mgr.resolve_symbol("MINDTREE") == "LTIM"
    assert mgr.resolve_symbol("TATAELXSI") == "TATAELXSI"


def test_get_corporate_actions_offline_benchmark():
    """Verify offline curated corporate actions return correct structured records."""
    mgr = CorporateActionsManager()
    tata_actions = mgr.get_corporate_actions("TATAELXSI")
    assert len(tata_actions) >= 2
    assert tata_actions[0]["type"] == "dividend"
    assert tata_actions[0]["cash_amount"] == 70.0

    dixon_actions = mgr.get_corporate_actions("DIXON")
    assert any(a["type"] == "split" for a in dixon_actions)

    ril_actions = mgr.get_corporate_actions("RELIANCE")
    assert any(a["type"] == "bonus" and a["ratio_num"] == 1 for a in ril_actions)


def test_convert_to_chunks():
    """Verify corporate actions are transformed into valid RAG Golden Chunks."""
    mgr = CorporateActionsManager()
    chunks = mgr.convert_to_chunks("TATAELXSI")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["metadata"]["doc_type"] == "Corporate_Action"
    assert chunk["metadata"]["company"] == "TATAELXSI"
    assert "₹70" in chunk["text"]
    assert "TATAELXSI_CORP_ACTIONS_c01" in chunk["chunk_id"]


def test_verify_action_claim():
    """Verify cross-validation of dividends and bonus claims against exchange records."""
    mgr = CorporateActionsManager()
    # Test Tata Elxsi ₹70 dividend
    act = mgr.verify_action_claim("TATAELXSI", "dividend", 70.0)
    assert act is not None
    assert act["cash_amount"] == 70.0

    # Test non-existent dividend ₹999.0
    fake_act = mgr.verify_action_claim("TATAELXSI", "dividend", 999.0)
    assert fake_act is None

    # Test Reliance 1:1 bonus
    bonus_act = mgr.verify_action_claim("RELIANCE", "bonus", 1.0)
    assert bonus_act is not None
    assert bonus_act["ratio_num"] == 1
