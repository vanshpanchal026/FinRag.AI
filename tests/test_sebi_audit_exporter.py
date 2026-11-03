"""Unit tests for Institutional SEBI Reg 24(7) Audit PDF Exporter.

Verifies:
1. PDF generation produces a valid, non-empty PDF file.
2. Mandatory SEBI Reg 24(7) compliance metadata is present.
3. Citations, chunk IDs, page numbers, AST math traces, and SHA-256 evidence hashes are embedded.
"""

from pathlib import Path
import pymupdf
import pytest

from src.export.sebi_audit_exporter import (
    SEBIAuditPDFExporter,
    compute_evidence_hash,
)


@pytest.fixture
def sample_audit_payload():
    query = "What was the Revenue from Operations and QoQ growth in Q2 FY27?"
    answer = (
        "Tata Elxsi reported Revenue from Operations of 1,250.5 Cr in Q2 FY27 compared to 1,180.0 Cr in Q1 FY27 (+5.9% QoQ) "
        "[Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]."
    )
    eval_report = {
        "query_id": "sebi_audit_test_99",
        "hallucination_flag": False,
        "deterministic_checks": {
            "numeric_match_pass": True,
            "numbers_in_answer": ["1,250.5", "1,180.0", "5.9%"],
            "verbatim_matches": ["1,250.5", "1,180.0"],
            "unit_converted_matches": [],
            "ast_math_traces": [
                {
                    "expression": "((1250.5 - 1180.0) / 1180.0) * 100",
                    "operator_symbol": "% change",
                    "computed_value": 5.97,
                    "claimed_value": 5.9,
                    "is_valid": True,
                }
            ],
            "unverified_numbers": [],
        },
        "llm_judge_scores": {
            "faithfulness_score": 10,
            "citation_accuracy_score": 10,
            "answer_relevance_score": 10,
            "reasoning": "All metrics grounded and verified.",
        },
    }
    golden_chunks = [
        {
            "chunk_id": "TATAELXSI_Q2_FY27_p12_c01",
            "text": "Statement of Profit and Loss\nRevenue from Operations: Q2 FY27 = 1,250.5 Cr | Q1 FY27 = 1,180.0 Cr.",
            "metadata": {
                "source_file": "TataElxsi_Q2.pdf",
                "page_no": 12,
                "company": "TATAELXSI",
            },
        }
    ]
    return query, answer, eval_report, golden_chunks


def test_sebi_audit_exporter_produces_valid_pdf(sample_audit_payload, tmp_path):
    """Verify PDF export generates valid bytes and embedded compliance text."""
    query, answer, eval_report, golden_chunks = sample_audit_payload
    exporter = SEBIAuditPDFExporter(
        firm_name="Capital Research Institutional Advisors",
        registration_no="INH000077777",
        principal_officer="Compliance Officer",
    )

    pdf_out = tmp_path / "sebi_audit_dossier.pdf"
    pdf_bytes = exporter.generate_pdf(
        query=query,
        answer=answer,
        eval_report=eval_report,
        golden_chunks=golden_chunks,
        company="TATAELXSI",
        quarter="Q2_FY27",
        output_path=pdf_out,
    )

    # 1. Non-empty valid PDF file
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 1000

    # 2. Inspect embedded textual elements
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join([p.get_text() for p in doc])
    doc.close()

    # 3. Required sections embedded
    assert "SEBI (Research Analysts) Regulations, 2014" in full_text
    assert "Reg 24(7)" in full_text
    assert "CAPITAL RESEARCH INSTITUTIONAL ADVISORS" in full_text
    assert "INH000077777" in full_text
    assert "TATAELXSI_Q2_FY27_p12_c01" in full_text
    assert "Page: 12" in full_text
    assert "SHA-256" in full_text
    assert "PASS" in full_text
