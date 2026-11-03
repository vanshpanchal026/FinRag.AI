"""Tests for Institutional SEBI Reg 24(7) Audit PDF Exporter."""

from pathlib import Path
import pymupdf
import pytest

from src.exports.sebi_audit_pdf import SEBIAuditPDFExporter, compute_evidence_hash


@pytest.fixture
def sample_audit_data():
    query = "What was the Revenue from Operations and QoQ growth in Q2 FY27?"
    answer = (
        "Tata Elxsi reported Revenue from Operations of 1,250.5 Cr in Q2 FY27 compared to 1,180.0 Cr in Q1 FY27 (+5.9% QoQ) "
        "[Source: TataElxsi_Q2.pdf | Page: 12 | Chunk: TATAELXSI_Q2_FY27_p12_c01]."
    )
    eval_report = {
        "query_id": "test_audit_001",
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
            "reasoning": "All numbers verified verbatim or via AST arithmetic.",
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


def test_compute_evidence_hash():
    """Verify SHA-256 evidence hashing is deterministic and non-empty."""
    h1 = compute_evidence_hash("C01", "Revenue 1250 Cr", "file.pdf", 12)
    h2 = compute_evidence_hash("C01", "Revenue 1250 Cr", "file.pdf", 12)
    h3 = compute_evidence_hash("C02", "Revenue 1250 Cr", "file.pdf", 12)

    assert len(h1) == 64
    assert h1 == h2
    assert h1 != h3


def test_generate_sebi_audit_pdf_bytes(sample_audit_data, tmp_path):
    """Verify complete PDF generation returns valid PDF bytes containing SEBI compliance headers."""
    query, answer, eval_report, golden_chunks = sample_audit_data
    exporter = SEBIAuditPDFExporter(
        firm_name="Alpha Vantage Institutional Research",
        registration_no="INH000098765",
        principal_officer="Dr. S. Ramanathan, CFA",
    )

    pdf_path = tmp_path / "sebi_audit_test.pdf"
    pdf_bytes = exporter.generate_pdf(
        query=query,
        answer=answer,
        eval_report=eval_report,
        golden_chunks=golden_chunks,
        company="TATAELXSI",
        quarter="Q2_FY27",
        output_path=pdf_path,
    )

    # 1. Check bytes header
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_path.exists()

    # 2. Open with PyMuPDF to inspect text content
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1
    full_text = "\n".join([page.get_text() for page in doc])

    # 3. Assert mandatory regulatory sections
    assert "SEBI (Research Analysts) Regulations, 2014" in full_text
    assert "Reg 24(7)" in full_text
    assert "ALPHA VANTAGE INSTITUTIONAL RESEARCH" in full_text
    assert "INH000098765" in full_text
    assert "TATAELXSI" in full_text
    assert "TATAELXSI_Q2_FY27_p12_c01" in full_text
    assert "PASS" in full_text
    assert "test_audit_001" in full_text

    doc.close()
