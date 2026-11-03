"""Unit tests for FinRAG-Eval smart chunking and PDF parsing pipeline.

Verifies:
1. Markdown table preservation & atomic chunking (contains_table: True) up to 1500 tokens.
2. Capture of up to 2 preceding header/unit lines with tables.
3. Concall speaker-turn grouping [Analyst Question + Management Response] up to 700 tokens.
4. Recursive splitting of commentary at 600 tokens with 80 tokens overlap.
5. Strict adherence to Chunk Object Schema and deterministic chunk_id formatting.

Sources:
- docs/DetectionLogic.md
- docs/DesignDocument.md Section 3A & 4.1
- Techstack.md Section 2
"""

import re
import pytest

from src.config import settings
from src.ingestion.smart_chunker import (
    chunk_document,
    chunk_page,
    chunk_recursive_text,
    chunk_table_block,
    count_tokens,
    extract_page_segments,
)
from src.ingestion.pdf_parser import (
    format_table_to_markdown,
    is_table_formatting_broken,
)

# Deterministic chunk ID pattern: {company}_{quarter}_p{page_no}_c{index:02d}
CHUNK_ID_PATTERN = re.compile(r"^[A-Z0-9_]+_[A-Za-z0-9_]+_p\d+_c\d{2}$")


@pytest.fixture
def sample_pl_table_markdown() -> str:
    """Synthetic financial Markdown containing P&L table and preceding unit/headers."""
    return (
        "### Standalone Financial Performance\n"
        "The Board of Directors approved the unaudited financial results for the quarter.\n\n"
        "(Rs. in Crores)\n"
        "Statement of Profit and Loss\n"
        "| Particulars | Q2 FY27 | Q1 FY27 | Q2 FY26 | YoY % | QoQ % |\n"
        "|---|---|---|---|---|---|\n"
        "| Revenue from Operations | 1,250.5 | 1,180.2 | 1,050.0 | +19.1% | +5.9% |\n"
        "| Other Income | 35.2 | 28.4 | 22.0 | +60.0% | +23.9% |\n"
        "| Total Income | 1,285.7 | 1,208.6 | 1,072.0 | +19.9% | +6.4% |\n"
        "| Operating Expenses | 895.0 | 850.1 | 770.5 | +16.2% | +5.3% |\n"
        "| EBITDA | 355.5 | 330.1 | 279.5 | +27.2% | +7.7% |\n"
        "| Depreciation & Amortization | 45.0 | 42.0 | 38.0 | +18.4% | +7.1% |\n"
        "| Finance Costs | 8.5 | 9.0 | 11.2 | -24.1% | -5.6% |\n"
        "| Profit Before Tax (PBT) | 302.0 | 279.1 | 230.3 | +31.1% | +8.2% |\n"
        "| Tax Expense | 76.0 | 70.3 | 58.0 | +31.0% | +8.1% |\n"
        "| Net Profit (PAT) | 226.0 | 208.8 | 172.3 | +31.2% | +8.2% |\n"
        "| Basic EPS (in Rs.) | 14.5 | 13.4 | 11.0 | +31.8% | +8.2% |\n\n"
        "Operating margins expanded by 180 bps YoY driven by favorable operating leverage."
    )


@pytest.fixture
def sample_concall_markdown() -> str:
    """Synthetic concall transcript exchange with Moderator, Analyst, and Management."""
    return (
        "Moderator: Good day ladies and gentlemen, and welcome to the Q2 FY27 Earnings Call. "
        "We have with us Mr. Rajesh Verma, Managing Director, and Ms. Priya Nair, Chief Financial Officer.\n\n"
        "Management: Thank you, and welcome everyone. We delivered an exceptional quarter with 19% YoY revenue growth "
        "and 31% expansion in PAT. Our international order book remains healthy at 3,500 Crores.\n\n"
        "Moderator: Thank you. We will now take questions. The first question is from Amit Shah from ICICI Securities.\n\n"
        "Amit Shah: Good afternoon and congratulations on a solid set of numbers. "
        "Could you provide some color on the order pipeline in Europe given the macroeconomic concerns? "
        "Are we seeing any client decision deferrals?\n\n"
        "Management: Thanks Amit. Regarding Europe, we actually saw 14% growth this quarter. "
        "The demand in automotive software and medical devices remains resilient. We have not witnessed "
        "any cancellations or margin concessions in our tier-1 accounts.\n\n"
        "Moderator: The next question is from Sneha Mehta from Kotak Institutional Equities.\n\n"
        "Sneha Mehta: Hi, thank you for taking my question. What is our capex guidance for FY28?\n\n"
        "Management: Our capex guidance for FY28 stands at 180-200 Crores, primarily allocated toward expanding "
        "our Bangalore development center and building GPU computing clusters."
    )


def test_table_preservation_and_header_capture(sample_pl_table_markdown: str):
    """Assert P&L table stays intact in single chunk with contains_table=True and captures unit lines."""
    page_data = {
        "page_no": 12,
        "source_file": "TATAELXSI_Q2FY27_Report.pdf",
        "text": sample_pl_table_markdown,
    }

    chunks = chunk_page(
        page_data=page_data,
        company="TATAELXSI",
        quarter="Q2_FY27",
        doc_type="Financial_Results",
    )

    # Must produce commentary chunks and one atomic table chunk
    assert len(chunks) >= 2

    # Find the table chunk
    table_chunks = [c for c in chunks if c["metadata"]["contains_table"] is True]
    assert len(table_chunks) == 1, "Table must not be fragmented into multiple chunks"

    table_chunk = table_chunks[0]

    # Verify unit and heading capture
    assert "(Rs. in Crores)" in table_chunk["text"]
    assert "Statement of Profit and Loss" in table_chunk["text"]

    # Verify table integrity: first row and last row must be in the same chunk
    assert "Revenue from Operations" in table_chunk["text"]
    assert "Basic EPS (in Rs.)" in table_chunk["text"]

    # Verify token constraints
    assert count_tokens(table_chunk["text"]) <= settings.max_table_tokens

    # Verify non-table chunks have contains_table=False
    non_table_chunks = [c for c in chunks if not c["metadata"]["contains_table"]]
    assert len(non_table_chunks) >= 1
    for ntc in non_table_chunks:
        assert "(Rs. in Crores)" not in ntc["text"], "Captured header should not leak into commentary"


def test_concall_speaker_turn_grouping(sample_concall_markdown: str):
    """Assert concall speaker turns keep Analyst Question + Management Response together."""
    page_data = {
        "page_no": 8,
        "source_file": "TATAELXSI_Q2FY27_Concall.pdf",
        "text": sample_concall_markdown,
    }

    chunks = chunk_page(
        page_data=page_data,
        company="TATAELXSI",
        quarter="Q2_FY27",
        doc_type="Concall_Transcript",
    )

    assert len(chunks) >= 1

    # Check for Analyst Q + Management A coherence
    amit_chunks = [c for c in chunks if "Amit Shah:" in c["text"]]
    assert len(amit_chunks) == 1, "Amit Shah question should be in exactly one chunk"

    # Crucial assertion: Amit Shah's question AND Management's answer must reside together
    assert "Amit Shah:" in amit_chunks[0]["text"]
    assert "Management: Thanks Amit" in amit_chunks[0]["text"]

    sneha_chunks = [c for c in chunks if "Sneha Mehta:" in c["text"]]
    assert len(sneha_chunks) == 1
    assert "Sneha Mehta:" in sneha_chunks[0]["text"]
    assert "Management: Our capex guidance" in sneha_chunks[0]["text"]

    # Assert all chunks obey max concall token limits
    for c in chunks:
        assert count_tokens(c["text"]) <= settings.max_concall_tokens
        assert c["metadata"]["contains_table"] is False


def test_chunk_metadata_schema_conformance(sample_pl_table_markdown: str):
    """Verify that every chunk strictly implements docs/DesignDocument.md Section 3A schema."""
    page_data = {
        "page_no": 14,
        "source_file": "TataElxsi_Q2_Presentation.pdf",
        "text": sample_pl_table_markdown,
    }

    chunks = chunk_page(
        page_data=page_data,
        company="TATAELXSI",
        quarter="Q2_FY27",
        doc_type="Investor_Presentation",
    )

    required_metadata_keys = {
        "company",
        "doc_type",
        "quarter",
        "page_no",
        "contains_table",
        "source_file",
        "doc_section",
        "detected_unit",
    }

    for idx, chunk in enumerate(chunks, start=1):
        # Top-level schema keys
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk

        # Deterministic chunk_id format
        assert CHUNK_ID_PATTERN.match(chunk["chunk_id"])
        expected_chunk_id = f"TATAELXSI_Q2_FY27_p14_c{idx:02d}"
        assert chunk["chunk_id"] == expected_chunk_id

        # Metadata keys and values
        metadata = chunk["metadata"]
        assert set(metadata.keys()) == required_metadata_keys
        assert metadata["company"] == "TATAELXSI"
        assert metadata["quarter"] == "Q2_FY27"
        assert metadata["doc_type"] == "Investor_Presentation"
        assert metadata["page_no"] == 14
        assert metadata["source_file"] == "TataElxsi_Q2_Presentation.pdf"
        assert isinstance(metadata["contains_table"], bool)


def test_recursive_commentary_splitting_with_overlap():
    """Verify recursive chunking splits long commentary at 600 tokens with 80 tokens overlap."""
    # Generate ~1500 tokens of financial commentary paragraphs
    paragraph = (
        "The Indian equity markets witnessed renewed sector rotation with capital goods, "
        "infrastructure, and electronics manufacturing services (EMS) leading quarterly outperformance. "
        "Management commentary indicated stable order backlogs and steady utilization rates across operating units. "
        "Working capital cycles normalized post monsoon disruptions, leading to robust operating cash generation.\n\n"
    )
    long_commentary = paragraph * 25  # ~1500+ tokens

    chunks = chunk_recursive_text(
        long_commentary, chunk_size=600, overlap=80
    )

    assert len(chunks) >= 3

    for c in chunks:
        # Every chunk must adhere to token budget
        assert count_tokens(c) <= 650  # 600 + buffer for overlap piece

    # Check for non-empty overlap between chunk 0 and chunk 1
    c0_words = set(chunks[0].split()[-15:])
    c1_words = set(chunks[1].split()[:25])
    common_words = c0_words.intersection(c1_words)
    assert len(common_words) >= 5, "Consecutive chunks must share semantic overlap"


def test_multi_table_page_segmentation():
    """Verify a page with two distinct tables keeps both tables intact as separate chunks."""
    multi_table_text = (
        "### Segment Overview\n\n"
        "(in Crores)\n"
        "Table 1: Revenue by Vertical\n"
        "| Vertical | Q2 FY27 | Q1 FY27 |\n"
        "|---|---|---|\n"
        "| Automotive | 550 | 510 |\n"
        "| Media | 320 | 315 |\n\n"
        "Interim commentary between tables discussing margin drivers.\n\n"
        "(in Crores)\n"
        "Table 2: Revenue by Geography\n"
        "| Geography | Q2 FY27 | Q1 FY27 |\n"
        "|---|---|---|\n"
        "| Americas | 450 | 420 |\n"
        "| Europe | 380 | 370 |\n"
    )

    page_data = {
        "page_no": 5,
        "source_file": "TataElxsi_Q2_Presentation.pdf",
        "text": multi_table_text,
    }

    chunks = chunk_page(
        page_data=page_data,
        company="TATAELXSI",
        quarter="Q2_FY27",
        doc_type="Investor_Presentation",
    )

    table_chunks = [c for c in chunks if c["metadata"]["contains_table"]]
    assert len(table_chunks) == 2, "Both tables must be extracted as separate atomic chunks"

    assert "Table 1: Revenue by Vertical" in table_chunks[0]["text"]
    assert "Automotive" in table_chunks[0]["text"]

    assert "Table 2: Revenue by Geography" in table_chunks[1]["text"]
    assert "Americas" in table_chunks[1]["text"]

    # Verify intermediate commentary is preserved
    commentary_chunks = [c for c in chunks if not c["metadata"]["contains_table"]]
    assert len(commentary_chunks) >= 1
    assert any("Interim commentary" in c["text"] for c in commentary_chunks)


def test_empty_and_whitespace_page():
    """Verify empty or whitespace-only page produces zero chunks without crashing."""
    page_data = {
        "page_no": 1,
        "source_file": "Blank.pdf",
        "text": "   \n\n   \t  \n  ",
    }
    chunks = chunk_page(
        page_data=page_data,
        company="TEST",
        quarter="Q1",
        doc_type="Report",
    )
    assert chunks == []


def test_chunk_document_multi_page():
    """Verify chunk_document aggregates chunks across multiple pages with correct page numbers."""
    pages = [
        {
            "page_no": 1,
            "source_file": "TestDoc.pdf",
            "text": "Chairman Statement.\n\nManagement: We welcome our shareholders to the AGM.",
        },
        {
            "page_no": 2,
            "source_file": "TestDoc.pdf",
            "text": "| Metric | Value |\n|---|---|\n| PAT | 100 Cr |",
        },
    ]

    all_chunks = chunk_document(
        pages=pages,
        company="RELIANCE",
        quarter="Q1_FY27",
        doc_type="Annual_Report",
    )

    assert len(all_chunks) == 2
    assert all_chunks[0]["metadata"]["page_no"] == 1
    assert all_chunks[0]["metadata"]["contains_table"] is False
    assert all_chunks[0]["chunk_id"] == "RELIANCE_Q1_FY27_p1_c01"

    assert all_chunks[1]["metadata"]["page_no"] == 2
    assert all_chunks[1]["metadata"]["contains_table"] is True
    assert all_chunks[1]["chunk_id"] == "RELIANCE_Q1_FY27_p2_c01"


def test_pdfplumber_fallback_formatting():
    """Verify format_table_to_markdown helper accurately converts 2D row lists."""
    raw_rows = [
        ["Segment", "Revenue (Cr)", "Margin %"],
        ["Auto", "500", "22.5%"],
        ["Medical", "200", "28.0%"],
    ]
    md_table = format_table_to_markdown(raw_rows)
    assert "| Segment | Revenue (Cr) | Margin % |" in md_table
    assert "|---|---|---|" in md_table
    assert "| Auto | 500 | 22.5% |" in md_table

    # Test broken formatting detector
    assert is_table_formatting_broken("Some text | col | col2 without delimiter")
    assert not is_table_formatting_broken(md_table)


def test_parse_pdf_end_to_end(tmp_path):
    """Verify parse_pdf generates page-by-page markdown and saves debug files."""
    import pymupdf
    from src.ingestion.pdf_parser import parse_pdf

    # Create synthetic multi-page PDF using pymupdf
    pdf_path = tmp_path / "TATAELXSI_Q2_Sample.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 72), "Tata Elxsi Limited - Q2 FY27 Results\nExecutive Commentary.")

    p2 = doc.new_page()
    p2.insert_text((50, 72), "Segment Breakdown\nTransportation: 512 Cr\nHealthcare: 120 Cr")

    doc.save(str(pdf_path))
    doc.close()

    # Parse using parse_pdf
    pages = parse_pdf(pdf_path, save_debug_md=True)

    assert len(pages) == 2
    assert pages[0]["page_no"] == 1
    assert pages[0]["source_file"] == "TATAELXSI_Q2_Sample.pdf"
    assert "Tata Elxsi Limited" in pages[0]["text"]

    assert pages[1]["page_no"] == 2
    assert pages[1]["source_file"] == "TATAELXSI_Q2_Sample.pdf"

    # Verify debug markdown was saved
    debug_file = settings.processed_md_dir / f"{pdf_path.stem}_processed.md"
    assert debug_file.exists()
    debug_content = debug_file.read_text(encoding="utf-8")
    assert "PAGE 1 START" in debug_content
    assert "PAGE 2 START" in debug_content

