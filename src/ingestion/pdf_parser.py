"""PDF parser module for FinRAG-Eval.

Extracts page-by-page Markdown text using PyMuPDF4LLM while preserving
exact 1-indexed page_no metadata. Includes a fallback helper using pdfplumber
for complex or broken table layouts, and saves intermediate Markdown output to
data/processed_md/ for debugging and verification.

Sources:
- PyMuPDF4LLM: https://pymupdf.readthedocs.io/en/latest/rag.html
- pdfplumber: https://github.com/jsvine/pdfplumber
- Techstack.md Section 2
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import pdfplumber
import pymupdf4llm

from src.config import settings


def format_table_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Convert raw 2D table rows into a formatted Markdown table.

    Args:
        rows: 2D list of table cells extracted from pdfplumber.

    Returns:
        Formatted Markdown table string.
    """
    if not rows or len(rows) < 1:
        return ""

    # Clean and normalize cells
    cleaned_rows: List[List[str]] = []
    for row in rows:
        cleaned_row = [
            " ".join((cell or "").split()).replace("|", "\\|") for cell in row
        ]
        cleaned_rows.append(cleaned_row)

    max_cols = max(len(r) for r in cleaned_rows)
    if max_cols == 0:
        return ""

    # Pad shorter rows to match column count
    padded_rows = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]

    # Build markdown table
    header_cells = padded_rows[0]
    header_line = "| " + " | ".join(header_cells) + " |"
    delimiter_line = "|" + "|".join(["---"] * max_cols) + "|"
    data_lines = ["| " + " | ".join(r) + " |" for r in padded_rows[1:]]

    return "\n".join([header_line, delimiter_line] + data_lines)


def fallback_extract_page_with_pdfplumber(
    pdf_path: Union[str, Path], page_no: int
) -> str:
    """Extract page text and convert tables to Markdown using pdfplumber fallback.

    Args:
        pdf_path: Path to the target PDF file.
        page_no: 1-indexed page number to extract.

    Returns:
        Extracted text with tables formatted as Markdown.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    with pdfplumber.open(str(pdf_file)) as pdf:
        if page_no < 1 or page_no > len(pdf.pages):
            raise IndexError(
                f"Page {page_no} out of bounds (1-{len(pdf.pages)}) in {pdf_file.name}"
            )

        page = pdf.pages[page_no - 1]
        extracted_tables = page.extract_tables()
        text = page.extract_text() or ""

        if not extracted_tables:
            return text

        # Format extracted tables into markdown
        markdown_tables = [
            format_table_to_markdown(t)
            for t in extracted_tables
            if t and len(t) >= 1
        ]
        table_block = "\n\n".join(markdown_tables)

        if text.strip():
            return f"{text.strip()}\n\n{table_block}"
        return table_block


def is_table_formatting_broken(text: str) -> bool:
    """Check if page text appears to have broken or unformatted tables.

    Heuristic check:
    - Contains loose pipe delimiters without proper markdown table header syntax (|---|)
    - Or has raw tabular financial patterns with multiple numbers and tabs/spaces but no markdown table
    """
    has_valid_table = bool(re.search(r"\|[ -:]+\|", text))
    if has_valid_table:
        return False

    # Check for orphaned table pipes or multi-column tab patterns with numbers
    has_orphaned_pipes = text.count("|") >= 2 and not has_valid_table
    has_financial_columns = bool(
        re.search(r"(?:\b\d+(?:\.\d+)?\b[ \t]+){3,}", text)
    )

    return has_orphaned_pipes or has_financial_columns


def parse_pdf(
    pdf_path: Union[str, Path],
    save_debug_md: bool = True,
    fallback_on_broken_tables: bool = True,
) -> List[Dict[str, Any]]:
    """Parse a PDF document page-by-page into Markdown text.

    Uses PyMuPDF4LLM as primary parser (preserving exact 1-indexed page_no),
    with an automated fallback to pdfplumber when broken table formatting
    is detected.

    Args:
        pdf_path: Path to the PDF file.
        save_debug_md: If True, saves intermediate Markdown to data/processed_md/.
        fallback_on_broken_tables: If True, triggers pdfplumber fallback on broken tables.

    Returns:
        List of dictionaries with keys:
            - page_no (int): 1-indexed page number
            - text (str): Markdown text of the page
            - source_file (str): Base name of the PDF file
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    source_file_name = pdf_file.name

    # Step 1: Parse using PyMuPDF4LLM page chunks
    try:
        raw_chunks = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)
    except Exception as e:
        # If pymupdf4llm fails entirely, fallback to pdfplumber for all pages
        raw_chunks = []

    pages: List[Dict[str, Any]] = []

    if raw_chunks:
        for idx, chunk in enumerate(raw_chunks, start=1):
            metadata = chunk.get("metadata", {})
            page_no = metadata.get("page_number", idx)
            text = chunk.get("text", "") or ""

            # Check if fallback is warranted
            if fallback_on_broken_tables and is_table_formatting_broken(text):
                try:
                    fallback_text = fallback_extract_page_with_pdfplumber(
                        pdf_file, page_no
                    )
                    if fallback_text.strip():
                        text = fallback_text
                except Exception:
                    pass  # Keep original text if fallback fails

            pages.append(
                {
                    "page_no": page_no,
                    "text": text,
                    "source_file": source_file_name,
                }
            )
    else:
        # Full fallback to pdfplumber
        with pdfplumber.open(str(pdf_file)) as pdf:
            for page_no in range(1, len(pdf.pages) + 1):
                text = fallback_extract_page_with_pdfplumber(pdf_file, page_no)
                pages.append(
                    {
                        "page_no": page_no,
                        "text": text,
                        "source_file": source_file_name,
                    }
                )

    # Step 2: Save intermediate Markdown output to data/processed_md/
    if save_debug_md:
        settings.ensure_directories()
        debug_output_path = (
            settings.processed_md_dir / f"{pdf_file.stem}_processed.md"
        )
        debug_lines = []
        for p in pages:
            debug_lines.append(
                f"<!-- PAGE {p['page_no']} START | FILE: {source_file_name} -->\n"
                f"### Page {p['page_no']}\n\n"
                f"{p['text'].strip()}\n\n"
                f"<!-- PAGE {p['page_no']} END -->\n\n---\n"
            )
        debug_output_path.write_text("\n".join(debug_lines), encoding="utf-8")

    return pages
