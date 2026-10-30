"""Smart financial chunker module for FinRAG-Eval.

Implements table-preserving and speaker-turn-aware chunking rules:
- Rule 1 (Table Preservation): Detects Markdown tables (|---|---|), captures up to 2
  preceding header/unit lines (e.g., "Rs. in Crores"), and locks the entire table as an
  atomic chunk (contains_table: True) up to 1500 tokens without splitting rows.
- Rule 2 (Commentary & Concall Splitting): For non-table text, groups speaker turns
  ("Moderator:", "Management:", Analyst names) preserving [Analyst Question + Management Response],
  or splits recursively using tiktoken (cl100k_base) at 600 tokens with 80 tokens overlap.
- Output: Strict Chunk Object Schema matching docs/DesignDocument.md (Section 3A)
  with deterministic chunk_id format: {company}_{quarter}_p{page_no}_c{index:02d}.

Sources:
- docs/DetectionLogic.md (Section 1.1 & 1.2)
- docs/DesignDocument.md (Section 3A & 4.1)
- Techstack.md Section 2
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import tiktoken

from src.config import settings

# Regex Patterns strictly defined in docs/DetectionLogic.md
# Section 1.1: Markdown Table Detection Pattern
TABLE_PATTERN = re.compile(
    r"((?:[ \t]*#+ .*\n+)?"
    r"(?:[ \t]*\|.*\|[ \t]*\n"
    r"[ \t]*\|[-:\s|]+\|[ \t]*\n"
    r"(?:[ \t]*\|.*\|[ \t]*\n?)+))",
    re.MULTILINE,
)

# Section 1.2: Concall Speaker-Turn Detection Pattern (lookahead delimiter)
SPEAKER_TURN_SPLIT_PATTERN = re.compile(
    r"(?=(?:Moderator|Management|[A-Z][a-z]+ [A-Z][a-z]+)\s*:)",
    re.MULTILINE,
)

# Check if text contains speaker turn markers
SPEAKER_MARKER_CHECK_PATTERN = re.compile(
    r"(?:Moderator|Management|[A-Z][a-z]+ [A-Z][a-z]+)\s*:",
    re.MULTILINE,
)

# Shared tiktoken encoder instance
_ENCODER = tiktoken.get_encoding(settings.tiktoken_encoding)


def count_tokens(text: str) -> int:
    """Return the number of tokens in text using cl100k_base encoding."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))


def split_text_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Slice text directly by tokens using tiktoken."""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    step = max(1, max_tokens - overlap)
    for i in range(0, len(tokens), step):
        slice_tokens = tokens[i : i + max_tokens]
        decoded = _ENCODER.decode(slice_tokens).strip()
        if decoded:
            chunks.append(decoded)
        if i + max_tokens >= len(tokens):
            break
    return chunks


def chunk_recursive_text(
    text: str,
    chunk_size: int = settings.chunk_size_tokens,
    overlap: int = settings.chunk_overlap_tokens,
) -> List[str]:
    """Split non-table commentary recursively using tiktoken at chunk_size with overlap.

    Hierarchy:
    1. Paragraphs (double newline)
    2. Sentences (single newline or period)
    3. Token slicing fallback

    Args:
        text: Input commentary text.
        chunk_size: Maximum tokens per chunk (default 600).
        overlap: Token overlap between consecutive chunks (default 80).

    Returns:
        List of chunk text strings.
    """
    text = text.strip()
    if not text:
        return []

    if count_tokens(text) <= chunk_size:
        return [text]

    # Split into paragraphs
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    atomic_units: List[str] = []

    for p in raw_paragraphs:
        if count_tokens(p) <= chunk_size:
            atomic_units.append(p)
        else:
            # Paragraph exceeds chunk_size; split into sentences or lines
            sentence_splits = re.split(r"(?<=[.\n])\s+", p)
            for s in sentence_splits:
                s_clean = s.strip()
                if not s_clean:
                    continue
                if count_tokens(s_clean) <= chunk_size:
                    atomic_units.append(s_clean)
                else:
                    # Token slice fallback for huge run-on sentences
                    atomic_units.extend(split_text_by_tokens(s_clean, chunk_size, overlap))

    if not atomic_units:
        return []

    chunks: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    for unit in atomic_units:
        u_tokens = count_tokens(unit)
        if current_units and (current_tokens + u_tokens > chunk_size):
            chunk_str = "\n\n".join(current_units).strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Extract trailing overlap tokens from chunk_str
            encoded = _ENCODER.encode(chunk_str)
            overlap_slice = encoded[-overlap:] if len(encoded) >= overlap else encoded
            overlap_prefix = _ENCODER.decode(overlap_slice).strip()

            current_units = [overlap_prefix, unit] if overlap_prefix else [unit]
            current_tokens = count_tokens("\n\n".join(current_units))
        else:
            current_units.append(unit)
            current_tokens += u_tokens

    if current_units:
        final_chunk = "\n\n".join(current_units).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)

    return chunks


def chunk_concall_speaker_turns(
    text: str,
    max_tokens: int = settings.max_concall_tokens,
) -> List[str]:
    """Group concall speaker turns preserving [Analyst Question + Management Response].

    Groups turns into atomic chunks up to max_tokens (700) so the question context
    is never detached from management's answers.

    Args:
        text: Concall transcript text containing speaker transitions.
        max_tokens: Upper token threshold (default 700 tokens).

    Returns:
        List of grouped speaker turn chunk strings.
    """
    raw_turns = [
        t.strip()
        for t in SPEAKER_TURN_SPLIT_PATTERN.split(text)
        if t.strip()
    ]
    if not raw_turns:
        return []

    chunks: List[str] = []
    current_turns: List[str] = []
    current_tokens = 0

    idx = 0
    while idx < len(raw_turns):
        turn = raw_turns[idx]
        t_tokens = count_tokens(turn)

        # Check if this turn is an Analyst question followed immediately by Management answer
        is_management = turn.startswith("Management:")
        is_next_management = (
            idx + 1 < len(raw_turns)
            and raw_turns[idx + 1].startswith("Management:")
        )

        if not is_management and is_next_management:
            next_turn = raw_turns[idx + 1]
            pair_text = f"{turn}\n\n{next_turn}"
            pair_tokens = count_tokens(pair_text)

            # If adding the pair exceeds max_tokens, finalize current chunk first
            if current_turns and (current_tokens + pair_tokens > max_tokens):
                chunks.append("\n\n".join(current_turns).strip())
                current_turns = []
                current_tokens = 0

            # If the pair itself fits under max_tokens, keep it bound together
            if pair_tokens <= max_tokens:
                current_turns.append(pair_text)
                current_tokens += pair_tokens
                idx += 2
                continue
            else:
                # The pair itself is larger than max_tokens (e.g. long answer)
                # Emit current_turns if any
                if current_turns:
                    chunks.append("\n\n".join(current_turns).strip())
                    current_turns = []
                    current_tokens = 0

                # If question + answer start can fit, or split the management answer
                question_tokens = t_tokens
                if question_tokens < max_tokens:
                    # Keep question with beginning of management answer
                    answer_subchunks = chunk_recursive_text(
                        next_turn,
                        chunk_size=max(200, max_tokens - question_tokens),
                        overlap=settings.chunk_overlap_tokens,
                    )
                    if answer_subchunks:
                        chunks.append(f"{turn}\n\n{answer_subchunks[0]}")
                        for extra_ans in answer_subchunks[1:]:
                            chunks.append(extra_ans)
                    else:
                        chunks.append(turn)
                else:
                    # Very long question
                    chunks.extend(chunk_recursive_text(turn))
                    chunks.extend(chunk_recursive_text(next_turn))
                idx += 2
                continue

        # Regular single turn (e.g., Moderator intro, consecutive management turns)
        if current_turns and (current_tokens + t_tokens > max_tokens):
            chunks.append("\n\n".join(current_turns).strip())
            current_turns = [turn]
            current_tokens = t_tokens
        else:
            current_turns.append(turn)
            current_tokens += t_tokens
        idx += 1

    if current_turns:
        final_chunk = "\n\n".join(current_turns).strip()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def chunk_table_block(
    table_text: str,
    max_tokens: int = settings.max_table_tokens,
) -> List[str]:
    """Lock entire table as atomic chunk up to max_tokens without splitting rows.

    If a table exceeds max_tokens (edge case of massive multi-row balance sheets),
    it splits row-by-row while preserving header and delimiter rows on every sub-chunk.

    Args:
        table_text: Complete markdown table string (with captured headers).
        max_tokens: Maximum tokens allowed per table chunk (default 1500).

    Returns:
        List of table chunk strings.
    """
    table_text = table_text.strip()
    if not table_text:
        return []

    if count_tokens(table_text) <= max_tokens:
        return [table_text]

    # Extreme edge case: Table exceeds max_tokens.
    # Split rows without breaking row integrity, repeating headers.
    lines = [line for line in table_text.splitlines() if line.strip()]
    header_lines: List[str] = []
    data_lines: List[str] = []
    found_delimiter = False

    for idx, line in enumerate(lines):
        if not found_delimiter:
            header_lines.append(line)
            if re.match(r"^[ \t]*\|[-:\s|]+\|[ \t]*$", line):
                found_delimiter = True
        else:
            data_lines.append(line)

    if not data_lines:
        # Unable to parse table rows structurally; return as-is
        return [table_text]

    header_block = "\n".join(header_lines)
    header_tokens = count_tokens(header_block)

    sub_chunks: List[str] = []
    current_data_rows: List[str] = []
    current_tokens = header_tokens

    for row in data_lines:
        row_tokens = count_tokens(row)
        if current_data_rows and (current_tokens + row_tokens > max_tokens):
            full_sub_chunk = f"{header_block}\n" + "\n".join(current_data_rows)
            sub_chunks.append(full_sub_chunk.strip())
            current_data_rows = [row]
            current_tokens = header_tokens + row_tokens
        else:
            current_data_rows.append(row)
            current_tokens += row_tokens

    if current_data_rows:
        full_sub_chunk = f"{header_block}\n" + "\n".join(current_data_rows)
        sub_chunks.append(full_sub_chunk.strip())

    return sub_chunks


def extract_page_segments(page_text: str) -> List[Tuple[str, str]]:
    """Segment page text into alternating non-table and table blocks.

    Captures up to 2 preceding lines immediately before each Markdown table
    to preserve table titles and units (e.g., "(Rs. in Crores)", "Standalone Results").

    Args:
        page_text: Raw Markdown text of a single PDF page.

    Returns:
        List of tuples: (segment_type, text_content) where segment_type is
        either 'table' or 'text'.
    """
    matches = list(TABLE_PATTERN.finditer(page_text))
    if not matches:
        return [("text", page_text.strip())] if page_text.strip() else []

    segments: List[Tuple[str, str]] = []
    cursor = 0

    for m in matches:
        preceding = page_text[cursor : m.start()]
        lines = preceding.splitlines()

        # Identify up to 2 preceding non-empty lines right before table
        non_empty_indices = [
            idx for idx, line in enumerate(lines) if line.strip()
        ]

        if len(non_empty_indices) >= 2:
            split_idx = non_empty_indices[-2]
            commentary_lines = lines[:split_idx]
            captured_header_lines = [
                lines[i].strip() for i in non_empty_indices[-2:]
            ]
        elif len(non_empty_indices) == 1:
            split_idx = non_empty_indices[-1]
            commentary_lines = lines[:split_idx]
            captured_header_lines = [lines[split_idx].strip()]
        else:
            commentary_lines = []
            captured_header_lines = []

        commentary_text = "\n".join(commentary_lines).strip()
        if commentary_text:
            segments.append(("text", commentary_text))

        table_body = m.group(0).strip()
        if captured_header_lines:
            table_text = "\n".join(captured_header_lines) + "\n" + table_body
        else:
            table_text = table_body

        segments.append(("table", table_text.strip()))
        cursor = m.end()

    trailing_text = page_text[cursor:].strip()
    if trailing_text:
        segments.append(("text", trailing_text))

    return segments


def chunk_page(
    page_data: Dict[str, Any],
    company: str,
    quarter: str,
    doc_type: str,
    source_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Chunk a single parsed page according to Table & Speaker-Turn rules.

    Args:
        page_data: Dictionary containing 'page_no' and 'text' (from pdf_parser).
        company: Company identifier (e.g., 'TATAELXSI').
        quarter: Financial reporting quarter (e.g., 'Q2_FY27').
        doc_type: Type of financial document ('Quarterly_Results', 'Concall_Transcript', etc.).
        source_file: Name of the PDF file (default: taken from page_data['source_file']).

    Returns:
        List of chunk objects conforming strictly to docs/DesignDocument.md Section 3A.
    """
    page_no = int(page_data.get("page_no", 1))
    page_text = page_data.get("text", "") or ""
    file_name = source_file or page_data.get("source_file", "unknown.pdf")

    # Sanitize identifiers for deterministic chunk_id
    clean_company = company.strip().replace(" ", "_").upper()
    clean_quarter = quarter.strip().replace(" ", "_")

    if not page_text.strip():
        return []

    segments = extract_page_segments(page_text)
    page_chunks: List[Dict[str, Any]] = []
    chunk_index = 1

    for seg_type, seg_content in segments:
        if not seg_content.strip():
            continue

        if seg_type == "table":
            table_subchunks = chunk_table_block(
                seg_content, max_tokens=settings.max_table_tokens
            )
            for sub_table in table_subchunks:
                chunk_id = (
                    f"{clean_company}_{clean_quarter}_p{page_no}_c{chunk_index:02d}"
                )
                page_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": sub_table,
                        "metadata": {
                            "company": clean_company,
                            "doc_type": doc_type,
                            "quarter": clean_quarter,
                            "page_no": page_no,
                            "contains_table": True,
                            "source_file": file_name,
                        },
                    }
                )
                chunk_index += 1
        else:
            # Non-table commentary
            if SPEAKER_MARKER_CHECK_PATTERN.search(seg_content):
                commentary_chunks = chunk_concall_speaker_turns(
                    seg_content, max_tokens=settings.max_concall_tokens
                )
            else:
                commentary_chunks = chunk_recursive_text(
                    seg_content,
                    chunk_size=settings.chunk_size_tokens,
                    overlap=settings.chunk_overlap_tokens,
                )

            for c_text in commentary_chunks:
                if not c_text.strip():
                    continue
                chunk_id = (
                    f"{clean_company}_{clean_quarter}_p{page_no}_c{chunk_index:02d}"
                )
                page_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": c_text,
                        "metadata": {
                            "company": clean_company,
                            "doc_type": doc_type,
                            "quarter": clean_quarter,
                            "page_no": page_no,
                            "contains_table": False,
                            "source_file": file_name,
                        },
                    }
                )
                chunk_index += 1

    return page_chunks


def chunk_document(
    pages: List[Dict[str, Any]],
    company: str,
    quarter: str,
    doc_type: str,
    source_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process an entire document page-by-page into a flat list of chunk objects.

    Args:
        pages: List of page dictionaries from pdf_parser.parse_pdf.
        company: Company identifier (e.g. 'TATAELXSI').
        quarter: Financial reporting quarter (e.g. 'Q2_FY27').
        doc_type: Document classification.
        source_file: Optional override for file name.

    Returns:
        Flat list of chunk objects strictly following Chunk Object Schema.
    """
    all_chunks: List[Dict[str, Any]] = []
    for page in pages:
        chunks = chunk_page(
            page_data=page,
            company=company,
            quarter=quarter,
            doc_type=doc_type,
            source_file=source_file,
        )
        all_chunks.extend(chunks)
    return all_chunks
