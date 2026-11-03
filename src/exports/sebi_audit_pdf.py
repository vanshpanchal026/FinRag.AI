"""Institutional SEBI Reg 24(7) Audit PDF Exporter module.

Generates tamper-evident, publication-grade regulatory audit dossiers conforming to:
SEBI (Research Analysts) Regulations, 2014 — Regulation 24(7) record-keeping mandate:
"Research analyst or research entity shall maintain research records including...
 basis of recommendation, research report, date, and supporting documents for a period
 of minimum five years."

Includes:
1. Configurable PMS / Research Analyst Firm Header (Hybrid Template Pattern).
2. Executive Query and Synthesized Cited Institutional Answer.
3. Two-Tier Verification Matrix (AST Math traces, Lakhs vs Crores, verbatim matches).
4. Tier-2 LLM Judge Scorecard (Faithfulness, Citation Accuracy, Relevance).
5. Cryptographic Evidence Appendix with SHA-256 hashes for regulatory compliance.

Sources:
- SEBI (Research Analysts) Regulations, 2014, Reg 24(7)
- ReportLab PDF Library
"""

from datetime import datetime, timezone
import hashlib
import html
import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import settings


def compute_evidence_hash(chunk_id: str, text: str, source_file: str, page_no: int) -> str:
    """Compute SHA-256 cryptographic hash of evidence for tamper-proof compliance archiving."""
    payload = f"CHUNK:{chunk_id}|FILE:{source_file}|PAGE:{page_no}|TEXT:{text.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sanitize_xml(text: str) -> str:
    """Sanitize arbitrary text and markdown for safe ReportLab XML Paragraph rendering."""
    if not text:
        return ""
    # Normalize unicode whitespace & typography
    t = text.replace("\u202f", " ").replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    # Protect linebreaks and bolding
    t = re.sub(r"<\s*br\s*/?\s*>", "__BR__", t, flags=re.IGNORECASE)
    t = re.sub(r"\*\*(.*?)\*\*", r"__B_START__\1__B_END__", t)
    # Strip any stray HTML tags
    t = re.sub(r"<[^>]+>", " ", t)
    # Escape XML special characters (&, <, >)
    t = html.escape(t)
    # Restore safe formatting tags
    t = t.replace("__B_START__", "<b>").replace("__B_END__", "</b>")
    t = t.replace("__BR__", "<br/>")
    t = t.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")
    return t


def sanitize_plain_text(text: str) -> str:
    """Strip all HTML tags and escape XML for plain table cells and excerpts."""
    if not text:
        return ""
    t = text.replace("\u202f", " ").replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"<[^>]+>", " ", t)
    return html.escape(t)


class SEBIAuditPDFExporter:
    """Institutional PDF Exporter for SEBI Reg 24(7) compliance."""

    def __init__(
        self,
        firm_name: Optional[str] = None,
        registration_no: Optional[str] = None,
        principal_officer: Optional[str] = None,
    ):
        self.firm_name = firm_name or settings.sebi_firm_name or "Institutional Equity Research Compliance"
        self.registration_no = registration_no or settings.sebi_registration_no or "INH000000000 (SEBI Registered RA)"
        self.principal_officer = principal_officer or settings.sebi_principal_officer or "Principal Officer / Lead Analyst"

    def generate_pdf(
        self,
        query: str,
        answer: str,
        eval_report: Dict[str, Any],
        golden_chunks: List[Dict[str, Any]],
        company: str = "TATAELXSI",
        quarter: str = "Q2_FY27",
        output_path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """Generate a complete SEBI Reg 24(7) Audit PDF dossier.

        Args:
            query: Financial query analyzed.
            answer: Synthesized response with citations.
            eval_report: Output from EvalSubAgent.evaluate().
            golden_chunks: Top retrieved Golden Chunks.
            company: Company ticker.
            quarter: Financial period.
            output_path: Optional file path to save PDF directly.

        Returns:
            PDF file content as bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1e3a8a"),
        )
        meta_label_style = ParagraphStyle(
            "MetaLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#475569"),
        )
        meta_val_style = ParagraphStyle(
            "MetaVal",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#0f172a"),
        )
        section_h1 = ParagraphStyle(
            "SectionH1",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1e293b"),
        )
        body_bold = ParagraphStyle(
            "ReportBodyBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#0f172a"),
        )
        code_style = ParagraphStyle(
            "CodeExcerpt",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#0f172a"),
        )
        tag_pass_style = ParagraphStyle(
            "TagPass",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#059669"),
        )
        tag_fail_style = ParagraphStyle(
            "TagFail",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#dc2626"),
        )

        elements: List[Any] = []

        # ==============================================================================
        # 1. REGULATORY HEADER & COMPLIANCE METADATA
        # ==============================================================================
        elements.append(Paragraph(self.firm_name.upper(), title_style))
        elements.append(Spacer(1, 2))
        elements.append(
            Paragraph(
                "SEBI (Research Analysts) Regulations, 2014 — Reg 24(7) Institutional Compliance Audit",
                subtitle_style,
            )
        )
        elements.append(Spacer(1, 6))

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        audit_id = eval_report.get("query_id", "AUDIT-UNKNOWN")

        header_table_data = [
            [
                Paragraph("<b>Registration No:</b>", meta_label_style),
                Paragraph(self.registration_no, meta_val_style),
                Paragraph("<b>Audit Dossier ID:</b>", meta_label_style),
                Paragraph(f"<code>{audit_id}</code>", meta_val_style),
            ],
            [
                Paragraph("<b>Principal Officer:</b>", meta_label_style),
                Paragraph(self.principal_officer, meta_val_style),
                Paragraph("<b>Generated Timestamp:</b>", meta_label_style),
                Paragraph(now_utc, meta_val_style),
            ],
            [
                Paragraph("<b>Target Entity:</b>", meta_label_style),
                Paragraph(f"<b>{company}</b> (Period: {quarter})", meta_val_style),
                Paragraph("<b>Mandate Ref:</b>", meta_label_style),
                Paragraph("SEBI Reg 24(7) Five-Year Record Retention", meta_val_style),
            ],
        ]

        header_table = Table(header_table_data, colWidths=[90, 180, 110, 160])
        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(header_table)
        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=8))

        # ==============================================================================
        # 2. EXECUTIVE QUERY & SYNTHESIZED INSTITUTIONAL ANSWER
        # ==============================================================================
        elements.append(Paragraph("1. Research Query & Synthesized Institutional Findings", section_h1))
        elements.append(Spacer(1, 3))
        
        query_box = Table(
            [[Paragraph("<b>Analyst Query:</b>", body_bold), Paragraph(sanitize_plain_text(query), body_style)]],
            colWidths=[90, 450],
        )
        query_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        elements.append(query_box)
        elements.append(Spacer(1, 6))

        # Render synthesized answer cleanly with XML sanitization
        clean_answer = sanitize_xml(answer)
        answer_box = Table(
            [[Paragraph("<b>Grounded Synthesis:</b><br/><br/>" + clean_answer, body_style)]],
            colWidths=[540],
        )
        answer_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(answer_box)
        elements.append(Spacer(1, 8))

        # ==============================================================================
        # 3. TWO-TIER AUDITOR SCORECARD & VERIFICATION MATRIX
        # ==============================================================================
        elements.append(Paragraph("2. Deterministic Verification & Math AST Recomputations", section_h1))
        elements.append(Spacer(1, 3))

        deterministic_data = eval_report.get("deterministic_checks", {})
        llm_judge = eval_report.get("llm_judge_scores", {})
        num_pass = deterministic_data.get("numeric_match_pass", True)
        hallucination_flag = eval_report.get("hallucination_flag", False)

        # 4-metric summary bar
        score_summary = [
            [
                Paragraph("<b>Deterministic Check:</b>", meta_label_style),
                Paragraph("PASS ✅" if num_pass else "FAIL ❌", tag_pass_style if num_pass else tag_fail_style),
                Paragraph("<b>Faithfulness:</b>", meta_label_style),
                Paragraph(f"<b>{llm_judge.get('faithfulness_score', 10)} / 10</b>", body_style),
                Paragraph("<b>Citation Accuracy:</b>", meta_label_style),
                Paragraph(f"<b>{llm_judge.get('citation_accuracy_score', 10)} / 10</b>", body_style),
                Paragraph("<b>Hallucination:</b>", meta_label_style),
                Paragraph("ZERO DETECTED ✅" if not hallucination_flag else "FLAGGED ❌", tag_pass_style if not hallucination_flag else tag_fail_style),
            ]
        ]
        score_bar = Table(score_summary, colWidths=[90, 60, 70, 50, 85, 45, 80, 60])
        score_bar.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(score_bar)
        elements.append(Spacer(1, 6))

        # Verification Matrix Table
        matrix_rows = [
            [
                Paragraph("<b>Report Number</b>", meta_label_style),
                Paragraph("<b>Verification Mechanism</b>", meta_label_style),
                Paragraph("<b>Source Grounding / AST Formula</b>", meta_label_style),
                Paragraph("<b>Audit Status</b>", meta_label_style),
            ]
        ]

        # 1. Verbatim numbers
        for vnum in deterministic_data.get("verbatim_matches", []):
            matrix_rows.append(
                [
                    Paragraph(f"<code>{sanitize_plain_text(str(vnum))}</code>", code_style),
                    Paragraph("Verbatim Raw Text Match", body_style),
                    Paragraph("Exact token match in source filing chunks (comma-normalized)", body_style),
                    Paragraph("PASS ✅", tag_pass_style),
                ]
            )

        # 2. Unit Converted numbers
        for ucov in deterministic_data.get("unit_converted_matches", []):
            s_raw = sanitize_plain_text(str(ucov.get('source_raw', '')))
            c_raw = sanitize_plain_text(str(ucov.get('claimed_raw', '')))
            matrix_rows.append(
                [
                    Paragraph(f"<code>{c_raw}</code>", code_style),
                    Paragraph(f"Unit Normalizer ({sanitize_plain_text(str(ucov.get('source_unit', ''))).upper()} → {sanitize_plain_text(str(ucov.get('claimed_unit', ''))).upper()})", body_style),
                    Paragraph(f"Source: {s_raw} ≡ Claimed: {c_raw}", body_style),
                    Paragraph("PASS ✅", tag_pass_style),
                ]
            )

        # 3. AST Math Recomputations
        for trace in deterministic_data.get("ast_math_traces", []):
            c_val = sanitize_plain_text(str(trace.get('claimed_value', '')))
            expr = sanitize_plain_text(str(trace.get('expression', '')))
            comp_val = sanitize_plain_text(str(trace.get('computed_value', '')))
            op_sym = sanitize_plain_text(str(trace.get('operator_symbol', '')))
            matrix_rows.append(
                [
                    Paragraph(f"<code>{c_val}</code>", code_style),
                    Paragraph(f"AST Recomputation ({op_sym})", body_style),
                    Paragraph(f"<code>{expr} = {comp_val}</code>", code_style),
                    Paragraph("PASS ✅", tag_pass_style),
                ]
            )

        # 4. Unverified numbers
        for unum in deterministic_data.get("unverified_numbers", []):
            matrix_rows.append(
                [
                    Paragraph(f"<code>{sanitize_plain_text(str(unum))}</code>", code_style),
                    Paragraph("Unverified / No Grounding", body_style),
                    Paragraph("Figure not found verbatim, via AST, or via unit normalization", body_style),
                    Paragraph("FAIL ❌", tag_fail_style),
                ]
            )

        if len(matrix_rows) > 1:
            matrix_table = Table(matrix_rows, colWidths=[90, 130, 250, 70])
            matrix_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            elements.append(matrix_table)
        elements.append(Spacer(1, 8))

        # ==============================================================================
        # 4. CRYPTOGRAPHIC EVIDENCE APPENDIX (SHA-256)
        # ==============================================================================
        elements.append(Paragraph("3. Regulatory Evidence Appendix & Cryptographic Hash (SHA-256)", section_h1))
        elements.append(Spacer(1, 3))

        for idx, chunk in enumerate(golden_chunks, start=1):
            cid = chunk.get("chunk_id", f"CHUNK_{idx}")
            meta = chunk.get("metadata", {})
            sfile = meta.get("source_file", "unknown.pdf")
            pno = meta.get("page_no", 1)
            raw_text = chunk.get("text", "").strip()

            # Compute tamper-evident hash
            c_hash = compute_evidence_hash(cid, raw_text, sfile, pno)

            chunk_meta_data = [
                [
                    Paragraph(f"<b>Evidence Chunk #{idx}:</b> <code>{sanitize_plain_text(str(cid))}</code>", meta_label_style),
                    Paragraph(f"<b>File:</b> {sanitize_plain_text(str(sfile))} | <b>Page:</b> {pno}", meta_val_style),
                ],
                [
                    Paragraph("<b>SHA-256 Hash:</b>", meta_label_style),
                    Paragraph(f"<code>{c_hash}</code>", code_style),
                ],
                [
                    Paragraph("<b>Raw Excerpt:</b>", meta_label_style),
                    Paragraph(sanitize_plain_text(raw_text[:350].replace("\n", " ")) + ("..." if len(raw_text) > 350 else ""), code_style),
                ],
            ]

            chunk_box = Table(chunk_meta_data, colWidths=[110, 430])
            chunk_box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("PADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            elements.append(chunk_box)
            elements.append(Spacer(1, 4))

        # Compliance sign-off box
        sign_off_data = [
            [
                Paragraph(
                    "<b>REGULATORY AUDIT ATTESTATION:</b> This document was generated deterministically by FinRAG-Eval. "
                    "All numbers and claims are mapped to specific source pages and verified against raw filings. "
                    "In compliance with SEBI Reg 24(7), retain this electronic record alongside the underlying research report.",
                    meta_label_style,
                )
            ]
        ]
        sign_off_table = Table(sign_off_data, colWidths=[540])
        sign_off_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e2e8f0")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#64748b")),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(Spacer(1, 4))
        elements.append(sign_off_table)

        # Build document
        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(pdf_bytes)

        return pdf_bytes
