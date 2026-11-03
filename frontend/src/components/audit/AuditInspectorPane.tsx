/**
 * Right Pane — Live Two-Tier Audit & Provenance Inspector
 * Features:
 *  - 4-Badge Scorecard (Numeric Integrity PASS/FAIL, Faithfulness, Citation, Relevance)
 *  - AST Math & Unit Verification Drawer with color-coded verification pills
 *  - Golden Chunks Evidence Accordion (raw Markdown, rerank score, page info)
 *  - 1-Click "Export SEBI Reg 24(7) Audit PDF" button
 */

import React, { useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  FileDown,
  Hash,
  Cpu,
  ArrowRightLeft,
  BookOpen,
  FlaskConical,
  AlertTriangle,
} from "lucide-react";
import { clsx } from "clsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { exportSEBIPDF } from "../../api/client";
import { useFinRAGStore } from "../../store";
import type {
  ASTMathTrace,
  GoldenChunk,
  UnitConvertedMatch,
  VerbatimMatch,
} from "../../types/api";

// ─── Score Badge ──────────────────────────────────────────────────────────────
function ScoreBadge({
  label,
  value,
  max,
  pass,
  variant = "numeric",
}: {
  label: string;
  value?: number;
  max?: number;
  pass?: boolean;
  variant?: "pass-fail" | "numeric";
}) {
  const isPass = variant === "pass-fail" ? pass : (value ?? 0) >= 7;
  return (
    <div className="flex flex-col items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 min-w-0">
      <span className="text-[10px] text-slate-500 uppercase tracking-widest text-center leading-tight">
        {label}
      </span>
      {variant === "pass-fail" ? (
        <div
          className={clsx(
            "flex items-center gap-1 text-xs font-bold",
            pass ? "text-emerald-400" : "text-rose-400"
          )}
        >
          {pass ? (
            <CheckCircle2 className="w-3.5 h-3.5" />
          ) : (
            <XCircle className="w-3.5 h-3.5" />
          )}
          {pass ? "PASS" : "FAIL"}
        </div>
      ) : (
        <span
          className={clsx(
            "text-base font-bold tabular-nums",
            isPass ? "text-emerald-400" : "text-amber-400"
          )}
        >
          {value ?? "—"}
          {max !== undefined && (
            <span className="text-slate-600 text-xs font-normal">/{max}</span>
          )}
        </span>
      )}
    </div>
  );
}

// ─── Verification Pill ────────────────────────────────────────────────────────
function VerifBadge({ type }: { type: string }) {
  const styles: Record<string, string> = {
    VERIFIED_VERBATIM:
      "bg-emerald-950/60 text-emerald-400 border-emerald-800/50",
    VERIFIED_UNIT_CONVERSION:
      "bg-amber-950/60 text-amber-400 border-amber-800/50",
    VERIFIED_AST_DERIVATION:
      "bg-sky-950/60 text-sky-400 border-sky-800/50",
    UNVERIFIED: "bg-rose-950/60 text-rose-400 border-rose-800/50",
  };
  const labels: Record<string, string> = {
    VERIFIED_VERBATIM: "✓ VERBATIM",
    VERIFIED_UNIT_CONVERSION: "⇄ UNIT CONV",
    VERIFIED_AST_DERIVATION: "∫ AST MATH",
    UNVERIFIED: "⚠ UNVERIFIED",
  };
  return (
    <span
      className={clsx(
        "inline-flex text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border",
        styles[type] ?? "bg-slate-800 text-slate-400 border-slate-700"
      )}
    >
      {labels[type] ?? type}
    </span>
  );
}

// ─── Collapsible Section ──────────────────────────────────────────────────────
function Collapsible({
  title,
  icon,
  count,
  defaultOpen = false,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-slate-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-slate-900 hover:bg-slate-800/70 transition-colors"
      >
        <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
          {icon}
          {title}
          {count !== undefined && (
            <span className="text-[10px] text-slate-600">({count})</span>
          )}
        </div>
        {open ? (
          <ChevronUp className="w-3.5 h-3.5 text-slate-500" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
        )}
      </button>
      {open && (
        <div className="px-3 py-3 space-y-2 bg-slate-950/40">{children}</div>
      )}
    </div>
  );
}

// ─── Chunk Card ───────────────────────────────────────────────────────────────
function ChunkCard({
  chunk,
  highlighted,
}: {
  chunk: GoldenChunk;
  highlighted: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (highlighted) {
      ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [highlighted]);

  return (
    <div
      ref={ref}
      className={clsx(
        "rounded-lg border p-3 transition-all duration-300",
        highlighted
          ? "border-emerald-500 bg-emerald-950/20 shadow-[0_0_16px_rgba(16,185,129,0.15)]"
          : "border-slate-800 bg-slate-950/40"
      )}
    >
      <div className="flex items-center justify-between mb-2 flex-wrap gap-1">
        <span className="text-[10px] font-mono text-sky-400 break-all">
          {chunk.chunk_id}
        </span>
        <div className="flex items-center gap-1.5">
          {chunk.metadata.contains_table && (
            <span className="text-[9px] font-medium px-1.5 py-0.5 bg-sky-950/60 text-sky-400 border border-sky-800/50 rounded">
              TABLE
            </span>
          )}
          <span className="text-[10px] text-slate-500">
            p.{chunk.metadata.page_no}
          </span>
          <span
            className={clsx(
              "text-[10px] font-mono tabular-nums",
              chunk.rerank_score >= 0.8
                ? "text-emerald-400"
                : chunk.rerank_score >= 0.5
                ? "text-amber-400"
                : "text-slate-500"
            )}
          >
            {(chunk.rerank_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <div className="text-[11px] text-slate-500 mb-2">
        {chunk.metadata.source_file} · {chunk.metadata.quarter}
      </div>
      <div className="text-xs text-slate-400 font-mono leading-relaxed max-h-40 overflow-y-auto">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            table: ({ children }) => (
              <div className="overflow-x-auto">
                <table className="min-w-full text-[10px] border-collapse border border-slate-800">
                  {children}
                </table>
              </div>
            ),
            th: ({ children }) => (
              <th className="border border-slate-800 px-2 py-1 text-left text-slate-400 bg-slate-900">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="border border-slate-800 px-2 py-1 text-slate-500">
                {children}
              </td>
            ),
            p: ({ children }) => <p className="mb-1">{children}</p>,
          }}
        >
          {chunk.text}
        </ReactMarkdown>
      </div>
    </div>
  );
}

// ─── Main AuditInspectorPane ──────────────────────────────────────────────────
export function AuditInspectorPane() {
  const { activeResponse, highlightedChunkId, sebiProfile, filters } =
    useFinRAGStore();
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  if (!activeResponse) {
    return (
      <aside className="flex flex-col h-full bg-slate-900 border-l border-slate-800 items-center justify-center gap-3 p-6">
        <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
          <FlaskConical className="w-5 h-5 text-slate-600" />
        </div>
        <p className="text-xs text-slate-600 text-center max-w-[180px]">
          Run a query or preset to see the live audit report here.
        </p>
      </aside>
    );
  }

  const { eval_report, golden_chunks, query, answer } = activeResponse;
  const det = eval_report.deterministic_checks;
  const llm = eval_report.llm_judge_scores;

  const handleExportPDF = async () => {
    setExporting(true);
    setExportError(null);
    try {
      const blob = await exportSEBIPDF({
        query,
        answer,
        eval_report,
        golden_chunks,
        company: filters.company || "UNKNOWN",
        quarter: filters.quarter || "UNKNOWN",
        firm_name: sebiProfile.firmName || undefined,
        registration_no: sebiProfile.registrationNo || undefined,
        principal_officer: sebiProfile.analystName || undefined,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `SEBI_Audit_${filters.company || "Report"}_${filters.quarter || ""}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setExportError("PDF export failed. Ensure the backend is running.");
    } finally {
      setExporting(false);
    }
  };

  return (
    <aside className="flex flex-col h-full bg-slate-900 border-l border-slate-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-emerald-400" />
          <span className="text-sm font-semibold text-slate-100 tracking-wide">
            AUDIT INSPECTOR
          </span>
        </div>
        <button
          onClick={handleExportPDF}
          disabled={exporting}
          className={clsx(
            "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition-all border",
            exporting
              ? "bg-slate-800 text-slate-600 border-slate-700 cursor-wait"
              : "bg-amber-950/50 text-amber-400 border-amber-800/50 hover:bg-amber-900/60 hover:text-amber-300 active:scale-95"
          )}
        >
          <FileDown className="w-3 h-3" />
          {exporting ? "Exporting…" : "Export SEBI PDF"}
        </button>
      </div>

      {exportError && (
        <div className="mx-3 mt-2 flex items-center gap-2 text-xs text-rose-400 bg-rose-950/40 border border-rose-800/50 rounded-md px-3 py-2">
          <AlertTriangle className="w-3 h-3 shrink-0" />
          {exportError}
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-4">
        {/* ── 4-Badge Scorecard ── */}
        <section>
          <label className="block text-[10px] font-medium text-slate-500 uppercase tracking-widest mb-2">
            Eval Scorecard
          </label>
          <div className="grid grid-cols-2 gap-2">
            <ScoreBadge
              label="Numeric Integrity"
              variant="pass-fail"
              pass={det.numeric_match_pass}
            />
            <ScoreBadge
              label="Faithfulness"
              value={llm.faithfulness_score}
              max={10}
            />
            <ScoreBadge
              label="Citation Accuracy"
              value={llm.citation_accuracy_score}
              max={10}
            />
            <ScoreBadge
              label="Answer Relevance"
              value={llm.answer_relevance_score}
              max={10}
            />
          </div>
          {llm.hallucination_flag && (
            <div className="mt-2 flex items-center gap-2 text-xs text-rose-400 bg-rose-950/30 border border-rose-800/50 rounded-md px-3 py-2">
              <AlertTriangle className="w-3 h-3 shrink-0" />
              Hallucination detected — unverified claims found.
            </div>
          )}
          {llm.audit_reasoning && (
            <p className="mt-2 text-[11px] text-slate-500 italic leading-relaxed">
              {llm.audit_reasoning}
            </p>
          )}
        </section>

        {/* ── AST Math & Unit Verification Drawer ── */}
        <Collapsible
          title="Math & Unit Verification"
          icon={<Cpu className="w-3.5 h-3.5 text-emerald-400" />}
          count={
            (det.verbatim_matches?.length ?? 0) +
            (det.unit_converted_matches?.length ?? 0) +
            (det.ast_math_traces?.length ?? 0)
          }
          defaultOpen
        >
          {/* Verbatim Matches */}
          {(det.verbatim_matches?.length ?? 0) > 0 && (
            <div>
              <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-1.5 flex items-center gap-1">
                <Hash className="w-3 h-3" /> Verbatim Numbers
              </p>
              <div className="flex flex-wrap gap-1.5">
                {det.verbatim_matches.map((m: VerbatimMatch, i: number) => (
                  <div
                    key={i}
                    className="flex items-center gap-1.5 bg-emerald-950/30 border border-emerald-800/40 rounded px-2 py-1"
                  >
                    <span className="text-[11px] font-mono text-emerald-300">
                      {m.number}
                    </span>
                    <VerifBadge type={m.verification_type} />
                    <span className="text-[9px] text-slate-600">
                      p.{m.page_no}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Unit Conversions */}
          {(det.unit_converted_matches?.length ?? 0) > 0 && (
            <div>
              <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-1.5 flex items-center gap-1">
                <ArrowRightLeft className="w-3 h-3" /> Unit Conversions
              </p>
              <div className="space-y-1.5">
                {det.unit_converted_matches.map(
                  (m: UnitConvertedMatch, i: number) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 bg-amber-950/20 border border-amber-800/30 rounded px-2.5 py-1.5 text-[11px] font-mono"
                    >
                      <span className="text-amber-300">{m.original}</span>
                      <ArrowRightLeft className="w-3 h-3 text-amber-600 shrink-0" />
                      <span className="text-amber-400">{m.converted}</span>
                      <VerifBadge type={m.verification_type} />
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          {/* AST Math Traces */}
          {(det.ast_math_traces?.length ?? 0) > 0 && (
            <div>
              <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-1.5 flex items-center gap-1">
                <FlaskConical className="w-3 h-3" /> AST Formula Traces
              </p>
              <div className="space-y-2">
                {det.ast_math_traces.map((t: ASTMathTrace, i: number) => (
                  <div
                    key={i}
                    className={clsx(
                      "rounded-md border px-3 py-2 text-[11px] font-mono",
                      t.match
                        ? "bg-sky-950/20 border-sky-800/40"
                        : "bg-rose-950/20 border-rose-800/40"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <VerifBadge type={t.verification_type} />
                      <span
                        className={
                          t.match ? "text-emerald-400" : "text-rose-400"
                        }
                      >
                        {t.claimed_value}
                      </span>
                    </div>
                    <div className="text-slate-400 break-all">
                      {t.formula_string}
                    </div>
                    {t.computed_value !== null && (
                      <div className="text-slate-600 mt-0.5">
                        = {(t.computed_value * 100).toFixed(4)}%
                      </div>
                    )}
                    {t.error && (
                      <div className="text-rose-500 mt-0.5 text-[10px]">
                        {t.error}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Unverified numbers */}
          {det.unverified_numbers.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              <p className="w-full text-[10px] text-rose-500/80 uppercase tracking-widest mb-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> Unverified
              </p>
              {det.unverified_numbers.map((n: string, i: number) => (
                <span
                  key={i}
                  className="text-[11px] font-mono text-rose-400 bg-rose-950/30 border border-rose-800/40 rounded px-2 py-0.5"
                >
                  {n}
                </span>
              ))}
            </div>
          )}

          {det.verbatim_matches?.length === 0 &&
            det.unit_converted_matches?.length === 0 &&
            det.ast_math_traces?.length === 0 && (
              <p className="text-xs text-slate-600 italic">
                No verification data yet (backend AST module pending in Phase 5).
              </p>
            )}
        </Collapsible>

        {/* ── Golden Chunks Evidence ── */}
        <Collapsible
          title="Golden Chunks Evidence"
          icon={<BookOpen className="w-3.5 h-3.5 text-sky-400" />}
          count={golden_chunks.length}
          defaultOpen
        >
          {golden_chunks.length === 0 ? (
            <p className="text-xs text-slate-600 italic">No chunks retrieved.</p>
          ) : (
            <div className="space-y-3">
              {golden_chunks.map((chunk: GoldenChunk) => (
                <ChunkCard
                  key={chunk.chunk_id}
                  chunk={chunk}
                  highlighted={highlightedChunkId === chunk.chunk_id}
                />
              ))}
            </div>
          )}
        </Collapsible>

        {/* ── LLM Unverified Claims ── */}
        {llm.unverified_claims?.length > 0 && (
          <Collapsible
            title="Unverified Claims"
            icon={<AlertTriangle className="w-3.5 h-3.5 text-rose-400" />}
            count={llm.unverified_claims.length}
          >
            <ul className="space-y-1">
              {llm.unverified_claims.map((claim: string, i: number) => (
                <li
                  key={i}
                  className="text-xs text-rose-400 bg-rose-950/20 border border-rose-800/30 rounded px-2.5 py-1.5"
                >
                  {claim}
                </li>
              ))}
            </ul>
          </Collapsible>
        )}
      </div>
    </aside>
  );
}
