/**
 * Center Pane — Analyst Synthesis Terminal
 * Features:
 *  - 1-Click Trader Preset Pills (🔍 Concall Red Flags, 📊 Guidance vs Actuals, 💡 Growth & Capex)
 *  - View Switcher (Workbench ↔ IndFinRAG-Bench Leaderboard)
 *  - Markdown-rendered chat stream with remark-gfm tables
 *  - Interactive clickable citation pills that highlight Right Pane chunks
 *  - Streaming-style message UI with skeleton loaders
 */

import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Search,
  BarChart2,
  Zap,
  Send,
  Trash2,
  BarChart,
  MessageSquare,
  Bot,
  User,
  RefreshCw,
} from "lucide-react";
import { clsx } from "clsx";
import { useQueryMutation, usePresetMutation } from "../../api/client";
import { useFinRAGStore } from "../../store";
import { BenchmarkDashboard } from "../benchmark/BenchmarkDashboard";
import type { ChatMessage, PresetName, QueryResponse } from "../../types/api";

function generateId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

// ─── Citation Pill Parser ─────────────────────────────────────────────────────
const CITATION_REGEX =
  /\[(?:Source:\s*)?([^|]+)\s*\|\s*Page[:\s]+(\d+)\s*\|\s*Chunk[:\s:]+([^\]]+)\]/g;

function CitationPill({
  source,
  page,
  chunk,
}: {
  source: string;
  page: string;
  chunk: string;
}) {
  const { setHighlightedChunkId } = useFinRAGStore();
  const trimmed = source.trim().replace(/\.pdf$/i, "");
  const display = trimmed.length > 20 ? trimmed.slice(0, 20) + "…" : trimmed;

  return (
    <button
      onClick={() => setHighlightedChunkId(chunk.trim())}
      title={`${source} | Page ${page} | Chunk ${chunk}`}
      className="inline-flex items-center gap-1 mx-0.5 px-2 py-0.5 rounded text-[11px] font-mono bg-sky-950/60 text-sky-400 border border-sky-800/50 hover:bg-sky-900/60 hover:text-sky-300 transition-colors cursor-pointer align-middle"
    >
      {display} · p{page} · {chunk.trim().slice(-6)}
    </button>
  );
}

// ─── Markdown Renderer with Citation Pills ────────────────────────────────────
function AnalystMarkdown({ content }: { content: string }) {
  // Replace citation tags with placeholders we can map back to React components
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  const regex = new RegExp(CITATION_REGEX.source, "g");

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <ReactMarkdown
          key={`md-${lastIndex}`}
          remarkPlugins={[remarkGfm]}
          components={{
            table: ({ children }) => (
              <div className="overflow-x-auto my-3">
                <table className="min-w-full text-xs border-collapse border border-slate-700">
                  {children}
                </table>
              </div>
            ),
            th: ({ children }) => (
              <th className="border border-slate-700 px-3 py-1.5 text-left text-slate-300 bg-slate-800 font-semibold">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="border border-slate-700 px-3 py-1.5 text-slate-400">
                {children}
              </td>
            ),
            code: ({ children, className }) => {
              const isBlock = className?.includes("language-");
              return isBlock ? (
                <pre className="bg-slate-950 rounded-md p-3 overflow-x-auto text-xs text-emerald-400 my-2">
                  <code>{children}</code>
                </pre>
              ) : (
                <code className="bg-slate-800 rounded px-1 py-0.5 text-emerald-400 text-[11px] font-mono">
                  {children}
                </code>
              );
            },
            strong: ({ children }) => (
              <strong className="text-slate-100 font-semibold">{children}</strong>
            ),
            p: ({ children }) => (
              <p className="mb-2 leading-relaxed text-slate-300">{children}</p>
            ),
            li: ({ children }) => (
              <li className="text-slate-300 leading-relaxed">{children}</li>
            ),
          }}
        >
          {content.slice(lastIndex, match.index)}
        </ReactMarkdown>
      );
    }
    parts.push(
      <CitationPill
        key={`cit-${match.index}`}
        source={match[1]}
        page={match[2]}
        chunk={match[3]}
      />
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push(
      <ReactMarkdown
        key={`md-end`}
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="overflow-x-auto my-3">
              <table className="min-w-full text-xs border-collapse border border-slate-700">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-slate-700 px-3 py-1.5 text-left text-slate-300 bg-slate-800 font-semibold">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-slate-700 px-3 py-1.5 text-slate-400">
              {children}
            </td>
          ),
          p: ({ children }) => (
            <p className="mb-2 leading-relaxed text-slate-300">{children}</p>
          ),
          strong: ({ children }) => (
            <strong className="text-slate-100 font-semibold">{children}</strong>
          ),
          li: ({ children }) => (
            <li className="text-slate-300 leading-relaxed">{children}</li>
          ),
        }}
      >
        {content.slice(lastIndex)}
      </ReactMarkdown>
    );
  }

  return <div className="text-sm leading-relaxed">{parts}</div>;
}

// ─── Message Bubble ───────────────────────────────────────────────────────────
function MessageBubble({
  msg,
  onOpenAudit,
}: {
  msg: ChatMessage;
  onOpenAudit: (r: QueryResponse) => void;
}) {
  const isUser = msg.role === "user";

  return (
    <div className={clsx("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      <div
        className={clsx(
          "w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5",
          isUser ? "bg-sky-700" : "bg-emerald-800"
        )}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5 text-sky-200" />
        ) : (
          <Bot className="w-3.5 h-3.5 text-emerald-200" />
        )}
      </div>
      <div
        className={clsx(
          "max-w-[85%] rounded-xl px-4 py-3",
          isUser
            ? "bg-sky-900/40 border border-sky-800/50 text-sky-100 text-sm"
            : "bg-slate-800/60 border border-slate-700/50"
        )}
      >
        {isUser ? (
          <p className="text-sm">{msg.content}</p>
        ) : (
          <AnalystMarkdown content={msg.content} />
        )}
        {msg.queryResponse && !isUser && (
          <button
            onClick={() => onOpenAudit(msg.queryResponse!)}
            className="mt-2 text-[11px] text-emerald-500 hover:text-emerald-400 transition-colors flex items-center gap-1"
          >
            <BarChart className="w-3 h-3" />
            View audit & evidence →
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Skeleton Loader ──────────────────────────────────────────────────────────
function ThinkingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="w-7 h-7 rounded-full bg-emerald-800 flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-3.5 h-3.5 text-emerald-200" />
      </div>
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-3 flex items-center gap-2">
        <RefreshCw className="w-3.5 h-3.5 text-emerald-400 animate-spin" />
        <span className="text-xs text-slate-400">Analyst is synthesizing…</span>
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export function AnalystWorkspace() {
  const {
    messages,
    addMessage,
    clearMessages,
    filters,
    setActiveResponse,
    rightPaneView,
    setRightPaneView,
  } = useFinRAGStore();

  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const queryMutation = useQueryMutation();
  const presetMutation = usePresetMutation();

  const isLoading = queryMutation.isPending || presetMutation.isPending;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleOpenAudit = (r: QueryResponse) => {
    setActiveResponse(r);
    setRightPaneView("audit");
  };

  const sendQuery = async (query: string) => {
    if (!query.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: query.trim(),
      timestamp: Date.now(),
    };
    addMessage(userMsg);
    setInput("");

    try {
      const res = await queryMutation.mutateAsync({
        query: query.trim(),
        company: filters.company || undefined,
        quarter: filters.quarter || undefined,
        doc_type: filters.docType || undefined,
      });

      const assistantMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        content: res.answer,
        timestamp: Date.now(),
        queryResponse: res,
      };
      addMessage(assistantMsg);
      setActiveResponse(res);
    } catch {
      addMessage({
        id: generateId(),
        role: "assistant",
        content: "Connection failed. Ensure the FastAPI backend is running on port 8000.",
        timestamp: Date.now(),
      });
    }
  };

  const runPreset = async (name: PresetName, label: string) => {
    if (!filters.company) {
      addMessage({
        id: generateId(),
        role: "assistant",
        content:
          "Set a **Company** filter in the left sidebar before running a preset.",
        timestamp: Date.now(),
      });
      return;
    }

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: `▶ ${label}`,
      timestamp: Date.now(),
      isPreset: true,
    };
    addMessage(userMsg);

    try {
      const res = await presetMutation.mutateAsync({
        name,
        req: {
          company: filters.company,
          quarter: filters.quarter || undefined,
          q_prev:
            name === "guidance_vs_actuals"
              ? filters.quarter
                ? filters.quarter.replace("Q2", "Q1")
                : undefined
              : undefined,
          q_curr: name === "guidance_vs_actuals" ? filters.quarter || undefined : undefined,
        },
      });

      const assistantMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        content: res.answer,
        timestamp: Date.now(),
        queryResponse: res,
        presetName: name,
      };
      addMessage(assistantMsg);
      setActiveResponse(res);
    } catch {
      addMessage({
        id: generateId(),
        role: "assistant",
        content: "Preset failed. Ensure documents are indexed and the backend is running.",
        timestamp: Date.now(),
      });
    }
  };

  return (
    <main className="flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* ── Top Bar ── */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-800 bg-slate-900 flex-wrap">
        {/* View Toggle */}
        <div className="flex rounded-lg overflow-hidden border border-slate-700 mr-2">
          <button
            onClick={() => setRightPaneView("audit")}
            className={clsx(
              "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-colors",
              rightPaneView !== "benchmark"
                ? "bg-emerald-700 text-white"
                : "bg-transparent text-slate-400 hover:text-slate-200"
            )}
          >
            <MessageSquare className="w-3 h-3" /> Workbench
          </button>
          <button
            onClick={() => setRightPaneView("benchmark")}
            className={clsx(
              "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-colors",
              rightPaneView === "benchmark"
                ? "bg-emerald-700 text-white"
                : "bg-transparent text-slate-400 hover:text-slate-200"
            )}
          >
            <BarChart2 className="w-3 h-3" /> IndFinRAG-Bench
          </button>
        </div>

        {/* Preset Pills */}
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => runPreset("red_flag_scanner", "Concall Red Flags")}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-rose-950/60 text-rose-400 border border-rose-800/50 hover:bg-rose-900/50 hover:text-rose-300 disabled:opacity-50 transition-all"
          >
            <Search className="w-3 h-3" /> Concall Red Flags
          </button>
          <button
            onClick={() => runPreset("guidance_vs_actuals", "Guidance vs Actuals")}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-sky-950/60 text-sky-400 border border-sky-800/50 hover:bg-sky-900/50 hover:text-sky-300 disabled:opacity-50 transition-all"
          >
            <BarChart2 className="w-3 h-3" /> Guidance vs Actuals
          </button>
          <button
            onClick={() => runPreset("growth_triggers_capex", "Growth Triggers & Capex")}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-amber-950/60 text-amber-400 border border-amber-800/50 hover:bg-amber-900/50 hover:text-amber-300 disabled:opacity-50 transition-all"
          >
            <Zap className="w-3 h-3" /> Growth & Capex
          </button>
        </div>

        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            title="Clear chat"
            className="ml-auto text-slate-600 hover:text-rose-400 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* ── Content Area ── */}
      {rightPaneView === "benchmark" ? (
        <div className="flex-1 overflow-y-auto">
          <BenchmarkDashboard />
        </div>
      ) : (
        <>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center gap-3 py-16">
                <div className="w-12 h-12 rounded-full bg-emerald-900/40 flex items-center justify-center">
                  <Bot className="w-6 h-6 text-emerald-400" />
                </div>
                <h2 className="text-slate-300 font-semibold text-base">
                  FinRAG-Eval Analyst
                </h2>
                <p className="text-slate-500 text-sm max-w-xs">
                  Upload financial filings in the left sidebar, then ask questions
                  or use the preset pills above.
                </p>
                <div className="mt-4 grid grid-cols-1 gap-2 w-full max-w-sm">
                  {[
                    "What is the YoY revenue growth and EBITDA margin trend?",
                    "Summarize management guidance for next 2 quarters.",
                    "Compare capex between Q1 and Q2 and explain the delta.",
                  ].map((q) => (
                    <button
                      key={q}
                      onClick={() => sendQuery(q)}
                      className="text-left text-xs text-slate-400 border border-slate-800 rounded-lg px-3 py-2 hover:border-emerald-700 hover:text-slate-200 transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                msg={msg}
                onOpenAudit={handleOpenAudit}
              />
            ))}
            {isLoading && <ThinkingIndicator />}
            <div ref={bottomRef} />
          </div>

          {/* ── Input Bar ── */}
          <div className="px-4 pb-4 pt-2 border-t border-slate-800">
            <div className="flex items-end gap-2 bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 focus-within:border-emerald-600 transition-colors">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendQuery(input);
                  }
                }}
                placeholder="Ask about financials, concall statements, ratios… (Enter to send)"
                rows={1}
                className="flex-1 bg-transparent text-sm text-slate-100 placeholder-slate-600 resize-none outline-none max-h-32 leading-relaxed"
                style={{ scrollbarWidth: "none" }}
              />
              <button
                onClick={() => sendQuery(input)}
                disabled={!input.trim() || isLoading}
                className={clsx(
                  "p-1.5 rounded-lg transition-all shrink-0",
                  input.trim() && !isLoading
                    ? "bg-emerald-600 hover:bg-emerald-500 text-white active:scale-95"
                    : "bg-slate-800 text-slate-600 cursor-not-allowed"
                )}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="text-[10px] text-slate-700 mt-1.5 text-center">
              Shift+Enter for newline · Citations are clickable
            </p>
          </div>
        </>
      )}
    </main>
  );
}
