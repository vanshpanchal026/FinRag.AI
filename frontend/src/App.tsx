/**
 * App.tsx — 3-Pane Institutional Terminal Root
 * Layout: [Left Sidebar 280px] | [Center Pane flex-1] | [Right Pane 360px]
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Sidebar } from "./components/layout/Sidebar";
import { AnalystWorkspace } from "./components/chat/AnalystWorkspace";
import { AuditInspectorPane } from "./components/audit/AuditInspectorPane";
import { useHealth } from "./api/client";
import { Activity, AlertCircle } from "lucide-react";
import { clsx } from "clsx";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5_000,
    },
  },
});

function StatusBar() {
  const { data, isError, isLoading } = useHealth();

  return (
    <div className="flex items-center gap-2 px-3 py-1 border-t border-slate-800 bg-slate-950 text-[10px] text-slate-600 shrink-0">
      <div
        className={clsx(
          "w-1.5 h-1.5 rounded-full",
          isLoading
            ? "bg-amber-500 animate-pulse"
            : isError
            ? "bg-rose-500"
            : "bg-emerald-500"
        )}
      />
      {isError ? (
        <span className="flex items-center gap-1 text-rose-600">
          <AlertCircle className="w-3 h-3" />
          Backend offline — start FastAPI on port 8000
        </span>
      ) : data ? (
        <span className="flex items-center gap-1">
          <Activity className="w-3 h-3 text-emerald-600" />
          {data.engine} v{data.version} · {data.indexed_chunks_count.toLocaleString()} chunks ·{" "}
          {data.models.embedding}
        </span>
      ) : (
        <span>Connecting to backend…</span>
      )}
      <span className="ml-auto">FinRAG-Eval Institutional Workbench</span>
    </div>
  );
}

function WorkbenchLayout() {
  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-3 px-4 py-2.5 border-b border-slate-800 bg-slate-900 shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-emerald-600 flex items-center justify-center">
            <span className="text-white text-[10px] font-bold">FR</span>
          </div>
          <span className="text-sm font-bold text-slate-100 tracking-tight">
            FinRAG<span className="text-emerald-400">-Eval</span>
          </span>
        </div>
        <div className="h-4 w-px bg-slate-800 mx-1" />
        <span className="text-xs text-slate-500">
          Zero-Hallucination Financial Intelligence Terminal
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/60 text-emerald-500 border border-emerald-800/50">
            SEBI Reg 24(7) Ready
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-950/60 text-sky-500 border border-sky-800/50">
            IndFinRAG-Bench
          </span>
        </div>
      </header>

      {/* 3-Pane Grid */}
      <div
        className="flex-1 grid overflow-hidden"
        style={{ gridTemplateColumns: "280px 1fr 360px" }}
      >
        <Sidebar />
        <AnalystWorkspace />
        <AuditInspectorPane />
      </div>

      <StatusBar />
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WorkbenchLayout />
    </QueryClientProvider>
  );
}
