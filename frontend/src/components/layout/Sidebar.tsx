/**
 * Left Sidebar — Corpus & Compliance Manager
 * Features:
 *  - Multi-Doc Drag-and-Drop PDF Uploader (6 document types)
 *  - Company / Quarter / DocType metadata filters
 *  - Indexed documents list
 *  - SEBI Reg 24(7) Compliance Profile collapsible drawer
 */

import React, { useCallback, useRef, useState } from "react";
import {
  Upload,
  FileText,
  Filter,
  ChevronDown,
  ChevronUp,
  Shield,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Folder,
  X,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { clsx } from "clsx";
import {
  useDocuments,
  useIngestMutation,
  QUERY_KEYS,
} from "../../api/client";
import { useFinRAGStore, DOC_TYPES } from "../../store";

const QUARTERS = [
  "Q1_FY25", "Q2_FY25", "Q3_FY25", "Q4_FY25",
  "Q1_FY26", "Q2_FY26", "Q3_FY26", "Q4_FY26",
  "Q1_FY27", "Q2_FY27", "Q3_FY27", "Q4_FY27",
];

export function Sidebar() {
  const queryClient = useQueryClient();
  const { filters, setFilters, sebiProfile, setSebiProfile } = useFinRAGStore();
  const { data: docsData, isLoading: docsLoading } = useDocuments();
  const ingestMutation = useIngestMutation();

  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadCompany, setUploadCompany] = useState("");
  const [uploadQuarter, setUploadQuarter] = useState(QUARTERS[9]); // Q2_FY27
  const [uploadDocType, setUploadDocType] = useState<string>(DOC_TYPES[0]);
  const [uploadResult, setUploadResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [sebiOpen, setSebiOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file?.name.toLowerCase().endsWith(".pdf")) {
        setSelectedFile(file);
        setUploadResult(null);
      }
    },
    []
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadResult(null);
    }
  };

  const handleIngest = async () => {
    if (!selectedFile || !uploadCompany.trim()) return;
    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("company", uploadCompany.trim().toUpperCase());
    fd.append("quarter", uploadQuarter);
    fd.append("doc_type", uploadDocType);

    try {
      const res = await ingestMutation.mutateAsync(fd);
      setUploadResult({
        ok: true,
        msg: `Indexed ${res.chunks_count} chunks (${res.tables_count} tables) from ${res.source_file}`,
      });
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.documents });
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Upload failed. Check backend.";
      setUploadResult({ ok: false, msg });
    }
  };

  return (
    <aside className="flex flex-col h-full bg-slate-900 border-r border-slate-800 overflow-y-auto">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2">
        <Folder className="w-4 h-4 text-emerald-400" />
        <span className="text-sm font-semibold text-slate-100 tracking-wide">
          CORPUS MANAGER
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-5">
        {/* ── Drop Zone ── */}
        <section>
          <label className="block text-xs font-medium text-slate-400 uppercase tracking-widest mb-2">
            Upload Filing
          </label>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleFileDrop}
            onClick={() => fileInputRef.current?.click()}
            className={clsx(
              "relative flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed cursor-pointer py-6 px-3 transition-colors",
              dragOver
                ? "border-emerald-500 bg-emerald-950/30"
                : "border-slate-700 hover:border-slate-500 bg-slate-950/40"
            )}
          >
            <Upload className="w-6 h-6 text-slate-500" />
            {selectedFile ? (
              <span className="text-xs text-emerald-400 text-center break-all px-2">
                {selectedFile.name}
              </span>
            ) : (
              <span className="text-xs text-slate-500 text-center">
                Drop PDF or click to browse
              </span>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleFileSelect}
            />
          </div>
        </section>

        {/* ── Upload Metadata ── */}
        <section className="space-y-2">
          <input
            type="text"
            placeholder="Company ticker (e.g. TATAELXSI)"
            value={uploadCompany}
            onChange={(e) => setUploadCompany(e.target.value.toUpperCase())}
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-emerald-500 font-mono"
          />
          <select
            value={uploadQuarter}
            onChange={(e) => setUploadQuarter(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
          >
            {QUARTERS.map((q) => (
              <option key={q} value={q}>{q}</option>
            ))}
          </select>
          <select
            value={uploadDocType}
            onChange={(e) => setUploadDocType(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
          >
            {DOC_TYPES.map((dt) => (
              <option key={dt} value={dt}>{dt.replace(/_/g, " ")}</option>
            ))}
          </select>

          <button
            onClick={handleIngest}
            disabled={!selectedFile || !uploadCompany.trim() || ingestMutation.isPending}
            className={clsx(
              "w-full py-2 rounded-md text-xs font-semibold transition-all flex items-center justify-center gap-2",
              !selectedFile || !uploadCompany.trim()
                ? "bg-slate-800 text-slate-600 cursor-not-allowed"
                : "bg-emerald-600 hover:bg-emerald-500 active:scale-[0.98] text-white"
            )}
          >
            {ingestMutation.isPending ? (
              <><RefreshCw className="w-3 h-3 animate-spin" /> Indexing...</>
            ) : (
              <><Upload className="w-3 h-3" /> Ingest Document</>
            )}
          </button>

          {uploadResult && (
            <div className={clsx(
              "flex items-start gap-2 rounded-md px-3 py-2 text-xs",
              uploadResult.ok
                ? "bg-emerald-950/50 text-emerald-400 border border-emerald-800"
                : "bg-rose-950/50 text-rose-400 border border-rose-800"
            )}>
              {uploadResult.ok
                ? <CheckCircle2 className="w-3 h-3 mt-0.5 shrink-0" />
                : <AlertCircle className="w-3 h-3 mt-0.5 shrink-0" />}
              <span>{uploadResult.msg}</span>
            </div>
          )}
        </section>

        {/* ── Active Context Filters ── */}
        <section>
          <div className="flex items-center gap-2 mb-2">
            <Filter className="w-3 h-3 text-slate-400" />
            <label className="text-xs font-medium text-slate-400 uppercase tracking-widest">
              Active Filters
            </label>
            {(filters.company || filters.quarter || filters.docType) && (
              <button
                onClick={() => setFilters({ company: "", quarter: "", docType: "" })}
                className="ml-auto text-slate-600 hover:text-rose-400 transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
          <div className="space-y-2">
            <input
              type="text"
              placeholder="Filter by company…"
              value={filters.company}
              onChange={(e) => setFilters({ company: e.target.value.toUpperCase() })}
              className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-sky-500 font-mono"
            />
            <select
              value={filters.quarter}
              onChange={(e) => setFilters({ quarter: e.target.value })}
              className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-sky-500"
            >
              <option value="">All quarters</option>
              {QUARTERS.map((q) => (
                <option key={q} value={q}>{q}</option>
              ))}
            </select>
            <select
              value={filters.docType}
              onChange={(e) => setFilters({ docType: e.target.value })}
              className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-sky-500"
            >
              <option value="">All document types</option>
              {DOC_TYPES.map((dt) => (
                <option key={dt} value={dt}>{dt.replace(/_/g, " ")}</option>
              ))}
            </select>
          </div>
        </section>

        {/* ── Indexed Documents ── */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-slate-400 uppercase tracking-widest">
              Indexed Corpus
            </label>
            {docsData && (
              <span className="text-xs text-slate-600">
                {docsData.total_chunks.toLocaleString()} chunks
              </span>
            )}
          </div>
          {docsLoading ? (
            <div className="space-y-1">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-8 bg-slate-800 animate-pulse rounded-md" />
              ))}
            </div>
          ) : docsData?.documents.length === 0 ? (
            <div className="text-xs text-slate-600 italic py-2 text-center">
              No documents indexed yet. Upload a PDF to begin.
            </div>
          ) : (
            <div className="space-y-1">
              {docsData?.documents.map((doc) => (
                <div
                  key={doc.source_file}
                  className="flex items-start gap-2 bg-slate-950/60 rounded-md px-2 py-1.5 border border-slate-800"
                >
                  <FileText className="w-3 h-3 text-sky-400 mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <p className="text-xs font-mono text-slate-300 truncate">{doc.company}</p>
                    <p className="text-[10px] text-slate-500 truncate">
                      {doc.quarter} · {doc.doc_type.replace(/_/g, " ")}
                    </p>
                    <p className="text-[10px] text-slate-600">
                      {doc.chunks_count} chunks · {doc.tables_count} tables
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* ── SEBI Compliance Profile ── */}
        <section>
          <button
            onClick={() => setSebiOpen((v) => !v)}
            className="w-full flex items-center justify-between text-xs font-medium text-slate-400 uppercase tracking-widest mb-2 hover:text-slate-200 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Shield className="w-3 h-3 text-amber-400" />
              SEBI Reg 24(7) Profile
            </div>
            {sebiOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
          {sebiOpen && (
            <div className="space-y-2 border border-amber-900/40 rounded-md p-3 bg-amber-950/10">
              <input
                type="text"
                placeholder="Firm name"
                value={sebiProfile.firmName}
                onChange={(e) => setSebiProfile({ firmName: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-amber-500"
              />
              <input
                type="text"
                placeholder="SEBI Reg. No. (INH...)"
                value={sebiProfile.registrationNo}
                onChange={(e) => setSebiProfile({ registrationNo: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-amber-500 font-mono"
              />
              <input
                type="text"
                placeholder="Principal analyst name"
                value={sebiProfile.analystName}
                onChange={(e) => setSebiProfile({ analystName: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-amber-500"
              />
              <p className="text-[10px] text-amber-600/80 italic">
                Saved locally. Prefilled into every audit PDF export.
              </p>
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}
