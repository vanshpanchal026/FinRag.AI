/**
 * BenchmarkDashboard — IndFinRAG-Bench Comparison View
 * Renders FinRAG-Eval vs Naive RAG baseline metrics using Recharts.
 * Falls back to illustrative mock data when benchmark hasn't been run.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
} from "recharts";
import { Trophy, Info, RefreshCw } from "lucide-react";
import { clsx } from "clsx";
import { useBenchmarkSummary } from "../../api/client";
import type { BenchmarkCategory } from "../../types/api";

// ─── Mock data shown before running benchmark ─────────────────────────────────
const MOCK_CATEGORIES: BenchmarkCategory[] = [
  { name: "Table Reasoning", finrag_score: 9.1, naive_rag_score: 5.4, questions: 25 },
  { name: "Guidance vs Actuals", finrag_score: 8.7, naive_rag_score: 4.8, questions: 25 },
  { name: "Multi-Doc Q&A", finrag_score: 8.4, naive_rag_score: 4.2, questions: 25 },
  { name: "Concall Red Flags", finrag_score: 8.9, naive_rag_score: 5.1, questions: 25 },
];

const MOCK_OVERALL = {
  finrag_faithfulness: 9.1,
  finrag_citation_accuracy: 8.8,
  finrag_relevance: 8.9,
  naive_faithfulness: 4.9,
  naive_citation_accuracy: 3.2,
  naive_relevance: 6.1,
};

const EMERALD = "#10b981";
const SLATE = "#475569";

function MetricCard({
  label,
  finrag,
  naive,
}: {
  label: string;
  finrag: number;
  naive: number;
}) {
  const delta = finrag - naive;
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg px-4 py-3">
      <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">{label}</p>
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="text-[10px] text-slate-600 mb-0.5">FinRAG-Eval</p>
          <p className="text-lg font-bold tabular-nums text-emerald-400">
            {finrag.toFixed(1)}<span className="text-slate-600 text-xs font-normal">/10</span>
          </p>
        </div>
        <div>
          <p className="text-[10px] text-slate-600 mb-0.5">Naive RAG</p>
          <p className="text-base font-semibold tabular-nums text-slate-500">
            {naive.toFixed(1)}<span className="text-slate-600 text-xs font-normal">/10</span>
          </p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-slate-600 mb-0.5">Delta</p>
          <p className={clsx("text-sm font-bold tabular-nums", delta > 0 ? "text-emerald-400" : "text-rose-400")}>
            {delta > 0 ? "+" : ""}{delta.toFixed(1)}
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="tabular-nums">
          {p.name}: {p.value.toFixed(1)}/10
        </p>
      ))}
    </div>
  );
}

export function BenchmarkDashboard() {
  const { data, isLoading, error } = useBenchmarkSummary();

  const isPending = data?.status === "pending" || error;
  const categories = data?.categories ?? MOCK_CATEGORIES;
  const overall = data?.overall_metrics ?? MOCK_OVERALL;

  const chartData = categories.map((c: BenchmarkCategory) => ({
    name: c.name,
    "FinRAG-Eval": c.finrag_score,
    "Naive RAG": c.naive_rag_score,
    questions: c.questions,
  }));

  const radialData = [
    { name: "Faithfulness", value: overall.finrag_faithfulness, fill: EMERALD },
    { name: "Citation", value: overall.finrag_citation_accuracy, fill: "#38bdf8" },
    { name: "Relevance", value: overall.finrag_relevance, fill: "#f59e0b" },
  ];

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Trophy className="w-5 h-5 text-amber-400 shrink-0" />
        <div>
          <h2 className="text-sm font-semibold text-slate-100">
            IndFinRAG-Bench Leaderboard
          </h2>
          <p className="text-[11px] text-slate-500">
            100-question Indian equities stress-test across 4 categories
          </p>
        </div>
        {isLoading && (
          <RefreshCw className="w-3.5 h-3.5 text-slate-500 animate-spin ml-auto" />
        )}
      </div>

      {isPending && (
        <div className="flex items-start gap-2 text-xs text-amber-400 bg-amber-950/20 border border-amber-800/40 rounded-md px-3 py-2">
          <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span>
            Benchmark not yet executed. Showing illustrative data.
            Run <code className="font-mono bg-slate-800 px-1 rounded">python benchmarks/ind_fin_bench.py</code> to generate real metrics.
          </span>
        </div>
      )}

      {/* Overall Score Cards */}
      <section>
        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-3">
          Overall Metrics — FinRAG-Eval vs Naive RAG
        </p>
        <div className="grid grid-cols-1 gap-3">
          <MetricCard label="Faithfulness" finrag={overall.finrag_faithfulness} naive={overall.naive_faithfulness} />
          <MetricCard label="Citation Accuracy" finrag={overall.finrag_citation_accuracy} naive={overall.naive_citation_accuracy} />
          <MetricCard label="Answer Relevance" finrag={overall.finrag_relevance} naive={overall.naive_relevance} />
        </div>
      </section>

      {/* Category Bar Chart */}
      <section>
        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-3">
          Per-Category Comparison (Score / 10)
        </p>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#64748b", fontSize: 9 }}
                axisLine={{ stroke: "#1e293b" }}
                tickLine={false}
                interval={0}
                width={60}
              />
              <YAxis
                domain={[0, 10]}
                tick={{ fill: "#64748b", fontSize: 9 }}
                axisLine={{ stroke: "#1e293b" }}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: "10px", color: "#64748b" }}
                iconType="square"
                iconSize={8}
              />
              <Bar
                dataKey="FinRAG-Eval"
                fill={EMERALD}
                radius={[3, 3, 0, 0]}
                maxBarSize={28}
              />
              <Bar
                dataKey="Naive RAG"
                fill={SLATE}
                radius={[3, 3, 0, 0]}
                maxBarSize={28}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Radial Score */}
      <section>
        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-3">
          FinRAG-Eval Score Profile
        </p>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col items-center">
          <ResponsiveContainer width="100%" height={180}>
            <RadialBarChart
              cx="50%"
              cy="50%"
              innerRadius="30%"
              outerRadius="90%"
              data={radialData}
              startAngle={90}
              endAngle={-270}
            >
              <RadialBar
                dataKey="value"
                cornerRadius={4}
                background={{ fill: "#1e293b" }}
                label={false}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.[0]) return null;
                  return (
                    <div className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300">
                      {payload[0].payload.name}: {payload[0].value}/10
                    </div>
                  );
                }}
              />
            </RadialBarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2">
            {radialData.map((d) => (
              <div key={d.name} className="flex items-center gap-1.5">
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ background: d.fill }}
                />
                <span className="text-[10px] text-slate-500">{d.name}</span>
                <span className="text-[10px] font-mono" style={{ color: d.fill }}>
                  {d.value.toFixed(1)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
