/**
 * TypeScript interfaces for FinRAG-Eval API.
 * Mirrors FastAPI Pydantic schemas 1-to-1 for full type safety.
 * Source: docs/DetectionLogic.md Section 5 & docs/DesignDocument.md Section 3
 */

export type VerificationType =
  | "VERIFIED_VERBATIM"
  | "VERIFIED_UNIT_CONVERSION"
  | "VERIFIED_AST_DERIVATION"
  | "UNVERIFIED";

export interface VerbatimMatch {
  number: string;
  chunk_id: string;
  page_no: number;
  verification_type: "VERIFIED_VERBATIM";
}

export interface UnitConvertedMatch {
  original: string;
  converted: string;
  factor: number;
  chunk_id: string;
  verification_type: "VERIFIED_UNIT_CONVERSION";
}

export interface ASTMathTrace {
  formula_string: string;
  operands: Record<string, number>;
  computed_value: number | null;
  claimed_value: string;
  tolerance: number;
  match: boolean;
  verification_type: "VERIFIED_AST_DERIVATION" | "UNVERIFIED";
  source_chunks: string[];
  error?: string;
}

export interface DeterministicReport {
  numbers_in_answer: string[];
  unverified_numbers: string[];
  numeric_match_pass: boolean;
  verbatim_matches: VerbatimMatch[];
  unit_converted_matches: UnitConvertedMatch[];
  ast_math_traces: ASTMathTrace[];
}

export interface LLMJudgeScores {
  faithfulness_score: number;
  citation_accuracy_score: number;
  answer_relevance_score: number;
  hallucination_flag: boolean;
  unverified_claims: string[];
  audit_reasoning: string;
  reasoning?: string; // legacy compat
}

export interface EvalReport {
  query_id: string;
  deterministic_checks: DeterministicReport;
  llm_judge_scores: LLMJudgeScores;
  hallucination_flag?: boolean;
}

export interface ChunkMetadata {
  company: string;
  doc_type: string;
  quarter: string;
  page_no: number;
  contains_table: boolean;
  source_file: string;
}

export interface GoldenChunk {
  chunk_id: string;
  text: string;
  rerank_score: number;
  metadata: ChunkMetadata;
}

export interface QueryRequest {
  query: string;
  company?: string;
  quarter?: string;
  doc_type?: string;
  top_k?: number;
  top_n?: number;
}

export interface QueryResponse {
  query: string;
  answer: string;
  golden_chunks: GoldenChunk[];
  eval_report: EvalReport;
  metadata_filter: Record<string, string>;
}

export interface PresetRequest {
  company: string;
  quarter?: string;
  q_prev?: string;
  q_curr?: string;
}

export type PresetName =
  | "red_flag_scanner"
  | "guidance_vs_actuals"
  | "growth_triggers_capex";

export interface DocumentMeta {
  source_file: string;
  company: string;
  quarter: string;
  doc_type: string;
  chunks_count: number;
  tables_count: number;
}

export interface DocumentsResponse {
  documents: DocumentMeta[];
  total_chunks: number;
}

export interface IngestResponse {
  status: string;
  message: string;
  source_file: string;
  company: string;
  quarter: string;
  doc_type: string;
  pages_count: number;
  chunks_count: number;
  tables_count: number;
  index_stats: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  engine: string;
  version: string;
  collection_name: string;
  indexed_chunks_count: number;
  models: {
    embedding: string;
    reranker: string;
    analyst_llm: string;
    eval_llm: string;
  };
}

export interface ExportPDFRequest {
  query: string;
  answer: string;
  eval_report: EvalReport;
  golden_chunks: GoldenChunk[];
  company: string;
  quarter: string;
  firm_name?: string;
  registration_no?: string;
  principal_officer?: string;
}

export interface BenchmarkCategory {
  name: string;
  finrag_score: number;
  naive_rag_score: number;
  questions: number;
}

export interface BenchmarkSummary {
  status: string;
  message?: string;
  categories?: BenchmarkCategory[];
  overall_metrics?: {
    finrag_faithfulness: number;
    finrag_citation_accuracy: number;
    finrag_relevance: number;
    naive_faithfulness: number;
    naive_citation_accuracy: number;
    naive_relevance: number;
  };
}

export interface SEBIProfile {
  firmName: string;
  registrationNo: string;
  analystName: string;
}

export interface ActiveFilters {
  company: string;
  quarter: string;
  docType: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  queryResponse?: QueryResponse;
  isPreset?: boolean;
  presetName?: PresetName;
}
