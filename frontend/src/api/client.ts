/**
 * Axios HTTP client + React Query hooks for FinRAG-Eval API.
 * Base URL: http://localhost:8000
 */

import axios from "axios";
import {
  useQuery,
  useMutation,
  type UseQueryOptions,
} from "@tanstack/react-query";
import type {
  QueryRequest,
  QueryResponse,
  PresetRequest,
  PresetName,
  DocumentsResponse,
  IngestResponse,
  HealthResponse,
  BenchmarkSummary,
  ExportPDFRequest,
} from "../types/api";

export const API_BASE = "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120_000,
});

// ─── Query Keys ──────────────────────────────────────────────────────────────
export const QUERY_KEYS = {
  health: ["health"] as const,
  documents: ["documents"] as const,
  benchmark: ["benchmark"] as const,
};

// ─── API Functions ────────────────────────────────────────────────────────────
export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/api/health");
  return data;
}

export async function fetchDocuments(): Promise<DocumentsResponse> {
  const { data } = await apiClient.get<DocumentsResponse>("/api/documents");
  return data;
}

export async function runQuery(req: QueryRequest): Promise<QueryResponse> {
  const { data } = await apiClient.post<QueryResponse>("/api/query", req);
  return data;
}

export async function runPreset(
  presetName: PresetName,
  req: PresetRequest
): Promise<QueryResponse> {
  const { data } = await apiClient.post<QueryResponse>(
    `/api/preset/${presetName}`,
    req
  );
  return data;
}

export async function ingestDocument(formData: FormData): Promise<IngestResponse> {
  const { data } = await apiClient.post<IngestResponse>("/api/ingest", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function fetchBenchmarkSummary(): Promise<BenchmarkSummary> {
  const { data } = await apiClient.get<BenchmarkSummary>("/api/benchmark/summary");
  return data;
}

export async function exportSEBIPDF(req: ExportPDFRequest): Promise<Blob> {
  const { data } = await apiClient.post("/api/audit/export-pdf", req, {
    responseType: "blob",
  });
  return data;
}

// ─── React Query Hooks ────────────────────────────────────────────────────────
export function useHealth(
  options?: Partial<UseQueryOptions<HealthResponse>>
) {
  return useQuery<HealthResponse>({
    queryKey: QUERY_KEYS.health,
    queryFn: fetchHealth,
    refetchInterval: 30_000,
    ...options,
  });
}

export function useDocuments(
  options?: Partial<UseQueryOptions<DocumentsResponse>>
) {
  return useQuery<DocumentsResponse>({
    queryKey: QUERY_KEYS.documents,
    queryFn: fetchDocuments,
    refetchInterval: 10_000,
    ...options,
  });
}

export function useBenchmarkSummary(
  options?: Partial<UseQueryOptions<BenchmarkSummary>>
) {
  return useQuery<BenchmarkSummary>({
    queryKey: QUERY_KEYS.benchmark,
    queryFn: fetchBenchmarkSummary,
    ...options,
  });
}

export function useQueryMutation() {
  return useMutation({
    mutationFn: runQuery,
  });
}

export function usePresetMutation() {
  return useMutation({
    mutationFn: ({ name, req }: { name: PresetName; req: PresetRequest }) =>
      runPreset(name, req),
  });
}

export function useIngestMutation() {
  return useMutation({
    mutationFn: ingestDocument,
  });
}
