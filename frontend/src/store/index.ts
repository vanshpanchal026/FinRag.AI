/**
 * Zustand global state for FinRAG-Eval Institutional Workbench.
 * Manages active filters, SEBI compliance profile, chat history,
 * and the highlighted chunk for Right Pane provenance inspector.
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  ActiveFilters,
  ChatMessage,
  SEBIProfile,
  QueryResponse,
} from "../types/api";

export const DOC_TYPES = [
  "Financial_Results",
  "Concall_Transcript",
  "Investor_Presentation",
  "Annual_Report",
  "DRHP_RHP",
  "Corporate_Announcement",
] as const;

export type DocType = (typeof DOC_TYPES)[number];

interface FinRAGState {
  // Active context filters for all queries
  filters: ActiveFilters;
  setFilters: (f: Partial<ActiveFilters>) => void;
  resetFilters: () => void;

  // SEBI Reg 24(7) compliance profile
  sebiProfile: SEBIProfile;
  setSebiProfile: (p: Partial<SEBIProfile>) => void;

  // Chat message history (center pane)
  messages: ChatMessage[];
  addMessage: (msg: ChatMessage) => void;
  clearMessages: () => void;

  // Currently active query response (drives right pane)
  activeResponse: QueryResponse | null;
  setActiveResponse: (r: QueryResponse | null) => void;

  // Highlighted chunk (triggered by citation pill click)
  highlightedChunkId: string | null;
  setHighlightedChunkId: (id: string | null) => void;

  // Right pane view
  rightPaneView: "audit" | "benchmark";
  setRightPaneView: (v: "audit" | "benchmark") => void;

  // Upload state
  isIngesting: boolean;
  setIsIngesting: (v: boolean) => void;

  // Query loading state
  isQuerying: boolean;
  setIsQuerying: (v: boolean) => void;
}

const DEFAULT_FILTERS: ActiveFilters = {
  company: "",
  quarter: "",
  docType: "",
};

const DEFAULT_SEBI: SEBIProfile = {
  firmName: "",
  registrationNo: "",
  analystName: "",
};

export const useFinRAGStore = create<FinRAGState>()(
  persist(
    (set) => ({
      filters: DEFAULT_FILTERS,
      setFilters: (f) =>
        set((s) => ({ filters: { ...s.filters, ...f } })),
      resetFilters: () => set({ filters: DEFAULT_FILTERS }),

      sebiProfile: DEFAULT_SEBI,
      setSebiProfile: (p) =>
        set((s) => ({ sebiProfile: { ...s.sebiProfile, ...p } })),

      messages: [],
      addMessage: (msg) =>
        set((s) => ({ messages: [...s.messages, msg] })),
      clearMessages: () => set({ messages: [] }),

      activeResponse: null,
      setActiveResponse: (r) => set({ activeResponse: r }),

      highlightedChunkId: null,
      setHighlightedChunkId: (id) => set({ highlightedChunkId: id }),

      rightPaneView: "audit",
      setRightPaneView: (v) => set({ rightPaneView: v }),

      isIngesting: false,
      setIsIngesting: (v) => set({ isIngesting: v }),

      isQuerying: false,
      setIsQuerying: (v) => set({ isQuerying: v }),
    }),
    {
      name: "finrag-store",
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        filters: s.filters,
        sebiProfile: s.sebiProfile,
        messages: s.messages.slice(-50), // persist last 50 messages
      }),
    }
  )
);
