/**
 * api.ts — Typed API client for the Legal GraphRAG backend.
 * Base URL: http://localhost:8000
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Citation {
  article: string;
  article_id: string;
  treaty: string;
  title: string | null;
  score: number;
  sources: ("vector" | "graph")[];
}

export interface QueryResponse {
  question: string;
  answer: string;
  citations: Citation[];
  context_used: number;
  model: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ArticleResponse {
  article_id: string;
  number: string;
  treaty: string;
  text: string;
  title_num?: string;
  title_name?: string;
  chapter_num?: string;
  chapter_name?: string;
  ex_article?: string;
  page_start: number;
  page_end: number;
}

export interface ArticleSearchResult {
  article_id: string;
  number: string;
  treaty: string;
  title: string | null;
  preview: string;
  score: number;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, string>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relationship: string;
  weight?: number;
}

export interface SubgraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  center_id: string;
}

export interface HealthResponse {
  status: string;
  neo4j: boolean;
  chromadb: boolean;
  ollama: boolean;
  ollama_model: string;
  chunks_count: number;
  articles_count: number;
}

// ─── API Client ───────────────────────────────────────────────────────────────

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetchJSON<HealthResponse>("/health"),

  query: (question: string, top_k = 5) =>
    fetchJSON<QueryResponse>("/api/query", {
      method: "POST",
      body: JSON.stringify({ question, top_k }),
    }),

  chat: (question: string, history: ChatMessage[], top_k = 5) =>
    fetchJSON<{ answer: string; citations: Citation[]; model: string }>(
      "/api/chat",
      {
        method: "POST",
        body: JSON.stringify({ question, history, top_k }),
      }
    ),

  getArticle: (articleId: string) =>
    fetchJSON<ArticleResponse>(`/api/articles/${articleId}`),

  searchArticles: (q: string, treaty?: string, limit = 10) => {
    const params = new URLSearchParams({ q, limit: String(limit) });
    if (treaty) params.set("treaty", treaty);
    return fetchJSON<ArticleSearchResult[]>(`/api/articles?${params}`);
  },

  getSubgraph: (articleId: string) =>
    fetchJSON<SubgraphResponse>(`/api/graph/${articleId}`),

  /**
   * Query with streaming (Server-Sent Events).
   * Returns an EventSource-compatible URL — caller must handle SSE.
   */
  streamQuery: async function* (
    question: string,
    top_k = 5,
    signal?: AbortSignal
  ): AsyncGenerator<{ type: string; content?: string; citations?: Citation[] }> {
    const res = await fetch(`${BASE_URL}/api/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k }),
      signal,
    });

    if (!res.ok || !res.body) throw new Error("Stream failed");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const data = line.replace(/^data: /, "").trim();
        if (data) {
          try {
            yield JSON.parse(data);
          } catch {}
        }
      }
    }
  },
};

