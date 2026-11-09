"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { api, Citation, ChatMessage } from "@/lib/api";
import { CitationCard } from "@/components/CitationCard";
import { ArticleDrawer } from "@/components/ArticleDrawer";
import { StatusBar } from "@/components/StatusBar";
import { Send, RotateCcw, Scale, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import clsx from "clsx";

// ─── Sample questions ─────────────────────────────────────────────────────────

const SAMPLE_QUESTIONS = [
  "Que dit l'article 5 sur le principe de subsidiarité ?",
  "Quelles sont les compétences exclusives de l'Union européenne ?",
  "Quel article traite de la protection des données personnelles ?",
  "Quelles sont les institutions de l'Union européenne mentionnées dans le TEU ?",
  "Qu'est-ce que la procédure législative ordinaire ?",
];

// ─── Types ────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  streaming?: boolean;
  model?: string;
  contextUsed?: number;
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedArticleId, setSelectedArticleId] = useState<string | null>(null);
  const [streamingContent, setStreamingContent] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  const send = useCallback(
    async (question: string) => {
      if (!question.trim() || loading) return;

      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: question,
      };
      setMessages((prev) => [...prev, userMessage]);
      setInput("");
      setLoading(true);
      setStreamingContent("");

      abortRef.current = new AbortController();

      const assistantId = (Date.now() + 1).toString();

      // Add empty assistant placeholder
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "", streaming: true },
      ]);

      let fullText = "";
      let citations: Citation[] = [];
      let model = "";
      let contextUsed = 0;

      try {
        const gen = api.streamQuery(
          question,
          5,
          abortRef.current.signal
        );

        for await (const chunk of gen) {
          if (chunk.type === "citations") {
            citations = chunk.citations ?? [];
          } else if (chunk.type === "token") {
            fullText += chunk.content ?? "";
            setStreamingContent(fullText);
          } else if (chunk.type === "done") {
            break;
          }
        }
      } catch (err: any) {
        if (err?.name !== "AbortError") {
          // Fallback to non-streaming
          try {
            const res = await api.query(question);
            fullText = res.answer;
            citations = res.citations;
            model = res.model;
            contextUsed = res.context_used;
          } catch {
            fullText = "❌ Erreur: Impossible de joindre l'API. Vérifiez que le backend est démarré.";
          }
        }
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: fullText,
                citations,
                streaming: false,
                model,
                contextUsed,
              }
            : m
        )
      );
      setStreamingContent("");
      setLoading(false);
    },
    [loading]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    send(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  const reset = () => {
    abortRef.current?.abort();
    setMessages([]);
    setStreamingContent("");
    setLoading(false);
  };

  return (
    <div className="flex h-screen flex-col bg-[#0d1117]">
      {/* ── Header ── */}
      <header className="flex shrink-0 items-center justify-between border-b border-[#30363d] bg-[#161b22] px-6 py-3">
        <div className="flex items-center gap-3">
          <Scale className="h-6 w-6 text-amber-400" />
          <div>
            <h1 className="text-lg font-bold text-amber-300 leading-none">
              Legal GraphRAG
            </h1>
            <p className="text-xs text-slate-500">
              EU Treaty Intelligence · TEU &amp; TFEU
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <StatusBar />
          {messages.length > 0 && (
            <button
              onClick={reset}
              className="flex items-center gap-1.5 rounded-md border border-slate-700 px-3 py-1.5 text-xs text-slate-400 hover:border-slate-500 hover:text-slate-300"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Nouvelle conversation
            </button>
          )}
        </div>
      </header>

      {/* ── Main area ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chat column */}
        <main className="flex flex-1 flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-6">
            {messages.length === 0 ? (
              <Welcome onSelect={send} />
            ) : (
              <div className="mx-auto max-w-3xl space-y-6">
                {messages.map((msg, i) => (
                  <MessageBubble
                    key={msg.id}
                    message={msg}
                    streamingContent={
                      msg.streaming ? streamingContent : undefined
                    }
                    onViewArticle={setSelectedArticleId}
                  />
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>

          {/* Input bar */}
          <div className="shrink-0 border-t border-[#30363d] bg-[#161b22] p-4">
            <form
              onSubmit={handleSubmit}
              className="mx-auto flex max-w-3xl gap-3"
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Posez une question juridique sur les traités européens..."
                rows={1}
                disabled={loading}
                className="flex-1 resize-none rounded-xl border border-[#30363d] bg-[#1c2330] px-4 py-3 text-sm text-slate-200 placeholder-slate-500 outline-none focus:border-amber-700 focus:ring-1 focus:ring-amber-700/50 disabled:opacity-50"
                style={{ maxHeight: "160px", overflowY: "auto" }}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-amber-600 text-white transition hover:bg-amber-500 disabled:opacity-40"
              >
                {loading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </button>
            </form>
            <p className="mt-2 text-center text-xs text-slate-600">
              Entrée pour envoyer · Maj+Entrée pour nouvelle ligne
            </p>
          </div>
        </main>
      </div>

      {/* Article Drawer */}
      {selectedArticleId && (
        <ArticleDrawer
          articleId={selectedArticleId}
          onClose={() => setSelectedArticleId(null)}
        />
      )}
    </div>
  );
}

// ─── Welcome screen ───────────────────────────────────────────────────────────

function Welcome({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center gap-8 pt-12 text-center">
      <div>
        <Scale className="mx-auto mb-4 h-16 w-16 text-amber-400 opacity-80" />
        <h2 className="text-3xl font-bold text-amber-300">
          Bienvenue dans Legal GraphRAG
        </h2>
        <p className="mt-2 text-slate-400">
          Interrogez les Traités sur l&apos;Union Européenne (TEU) et sur le
          Fonctionnement de l&apos;Union Européenne (TFUE) avec la précision du
          Graph RAG.
        </p>
      </div>

      <div className="grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
        {SAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="rounded-lg border border-[#30363d] bg-[#1c2330] p-3 text-left text-sm text-slate-300 transition hover:border-amber-700/50 hover:bg-[#242d3d] hover:text-amber-200"
          >
            {q}
          </button>
        ))}
      </div>

      <p className="text-xs text-slate-600">
        Alimenté par Neo4j · ChromaDB · Ollama
      </p>
    </div>
  );
}

// ─── Message Bubble ───────────────────────────────────────────────────────────

function MessageBubble({
  message,
  streamingContent,
  onViewArticle,
}: {
  message: Message;
  streamingContent?: string;
  onViewArticle: (id: string) => void;
}) {
  const isUser = message.role === "user";
  const content = message.streaming ? streamingContent ?? "" : message.content;

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-amber-800/30 border border-amber-800/40 px-4 py-3 text-sm text-amber-100">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-amber-900/50 text-amber-400">
        <Scale className="h-4 w-4" />
      </div>
      <div className="flex-1 space-y-4">
        {/* Answer */}
        <div className="rounded-2xl rounded-tl-sm border border-[#30363d] bg-[#1c2330] px-5 py-4">
          {content ? (
            <div className="prose-legal text-sm text-slate-200">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
              {message.streaming && (
                <span className="cursor-blink ml-0.5 inline-block h-4 w-0.5 bg-amber-400" />
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Réflexion en cours...
            </div>
          )}

          {/* Meta */}
          {!message.streaming && message.model && (
            <p className="mt-3 text-xs text-slate-600">
              {message.model} · {message.contextUsed} articles utilisés
            </p>
          )}
        </div>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-500">
              Sources · {message.citations.length} articles
            </p>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {message.citations.map((cit, i) => (
                <CitationCard
                  key={cit.article_id}
                  citation={cit}
                  index={i}
                  onViewArticle={onViewArticle}
                  compact
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
