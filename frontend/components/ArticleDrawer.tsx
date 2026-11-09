"use client";

import { useEffect, useState } from "react";
import { api, ArticleResponse } from "@/lib/api";
import { X, FileText, Loader2, ExternalLink } from "lucide-react";
import clsx from "clsx";

interface ArticleDrawerProps {
  articleId: string;
  onClose: () => void;
}

const treatyColors: Record<string, string> = {
  TEU: "bg-amber-900/40 text-amber-300 border-amber-700",
  TFEU: "bg-blue-900/40 text-blue-300 border-blue-700",
};

export function ArticleDrawer({ articleId, onClose }: ArticleDrawerProps) {
  const [article, setArticle] = useState<ArticleResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .getArticle(articleId)
      .then(setArticle)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [articleId]);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Drawer */}
      <aside className="fixed right-0 top-0 z-50 flex h-full w-full max-w-xl flex-col border-l border-[#30363d] bg-[#161b22] shadow-2xl">
        {/* Header */}
        <div className="flex shrink-0 items-center justify-between border-b border-[#30363d] px-6 py-4">
          <div className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-amber-400" />
            <span className="font-semibold text-amber-200">
              {article ? article.article_id : articleId}
            </span>
            {article && (
              <span
                className={clsx(
                  "ml-1 rounded border px-2 py-0.5 text-xs font-semibold",
                  treatyColors[article.treaty] ??
                    "bg-slate-700 text-slate-300 border-slate-600"
                )}
              >
                {article.treaty}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {loading && (
            <div className="flex h-40 items-center justify-center gap-2 text-slate-500">
              <Loader2 className="h-5 w-5 animate-spin" />
              Chargement de l&apos;article...
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-800 bg-red-900/20 p-4 text-sm text-red-300">
              ❌ {error}
            </div>
          )}

          {article && (
            <div className="space-y-5">
              {/* Breadcrumb */}
              {(article.title_name || article.chapter_name) && (
                <div className="text-xs text-slate-500">
                  {article.title_name && (
                    <span>
                      Titre {article.title_num} — {article.title_name}
                    </span>
                  )}
                  {article.chapter_name && (
                    <span>
                      {" "}
                      › Chapitre {article.chapter_num} — {article.chapter_name}
                    </span>
                  )}
                </div>
              )}

              {/* Ex-article */}
              {article.ex_article && (
                <p className="text-xs italic text-slate-500">
                  {article.ex_article}
                </p>
              )}

              {/* Full text */}
              <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-4">
                <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-slate-200">
                  {article.text}
                </pre>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-3 text-xs text-slate-500">
                <div className="rounded-md border border-[#30363d] bg-[#1c2330] p-3">
                  <p className="font-semibold text-slate-400">Article</p>
                  <p className="mt-1 text-slate-300">{article.number}</p>
                </div>
                <div className="rounded-md border border-[#30363d] bg-[#1c2330] p-3">
                  <p className="font-semibold text-slate-400">Traité</p>
                  <p className="mt-1 text-slate-300">{article.treaty}</p>
                </div>
                <div className="rounded-md border border-[#30363d] bg-[#1c2330] p-3">
                  <p className="font-semibold text-slate-400">Page début</p>
                  <p className="mt-1 text-slate-300">p. {article.page_start}</p>
                </div>
                <div className="rounded-md border border-[#30363d] bg-[#1c2330] p-3">
                  <p className="font-semibold text-slate-400">Page fin</p>
                  <p className="mt-1 text-slate-300">p. {article.page_end}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}

