import { Citation } from "@/lib/api";
import { clsx } from "clsx";

interface CitationCardProps {
  citation: Citation;
  index: number;
  onViewArticle?: (articleId: string) => void;
  compact?: boolean;
}

const treatyColors: Record<string, string> = {
  TEU: "bg-amber-900/40 text-amber-300 border-amber-700",
  TFEU: "bg-blue-900/40 text-blue-300 border-blue-700",
};

const sourceIcons: Record<string, string> = {
  vector: "🔍",
  graph: "🕸",
};

export function CitationCard({
  citation,
  index,
  onViewArticle,
  compact = false,
}: CitationCardProps) {
  const treatyStyle =
    treatyColors[citation.treaty] ??
    "bg-slate-700/40 text-slate-300 border-slate-600";
  const scorePercent = Math.min(100, Math.round((citation.score / 15) * 100));

  return (
    <div
      className={clsx(
        "rounded-lg border border-slate-700 bg-slate-800/50 transition-colors hover:border-amber-700/50",
        compact ? "p-3" : "p-4"
      )}
    >
      <div className="flex items-start justify-between gap-2">
        {/* Number badge */}
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-amber-900/60 text-xs font-bold text-amber-300">
          {index + 1}
        </span>

        {/* Treaty badge */}
        <span
          className={clsx(
            "rounded border px-2 py-0.5 text-xs font-semibold",
            treatyStyle
          )}
        >
          {citation.treaty}
        </span>
      </div>

      {/* Article name */}
      <button
        onClick={() => onViewArticle?.(citation.article_id)}
        className="mt-2 w-full text-left font-semibold text-amber-200 hover:text-amber-100 hover:underline"
      >
        {citation.article}
      </button>

      {/* Title */}
      {citation.title && !compact && (
        <p className="mt-0.5 text-xs text-slate-400 line-clamp-1">
          {citation.title}
        </p>
      )}

      {/* Score bar */}
      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
          <span>Relevance</span>
          <span>{citation.score.toFixed(2)}</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-slate-700">
          <div
            className="h-1.5 rounded-full bg-amber-500 transition-all"
            style={{ width: `${scorePercent}%` }}
          />
        </div>
      </div>

      {/* Sources */}
      <div className="mt-2 flex gap-1">
        {citation.sources.map((s) => (
          <span
            key={s}
            className="rounded bg-slate-700 px-1.5 py-0.5 text-xs text-slate-400"
            title={s === "vector" ? "Vector similarity" : "Graph traversal"}
          >
            {sourceIcons[s]} {s}
          </span>
        ))}
      </div>
    </div>
  );
}

