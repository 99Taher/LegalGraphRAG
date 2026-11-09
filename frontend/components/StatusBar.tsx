"use client";

import { useEffect, useState } from "react";
import { api, HealthResponse } from "@/lib/api";
import clsx from "clsx";

export function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const check = () =>
      api
        .health()
        .then(setHealth)
        .catch(() => setHealth(null));
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  if (!health) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-500">
        <span className="h-2 w-2 rounded-full bg-slate-600 animate-pulse" />
        Connexion...
      </div>
    );
  }

  const allOk = health.neo4j && health.chromadb;

  return (
    <div className="flex items-center gap-3 text-xs text-slate-500">
      <StatusDot label="Neo4j" ok={health.neo4j} />
      <StatusDot label="ChromaDB" ok={health.chromadb} />
      <StatusDot label="LLM" ok={health.ollama} />
      {allOk && (
        <span className="hidden text-slate-600 sm:inline">
          {health.articles_count} articles · {health.chunks_count} chunks
        </span>
      )}
    </div>
  );
}

function StatusDot({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="flex items-center gap-1">
      <span
        className={clsx(
          "h-2 w-2 rounded-full",
          ok ? "bg-green-500" : "bg-red-500"
        )}
      />
      <span className={ok ? "text-slate-400" : "text-red-400"}>{label}</span>
    </div>
  );
}

