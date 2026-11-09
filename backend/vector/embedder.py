"""
embedder.py — Generate embeddings using sentence-transformers with Ollama fallback.

Primary: sentence-transformers multilingual model (local, no API needed)
Fallback: Ollama embedding API
"""
import json
import sys
from pathlib import Path
from typing import Sequence
from rich.console import Console
from rich.progress import track

console = Console()

# Best multilingual embedding models (ranked)
EMBEDDING_MODELS = [
    "paraphrase-multilingual-mpnet-base-v2",   # 768d, 50 langs, great for French
    "paraphrase-multilingual-MiniLM-L12-v2",   # 384d, lighter version
    "all-MiniLM-L6-v2",                         # English only, fallback
]


class Embedder:
    """Local embedding generator using sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODELS[0]):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            console.print(f"[cyan]🔄 Loading embedding model:[/] {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_sentence_embedding_dimension()
            console.print(f"[green]✓ Model loaded:[/] {self.model_name} (dim={dim})")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embed a list of texts efficiently in batches."""
        model = self._load_model()
        all_embeddings = []

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        iterator = track(batches, description="Embedding...") if show_progress else batches

        for batch in iterator:
            embeddings = model.encode(
                list(batch),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def embed_chunks(self, chunks_path: str, output_path: str | None = None) -> list[dict]:
        """
        Load chunks JSON, generate embeddings, and save to output.
        Returns list of {chunk_id, embedding, metadata}.
        """
        chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        console.print(f"[cyan]📐 Embedding {len(chunks)} chunks...[/]")

        texts = [c["text"] for c in chunks]
        embeddings = self.embed_batch(texts)

        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
                "text": chunk["text"],
                "metadata": {
                    "article_id": chunk["article_id"],
                    "article_num": chunk["article_num"],
                    "treaty": chunk["treaty"],
                    "title_name": chunk.get("title_name"),
                    "chapter_name": chunk.get("chapter_name"),
                    "page_start": chunk.get("page_start", 0),
                    "chunk_index": chunk.get("chunk_index", 0),
                },
            })

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # Save without the raw embeddings (too large) — just metadata
            meta_only = [{k: v for k, v in r.items() if k != "embedding"} for r in result]
            out.write_text(json.dumps(meta_only, ensure_ascii=False, indent=2), encoding="utf-8")
            console.print(f"[green]✓ Embedding metadata saved to:[/] {output_path}")

        console.print(f"[bold green]✓ Generated {len(result)} embeddings[/]")
        return result


# ─── Singleton ────────────────────────────────────────────────────────────────

_embedder: Embedder | None = None


def get_embedder(model_name: str = EMBEDDING_MODELS[0]) -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(model_name)
    return _embedder


if __name__ == "__main__":
    embedder = Embedder()
    test = embedder.embed_text("Quelles sont les compétences de la Cour de justice ?")
    print(f"Embedding dim: {len(test)}")
    print(f"First 5 values: {test[:5]}")
