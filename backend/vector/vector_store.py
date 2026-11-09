"""
vector_store.py — ChromaDB vector store for semantic retrieval.
Stores chunk embeddings with rich metadata for hybrid retrieval.
"""
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass
class VectorResult:
    """Result from a vector similarity search."""
    chunk_id: str
    text: str
    article_id: str
    article_num: str
    treaty: str
    title_name: str | None
    chapter_name: str | None
    page_start: int
    distance: float
    score: float = 0.0
    source: str = "vector"

    def to_context_str(self) -> str:
        """Format for LLM context."""
        parts = [f"[Article {self.article_num} {self.treaty}]"]
        if self.title_name:
            parts.append(f"Titre: {self.title_name}")
        parts.append("")
        parts.append(self.text)
        return "\n".join(parts)


class VectorStore:
    """ChromaDB-backed vector store for legal chunks."""

    COLLECTION_NAME = "legal_chunks"

    def __init__(self, persist_dir: str = "./data/chroma"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy-init ChromaDB client."""
        if self._client is None:
            import chromadb
            console.print(f"[cyan]🗄️  Initializing ChromaDB:[/] {self.persist_dir}")
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self):
        """Get or create the chunks collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            console.print(f"[green]✓ Collection '{self.COLLECTION_NAME}':[/] {self._collection.count()} items")
        return self._collection

    def add_embeddings(
        self,
        embedded_chunks: list[dict],
        batch_size: int = 100,
    ):
        """
        Add embedded chunks to ChromaDB.
        embedded_chunks: list of {chunk_id, embedding, text, metadata}
        """
        # Deduplicate by chunk_id to prevent ChromaDB upsert errors
        unique_chunks = {}
        for chunk in embedded_chunks:
            unique_chunks[chunk["chunk_id"]] = chunk
        deduped_list = list(unique_chunks.values())

        collection = self._get_collection()
        console.print(f"[cyan]📥 Adding {len(deduped_list)} unique embeddings to ChromaDB...[/]")

        for i in track(range(0, len(deduped_list), batch_size), description="Storing..."):
            batch = deduped_list[i:i + batch_size]

            ids = [c["chunk_id"] for c in batch]
            embeddings = [c["embedding"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = []
            for c in batch:
                meta = c.get("metadata", {})
                # ChromaDB requires all metadata values to be str/int/float/bool
                clean_meta = {
                    k: (str(v) if v is not None else "") if not isinstance(v, (int, float, bool)) else v
                    for k, v in meta.items()
                }
                metadatas.append(clean_meta)

            # Upsert (add or update)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        total = collection.count()
        console.print(f"[bold green]✓ ChromaDB now has {total} chunks[/]")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        treaty_filter: str | None = None,
    ) -> list[VectorResult]:
        """
        Semantic similarity search.
        Returns top-k most similar chunks.
        """
        collection = self._get_collection()

        where = None
        if treaty_filter:
            where = {"treaty": {"$eq": treaty_filter}}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        vector_results = []
        if not results["ids"] or not results["ids"][0]:
            return vector_results

        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance → similarity score
            score = 1.0 - dist

            vector_results.append(VectorResult(
                chunk_id=chunk_id,
                text=doc,
                article_id=meta.get("article_id", ""),
                article_num=meta.get("article_num", ""),
                treaty=meta.get("treaty", ""),
                title_name=meta.get("title_name") or None,
                chapter_name=meta.get("chapter_name") or None,
                page_start=int(meta.get("page_start", 0)),
                distance=dist,
                score=score,
            ))

        return vector_results

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._get_collection().count()

    def clear(self):
        """Delete and recreate the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        console.print("[yellow]⚠️  ChromaDB collection cleared[/]")


# ─── Singleton ────────────────────────────────────────────────────────────────

_store: VectorStore | None = None


def get_vector_store(persist_dir: str = "./data/chroma") -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(persist_dir)
    return _store


if __name__ == "__main__":
    store = VectorStore()
    print(f"Chunks in store: {store.count()}")
