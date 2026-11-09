"""
03_embed.py — Phase 3: Generate embeddings and populate ChromaDB.

Prerequisites:
  - Run 01_extract.py first
  - sentence-transformers installed (pip install sentence-transformers)

Run: python scripts/03_embed.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from backend.config import settings
from backend.vector.embedder import Embedder
from backend.vector.vector_store import VectorStore

console = Console()


def main():
    console.print(Panel.fit(
        "⚖️  [bold cyan]Legal GraphRAG — Phase 3: Embeddings & Vector Store[/bold cyan]",
        border_style="cyan"
    ))

    chunks_path = f"{settings.structured_dir}/chunks.json"
    if not Path(chunks_path).exists():
        console.print(f"[red]❌ Missing: {chunks_path}[/]")
        console.print("[yellow]Run: python scripts/01_extract.py first[/]")
        return

    # ── Embedding ──────────────────────────────────────────────────────────
    console.rule("[bold]Step 1: Generate Embeddings[/bold]")
    console.print("[yellow]Note: First run will download the model (~400MB)[/]")

    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(
        chunks_path=chunks_path,
        output_path=f"{settings.structured_dir}/embeddings_meta.json",
    )

    # ── Store in ChromaDB ──────────────────────────────────────────────────
    console.rule("[bold]Step 2: Store in ChromaDB[/bold]")
    store = VectorStore(settings.chroma_persist_dir)
    store.add_embeddings(embedded_chunks)

    console.print()
    console.print(Panel.fit(
        f"✅ [bold green]Embeddings Ready![/bold green]\n\n"
        f"  🧩 Chunks embedded: {len(embedded_chunks)}\n"
        f"  🗄️  ChromaDB items:  {store.count()}\n\n"
        "  Next: [cyan]python scripts/04_test_retrieval.py[/cyan]\n"
        "  Then: [cyan]python -m backend.api.main[/cyan]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
