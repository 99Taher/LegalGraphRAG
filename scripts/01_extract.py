"""
01_extract.py — Phase 1: Extract text from PDF and parse article structure.

Run: python scripts/01_extract.py

Output:
  data/raw/full_text.txt          ← Raw extracted text
  data/structured/articles.json  ← Parsed articles
  data/structured/protocols.json ← Parsed protocols
  data/structured/chunks.json    ← Text chunks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from backend.config import settings
from backend.ingestion.pdf_extractor import extract_pdf, save_raw_text
from backend.ingestion.article_parser import parse_document, save_structured_data
from backend.ingestion.chunker import chunk_articles, save_chunks

console = Console()


def main():
    settings.ensure_dirs()

    console.print(Panel.fit(
        "⚖️  [bold cyan]Legal GraphRAG — Phase 1: Document Extraction[/bold cyan]",
        border_style="cyan"
    ))

    # ── Step 1: Extract PDF ────────────────────────────────────────────────
    console.rule("[bold]Step 1: PDF Extraction[/bold]")
    pdf_path = settings.pdf_path
    if not Path(pdf_path).exists():
        console.print(f"[red]❌ PDF not found: {pdf_path}[/]")
        console.print("[yellow]Place your document.pdf in the project root.[/]")
        return

    extracted = extract_pdf(pdf_path)
    raw_txt_path = save_raw_text(extracted, settings.raw_dir)

    # ── Step 2: Parse Article Structure ───────────────────────────────────
    console.rule("[bold]Step 2: Article Parsing[/bold]")
    full_text = Path(raw_txt_path).read_text(encoding="utf-8")
    articles, protocols = parse_document(full_text)
    save_structured_data(articles, protocols, settings.structured_dir)

    # ── Step 3: Chunking ───────────────────────────────────────────────────
    console.rule("[bold]Step 3: Smart Chunking[/bold]")
    chunks = chunk_articles(
        f"{settings.structured_dir}/articles.json",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    save_chunks(chunks, settings.structured_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        f"✅ [bold green]Extraction Complete![/bold green]\n\n"
        f"  📄 Pages:     {extracted.total_pages}\n"
        f"  📖 Articles:  {len(articles)}\n"
        f"  📋 Protocols: {len(protocols)}\n"
        f"  🧩 Chunks:    {len(chunks)}\n\n"
        f"  Next: [cyan]python scripts/02_build_graph.py[/cyan]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
