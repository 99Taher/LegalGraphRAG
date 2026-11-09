"""
04_test_retrieval.py — Phase 4: Test the full retrieval pipeline.

Run: python scripts/04_test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.config import settings
from backend.graph.neo4j_client import Neo4jClient
from backend.graph.graph_retriever import GraphRetriever
from backend.vector.vector_store import VectorStore
from backend.vector.embedder import Embedder
from backend.retrieval.hybrid_retriever import HybridRetriever

console = Console()

TEST_QUERIES = [
    "Quelles sont les compétences de la Cour de justice de l'Union européenne ?",
    "Qu'est-ce que le principe de subsidiarité ?",
    "Quelles sont les missions du Parlement européen ?",
    "Comment fonctionne la Banque centrale européenne ?",
    "Quels sont les droits fondamentaux reconnus par l'Union européenne ?",
]


def main():
    console.print(Panel.fit(
        "⚖️  [bold cyan]Legal GraphRAG — Phase 4: Retrieval Test[/bold cyan]",
        border_style="cyan"
    ))

    # Initialize services
    client = Neo4jClient(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
    )
    neo4j_ok = client.connect()

    store = VectorStore(settings.chroma_persist_dir)
    embedder = Embedder()

    retriever = HybridRetriever(
        graph_retriever=GraphRetriever(client) if neo4j_ok else None,
        vector_store=store,
        embedder=embedder,
    )

    # Print system status
    status_table = Table(title="System Status")
    status_table.add_column("Service", style="cyan")
    status_table.add_column("Status", style="bold")
    status_table.add_column("Info", style="yellow")
    status_table.add_row(
        "Neo4j", "✅ Connected" if neo4j_ok else "❌ Offline",
        "Run: docker-compose up -d" if not neo4j_ok else ""
    )
    status_table.add_row("ChromaDB", f"✅ {store.count()} chunks", "")
    console.print(status_table)

    if store.count() == 0:
        console.print("[red]❌ ChromaDB is empty. Run: python scripts/03_embed.py[/]")
        return

    # Run test queries
    console.rule("[bold]Running Test Queries[/bold]")
    for i, query in enumerate(TEST_QUERIES, 1):
        console.print(f"\n[bold cyan]Query {i}:[/bold cyan] {query}")

        results = retriever.retrieve(query, top_k=3)

        if not results:
            console.print("  [yellow]No results found[/]")
            continue

        for j, r in enumerate(results, 1):
            console.print(
                f"  [green]{j}.[/green] [{r.hybrid_score:.3f}] "
                f"[bold]Article {r.article_num} ({r.treaty})[/bold] "
                f"| Sources: {', '.join(r.sources)}"
            )
            preview = r.text[:150].replace('\n', ' ')
            console.print(f"     [dim]{preview}...[/dim]")

    if neo4j_ok:
        client.close()

    console.print()
    console.print(Panel.fit(
        "✅ [bold green]Retrieval Test Complete![/bold green]\n\n"
        "  Ready to start API:\n"
        "  [cyan]python -m backend.api.main[/cyan]\n\n"
        "  API docs: [link=http://localhost:8000/docs]http://localhost:8000/docs[/link]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
