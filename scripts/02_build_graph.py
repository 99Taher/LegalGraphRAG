"""
02_build_graph.py — Phase 2: Build Neo4j knowledge graph.

Prerequisites:
  - Run 01_extract.py first
  - Neo4j must be running: docker-compose up -d

Run: python scripts/02_build_graph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from backend.config import settings
from backend.graph.neo4j_client import Neo4jClient
from backend.graph.graph_builder import GraphBuilder

console = Console()


def main():
    console.print(Panel.fit(
        "⚖️  [bold cyan]Legal GraphRAG — Phase 2: Knowledge Graph Construction[/bold cyan]",
        border_style="cyan"
    ))

    # Check prerequisites
    required_files = [
        f"{settings.structured_dir}/articles.json",
        f"{settings.structured_dir}/protocols.json",
    ]
    for f in required_files:
        if not Path(f).exists():
            console.print(f"[red]❌ Missing: {f}[/]")
            console.print("[yellow]Run: python scripts/01_extract.py first[/]")
            return

    # Connect to Neo4j
    console.rule("[bold]Connecting to Neo4j[/bold]")
    client = Neo4jClient(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )

    if not client.connect():
        console.print(Panel(
            "[red]❌ Cannot connect to Neo4j[/red]\n\n"
            "Start Neo4j with Docker:\n"
            "[cyan]  docker-compose up -d[/cyan]\n\n"
            "Then wait ~30 seconds and retry.",
            border_style="red"
        ))
        return

    # Wipe existing graph to avoid constraint errors from stale data
    console.print("[yellow]🗑️  Clearing existing graph data...[/]")
    client.clear_all(confirm=True)
    console.print("[green]✓ Graph cleared[/]")

    # Build graph
    console.rule("[bold]Building Knowledge Graph[/bold]")
    builder = GraphBuilder(client)
    builder.build_all(
        articles_path=f"{settings.structured_dir}/articles.json",
        protocols_path=f"{settings.structured_dir}/protocols.json",
        chunks_path=f"{settings.structured_dir}/chunks.json"
        if Path(f"{settings.structured_dir}/chunks.json").exists() else None,
    )

    client.close()

    console.print()
    console.print(Panel.fit(
        "✅ [bold green]Knowledge Graph Built![/bold green]\n\n"
        "  🌐 View graph: [link=http://localhost:7474]http://localhost:7474[/link]\n"
        "  📊 Try Cypher: [cyan]MATCH (n:Article) RETURN n LIMIT 25[/cyan]\n\n"
        "  Next: [cyan]python scripts/03_embed.py[/cyan]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
