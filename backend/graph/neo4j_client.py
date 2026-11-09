"""
neo4j_client.py — Neo4j connection management and base query utilities.
Uses the official neo4j Python driver.
"""
import sys
import time
from contextlib import contextmanager
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import TransientError
from rich.console import Console

console = Console()


class Neo4jClient:
    """Thread-safe Neo4j connection wrapper."""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Driver | None = None

    def connect(self) -> bool:
        """Establish connection and verify connectivity."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            self._driver.verify_connectivity()
            console.print(f"[bold green]✓ Connected to Neo4j:[/] {self.uri}")
            return True
        except Exception as e:
            console.print(f"[bold red]✗ Neo4j connection failed:[/] {e}")
            console.print("[yellow]Make sure Neo4j is running: docker-compose up -d[/]")
            return False

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self, database: str = "neo4j"):
        """Context manager for a Neo4j session."""
        if not self._driver:
            raise RuntimeError("Not connected. Call connect() first.")
        session: Session = self._driver.session(database=database)
        try:
            yield session
        finally:
            session.close()

    def run(self, query: str, parameters: dict = None, **kwargs) -> list[dict]:
        """Run a Cypher query and return results as list of dicts.
        Retries up to 3 times on TransientError (e.g. deadlock).
        """
        params = parameters or {}
        params.update(kwargs)
        for attempt in range(3):
            try:
                with self.session() as session:
                    result = session.run(query, params)
                    return [dict(record) for record in result]
            except TransientError as e:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    raise

    def run_write(self, query: str, parameters: dict = None, **kwargs) -> list[dict]:
        """Run a write Cypher query."""
        return self.run(query, parameters, **kwargs)

    def clear_all(self, confirm: bool = False):
        """Delete all nodes and relationships. Use with caution!"""
        if not confirm:
            console.print("[yellow]Pass confirm=True to clear the database[/]")
            return
        self.run("MATCH (n) DETACH DELETE n")
        console.print("[bold red]⚠️  Database cleared[/]")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


# ─── Schema Setup ─────────────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    # Constraints (unique IDs)
    "CREATE CONSTRAINT treaty_id IF NOT EXISTS FOR (t:Treaty) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT title_id IF NOT EXISTS FOR (t:Title) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT institution_id IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT country_id IF NOT EXISTS FOR (c:Country) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:LegalConcept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT protocol_id IF NOT EXISTS FOR (p:Protocol) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",

    # Indexes for fast lookups
    "CREATE INDEX article_number IF NOT EXISTS FOR (a:Article) ON (a.number)",
    "CREATE INDEX article_treaty IF NOT EXISTS FOR (a:Article) ON (a.treaty)",
    "CREATE FULLTEXT INDEX article_text IF NOT EXISTS FOR (a:Article) ON EACH [a.text]",
]


def setup_schema(client: Neo4jClient):
    """Create all constraints and indexes."""
    console.print("[cyan]⚙️  Setting up Neo4j schema...[/]")
    for query in SCHEMA_QUERIES:
        try:
            client.run(query)
        except Exception as e:
            # Constraint may already exist
            console.print(f"[yellow]  Schema note:[/] {e}")
    console.print("[bold green]✓ Schema ready[/]")


# ─── Singleton ────────────────────────────────────────────────────────────────

_client: Neo4jClient | None = None


def get_client() -> Neo4jClient:
    """Get or create the singleton Neo4j client."""
    global _client
    if _client is None:
        from backend.config import settings
        _client = Neo4jClient(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        _client.connect()
    return _client


if __name__ == "__main__":
    from backend.config import settings
    client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    if client.connect():
        setup_schema(client)
        result = client.run("RETURN 'Neo4j is ready!' as msg")
        print(result)
        client.close()
