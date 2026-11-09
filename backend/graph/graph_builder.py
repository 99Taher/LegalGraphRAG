"""
graph_builder.py — Build the Neo4j legal knowledge graph.

Graph schema:

    (Treaty)-[:CONTAINS]->(Title)-[:CONTAINS]->(Chapter)
        -[:CONTAINS]->(Section)-[:CONTAINS]->(Article)
        -[:CONTAINS]->(Paragraph)

    (Protocol)-[:ANNEXED_TO]->(Treaty)
    (Protocol)-[:CONTAINS]->(ProtocolArticle)
    (ProtocolArticle)-[:CONTAINS]->(Paragraph)

    (Article)-[:CITES]->(Article|Paragraph)
    (Article)-[:HAS_CHUNK]->(Chunk)

    (Protocol)-[:RELATED_TO]->(Article|Paragraph)
    (Declaration)-[:REFERS_TO]->(Article|Paragraph)

The builder never guesses a TUE/TFUE target when the parser supplies an
exact target_id. This is important for references such as Article 16 TUE
vs Article 16 TFUE and Article 6(2) TUE vs Article 6 of a protocol.
"""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import settings
from backend.graph.neo4j_client import Neo4jClient, setup_schema

console = Console()


EU_INSTITUTIONS = [
    {"id": "inst-parliament", "name": "Parlement européen", "type": "Legislative"},
    {"id": "inst-council-eu", "name": "Conseil européen", "type": "Executive"},
    {"id": "inst-council", "name": "Conseil", "type": "Legislative"},
    {"id": "inst-commission", "name": "Commission européenne", "type": "Executive"},
    {"id": "inst-cjeu", "name": "Cour de justice de l'Union européenne", "type": "Judicial"},
    {"id": "inst-ecb", "name": "Banque centrale européenne", "type": "Financial"},
    {"id": "inst-court-auditors", "name": "Cour des comptes", "type": "Audit"},
    {"id": "inst-eesc", "name": "Comité économique et social européen", "type": "Advisory"},
    {"id": "inst-cor", "name": "Comité des régions", "type": "Advisory"},
    {"id": "inst-eib", "name": "Banque européenne d'investissement", "type": "Financial"},
]

EU_COUNTRIES = [
    {"id": "country-be", "name": "Belgique", "iso": "BE"},
    {"id": "country-de", "name": "Allemagne", "iso": "DE"},
    {"id": "country-fr", "name": "France", "iso": "FR"},
    {"id": "country-it", "name": "Italie", "iso": "IT"},
    {"id": "country-nl", "name": "Pays-Bas", "iso": "NL"},
    {"id": "country-lu", "name": "Luxembourg", "iso": "LU"},
    {"id": "country-dk", "name": "Danemark", "iso": "DK"},
    {"id": "country-ie", "name": "Irlande", "iso": "IE"},
    {"id": "country-gr", "name": "Grèce", "iso": "GR"},
    {"id": "country-es", "name": "Espagne", "iso": "ES"},
    {"id": "country-pt", "name": "Portugal", "iso": "PT"},
    {"id": "country-at", "name": "Autriche", "iso": "AT"},
    {"id": "country-fi", "name": "Finlande", "iso": "FI"},
    {"id": "country-se", "name": "Suède", "iso": "SE"},
    {"id": "country-pl", "name": "Pologne", "iso": "PL"},
    {"id": "country-cz", "name": "République tchèque", "iso": "CZ"},
    {"id": "country-hu", "name": "Hongrie", "iso": "HU"},
    {"id": "country-sk", "name": "Slovaquie", "iso": "SK"},
    {"id": "country-si", "name": "Slovénie", "iso": "SI"},
    {"id": "country-ee", "name": "Estonie", "iso": "EE"},
    {"id": "country-lv", "name": "Lettonie", "iso": "LV"},
    {"id": "country-lt", "name": "Lituanie", "iso": "LT"},
    {"id": "country-cy", "name": "Chypre", "iso": "CY"},
    {"id": "country-mt", "name": "Malte", "iso": "MT"},
    {"id": "country-bg", "name": "Bulgarie", "iso": "BG"},
    {"id": "country-ro", "name": "Roumanie", "iso": "RO"},
    {"id": "country-hr", "name": "Croatie", "iso": "HR"},
]


class GraphBuilder:
    """Build the legal knowledge graph in Neo4j."""

    def __init__(self, client: Neo4jClient):
        self.client = client
        self._title_cache: dict[str, str] = {}
        self._chapter_cache: dict[str, str] = {}
        self._section_cache: dict[str, str] = {}

    def build_all(
        self,
        articles_path: str,
        protocols_path: str,
        chunks_path: str | None = None,
        declarations_path: str | None = None,
    ) -> None:
        """Build all legal entities and relationships."""

        console.print("\n[bold cyan]🏗️  Building Knowledge Graph[/]")
        console.rule()

        setup_schema(self.client)

        self._create_treaties()
        self._create_charter()
        self._create_institutions()
        self._create_countries()

        protocols = self._load_json(protocols_path)
        articles = self._load_json(articles_path)

        if declarations_path is None:
            declarations_path = str(Path(articles_path).parent / "declarations.json")

        declarations = []
        if Path(declarations_path).exists():
            declarations = self._load_json(declarations_path)
        else:
            console.print(
                f"[yellow]⚠ declarations.json not found:[/] {declarations_path}"
            )

        # Base nodes first.
        self._create_protocols(protocols)
        self._create_articles(articles)
        self._create_declarations(declarations)

        # Relationships after all targets exist.
        self._create_citations(articles)
        self._link_protocol_references(protocols)
        self._link_declaration_references(declarations)

        if chunks_path and Path(chunks_path).exists():
            chunks = self._load_json(chunks_path)
            self._create_chunks(chunks)

        self._link_institutions_to_articles(articles)

        console.rule()
        self._print_stats()

    @staticmethod
    def _load_json(path: str) -> list[dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Base entities
    # ------------------------------------------------------------------

    def _create_treaties(self) -> None:
        treaties = [
            {
                "id": "TEU",
                "name": "Traité sur l'Union européenne",
                "abbr": "TUE",
                "date": "2007-12-13",
                "in_force": "2009-12-01",
                "official_ref": "2016/C 202/01",
            },
            {
                "id": "TFEU",
                "name": "Traité sur le fonctionnement de l'Union européenne",
                "abbr": "TFUE",
                "date": "1957-03-25",
                "consolidated": "2016",
                "official_ref": "2016/C 202/01",
            },
        ]

        console.print("[cyan]📜 Creating Treaty nodes...[/]")
        for treaty in treaties:
            self.client.run(
                "MERGE (t:Treaty {id:$id}) SET t += $props",
                {"id": treaty["id"], "props": treaty},
            )
        console.print(f"[green]  ✓ {len(treaties)} treaties created[/]")

    def _create_charter(self) -> None:
        """Create the EU Charter as a separate legal document root."""
        self.client.run(
            """
            MERGE (c:Charter {id:'CHARTER'})
            SET c.name=$name,
                c.abbr='CHARTE',
                c.official_ref='2016/C 202/02'
            """,
            {"name": "Charte des droits fondamentaux de l'Union européenne"},
        )

    def _create_institutions(self) -> None:
        console.print("[cyan]🏛️  Creating Institution nodes...[/]")
        for institution in EU_INSTITUTIONS:
            self.client.run(
                "MERGE (i:Institution {id:$id}) SET i += $props",
                {"id": institution["id"], "props": institution},
            )
        console.print(f"[green]  ✓ {len(EU_INSTITUTIONS)} institutions created[/]")

    def _create_countries(self) -> None:
        console.print("[cyan]🌍 Creating Country nodes...[/]")
        for country in EU_COUNTRIES:
            self.client.run(
                "MERGE (c:Country {id:$id}) SET c += $props",
                {"id": country["id"], "props": country},
            )
        console.print(f"[green]  ✓ {len(EU_COUNTRIES)} countries created[/]")

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------

    def _create_protocols(self, protocols: list[dict]) -> None:
        console.print(f"[cyan]📋 Creating {len(protocols)} Protocol nodes...[/]")

        for protocol in protocols:
            parent_treaties = {"TEU", "TFEU"}
            if protocol.get("parent_treaty") in {"TEU", "TFEU"}:
                parent_treaties.add(protocol["parent_treaty"])
            for treaty_id in protocol.get("parent_treaties") or []:
                if treaty_id in {"TEU", "TFEU"}:
                    parent_treaties.add(treaty_id)

            parent_treaties = sorted(parent_treaties)

            self.client.run(
                """
                MERGE (p:Protocol {id:$id})
                SET p.number=$number,
                    p.name=$name,
                    p.page=$page,
                    p.parent_treaties=$parent_treaties
                """,
                {
                    "id": protocol["protocol_id"],
                    "number": str(protocol["number"]),
                    "name": protocol.get("name", ""),
                    "page": protocol.get("page", 0),
                    "parent_treaties": parent_treaties,
                },
            )

            for treaty_id in parent_treaties:
                self.client.run(
                    """
                    MATCH (p:Protocol {id:$protocol_id})
                    MATCH (t:Treaty {id:$treaty_id})
                    MERGE (p)-[:ANNEXED_TO]->(t)
                    """,
                    {
                        "protocol_id": protocol["protocol_id"],
                        "treaty_id": treaty_id,
                    },
                )

        console.print(f"[green]  ✓ {len(protocols)} Protocol nodes created[/]")

    def _create_charter_article(self, article: dict) -> None:
        """Create a Charter article under the separate Charter root."""
        self.client.run(
            """
            MATCH (c:Charter {id:'CHARTER'})
            MERGE (a:Article:CharterArticle {id:$id})
            SET a.number=$number,
                a.treaty='CHARTER',
                a.document_type='charter',
                a.text=$text,
                a.ex_article=$ex_article,
                a.page_start=$page_start,
                a.page_end=$page_end
            MERGE (c)-[:CONTAINS]->(a)
            """,
            {
                "id": article["article_id"],
                "number": article["number"],
                "text": article.get("text", ""),
                "ex_article": article.get("ex_article"),
                "page_start": article.get("page_start", 0),
                "page_end": article.get("page_end", 0),
            },
        )
        self._create_paragraphs(article)

    def _create_protocol_article(self, article: dict) -> None:
        protocol_id = f"Protocol-{article['protocol_num']}"

        self.client.run(
            """
            MATCH (p:Protocol {id:$protocol_id})
            MERGE (a:Article:ProtocolArticle {id:$id})
            SET a.number=$number,
                a.treaty=$treaty,
                a.document_type='protocol',
                a.protocol_num=$protocol_num,
                a.parent_treaty=$parent_treaty,
                a.text=$text,
                a.ex_article=$ex_article,
                a.page_start=$page_start,
                a.page_end=$page_end
            MERGE (p)-[:CONTAINS]->(a)
            """,
            {
                "protocol_id": protocol_id,
                "id": article["article_id"],
                "number": article["number"],
                "treaty": article.get("treaty", protocol_id),
                "protocol_num": article.get("protocol_num"),
                "parent_treaty": article.get("parent_treaty"),
                "text": article.get("text", ""),
                "ex_article": article.get("ex_article"),
                "page_start": article.get("page_start", 0),
                "page_end": article.get("page_end", 0),
            },
        )

        self._create_paragraphs(article)

    def _link_protocol_references(self, protocols: list[dict]) -> None:
        console.print("[cyan]🔗 Linking protocol references...[/]")
        count = 0

        for protocol in protocols:
            for reference in protocol.get("references", []):
                target_id = self._reference_target_id(reference)
                if not target_id:
                    continue

                result = self.client.run(
                    """
                    MATCH (p:Protocol {id:$protocol_id})
                    MATCH (target {id:$target_id})
                    MERGE (p)-[r:RELATED_TO]->(target)
                    SET r.source_text=$source_text
                    RETURN count(r) AS c
                    """,
                    {
                        "protocol_id": protocol["protocol_id"],
                        "target_id": target_id,
                        "source_text": str(reference.get("text", ""))[:300],
                    },
                )
                if result:
                    count += result[0].get("c", 0)

        console.print(
            f"[green]  ✓ {count} protocol→article/paragraph links created[/]"
        )

    # ------------------------------------------------------------------
    # Treaty article hierarchy
    # ------------------------------------------------------------------

    def _create_articles(self, articles: list[dict]) -> None:
        console.print(f"[cyan]📖 Creating {len(articles)} Article nodes...[/]")

        for article in track(articles, description="Articles..."):
            if article.get("document_type") == "protocol":
                self._create_protocol_article(article)
            elif article.get("document_type") == "charter":
                self._create_charter_article(article)
            else:
                self._create_treaty_article(article)

        console.print("[green]  ✓ Articles created with hierarchy[/]")

    def _create_treaty_article(self, article: dict) -> None:
        treaty_id = article["treaty"]
        parent_id = treaty_id

        if article.get("title_num"):
            title_id = f"{treaty_id}-Title-{article['title_num']}"
            if title_id not in self._title_cache:
                self.client.run(
                    """
                    MERGE (t:Title {id:$id})
                    SET t.number=$number, t.name=$name, t.treaty=$treaty
                    WITH t
                    MATCH (tr:Treaty {id:$treaty})
                    MERGE (tr)-[:CONTAINS]->(t)
                    """,
                    {
                        "id": title_id,
                        "number": article["title_num"],
                        "name": article.get("title_name", ""),
                        "treaty": treaty_id,
                    },
                )
                self._title_cache[title_id] = title_id
            parent_id = title_id

        if article.get("chapter_num"):
            chapter_id = f"{parent_id}-Chapter-{article['chapter_num']}"
            if chapter_id not in self._chapter_cache:
                self.client.run(
                    """
                    MERGE (c:Chapter {id:$id})
                    SET c.number=$number, c.name=$name, c.treaty=$treaty
                    WITH c
                    MATCH (parent {id:$parent_id})
                    MERGE (parent)-[:CONTAINS]->(c)
                    """,
                    {
                        "id": chapter_id,
                        "number": article["chapter_num"],
                        "name": article.get("chapter_name", ""),
                        "treaty": treaty_id,
                        "parent_id": parent_id,
                    },
                )
                self._chapter_cache[chapter_id] = chapter_id
            parent_id = chapter_id

        if article.get("section_num"):
            section_id = f"{parent_id}-Section-{article['section_num']}"
            if section_id not in self._section_cache:
                self.client.run(
                    """
                    MERGE (s:Section {id:$id})
                    SET s.number=$number, s.name=$name, s.treaty=$treaty
                    WITH s
                    MATCH (parent {id:$parent_id})
                    MERGE (parent)-[:CONTAINS]->(s)
                    """,
                    {
                        "id": section_id,
                        "number": article["section_num"],
                        "name": article.get("section_name", ""),
                        "treaty": treaty_id,
                        "parent_id": parent_id,
                    },
                )
                self._section_cache[section_id] = section_id
            parent_id = section_id

        self.client.run(
            """
            MERGE (a:Article {id:$id})
            SET a.number=$number,
                a.treaty=$treaty,
                a.document_type='treaty',
                a.text=$text,
                a.ex_article=$ex_article,
                a.page_start=$page_start,
                a.page_end=$page_end,
                a.title_num=$title_num,
                a.title_name=$title_name,
                a.chapter_num=$chapter_num,
                a.chapter_name=$chapter_name,
                a.section_num=$section_num,
                a.section_name=$section_name
            WITH a
            MATCH (parent {id:$parent_id})
            MERGE (parent)-[:CONTAINS]->(a)
            """,
            {
                "id": article["article_id"],
                "number": article["number"],
                "treaty": treaty_id,
                "text": article.get("text", ""),
                "ex_article": article.get("ex_article"),
                "page_start": article.get("page_start", 0),
                "page_end": article.get("page_end", 0),
                "title_num": article.get("title_num"),
                "title_name": article.get("title_name"),
                "chapter_num": article.get("chapter_num"),
                "chapter_name": article.get("chapter_name"),
                "section_num": article.get("section_num"),
                "section_name": article.get("section_name"),
                "parent_id": parent_id,
            },
        )

        self._create_paragraphs(article)

    def _create_paragraphs(self, article: dict) -> None:
        paragraphs = article.get("paragraphs", [])

        for index, paragraph_text in enumerate(paragraphs, start=1):
            text = paragraph_text.strip()
            if not text:
                continue
            if not text[0].isdigit():
                continue

            paragraph_id = f"{article['article_id']}-P{index}"

            self.client.run(
                """
                MATCH (a:Article {id:$article_id})
                MERGE (p:Paragraph {id:$id})
                SET p.number=$number,
                    p.text=$text,
                    p.article_id=$article_id,
                    p.treaty=$treaty,
                    p.page_start=$page_start
                MERGE (a)-[:CONTAINS]->(p)
                """,
                {
                    "id": paragraph_id,
                    "number": index,
                    "text": text,
                    "article_id": article["article_id"],
                    "treaty": article.get("treaty"),
                    "page_start": article.get("page_start", 0),
                },
            )

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _create_declarations(self, declarations: list[dict]) -> None:
        console.print(
            f"[cyan]📑 Creating {len(declarations)} Declaration nodes...[/]"
        )

        for declaration in declarations:
            self.client.run(
                """
                MERGE (d:Declaration {id:$id})
                SET d.number=$number,
                    d.title=$title,
                    d.text=$text,
                    d.page_start=$page_start,
                    d.page_end=$page_end
                """,
                {
                    "id": declaration["declaration_id"],
                    "number": str(declaration["number"]),
                    "title": declaration.get("title", ""),
                    "text": declaration.get("text", ""),
                    "page_start": declaration.get("page_start", 0),
                    "page_end": declaration.get("page_end", 0),
                },
            )

        console.print(
            f"[green]  ✓ {len(declarations)} Declaration nodes created[/]"
        )

    def _link_declaration_references(self, declarations: list[dict]) -> None:
        console.print("[cyan]🔗 Linking declaration references...[/]")
        count = 0

        for declaration in declarations:
            for reference in declaration.get("references", []):
                target_id = self._reference_target_id(reference)
                if not target_id:
                    continue

                result = self.client.run(
                    """
                    MATCH (d:Declaration {id:$declaration_id})
                    MATCH (target {id:$target_id})
                    MERGE (d)-[r:REFERS_TO]->(target)
                    SET r.source_text=$source_text
                    RETURN count(r) AS c
                    """,
                    {
                        "declaration_id": declaration["declaration_id"],
                        "target_id": target_id,
                        "source_text": str(reference.get("text", ""))[:300],
                    },
                )
                if result:
                    count += result[0].get("c", 0)

        console.print(
            f"[green]  ✓ {count} declaration→article/paragraph links created[/]"
        )

    # ------------------------------------------------------------------
    # Article citations
    # ------------------------------------------------------------------

    @staticmethod
    def _reference_target_id(reference) -> str | None:
        """Return only an exact target ID; never guess TUE/TFUE."""
        if not isinstance(reference, dict):
            return None

        target_id = reference.get("target_id")
        if target_id:
            return str(target_id)

        treaty = reference.get("treaty") or reference.get("target_treaty")
        article = reference.get("article") or reference.get("target_num")
        paragraph = reference.get("paragraph") or reference.get("target_paragraph")

        if not treaty or not article:
            return None

        target_id = f"{treaty}-Art-{str(article).replace(' ', '')}"
        if paragraph is not None:
            target_id += f"-P{paragraph}"
        return target_id

    def _create_citations(self, articles: list[dict]) -> None:
        console.print("[cyan]🔗 Creating article citation relationships...[/]")
        count = 0
        skipped = 0

        for article in articles:
            source_id = article["article_id"]

            for reference in article.get("cites", []):
                target_id = self._reference_target_id(reference)
                if not target_id:
                    skipped += 1
                    continue

                result = self.client.run(
                    """
                    MATCH (source:Article {id:$source_id})
                    MATCH (target {id:$target_id})
                    MERGE (source)-[r:CITES]->(target)
                    SET r.source_text=$source_text
                    RETURN count(r) AS c
                    """,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_text": (
                            reference.get("text", "")[:300]
                            if isinstance(reference, dict)
                            else ""
                        ),
                    },
                )

                if result:
                    count += result[0].get("c", 0)

        console.print(f"[green]  ✓ {count} citation relationships created[/]")
        if skipped:
            console.print(
                f"[yellow]  ⚠ {skipped} ambiguous references skipped[/]"
            )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def _create_chunks(self, chunks: list[dict]) -> None:
        console.print(f"[cyan]🧩 Creating {len(chunks)} Chunk nodes...[/]")

        batch_size = 100

        for start in track(
            range(0, len(chunks), batch_size),
            description="Chunks...",
        ):
            batch = chunks[start : start + batch_size]

            self.client.run(
                """
                UNWIND $chunks AS chunk
                MERGE (c:Chunk {id:chunk.chunk_id})
                SET c.text=chunk.text,
                    c.chunk_index=chunk.chunk_index,
                    c.page_start=chunk.page_start,
                    c.treaty=chunk.treaty,
                    c.article_num=chunk.article_num,
                    c.article_id=chunk.article_id,
                    c.title_num=chunk.title_num,
                    c.title_name=chunk.title_name,
                    c.chapter_num=chunk.chapter_num,
                    c.chapter_name=chunk.chapter_name,
                    c.section_num=chunk.section_num,
                    c.section_name=chunk.section_name
                WITH c, chunk
                MATCH (a:Article {id:chunk.article_id})
                MERGE (a)-[:HAS_CHUNK]->(c)
                """,
                {"chunks": batch},
            )

        console.print(
            f"[green]  ✓ {len(chunks)} chunks linked to articles[/]"
        )

    # ------------------------------------------------------------------
    # Institution heuristic
    # ------------------------------------------------------------------

    def _link_institutions_to_articles(self, articles: list[dict]) -> None:
        console.print("[cyan]🏛️  Linking institutions to articles...[/]")

        institution_keywords = {
            "inst-parliament": ["parlement européen"],
            "inst-council-eu": ["conseil européen"],
            "inst-council": ["conseil de l'union", "le conseil"],
            "inst-commission": ["commission européenne", "la commission"],
            "inst-cjeu": ["cour de justice", "tribunal", "cjue"],
            "inst-ecb": ["banque centrale européenne", "bce"],
            "inst-court-auditors": ["cour des comptes"],
        }

        count = 0

        for article in articles:
            text_lower = article.get("text", "").lower()

            for institution_id, keywords in institution_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    self.client.run(
                        """
                        MATCH (a:Article {id:$article_id})
                        MATCH (i:Institution {id:$institution_id})
                        MERGE (a)-[:REFERENCES]->(i)
                        """,
                        {
                            "article_id": article["article_id"],
                            "institution_id": institution_id,
                        },
                    )
                    count += 1

        console.print(
            f"[green]  ✓ {count} article→institution links created[/]"
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _print_stats(self) -> None:
        queries = {
            "Treaties": "MATCH (n:Treaty) RETURN count(n) AS c",
            "Charter": "MATCH (n:Charter) RETURN count(n) AS c",
            "Titles": "MATCH (n:Title) RETURN count(n) AS c",
            "Chapters": "MATCH (n:Chapter) RETURN count(n) AS c",
            "Sections": "MATCH (n:Section) RETURN count(n) AS c",
            "Articles": "MATCH (n:Article) RETURN count(n) AS c",
            "Protocol articles": "MATCH (n:ProtocolArticle) RETURN count(n) AS c",
            "Paragraphs": "MATCH (n:Paragraph) RETURN count(n) AS c",
            "Protocols": "MATCH (n:Protocol) RETURN count(n) AS c",
            "Declarations": "MATCH (n:Declaration) RETURN count(n) AS c",
            "Institutions": "MATCH (n:Institution) RETURN count(n) AS c",
            "Countries": "MATCH (n:Country) RETURN count(n) AS c",
            "Chunks": "MATCH (n:Chunk) RETURN count(n) AS c",
            "CONTAINS rels": "MATCH ()-[r:CONTAINS]->() RETURN count(r) AS c",
            "CITES rels": "MATCH ()-[r:CITES]->() RETURN count(r) AS c",
            "RELATED_TO rels": "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS c",
            "REFERS_TO rels": "MATCH ()-[r:REFERS_TO]->() RETURN count(r) AS c",
            "REFERENCES rels": "MATCH ()-[r:REFERENCES]->() RETURN count(r) AS c",
            "HAS_CHUNK rels": "MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS c",
        }

        from rich.table import Table

        table = Table(title="📊 Knowledge Graph Statistics")
        table.add_column("Entity / Relationship", style="cyan")
        table.add_column("Count", style="bold green", justify="right")

        for label, query in queries.items():
            result = self.client.run(query)
            count = result[0]["c"] if result else 0
            table.add_row(label, str(count))

        console.print(table)


if __name__ == "__main__":
    client = Neo4jClient(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )

    if client.connect():
        builder = GraphBuilder(client)
        builder.build_all(
            articles_path=f"{settings.structured_dir}/articles.json",
            protocols_path=f"{settings.structured_dir}/protocols.json",
            chunks_path=f"{settings.structured_dir}/chunks.json",
            declarations_path=f"{settings.structured_dir}/declarations.json",
        )
        client.close()
    else:
        console.print("[red]Could not connect to Neo4j. Is it running?[/]")
        console.print("[yellow]Run: docker-compose up -d[/]")
