"""
graph_retriever.py — Graph-based retrieval from Neo4j.

Main strategies:
1. Exact article lookup by ID
2. Exact article lookup by number + treaty
3. Exact paragraph lookup
4. Protocol lookup and Protocol -> Article/Paragraph retrieval
5. Article neighborhood expansion
6. Citation / cited-by retrieval
7. Neo4j full-text search
8. Institution lookup
9. Shortest legal path search
10. JSON-friendly subgraph for visualization
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.graph.neo4j_client import Neo4jClient

console = Console()


@dataclass
class GraphResult:
    """A retrieved legal node formatted for LLM context."""

    article_id: str
    article_num: str
    treaty: str
    text: str
    title_name: str | None = None
    chapter_name: str | None = None
    section_name: str | None = None
    ex_article: str | None = None
    page_start: int = 0
    page_end: int = 0
    document_type: str | None = None
    protocol_num: str | None = None
    score: float = 1.0
    source: str = "graph"
    related: list[dict] = field(default_factory=list)
    paragraphs: list[dict] = field(default_factory=list)

    def to_context_str(self) -> str:
        """Format the result for the LLM context window."""
        parts = [f"[Article {self.article_num} {self.treaty}]"]

        if self.document_type == "protocol" and self.protocol_num:
            parts.append(f"Protocol: {self.protocol_num}")
        if self.title_name:
            parts.append(f"Titre: {self.title_name}")
        if self.chapter_name:
            parts.append(f"Chapitre: {self.chapter_name}")
        if self.section_name:
            parts.append(f"Section: {self.section_name}")
        if self.ex_article:
            parts.append(f"({self.ex_article})")

        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                parts.append(f"Pages: {self.page_start}-{self.page_end}")
            else:
                parts.append(f"Page: {self.page_start}")

        parts.append("")
        parts.append(self.text)

        if self.paragraphs:
            parts.append("")
            parts.append("Paragraphes structurés:")
            for paragraph in self.paragraphs:
                parts.append(f"- §{paragraph['number']}: {paragraph['text']}")

        if self.related:
            parts.append("")
            parts.append("Relations:")
            for item in self.related:
                relation = item.get("relation", "RELATED_TO")
                display = item.get("name") or item.get("id") or item.get("number", "")
                treaty = item.get("treaty")
                if treaty:
                    display = f"{display} ({treaty})"
                parts.append(f"- {relation} → {display}")

        return "\n".join(parts)


class GraphRetriever:
    """Retrieve relevant legal context from the Neo4j knowledge graph."""

    def __init__(self, client: Neo4jClient):
        self.client = client

    # ------------------------------------------------------------------
    # Exact legal reference lookup
    # ------------------------------------------------------------------

    def get_article_by_id(self, article_id: str) -> list[GraphResult]:
        """Exact lookup, e.g. TEU-Art-16 or TFEU-Art-16."""
        results = self.client.run(
            """
            MATCH (a:Article {id:$article_id})
            OPTIONAL MATCH (a)-[:CONTAINS]->(p:Paragraph)
            RETURN a,
                   collect({
                       id:p.id,
                       number:p.number,
                       text:p.text
                   }) AS paragraphs
            """,
            {"article_id": article_id},
        )

        output = []
        for record in results:
            result = self._to_result(record)
            if result:
                output.append(result)
        return output

    def get_article_by_number(
        self,
        number: str,
        treaty: str | None = None,
        document_type: str = "treaty",
        limit: int = 10,
    ) -> list[GraphResult]:
        """
        Exact lookup by article number.

        Examples:
            get_article_by_number("16", "TEU")
            get_article_by_number("16", "TFEU")

        Keeping the treaty explicit avoids confusing TUE Article 16 with
        TFUE Article 16.
        """
        query = """
        MATCH (a:Article)
        WHERE a.number = $number
          AND ($treaty IS NULL OR a.treaty = $treaty)
          AND ($document_type IS NULL OR a.document_type = $document_type)
        OPTIONAL MATCH (a)-[:CONTAINS]->(p:Paragraph)
        RETURN a,
               collect({
                   id:p.id,
                   number:p.number,
                   text:p.text
               }) AS paragraphs
        ORDER BY a.treaty, a.id
        LIMIT $limit
        """

        results = self.client.run(
            query,
            {
                "number": str(number),
                "treaty": treaty,
                "document_type": document_type,
                "limit": limit,
            },
        )

        output = []
        for record in results:
            result = self._to_result(record)
            if result:
                output.append(result)
        return output

    def get_paragraph(
        self,
        article_id: str,
        paragraph_number: int | str,
    ) -> dict | None:
        """Retrieve an exact paragraph such as TUE Article 6 paragraph 2."""
        paragraph_id = f"{article_id}-P{paragraph_number}"

        results = self.client.run(
            """
            MATCH (a:Article {id:$article_id})
            MATCH (p:Paragraph {id:$paragraph_id})
            RETURN p, a
            """,
            {
                "article_id": article_id,
                "paragraph_id": paragraph_id,
            },
        )

        if not results:
            return None

        paragraph = results[0]["p"]
        article = results[0]["a"]

        return {
            "id": paragraph.get("id", ""),
            "article_id": article_id,
            "article": str(article.get("number", "")),
            "number": paragraph.get("number", ""),
            "text": paragraph.get("text", ""),
            "treaty": paragraph.get("treaty", ""),
        }

    # ------------------------------------------------------------------
    # Declaration retrieval
    # ------------------------------------------------------------------

    def get_declaration_by_number(self, number: str) -> list[dict]:
        """Retrieve an exact declaration such as Declaration 19."""
        results = self.client.run(
            """
            MATCH (d:Declaration {number:$number})
            OPTIONAL MATCH (d)-[:REFERS_TO]->(target)
            RETURN d, collect(target) AS references
            ORDER BY d.number
            """,
            {"number": str(number)},
        )

        output = []
        for record in results:
            declaration = record.get("d")
            if declaration is None:
                continue

            references = []
            for target in record.get("references", []):
                if target is not None:
                    references.append(self._node_summary(target))

            output.append({
                "id": declaration.get("id", ""),
                "number": declaration.get("number", ""),
                "title": declaration.get("title", ""),
                "text": declaration.get("text", ""),
                "page_start": declaration.get("page_start", 0),
                "page_end": declaration.get("page_end", 0),
                "references": references,
            })

        return output

    def get_declarations_for_article(self, article_id: str) -> list[dict]:
        """Find declarations that explicitly refer to an article or paragraph."""
        results = self.client.run(
            """
            MATCH (d:Declaration)-[:REFERS_TO]->(target)
            WHERE target.id = $article_id
               OR (target:Paragraph AND target.article_id = $article_id)
            RETURN d, target
            ORDER BY d.number
            """,
            {"article_id": article_id},
        )

        output = []
        for record in results:
            declaration = record.get("d")
            target = record.get("target")
            if declaration is None:
                continue

            output.append({
                "declaration_id": declaration.get("id", ""),
                "declaration_number": declaration.get("number", ""),
                "declaration_title": declaration.get("title", ""),
                "target_id": target.get("id", "") if target else "",
                "target_number": target.get("number", "") if target else "",
                "target_treaty": target.get("treaty", "") if target else "",
                "target_type": self._node_type(target) if target else "Node",
            })

        return output

    # ------------------------------------------------------------------
    # Protocol retrieval
    # ------------------------------------------------------------------

    def get_protocol_by_number(self, number: str) -> list[dict]:
        """Retrieve a protocol and the exact provisions it relates to."""
        results = self.client.run(
            """
            MATCH (p:Protocol {number:$number})
            OPTIONAL MATCH (p)-[:RELATED_TO]->(target)
            RETURN p,
                   collect(target) AS related
            ORDER BY p.number
            """,
            {"number": str(number)},
        )

        output = []
        for record in results:
            protocol = record["p"]
            related = []

            for target in record.get("related", []):
                if target is None:
                    continue
                related.append(self._node_summary(target))

            output.append(
                {
                    "id": protocol.get("id", ""),
                    "number": protocol.get("number", ""),
                    "name": protocol.get("name", ""),
                    "page": protocol.get("page", 0),
                    "related": related,
                }
            )

        return output

    def get_protocol_for_article(self, article_id: str) -> list[dict]:
        """Find protocols related to an article or one of its paragraphs."""
        results = self.client.run(
            """
            MATCH (p:Protocol)-[:RELATED_TO]->(target)
            WHERE target.id = $article_id
               OR (target:Paragraph AND target.article_id = $article_id)
            RETURN p, target
            ORDER BY p.number
            """,
            {"article_id": article_id},
        )

        output = []
        for record in results:
            protocol = record["p"]
            target = record["target"]
            output.append(
                {
                    "protocol_id": protocol.get("id", ""),
                    "protocol_number": protocol.get("number", ""),
                    "protocol_name": protocol.get("name", ""),
                    "target_id": target.get("id", ""),
                    "target_number": target.get("number", ""),
                    "target_treaty": target.get("treaty", ""),
                    "target_type": self._node_type(target),
                }
            )

        return output

    # ------------------------------------------------------------------
    # Graph relationships
    # ------------------------------------------------------------------

    def get_cited_articles(
        self,
        article_id: str,
        limit: int = 20,
    ) -> list[GraphResult]:
        """Return legal provisions directly cited by an article."""
        results = self.client.run(
            """
            MATCH (a:Article {id:$article_id})-[:CITES]->(target:Article)
            RETURN target AS a
            ORDER BY target.treaty, target.number
            LIMIT $limit
            """,
            {"article_id": article_id, "limit": limit},
        )

        output = []
        for record in results:
            result = self._to_result(record)
            if result:
                output.append(result)
        return output

    def get_citing_articles(
        self,
        article_id: str,
        limit: int = 20,
    ) -> list[GraphResult]:
        """Return articles that cite the given article."""
        results = self.client.run(
            """
            MATCH (source:Article)-[:CITES]->(a:Article {id:$article_id})
            RETURN source AS a
            ORDER BY source.treaty, source.number
            LIMIT $limit
            """,
            {"article_id": article_id, "limit": limit},
        )

        output = []
        for record in results:
            result = self._to_result(record)
            if result:
                output.append(result)
        return output

    def get_neighborhood(
        self,
        article_id: str,
        depth: int = 2,
        limit: int = 30,
    ) -> list[GraphResult]:
        """Expand around an article using legal graph relationships."""
        depth = max(1, min(int(depth), 3))

        query = f"""
        MATCH (center:Article {{id:$article_id}})
        OPTIONAL MATCH path =
            (center)-[:CITES|RELATED_TO|CONTAINS*1..{depth}]-(node)
        WITH center,
             collect(DISTINCT node) AS neighbors
        RETURN center,
               [
                   n IN neighbors
                   WHERE n IS NOT NULL
                   | {{
                       id:n.id,
                       number:n.number,
                       treaty:n.treaty,
                       name:n.name,
                       type:
                           CASE
                               WHEN n:Protocol THEN 'Protocol'
                               WHEN n:Paragraph THEN 'Paragraph'
                               WHEN n:Article THEN 'Article'
                               ELSE 'Node'
                           END
                   }}
               ][..{limit}] AS neighbors
        """

        results = self.client.run(query, {"article_id": article_id})
        if not results:
            return []

        main = self._to_result_raw(results[0].get("center"))
        if main is None:
            return []

        main.related = self._clean_list(results[0].get("neighbors", []))
        return [main]

    def get_article_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
    ) -> list[dict]:
        """Find a shortest legal graph path between two node IDs."""
        max_depth = max(1, min(int(max_depth), 10))

        query = f"""
        MATCH path = shortestPath(
            (a {{id:$from_id}})
            -[:CITES|RELATED_TO|CONTAINS*..{max_depth}]-
            (b {{id:$to_id}})
        )
        RETURN [
            node IN nodes(path) |
            {{
                id:node.id,
                number:node.number,
                treaty:node.treaty,
                name:node.name,
                type:
                    CASE
                        WHEN node:Protocol THEN 'Protocol'
                        WHEN node:Paragraph THEN 'Paragraph'
                        WHEN node:Article THEN 'Article'
                        ELSE 'Node'
                    END
            }}
        ] AS path
        """

        results = self.client.run(
            query,
            {"from_id": from_id, "to_id": to_id},
        )

        return self._clean_list(results[0]["path"]) if results else []

    # ------------------------------------------------------------------
    # Full-text retrieval
    # ------------------------------------------------------------------

    def fulltext_search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> list[GraphResult]:
        """Search Neo4j's configured full-text article index."""
        try:
            results = self.client.run(
                """
                CALL db.index.fulltext.queryNodes('article_text', $query)
                YIELD node AS a, score
                RETURN a, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query": query_text, "limit": limit},
            )

            output = []
            for record in results:
                result = self._to_result(
                    record,
                    score=float(record.get("score", 1.0)),
                )
                if result:
                    output.append(result)
            return output

        except Exception as exc:
            console.print(f"[yellow]Fulltext search error:[/] {exc}")
            return []

    # ------------------------------------------------------------------
    # Institution retrieval
    # ------------------------------------------------------------------

    def get_institution_articles(
        self,
        institution_name: str,
        limit: int = 5,
    ) -> list[GraphResult]:
        """Retrieve articles that explicitly reference an institution."""
        results = self.client.run(
            """
            MATCH (i:Institution)
            WHERE toLower(i.name) CONTAINS toLower($name)
            MATCH (a:Article)-[:REFERENCES]->(i)
            RETURN a
            ORDER BY a.treaty, a.number
            LIMIT $limit
            """,
            {"name": institution_name, "limit": limit},
        )

        output = []
        for record in results:
            result = self._to_result(record)
            if result:
                output.append(result)
        return output

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_subgraph_for_viz(
        self,
        article_id: str,
        depth: int = 1,
    ) -> dict:
        """Return JSON-friendly nodes and edges for a React graph view."""
        depth = max(1, min(int(depth), 2))

        query = f"""
        MATCH (center:Article {{id:$article_id}})
        OPTIONAL MATCH path =
            (center)-[:CITES|REFERENCES|RELATED_TO|CONTAINS|HAS_CHUNK*0..{depth}]-(node)
        WITH center, collect(DISTINCT node) AS raw_nodes
        UNWIND raw_nodes AS n
        WITH center,
             collect(DISTINCT CASE
                 WHEN n IS NULL THEN NULL
                 ELSE {{
                     id:n.id,
                     label:
                         CASE
                             WHEN n:Article THEN 'Article'
                             WHEN n:Paragraph THEN 'Paragraph'
                             WHEN n:Protocol THEN 'Protocol'
                             WHEN n:Institution THEN 'Institution'
                             WHEN n:Chunk THEN 'Chunk'
                             WHEN n:Title THEN 'Title'
                             WHEN n:Chapter THEN 'Chapter'
                             WHEN n:Section THEN 'Section'
                             WHEN n:Treaty THEN 'Treaty'
                             ELSE 'Node'
                         END,
                     number:n.number,
                     name:n.name,
                     treaty:n.treaty
                 }}
             END) AS nodes
        RETURN center.id AS center_id, nodes
        """

        results = self.client.run(query, {"article_id": article_id})
        if not results:
            return {"nodes": [], "edges": [], "center_id": article_id}

        node_items = self._clean_list(results[0].get("nodes", []))
        node_ids = [item["id"] for item in node_items if item.get("id")]

        if not node_ids:
            return {"nodes": [], "edges": [], "center_id": article_id}

        edge_results = self.client.run(
            """
            MATCH (a)-[r]->(b)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id AS source,
                   b.id AS target,
                   type(r) AS type
            """,
            {"ids": node_ids},
        )

        edges = [
            {
                "source": row["source"],
                "target": row["target"],
                "type": row["type"],
            }
            for row in edge_results
        ]

        return {
            "nodes": node_items,
            "edges": edges,
            "center_id": article_id,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_result(
        self,
        record: dict,
        score: float = 1.0,
    ) -> GraphResult | None:
        node = record.get("a")
        if node is None:
            return None

        result = self._to_result_raw(node, score)
        if result is not None:
            result.paragraphs = self._clean_list(record.get("paragraphs", []))
        return result

    def _to_result_raw(
        self,
        node,
        score: float = 1.0,
    ) -> GraphResult | None:
        if node is None:
            return None

        return GraphResult(
            article_id=node.get("id", ""),
            article_num=str(node.get("number", "")),
            treaty=node.get("treaty", ""),
            text=node.get("text", ""),
            title_name=node.get("title_name"),
            chapter_name=node.get("chapter_name"),
            section_name=node.get("section_name"),
            ex_article=node.get("ex_article"),
            page_start=int(node.get("page_start", 0) or 0),
            page_end=int(node.get("page_end", 0) or 0),
            document_type=node.get("document_type"),
            protocol_num=node.get("protocol_num"),
            score=score,
            source="graph",
        )

    @staticmethod
    def _node_summary(node) -> dict:
        return {
            "id": node.get("id", ""),
            "number": node.get("number", ""),
            "treaty": node.get("treaty", ""),
            "name": node.get("name", ""),
            "type": GraphRetriever._node_type(node),
        }

    @staticmethod
    def _node_type(node) -> str:
        labels = set(node.labels) if hasattr(node, "labels") else set()
        if "Protocol" in labels:
            return "Protocol"
        if "Paragraph" in labels:
            return "Paragraph"
        if "Article" in labels:
            return "Article"
        if "Chunk" in labels:
            return "Chunk"
        if "Declaration" in labels:
            return "Declaration"
        if "Section" in labels:
            return "Section"
        if "Chapter" in labels:
            return "Chapter"
        if "Title" in labels:
            return "Title"
        if "Treaty" in labels:
            return "Treaty"
        if "Charter" in labels:
            return "Charter"
        return "Node"

    @staticmethod
    def _clean_list(items) -> list[dict]:
        if not items:
            return []
        return [item for item in items if item is not None]


if __name__ == "__main__":
    from backend.config import settings

    client = Neo4jClient(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )

    if client.connect():
        retriever = GraphRetriever(client)

        print("\n=== Article lookup: Article 6 TEU ===")
        for result in retriever.get_article_by_number("6", "TEU"):
            print(result.to_context_str()[:1000])

        print("\n=== Article lookup: Article 16 TFEU ===")
        for result in retriever.get_article_by_number("16", "TFEU"):
            print(result.to_context_str()[:1000])

        print("\n=== Protocol related to Article 6 TEU ===")
        for item in retriever.get_protocol_for_article("TEU-Art-6"):
            print(item)

        print("\n=== Fulltext search: 'subsidiarité' ===")
        for result in retriever.fulltext_search("subsidiarité")[:3]:
            print(result.to_context_str()[:700])

        client.close()
    else:
        console.print("[red]Could not connect to Neo4j. Is it running?[/]")
