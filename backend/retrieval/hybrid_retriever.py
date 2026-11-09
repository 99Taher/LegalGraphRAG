"""
hybrid_retriever.py — Combines graph and vector retrieval.

Strategy:
1. Vector search for semantic similarity
2. Graph search for exact legal references
3. Expand explicit article references to protocols/declarations/citations
4. Merge and deduplicate results
5. Format the complete legal context for the LLM
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.graph.graph_retriever import GraphRetriever, GraphResult
from backend.vector.vector_store import VectorStore
from backend.vector.embedder import Embedder

console = Console()


@dataclass
class HybridResult:
    """Merged result from graph + vector retrieval."""

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
    vector_score: float = 0.0
    graph_score: float = 0.0
    hybrid_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    chunk_id: str | None = None
    relations: list[dict] = field(default_factory=list)
    extra_context: list[str] = field(default_factory=list)

    def to_context_str(self) -> str:
        """Format the result for the LLM context."""
        parts = [f"### Article {self.article_num} ({self.treaty})"]

        if self.title_name:
            parts.append(f"*Titre: {self.title_name}*")
        if self.chapter_name:
            parts.append(f"*Chapitre: {self.chapter_name}*")
        if self.section_name:
            parts.append(f"*Section: {self.section_name}*")
        if self.ex_article:
            parts.append(f"*{self.ex_article}*")

        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                parts.append(f"*Pages: {self.page_start}-{self.page_end}*")
            else:
                parts.append(f"*Page: {self.page_start}*")

        parts.append("")
        parts.append(self.text)

        if self.relations:
            parts.append("")
            parts.append("Relations juridiques:")
            for relation in self.relations:
                relation_type = relation.get("relation", "RELATED_TO")
                label = relation.get("name") or relation.get("label") or relation.get("id")
                number = relation.get("number")
                treaty = relation.get("treaty")
                protocol = relation.get("protocol")

                if protocol:
                    label = f"Protocole {protocol}: {label or ''}".strip()
                elif number and label and str(number) != str(label):
                    label = f"{label} {number}"

                if treaty:
                    label = f"{label} ({treaty})"

                parts.append(f"- {relation_type} → {label}")

        for block in self.extra_context:
            if block:
                parts.append("")
                parts.append(block)

        return "\n".join(parts)

    @property
    def citation(self) -> str:
        """Short citation string."""
        return f"Article {self.article_num} {self.treaty}"


class HybridRetriever:
    """Combines graph traversal and vector similarity for hybrid retrieval."""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_store: VectorStore,
        embedder: Embedder,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.graph = graph_retriever
        self.vector = vector_store
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        top_k_vector: int = 10,
        top_k_graph: int = 5,
    ) -> list[HybridResult]:
        """
        Main hybrid retrieval method.

        Explicit references are handled before semantic search so that
        legal identifiers such as "Article 6(2) TUE" remain unambiguous.
        """
        results: dict[str, HybridResult] = {}

        # --------------------------------------------------------------
        # 1. Vector search
        # --------------------------------------------------------------
        try:
            query_embedding = self.embedder.embed_text(query)
            vector_results = self.vector.search(
                query_embedding,
                top_k=top_k_vector,
            )
        except Exception as exc:
            console.print(f"[yellow]Vector search error:[/] {exc}")
            vector_results = []

        for vr in vector_results:
            key = vr.article_id
            if not key:
                continue

            if key not in results:
                results[key] = HybridResult(
                    article_id=vr.article_id,
                    article_num=vr.article_num,
                    treaty=vr.treaty,
                    text=vr.text,
                    title_name=vr.title_name,
                    chapter_name=vr.chapter_name,
                    page_start=getattr(vr, "page_start", 0) or 0,
                    chunk_id=vr.chunk_id,
                )

            results[key].vector_score = max(
                results[key].vector_score,
                float(vr.score),
            )
            results[key].sources.append("vector")

        # --------------------------------------------------------------
        # 2. Exact article references from the query
        # --------------------------------------------------------------
        article_refs = self._extract_article_refs(query)

        for ref in article_refs:
            number = ref["number"]
            treaty = ref["treaty"]
            paragraph = ref["paragraph"]

            graph_results = self.graph.get_article_by_number(
                number,
                treaty=treaty,
                document_type="treaty",
            )

            # If a treaty was not explicitly specified, allow both TEU/TFEU.
            if not graph_results and treaty is None:
                graph_results = self.graph.get_article_by_number(
                    number,
                    treaty=None,
                    document_type="treaty",
                )

            for gr in graph_results:
                self._merge_graph_result(
                    gr,
                    results,
                    score_boost=10.0,
                )

                # Preserve an exact paragraph reference in the retrieved context.
                if paragraph is not None:
                    paragraph_data = self.graph.get_paragraph(
                        gr.article_id,
                        paragraph,
                    )
                    if paragraph_data:
                        self._add_relation(
                            results[gr.article_id],
                            {
                                "relation": "TARGET_PARAGRAPH",
                                "id": paragraph_data["id"],
                                "number": paragraph_data["number"],
                            },
                        )

                # ----------------------------------------------------------
                # IMPORTANT: protocol expansion
                # ----------------------------------------------------------
                protocols = self.graph.get_protocol_for_article(gr.article_id)
                for protocol in protocols:
                    self._add_relation(
                        results[gr.article_id],
                        {
                            "relation": "RELATED_TO",
                            "id": protocol.get("protocol_id"),
                            "name": protocol.get("protocol_name"),
                            "number": protocol.get("protocol_number"),
                            "protocol": protocol.get("protocol_number"),
                        },
                    )

                # ----------------------------------------------------------
                # Declaration expansion
                # ----------------------------------------------------------
                declarations = self.graph.get_declarations_for_article(gr.article_id)
                for declaration in declarations:
                    self._add_relation(
                        results[gr.article_id],
                        {
                            "relation": "REFERS_TO",
                            "id": declaration.get("declaration_id"),
                            "name": declaration.get("declaration_title"),
                            "number": declaration.get("declaration_number"),
                        },
                    )

        # --------------------------------------------------------------
        # 3. Explicit declaration references
        # --------------------------------------------------------------
        declaration_refs = self._extract_declaration_refs(query)
        for declaration_number in declaration_refs:
            declarations = self.graph.get_declaration_by_number(declaration_number)
            for declaration in declarations:
                block = self._format_declaration_context(declaration)
                self._attach_extra_context(results, block)

                for reference in declaration.get("references", []):
                    target_id = reference.get("id")
                    if not target_id:
                        continue

                    target_results = self.graph.get_article_by_id(target_id)
                    for gr in target_results:
                        self._merge_graph_result(
                            gr,
                            results,
                            score_boost=8.0,
                        )

        # --------------------------------------------------------------
        # 4. Graph full-text search
        # --------------------------------------------------------------
        try:
            graph_ft_results = self.graph.fulltext_search(
                query,
                limit=top_k_graph,
            )
        except Exception as exc:
            console.print(f"[yellow]Graph fulltext error:[/] {exc}")
            graph_ft_results = []

        for gr in graph_ft_results:
            if gr:
                self._merge_graph_result(gr, results)

        # --------------------------------------------------------------
        # 5. Neighborhood expansion for top vector results
        # --------------------------------------------------------------
        top_vector_ids = [
            vr.article_id
            for vr in vector_results[:3]
            if getattr(vr, "article_id", None)
        ]

        for art_id in top_vector_ids:
            try:
                neighborhood = self.graph.get_neighborhood(art_id)
            except Exception:
                neighborhood = []

            for gr in neighborhood:
                if gr:
                    self._merge_graph_result(
                        gr,
                        results,
                        score_boost=0.1,
                    )

        # --------------------------------------------------------------
        # 6. Compute hybrid scores
        # --------------------------------------------------------------
        for result in results.values():
            result.hybrid_score = (
                self.vector_weight * result.vector_score
                + self.graph_weight * result.graph_score
            )
            result.sources = sorted(set(result.sources))

        sorted_results = sorted(
            results.values(),
            key=lambda r: r.hybrid_score,
            reverse=True,
        )

        return sorted_results[:top_k]

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def _merge_graph_result(
        self,
        gr: GraphResult,
        results: dict[str, HybridResult],
        score_boost: float = 0.0,
    ) -> None:
        """Merge one GraphResult into hybrid results."""
        key = gr.article_id
        if not key:
            return

        if key not in results:
            results[key] = HybridResult(
                article_id=gr.article_id,
                article_num=gr.article_num,
                treaty=gr.treaty,
                text=gr.text,
                title_name=gr.title_name,
                chapter_name=gr.chapter_name,
                section_name=getattr(gr, "section_name", None),
                ex_article=gr.ex_article,
                page_start=getattr(gr, "page_start", 0) or 0,
                page_end=getattr(gr, "page_end", 0) or 0,
            )

        results[key].graph_score = max(
            results[key].graph_score,
            float(gr.score) + score_boost,
        )
        results[key].sources.append("graph")

        # Keep graph relationships already attached to GraphResult.
        for relation in getattr(gr, "related", []) or []:
            self._add_relation(results[key], relation)

    def _add_relation(self, result: HybridResult, relation: dict) -> None:
        """Add a relation only once."""
        if not relation:
            return

        relation_id = relation.get("id")
        relation_type = relation.get("relation", "RELATED_TO")

        for existing in result.relations:
            if (
                existing.get("id") == relation_id
                and existing.get("relation", "RELATED_TO") == relation_type
            ):
                return

        result.relations.append(relation)

    def _attach_extra_context(
        self,
        results: dict[str, HybridResult],
        block: str,
    ) -> None:
        """
        Attach non-article context to the strongest existing result.

        This keeps the public retrieve() return type unchanged.
        """
        if not block:
            return

        if results:
            target = max(
                results.values(),
                key=lambda item: item.graph_score + item.vector_score,
            )
            if block not in target.extra_context:
                target.extra_context.append(block)

    # ------------------------------------------------------------------
    # Query parsing
    # ------------------------------------------------------------------

    def _extract_article_refs(self, query: str) -> list[dict]:
        """
        Extract references such as:

            Article 6(2) TUE
            Article 16 TFUE
            articles 123 et 124 TFUE
            art. 17 TFEU

        Returns dictionaries so treaty and paragraph information are retained.
        """
        text = query.replace("‐", "-").replace("‑", "-")
        pattern = re.compile(
            r"\b(?:articles?|art\.)\s+"
            r"(?P<refs>\d+(?:\s*\(\s*\d+\s*\))?"
            r"(?:\s*(?:,|et|and)\s*"
            r"\d+(?:\s*\(\s*\d+\s*\))?)*)"
            r"(?:\s+(?P<treaty>TUE|TEU|TFUE|TFEU))?\b",
            re.IGNORECASE,
        )

        matches = pattern.finditer(text)
        refs: list[dict] = []

        for match in matches:
            raw_refs = match.group("refs")
            raw_treaty = match.group("treaty")
            treaty = self._normalize_treaty(raw_treaty)

            # Determine an optional treaty from nearby query context when
            # the treaty appears just before/after the article phrase.
            if treaty is None:
                treaty = self._infer_treaty_from_context(text, match.start(), match.end())

            for item in re.split(r"\s*(?:,|et|and)\s*", raw_refs, flags=re.IGNORECASE):
                item = item.strip()
                number_match = re.match(r"^(\d+)", item)
                if not number_match:
                    continue

                paragraph_match = re.search(r"\(\s*(\d+)\s*\)", item)
                paragraph = int(paragraph_match.group(1)) if paragraph_match else None

                refs.append({
                    "number": number_match.group(1),
                    "paragraph": paragraph,
                    "treaty": treaty,
                })

        # Remove duplicates while keeping order.
        unique = []
        seen = set()
        for ref in refs:
            key = (ref["number"], ref["paragraph"], ref["treaty"])
            if key not in seen:
                seen.add(key)
                unique.append(ref)

        return unique

    def _extract_declaration_refs(self, query: str) -> list[str]:
        """Extract Declaration references such as Declaration 19."""
        matches = re.findall(
            r"\b(?:declaration|declaration no\.?|declaration n°)\s*(\d+)\b",
            query,
            re.IGNORECASE,
        )
        return list(dict.fromkeys(matches))

    def _infer_treaty_from_context(
        self,
        text: str,
        start: int,
        end: int,
    ) -> str | None:
        """Infer TEU/TFEU from nearby text when it was not captured directly."""
        window = text[max(0, start - 30): min(len(text), end + 30)]
        return self._normalize_treaty(
            re.search(r"\b(TUE|TEU|TFUE|TFEU)\b", window, re.IGNORECASE).group(1)
            if re.search(r"\b(TUE|TEU|TFUE|TFEU)\b", window, re.IGNORECASE)
            else None
        )

    @staticmethod
    def _normalize_treaty(value: str | None) -> str | None:
        """Normalize French/English treaty abbreviations."""
        if not value:
            return None

        value = value.upper()
        if value == "TEU" or value == "TUE":
            return "TEU"
        if value == "TFEU" or value == "TFUE":
            return "TFEU"
        return None

    # ------------------------------------------------------------------
    # Declaration formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_declaration_context(declaration: dict) -> str:
        """Format one declaration for the LLM."""
        number = declaration.get("number", "")
        title = declaration.get("title", "")
        text = declaration.get("text", "")

        lines = [f"### Declaration {number}"]
        if title:
            lines.append(title)
        lines.append("")
        if text:
            lines.append(text)

        references = declaration.get("references", [])
        if references:
            lines.append("")
            lines.append("Références:")
            for reference in references:
                ref_id = reference.get("id", "")
                ref_num = reference.get("number", "")
                ref_treaty = reference.get("treaty", "")
                label = ref_id or ref_num
                if ref_treaty:
                    label = f"{label} ({ref_treaty})"
                lines.append(f"- REFERS_TO → {label}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context and citations
    # ------------------------------------------------------------------

    def format_context(self, results: list[HybridResult]) -> str:
        """Format results into the LLM context string."""
        if not results:
            return "Aucun élément juridique pertinent trouvé."

        parts = ["## Contexte juridique pertinent\n"]
        for i, result in enumerate(results, 1):
            parts.append(f"{i}. {result.to_context_str()}")
            parts.append("")

        return "\n".join(parts)

    def get_citations(self, results: list[HybridResult]) -> list[dict]:
        """Return structured article citations for the API response."""
        citations = []

        for result in results:
            citations.append(
                {
                    "article": result.citation,
                    "article_id": result.article_id,
                    "treaty": result.treaty,
                    "title": result.title_name,
                    "score": round(result.hybrid_score, 3),
                    "sources": result.sources,
                    "relations": result.relations,
                }
            )

        return citations


if __name__ == "__main__":
    from backend.config import settings
    from backend.graph.neo4j_client import Neo4jClient

    client = Neo4jClient(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )

    if client.connect():
        retriever = HybridRetriever(
            graph_retriever=GraphRetriever(client),
            vector_store=VectorStore(settings.chroma_persist_dir),
            embedder=Embedder(),
        )

        query = "Which protocol is related to Article 6(2) TUE ?"
        results = retriever.retrieve(query, top_k=5)

        print(f"\nQuery: {query}")
        print(f"Results: {len(results)}")
        print("\n--- Context ---")
        print(retriever.format_context(results))

        print("\n--- Citations ---")
        for citation in retriever.get_citations(results):
            print(citation)

        client.close()
