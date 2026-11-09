"""
article_parser.py — Parse the TEU/TFEU legal corpus into structured objects.

Important:
- Article IDs are treaty-aware: TEU-Art-16 != TFEU-Art-16
- Paragraph IDs are supported: TUE-Art-6-P2
- Protocol articles are separate: Protocol-8-Art-1
- Cross-references keep their treaty + paragraph context

Hierarchy:
    Treaty -> Title -> Chapter -> Section -> Article -> Paragraph
    Protocol -> ProtocolArticle -> Paragraph
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class ArticleNode:
    """Represents one legal article."""

    # Identity
    article_id: str
    number: str
    treaty: str

    # Document type
    document_type: str = "treaty"       # treaty | protocol
    protocol_num: Optional[str] = None
    parent_treaty: Optional[str] = None # TUE or TFEU for protocol articles

    # Hierarchy
    title_num: Optional[str] = None
    title_name: Optional[str] = None
    chapter_num: Optional[str] = None
    chapter_name: Optional[str] = None
    section_num: Optional[str] = None
    section_name: Optional[str] = None

    # Content
    text: str = ""
    paragraphs: list[str] = field(default_factory=list)

    # Cross references
    ex_article: Optional[str] = None
    cites: list[dict] = field(default_factory=list)

    # Source
    page_start: int = 0
    page_end: int = 0


@dataclass
class Treaty:
    name: str
    full_name: str
    articles: list[ArticleNode] = field(default_factory=list)
    protocols: list[dict] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------

RE_ARTICLE = re.compile(
    r"^Article\s+(\d+(?:\s*bis|\s*ter)?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

RE_EX_ARTICLE = re.compile(
    r"\(ex-article\s+(\d+[A-Za-z]*)\s+(TUE|TCE|TFUE|CE)\)",
    re.IGNORECASE,
)

RE_TITLE = re.compile(
    r"^TITRE\s+([IVX]+)\s*[—–-]\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)

RE_CHAPTER = re.compile(
    r"^CHAPITRE\s+(\d+|[IVX]+)\s*[—–-]?\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)

RE_SECTION = re.compile(
    r"^SECTION\s+(\d+|[IVX]+)\s*[—–-]?\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)

RE_PROTOCOL = re.compile(
    r"^PROTOCOLE\s+\([Nn]\s*[°oº]\s*(\d+)\)\s*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)

RE_TREATY_TEU = re.compile(
    r"^TRAIT[EÉ]\s+SUR\s+L.UNION\s+EUROP[EÉ]ENNE$",
    re.IGNORECASE,
)

RE_TREATY_TFEU = re.compile(
    r"^TRAIT[EÉ]\s+SUR\s+LE\s+FONCTIONNEMENT",
    re.IGNORECASE,
)

# "article 16 TUE" / "article 16 TFUE"
RE_ABBR_REF = re.compile(
    r"\b(?:l'article|les articles?|article)\s+([0-9]+(?:\s*(?:bis|ter))?)"
    r"(?:\s*,\s*paragraphe\s+(\d+))?\s+(TUE|TFUE)\b",
    re.IGNORECASE,
)

# Qualified references:
#   article 16, paragraphe 4, du traité sur l'Union européenne
#   articles 48 et 79 du traité sur le fonctionnement de l'Union européenne
RE_QUALIFIED_REF = re.compile(
    r"\b(?:l'article|les articles?|article)\s+([^.;\n]{1,100}?)\s+"
    r"du\s+trait[ée]\s+sur\s+"
    r"(l'Union\s+europ[ée]enne|le\s+fonctionnement\s+de\s+l'Union\s+europ[ée]enne)",
    re.IGNORECASE,
)

# "article 6 du présent protocole"
RE_PROTOCOL_REF = re.compile(
    r"\b(?:l'article|les articles?|article)\s+([^.;\n]{1,80}?)\s+"
    r"du\s+présent\s+protocole\b",
    re.IGNORECASE,
)

# Unqualified article references. Qualified references are masked before this
# pattern is used so we do not incorrectly assign a TUE reference to a TFUE
# reference (or vice versa).
RE_GENERIC_REF = re.compile(
    r"\b(?:l'article|les articles?|article)\s+"
    r"(\d+(?:\s*(?:bis|ter))?(?:\s*(?:,|et)\s*\d+(?:\s*(?:bis|ter))?)*)",
    re.IGNORECASE,
)

ORDINAL_PARAGRAPHS = {
    "premier": 1,
    "première": 1,
    "premiere": 1,
    "second": 2,
    "seconde": 2,
    "deuxième": 2,
    "deuxieme": 2,
    "troisième": 3,
    "troisieme": 3,
    "quatrième": 4,
    "quatrieme": 4,
    "cinquième": 5,
    "cinquieme": 5,
}


def normalize_article_number(value: str) -> str:
    """Normalize '6 bis' -> '6bis'."""
    return re.sub(r"\s+", "", value.strip()).lower()


def treaty_from_text(value: str) -> str:
    """Convert a treaty description into TUE or TFEU."""
    value = value.lower()
    if "fonctionnement" in value:
        return "TFEU"
    return "TUE"


def paragraph_from_segment(segment: str) -> Optional[int]:
    """Read an explicit paragraph/alinéa number from a reference segment."""
    m = re.search(r"\bparagraphes?\s+(\d+)\b", segment, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"\b(deuxième|deuxieme|second|seconde|premier|première|premiere|"
                  r"troisième|troisieme|quatrième|quatrieme|cinquième|cinquieme)\s+alinéa\b",
                  segment, re.IGNORECASE)
    if m:
        return ORDINAL_PARAGRAPHS[m.group(1).lower()]

    return None


def article_numbers_from_segment(segment: str) -> list[str]:
    """Extract article numbers from a segment such as '48 et 79' or '3 à 6'."""
    cleaned = re.sub(
        r"\bparagraphes?\s+\d+(?:\s*(?:à|-)\s*\d+)?",
        " ",
        segment,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:premier|première|premiere|second|seconde|deuxième|deuxieme|"
        r"troisième|troisieme|quatrième|quatrieme|cinquième|cinquieme)\s+alinéa\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    range_match = re.search(r"\b(\d+)\s*(?:à|-)\s*(\d+)\b", cleaned)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start <= end and end - start <= 30:
            return [str(n) for n in range(start, end + 1)]

    numbers = re.findall(r"\b\d+(?:\s*(?:bis|ter))?\b", cleaned, re.IGNORECASE)
    return [n.strip() for n in numbers]


class LegalDocumentParser:
    """Parse the full TEU/TFEU document into structured articles."""

    def __init__(self, full_text: str):
        self.full_text = full_text
        self.lines = full_text.split("\n")
        self.articles: list[ArticleNode] = []
        self.protocols: list[dict] = []

        # Current context
        self._treaty = "TEU"
        self._title_num = None
        self._title_name = None
        self._chapter_num = None
        self._chapter_name = None
        self._section_num = None
        self._section_name = None
        self._current_protocol: Optional[str] = None
        self._protocol_parent_treaty: Optional[str] = None

    def parse(self) -> tuple[list[ArticleNode], list[dict]]:
        console.print("[bold cyan]🔍 Parsing document structure...[/]")

        i = 0
        current_article: Optional[ArticleNode] = None
        current_article_lines: list[str] = []
        current_page = 1

        while i < len(self.lines):
            line = self.lines[i].strip()

            if not line:
                i += 1
                continue

            # Track page markers generated by 01_extract.py
            page_match = re.match(r"^\[PAGE\s+(\d+)\]$", line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))
                i += 1
                continue

            # Treaty switch
            if RE_TREATY_TFEU.match(line):
                self._treaty = "TFEU"
                self._current_protocol = None
                self._protocol_parent_treaty = None
                self._reset_hierarchy()
                i += 1
                continue

            if RE_TREATY_TEU.match(line):
                self._treaty = "TEU"
                self._current_protocol = None
                self._protocol_parent_treaty = None
                self._reset_hierarchy()
                i += 1
                continue

            # Title
            title_m = RE_TITLE.match(line)
            if title_m and not self._current_protocol:
                self._title_num = title_m.group(1)
                self._title_name = title_m.group(2).strip()
                self._chapter_num = None
                self._chapter_name = None
                self._section_num = None
                self._section_name = None
                i += 1
                continue

            # Chapter
            chapter_m = RE_CHAPTER.match(line)
            if chapter_m and not self._current_protocol:
                self._chapter_num = chapter_m.group(1)
                self._chapter_name = chapter_m.group(2).strip()
                self._section_num = None
                self._section_name = None
                i += 1
                continue

            # Section
            section_m = RE_SECTION.match(line)
            if section_m and not self._current_protocol:
                self._section_num = section_m.group(1)
                self._section_name = section_m.group(2).strip()
                i += 1
                continue

            # Protocol header
            protocol_m = RE_PROTOCOL.match(line)
            if protocol_m:
                if current_article and current_article_lines:
                    self._finalize_article(current_article, current_article_lines, current_page)
                    current_article = None
                    current_article_lines = []

                protocol_num = protocol_m.group(1)
                protocol_name = protocol_m.group(2).strip()
                protocol_id = f"Protocol-{protocol_num}"
                parent_treaty = self._treaty

                protocol_refs = self._extract_references(
                    protocol_name,
                    default_treaty=None,
                    protocol_id=protocol_id,
                )

                self.protocols.append({
                    "protocol_id": protocol_id,
                    "number": protocol_num,
                    "name": protocol_name,
                    "page": current_page,
                    "parent_treaty": parent_treaty,
                    "references": protocol_refs,
                })

                self._current_protocol = protocol_num
                self._protocol_parent_treaty = parent_treaty
                self._reset_hierarchy()
                i += 1
                continue

            # Article header
            article_m = RE_ARTICLE.match(line)
            if article_m:
                if current_article and current_article_lines:
                    self._finalize_article(current_article, current_article_lines, current_page)

                art_num = article_m.group(1).strip()
                normalized_num = normalize_article_number(art_num)

                if self._current_protocol:
                    article_id = f"Protocol-{self._current_protocol}-Art-{normalized_num}"
                    article_treaty = f"Protocol-{self._current_protocol}"
                    document_type = "protocol"
                    protocol_num = self._current_protocol
                    parent_treaty = self._protocol_parent_treaty
                    title_num = None
                    title_name = None
                    chapter_num = None
                    chapter_name = None
                    section_num = None
                    section_name = None
                else:
                    article_id = f"{self._treaty}-Art-{normalized_num}"
                    article_treaty = self._treaty
                    document_type = "treaty"
                    protocol_num = None
                    parent_treaty = None
                    title_num = self._title_num
                    title_name = self._title_name
                    chapter_num = self._chapter_num
                    chapter_name = self._chapter_name
                    section_num = self._section_num
                    section_name = self._section_name

                current_article = ArticleNode(
                    article_id=article_id,
                    number=art_num,
                    treaty=article_treaty,
                    document_type=document_type,
                    protocol_num=protocol_num,
                    parent_treaty=parent_treaty,
                    title_num=title_num,
                    title_name=title_name,
                    chapter_num=chapter_num,
                    chapter_name=chapter_name,
                    section_num=section_num,
                    section_name=section_name,
                    page_start=current_page,
                )
                current_article_lines = []
                i += 1
                continue

            # Collect article body
            if current_article is not None:
                current_article_lines.append(line)

            i += 1

        if current_article and current_article_lines:
            self._finalize_article(current_article, current_article_lines, current_page)

        console.print(
            f"[bold green]✓ Parsed:[/] {len(self.articles)} articles, "
            f"{len(self.protocols)} protocols"
        )
        return self.articles, self.protocols

    def _reset_hierarchy(self):
        self._title_num = None
        self._title_name = None
        self._chapter_num = None
        self._chapter_name = None
        self._section_num = None
        self._section_name = None

    def _finalize_article(self, article: ArticleNode, lines: list[str], page_end: int):
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        article.text = "\n".join(lines)
        article.page_end = page_end

        ex_m = RE_EX_ARTICLE.search(article.text)
        if ex_m:
            article.ex_article = f"ex-article {ex_m.group(1)} {ex_m.group(2)}"

        default_treaty = (
            article.parent_treaty
            if article.document_type == "protocol" and article.parent_treaty
            else article.treaty
        )

        article.cites = self._extract_references(
            article.text,
            default_treaty=default_treaty,
            protocol_id=(f"Protocol-{article.protocol_num}" if article.protocol_num else None),
        )

        article.paragraphs = self._split_paragraphs(article.text)

        if len(article.text.strip()) > 20:
            if not any(a.article_id == article.article_id for a in self.articles):
                self.articles.append(article)

    def _extract_references(
        self,
        text: str,
        default_treaty: Optional[str],
        protocol_id: Optional[str] = None,
    ) -> list[dict]:
        """Extract treaty-aware article references."""
        refs: list[dict] = []
        seen: set[tuple[str, Optional[int]]] = set()

        def add_ref(
            target_treaty: str,
            article_number: str,
            paragraph: Optional[int],
            source_text: str,
        ):
            target_article = normalize_article_number(article_number)
            target_id = f"{target_treaty}-Art-{target_article}"
            if paragraph is not None:
                target_id += f"-P{paragraph}"

            key = (target_id, paragraph)
            if key in seen:
                return

            seen.add(key)
            refs.append({
                "target_id": target_id,
                "treaty": target_treaty,
                "article": article_number.strip(),
                "paragraph": paragraph,
                "text": source_text,
            })

        # Start with a masked copy. Ex-article labels are historical
        # numbering, not citations to the current article.
        masked = list(text)
        for m in RE_EX_ARTICLE.finditer(text):
            for idx in range(m.start(), m.end()):
                masked[idx] = " "

        # 1) "article X du présent protocole"
        if protocol_id:
            for m in RE_PROTOCOL_REF.finditer(text):
                segment = m.group(1)
                paragraph = paragraph_from_segment(segment)
                nums = article_numbers_from_segment(segment)
                for num in nums:
                    add_ref(protocol_id, num, paragraph if len(nums) == 1 else None, m.group(0))
                for idx in range(m.start(), m.end()):
                    masked[idx] = " "

        # 2) "article X ... du traité sur ..."
        for m in RE_QUALIFIED_REF.finditer(text):
            segment = m.group(1)
            treaty = treaty_from_text(m.group(2))
            paragraph = paragraph_from_segment(segment)
            nums = article_numbers_from_segment(segment)
            for num in nums:
                # If one article has one explicit paragraph, preserve it.
                p = paragraph if len(nums) == 1 else None
                add_ref(treaty, num, p, m.group(0))
            for idx in range(m.start(), m.end()):
                masked[idx] = " "

        # 3) "article X TUE/TFUE"
        masked_text = "".join(masked)
        for m in RE_ABBR_REF.finditer(masked_text):
            nums = [m.group(1)]
            paragraph = int(m.group(2)) if m.group(2) else None
            treaty = m.group(3).upper()
            for num in nums:
                add_ref(treaty, num, paragraph, m.group(0))
            for idx in range(m.start(), m.end()):
                masked[idx] = " "

        # 4) Remaining unqualified references -> current treaty/document.
        if default_treaty:
            remaining = "".join(masked)
            for m in RE_GENERIC_REF.finditer(remaining):
                nums = re.findall(r"\d+(?:\s*(?:bis|ter))?", m.group(1), re.IGNORECASE)
                for num in nums:
                    add_ref(default_treaty, num, None, m.group(0))

        return refs

    def _split_paragraphs(self, text: str) -> list[str]:
        parts = re.split(r"\n(?=\d+\.\s)", text)
        if len(parts) <= 1:
            parts = re.split(r"\n(?=[a-z]\)\s)", text)
        return [p.strip() for p in parts if p.strip()]


def parse_document(full_text: str) -> tuple[list[ArticleNode], list[dict]]:
    parser = LegalDocumentParser(full_text)
    return parser.parse()


def save_structured_data(
    articles: list[ArticleNode],
    protocols: list[dict],
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    articles_path = out / "articles.json"
    articles_path.write_text(
        json.dumps([asdict(a) for a in articles], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    protocols_path = out / "protocols.json"
    protocols_path.write_text(
        json.dumps(protocols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    console.print(f"[green]✓ Saved {len(articles)} articles to:[/] {articles_path}")
    console.print(f"[green]✓ Saved {len(protocols)} protocols to:[/] {protocols_path}")

    table = Table(title="Document Structure Summary")
    table.add_column("Treaty", style="cyan")
    table.add_column("Articles", style="green")
    table.add_column("Unique Titles", style="yellow")

    for treaty in ["TEU", "TFEU"]:
        treaty_articles = [
            a for a in articles
            if a.document_type == "treaty" and a.treaty == treaty
        ]
        titles = {a.title_num for a in treaty_articles if a.title_num}
        table.add_row(treaty, str(len(treaty_articles)), str(len(titles)))

    console.print(table)


if __name__ == "__main__":
    import sys

    text_path = sys.argv[1] if len(sys.argv) > 1 else "./data/raw/full_text.txt"
    text = Path(text_path).read_text(encoding="utf-8")
    articles, protocols = parse_document(text)
    save_structured_data(articles, protocols, "./data/structured")

    for art in articles[:3]:
        print(f"\n{'=' * 60}")
        print(f"ID: {art.article_id} | Treaty: {art.treaty} | Page: {art.page_start}")
        print(f"Document type: {art.document_type}")
        print(f"Title: {art.title_num} — {art.title_name}")
        print(f"Ex-article: {art.ex_article}")
        print(f"References: {art.cites[:3]}")
        print(f"Text preview: {art.text[:300]}...")
