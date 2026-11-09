"""
chunker.py — Smart chunking that respects article boundaries.

Strategy:
- Each article is a primary unit (never split across chunks)
- Long articles (> chunk_size tokens) are split by paragraph
- Each chunk carries full metadata for graph + vector linking
"""
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass
class Chunk:
    """A text chunk ready for embedding and graph linking."""
    chunk_id: str
    article_id: str
    treaty: str
    article_num: str
    chunk_index: int
    text: str
    title_num: str | None = None
    title_name: str | None = None
    chapter_num: str | None = None
    chapter_name: str | None = None
    page_start: int = 0
    metadata: dict = field(default_factory=dict)


def chunk_articles(
    articles_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Load parsed articles and create chunks.
    Articles shorter than chunk_size are kept as single chunks.
    Longer articles are split by paragraphs, then by character count.
    """
    articles = json.loads(Path(articles_path).read_text(encoding="utf-8"))
    console.print(f"[cyan]📦 Chunking {len(articles)} articles...[/]")

    chunks: list[Chunk] = []

    for article in track(articles, description="Chunking..."):
        text = article["text"].strip()
        if not text:
            continue

        paragraphs = article.get("paragraphs") or [text]

        # If single paragraph fits in chunk_size, keep whole article
        if len(text) <= chunk_size * 4:  # approximate chars (4 chars/token avg)
            chunk = Chunk(
                chunk_id=f"{article['article_id']}-chunk-0",
                article_id=article["article_id"],
                treaty=article["treaty"],
                article_num=article["number"],
                chunk_index=0,
                text=text,
                title_num=article.get("title_num"),
                title_name=article.get("title_name"),
                chapter_num=article.get("chapter_num"),
                chapter_name=article.get("chapter_name"),
                page_start=article.get("page_start", 0),
                metadata={
                    "ex_article": article.get("ex_article"),
                    "cites": article.get("cites", []),
                    "treaty": article["treaty"],
                }
            )
            chunks.append(chunk)
        else:
            # Split by paragraphs with overlap
            current_chunk_text = ""
            chunk_idx = 0
            prev_para = ""

            for para in paragraphs:
                candidate = (current_chunk_text + "\n" + para).strip()

                if len(candidate) > chunk_size * 4 and current_chunk_text:
                    # Save current chunk
                    chunk = Chunk(
                        chunk_id=f"{article['article_id']}-chunk-{chunk_idx}",
                        article_id=article["article_id"],
                        treaty=article["treaty"],
                        article_num=article["number"],
                        chunk_index=chunk_idx,
                        text=current_chunk_text.strip(),
                        title_num=article.get("title_num"),
                        title_name=article.get("title_name"),
                        chapter_num=article.get("chapter_num"),
                        chapter_name=article.get("chapter_name"),
                        page_start=article.get("page_start", 0),
                        metadata={
                            "ex_article": article.get("ex_article"),
                            "cites": article.get("cites", []),
                            "treaty": article["treaty"],
                        }
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    # Start next chunk with overlap (previous paragraph)
                    current_chunk_text = prev_para + "\n" + para
                else:
                    current_chunk_text = candidate
                prev_para = para

            # Save remaining text
            if current_chunk_text.strip():
                chunk = Chunk(
                    chunk_id=f"{article['article_id']}-chunk-{chunk_idx}",
                    article_id=article["article_id"],
                    treaty=article["treaty"],
                    article_num=article["number"],
                    chunk_index=chunk_idx,
                    text=current_chunk_text.strip(),
                    title_num=article.get("title_num"),
                    title_name=article.get("title_name"),
                    chapter_num=article.get("chapter_num"),
                    chapter_name=article.get("chapter_name"),
                    page_start=article.get("page_start", 0),
                    metadata={
                        "ex_article": article.get("ex_article"),
                        "cites": article.get("cites", []),
                        "treaty": article["treaty"],
                    }
                )
                chunks.append(chunk)

    console.print(f"[bold green]✓ Created {len(chunks)} chunks from {len(articles)} articles[/]")
    return chunks


def save_chunks(chunks: list[Chunk], output_dir: str) -> Path:
    """Save chunks to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "chunks.json"
    path.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    console.print(f"[green]✓ Saved chunks to:[/] {path}")
    return path


if __name__ == "__main__":
    chunks = chunk_articles("./data/structured/articles.json")
    save_chunks(chunks, "./data/structured")
    print(f"\nSample chunk:\n{chunks[5].text[:300]}")
