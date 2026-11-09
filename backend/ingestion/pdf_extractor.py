"""
pdf_extractor.py — Extract text from legal PDF using PyMuPDF.
Preserves page numbers and detects section boundaries.
"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass
class PageContent:
    page_num: int
    raw_text: str
    cleaned_text: str


@dataclass
class ExtractedDocument:
    pdf_path: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Fix hyphenation at line breaks
    - Normalize whitespace
    - Remove page headers/footers patterns
    - Preserve article structure markers
    """
    # Fix soft hyphens and broken words across lines
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Remove repeated page header patterns (Journal officiel de l'Union européenne)
    text = re.sub(r'(?i)C \d+/\d+\s+Journal officiel de l.Union europ.enne\s+\d+\.\d+\.\d+\s*', '', text)
    text = re.sub(r'(?i)\d+\.\d+\.\d+\s+Journal officiel de l.Union europ.enne\s+C \d+/\d+\s*', '', text)

    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Normalize line breaks: keep paragraph breaks, merge soft wraps
    # Preserve intentional line breaks (after period or article markers)
    text = re.sub(r'([^.\n])\n([a-zàâäéèêëîïôùûüç])', r'\1 \2', text)

    # Normalize multiple spaces
    text = re.sub(r'  +', ' ', text)

    # Normalize multiple blank lines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_pdf(pdf_path: str) -> ExtractedDocument:
    """
    Extract all text from PDF with page tracking.
    Returns structured ExtractedDocument.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    console.print(f"[bold cyan]📄 Opening PDF:[/] {path.name}")
    doc = fitz.open(str(path))
    total_pages = len(doc)
    console.print(f"[green]✓ Pages detected:[/] {total_pages}")

    pages = []
    full_text_parts = []

    for page_num in track(range(total_pages), description="Extracting pages..."):
        page = doc[page_num]
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        page_content = PageContent(
            page_num=page_num + 1,
            raw_text=raw_text,
            cleaned_text=cleaned,
        )
        pages.append(page_content)
        if cleaned:
            full_text_parts.append(f"\n[PAGE {page_num + 1}]\n{cleaned}")

    doc.close()

    full_text = "\n".join(full_text_parts)

    result = ExtractedDocument(
        pdf_path=str(path),
        total_pages=total_pages,
        pages=pages,
        full_text=full_text,
    )

    console.print(f"[bold green]✓ Extraction complete:[/] {total_pages} pages, {len(full_text):,} chars")
    return result


def save_raw_text(doc: ExtractedDocument, output_dir: str) -> Path:
    """Save full extracted text to file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "full_text.txt"
    out_file.write_text(doc.full_text, encoding="utf-8")
    console.print(f"[green]✓ Saved raw text to:[/] {out_file}")
    return out_file


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "./document.pdf"
    extracted = extract_pdf(pdf)
    save_raw_text(extracted, "./data/raw")
    print(f"\nSample (page 20):\n{extracted.pages[19].cleaned_text[:500]}")
