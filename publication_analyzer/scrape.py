"""Scrape a conference's program from its own web page(s) or PDF.

Conference programs have no common machine-readable format, so rather than write
a parser per site, this fetches the page you point it at, converts it to plain
text (HTML via the stdlib; PDF via ``pypdf``), and lets Gemini's structured
extraction pull out the paper list. The page is the conference's *actual*
program, so the resulting denominator is as complete as that page — far more so
than ad-hoc web search.

Provide the pages in a sources CSV with columns ``conference, year, url`` (one
row per page; a conference may have several rows for several years).
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Tuple

from google import genai

from .programs import Paper, dedupe_papers, parse_program_text

_USER_AGENT = (
    "Mozilla/5.0 (compatible; publication-analyzer/1.0; +https://openalex.org)"
)

# Fetcher returns (raw_bytes, content_type). Isolated so tests can inject one.
Fetcher = Callable[[str], Tuple[bytes, str]]


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, inserting newlines at block boundaries."""

    _SKIP = {"script", "style", "noscript", "head"}
    _BREAK = {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "td"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BREAK:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text + " ")

    def get_text(self) -> str:
        s = "".join(self._chunks)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n[ \t]*", "\n", s)
        s = re.sub(r"\n{2,}", "\n", s)
        return s.strip()


def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text (drops script/style, keeps line breaks)."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def pdf_to_text(data: bytes) -> str:
    """Extract text from a PDF byte string (requires the optional ``pypdf``)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without pypdf
        raise RuntimeError(
            "Reading PDF programs requires 'pypdf' (pip install pypdf)."
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _default_fetch(url: str, *, timeout: float = 30.0) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get_content_type()


def fetch_text(url: str, *, fetch: Fetcher = _default_fetch) -> str:
    """Fetch ``url`` and return its text, handling HTML and PDF content."""
    data, content_type = fetch(url)
    is_pdf = content_type == "application/pdf" or url.lower().split("?")[0].endswith(".pdf")
    if is_pdf:
        return pdf_to_text(data)
    html = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    return html_to_text(html)


def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    """Split text into <=``max_chars`` chunks on line boundaries.

    Large programs (hundreds of papers) would overflow the model's output
    budget in one parse, so the page text is chunked and parsed piece by piece.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and current:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return [c for c in chunks if c.strip()]


def scrape_program(
    client: genai.Client,
    model: str,
    urls: List[str],
    *,
    year: Optional[int] = None,
    fetch: Fetcher = _default_fetch,
    parse: Optional[Callable[[str], List[Paper]]] = None,
    max_chars: int = 12000,
) -> List[Paper]:
    """Fetch program page(s), extract papers, and return a deduplicated list.

    ``fetch`` and ``parse`` are injectable for testing. ``year`` is stamped onto
    any paper whose year the page didn't make explicit. Pages that fail to fetch
    are reported and skipped so one bad URL doesn't sink the run.
    """
    do_parse = parse or (lambda text: parse_program_text(client, model, text))
    papers: List[Paper] = []
    for url in urls:
        try:
            text = fetch_text(url, fetch=fetch)
        except Exception as exc:  # network / decode / PDF errors
            print(f"  ! failed to fetch {url}: {exc}")
            continue
        for chunk in chunk_text(text, max_chars):
            papers.extend(do_parse(chunk))
    if year is not None:
        for p in papers:
            if p.year is None:
                p.year = year
    return dedupe_papers(papers)


def read_scrape_sources(path: str) -> Dict[str, List[Tuple[Optional[int], str]]]:
    """Read a sources CSV (columns: ``conference, url`` and optional ``year``).

    Returns ``{conference: [(year, url), ...]}`` preserving row order.
    """
    sources: Dict[str, List[Tuple[Optional[int], str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        field_map = {(n or "").strip().lower(): n for n in (reader.fieldnames or [])}
        conf_col = field_map.get("conference")
        url_col = field_map.get("url")
        year_col = field_map.get("year")
        if not conf_col or not url_col:
            raise ValueError(
                "Sources CSV must have at least 'conference' and 'url' columns."
            )
        for row in reader:
            conference = (row.get(conf_col) or "").strip()
            url = (row.get(url_col) or "").strip()
            if not conference or not url:
                continue
            year_raw = (row.get(year_col) or "").strip() if year_col else ""
            try:
                year = int(year_raw) if year_raw else None
            except ValueError:
                year = None
            sources.setdefault(conference, []).append((year, url))
    return sources
