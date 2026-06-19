"""Acquire a conference's program — the list of papers presented (the denominator).

Two sources are supported:

* **Authoritative CSV** (``read_program_csv``): a file you supply with columns
  ``conference, year, title, authors`` (authors separated by ``;``). This gives
  an exact denominator and is the recommended input for a rigorous rate.
* **Best-effort discovery** (``discover_program_via_search``): when no program
  is supplied, ask Gemini — grounded in Google Search — to list the papers. This
  is incomplete by nature (a search can't reliably enumerate a whole program),
  so prefer the CSV when you have it.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from google import genai
from google.genai import errors, types

from .models import ProgramPaperList

_TRANSIENT_CODES = {429, 500, 502, 503, 504}


@dataclass
class Paper:
    """One presented paper (the unit of the denominator)."""

    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None


def _split_authors(raw: str) -> List[str]:
    return [a.strip() for a in (raw or "").replace("|", ";").split(";") if a.strip()]


def read_program_csv(path: str) -> Dict[str, List[Paper]]:
    """Read an authoritative program CSV into ``{conference: [Paper, ...]}``.

    Expected columns (header row, case-insensitive): ``conference``, ``title``,
    and optionally ``year`` and ``authors`` (authors separated by ``;`` or
    ``|``). Rows without a conference or title are skipped.
    """
    programs: Dict[str, List[Paper]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        # Map headers case-insensitively.
        field_map = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}
        conf_col = field_map.get("conference")
        title_col = field_map.get("title")
        year_col = field_map.get("year")
        authors_col = field_map.get("authors")
        if not conf_col or not title_col:
            raise ValueError(
                "Program CSV must have at least 'conference' and 'title' columns."
            )
        for row in reader:
            conference = (row.get(conf_col) or "").strip()
            title = (row.get(title_col) or "").strip()
            if not conference or not title:
                continue
            year_raw = (row.get(year_col) or "").strip() if year_col else ""
            try:
                year = int(year_raw) if year_raw else None
            except ValueError:
                year = None
            authors = _split_authors(row.get(authors_col, "")) if authors_col else []
            programs.setdefault(conference, []).append(
                Paper(title=title, authors=authors, year=year)
            )
    return programs


_DISCOVERY_PROMPT = """\
Using Google Search, find the papers that were presented at the academic \
conference "{name}" in the following year(s): {years}.

Consult the conference program, proceedings, or schedule pages. For every paper \
you can identify, report on its own line: the paper title exactly as presented, \
its authors, and the year it was presented. List the actual papers — do not \
summarize with counts. If you cannot find the program for a given year, say so \
for that year. Do not invent papers.\
"""

_PARSE_SYSTEM_PROMPT = """\
You turn research notes about a conference's program into structured records. \
For each distinct paper mentioned, output its title, its authors (as a list of \
names, empty if unknown), and the year it was presented. Do not invent papers \
that are not in the notes.\
"""


def parse_program_text(
    client: genai.Client,
    model: str,
    text: str,
    *,
    max_output_tokens: int = 8000,
    max_retries: int = 4,
) -> List[Paper]:
    """Parse a blob of program text into structured ``Paper`` records via Gemini.

    Shared by search-based discovery and the page scraper. Raises
    ``errors.APIError`` if the API keeps failing on a transient error.
    """
    if not text.strip():
        return []
    config = types.GenerateContentConfig(
        system_instruction=_PARSE_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=ProgramPaperList,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=max_output_tokens,
    )
    contents = (
        "Extract every paper described in the following program text.\n\n"
        "<program>\n" + text.strip() + "\n</program>"
    )
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            break
        except errors.APIError as exc:
            transient = getattr(exc, "code", None) in _TRANSIENT_CODES
            if not transient or attempt == max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    parsed = response.parsed
    if not isinstance(parsed, ProgramPaperList):
        return []
    return [
        Paper(title=p.title.strip(), authors=list(p.authors or []), year=p.year)
        for p in parsed.papers
        if (p.title or "").strip()
    ]


def dedupe_papers(papers: List[Paper]) -> List[Paper]:
    """Drop duplicate papers (same normalized title), keeping the first seen.

    A later duplicate can still fill in a missing year or extra authors on the
    record that's kept.
    """
    seen: Dict[str, Paper] = {}
    for p in papers:
        key = re.sub(r"[^a-z0-9]+", " ", (p.title or "").lower()).strip()
        if not key:
            continue
        if key not in seen:
            seen[key] = p
        else:
            kept = seen[key]
            if kept.year is None and p.year is not None:
                kept.year = p.year
            if not kept.authors and p.authors:
                kept.authors = p.authors
    return list(seen.values())


def discover_program_via_search(
    client: genai.Client,
    model: str,
    name: str,
    years: List[int],
    *,
    max_research_tokens: int = 8000,
    max_output_tokens: int = 8000,
    max_retries: int = 4,
) -> List[Paper]:
    """Best-effort: discover a conference's program via Gemini + Google Search.

    Returns a (likely incomplete) list of papers. Raises ``errors.APIError`` if
    the API keeps failing, so the caller can record the failure.
    """
    years_str = ", ".join(str(y) for y in years)
    notes = client.models.generate_content(
        model=model,
        contents=_DISCOVERY_PROMPT.format(name=name, years=years_str),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=max_research_tokens,
        ),
    )
    text = (notes.text or "").strip()
    if not text:
        return []
    return dedupe_papers(
        parse_program_text(
            client, model, text,
            max_output_tokens=max_output_tokens, max_retries=max_retries,
        )
    )
