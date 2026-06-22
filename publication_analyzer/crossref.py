"""Resolve where a paper was published via the Crossref API.

A drop-in alternative to the OpenAlex resolver (``openalex.lookup_publication``)
with the same signature and return type, used when OpenAlex rate-limits the
caller's IP. Crossref (https://www.crossref.org) is a free DOI registry; the
polite pool (set ``mailto``) is generous with request volume.

Matching mirrors the OpenAlex resolver: accept a candidate only when its title is
a close normalized match and (unless the title match is near-exact) an author
surname overlaps. Crossref lists preprints (notably SSRN, arXiv) as
``journal-article`` too, so those venues are treated as NON-journal here — only a
real journal counts as "published in a journal", matching the OpenAlex behavior.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from typing import Callable, List, Optional

from .openalex import (
    OpenAlexUnavailable,
    PublicationMatch,
    _TITLE_MATCH_THRESHOLD,
    _TITLE_NEAR_EXACT,
    _normalize_title,
    _pace,
    _surnames,
)

CROSSREF_WORKS_URL = "https://api.crossref.org/works"

# Container names that are preprint/working-paper venues, not real journals.
# Crossref mislabels these as "journal-article", so exclude them explicitly.
_PREPRINT_VENUES = (
    "ssrn", "arxiv", "preprint", "working paper", "nber", "repec",
    "research papers in economics", "cepr discussion", "iza discussion",
)


def _is_real_journal(container: Optional[str], work_type: str) -> bool:
    if work_type != "journal-article" or not container:
        return False
    low = container.lower()
    return not any(v in low for v in _PREPRINT_VENUES)


def _http_get_json(url: str, *, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "publication-analyzer"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cr_surnames(item: dict) -> set:
    out = set()
    for au in item.get("author", []) or []:
        fam = (au.get("family") or "").strip().lower()
        if fam:
            out.add(fam)
    return out


def _cr_year(item: dict) -> Optional[int]:
    for key in ("published", "published-print", "published-online", "issued"):
        parts = (item.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def lookup_publication(
    title: str,
    *,
    year: Optional[int] = None,
    authors: Optional[List[str]] = None,
    mailto: str = "",
    fetch: Callable[[str], dict] = _http_get_json,
    max_retries: int = 4,
) -> Optional[PublicationMatch]:
    """Find the Crossref work best matching ``title`` and report its venue.

    Same contract as ``openalex.lookup_publication``: returns ``None`` for a
    genuine no-match, raises ``OpenAlexUnavailable`` if Crossref can't be reached,
    and prefers a real journal version over a preprint at equal title similarity.
    """
    title = (title or "").strip()
    if not title:
        return None

    params = {"query.bibliographic": title, "rows": "5",
              "select": "DOI,title,container-title,type,issued,author"}
    if mailto:
        params["mailto"] = mailto
    url = CROSSREF_WORKS_URL + "?" + urllib.parse.urlencode(params)

    delay = 1.0
    data: dict = {}
    for attempt in range(max_retries + 1):
        try:
            _pace()
            data = fetch(url)
            break
        except Exception as exc:
            if attempt == max_retries:
                raise OpenAlexUnavailable(str(exc)) from exc
            wait = delay
            if isinstance(exc, urllib.error.HTTPError) and exc.headers:
                ra = exc.headers.get("Retry-After")
                if ra and ra.isdigit():
                    wait = max(wait, min(float(ra), 60.0))
            time.sleep(wait)
            delay = min(delay * 2, 30.0)

    want_title = _normalize_title(title)
    want_surnames = _surnames(authors)

    best: Optional[PublicationMatch] = None
    for item in (data.get("message", {}) or {}).get("items", []) or []:
        titles = item.get("title") or []
        cand_title = titles[0] if titles else ""
        if not cand_title:
            continue
        sim = SequenceMatcher(None, want_title, _normalize_title(cand_title)).ratio()
        if sim < _TITLE_MATCH_THRESHOLD:
            continue
        near_exact = sim >= _TITLE_NEAR_EXACT and len(want_title.split()) >= 4
        if want_surnames and not near_exact and not (
            want_surnames & _cr_surnames(item)
        ):
            continue
        wy = _cr_year(item)
        if year and wy and wy < year - 1:
            continue
        container = (item.get("container-title") or [None])[0]
        work_type = item.get("type", "")
        is_journal = _is_real_journal(container, work_type)
        match = PublicationMatch(
            openalex_id=("https://doi.org/" + item["DOI"]) if item.get("DOI") else "",
            matched_title=cand_title,
            work_type=work_type,
            source_name=container,
            source_type="journal" if is_journal else (work_type or "other"),
            publication_year=wy,
            title_similarity=round(sim, 3),
        )
        # Prefer the closest title; at a tie prefer a real journal over a preprint
        # (so the published version wins over an SSRN/arXiv copy of the same paper).
        if (
            best is None
            or match.title_similarity > best.title_similarity
            or (match.title_similarity == best.title_similarity
                and match.is_journal_article and not best.is_journal_article)
        ):
            best = match
    return best
