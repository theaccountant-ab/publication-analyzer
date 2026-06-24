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
    genuine no-match, raises ``OpenAlexUnavailable`` if Crossref can't be reached.

    Matching uses two queries — by title, and by author+title — then prefers a
    real journal over a preprint. A candidate is accepted when its title is a
    near-match (>=0.90), or a looser match (>=0.72) backed by an author-surname
    overlap. The author-anchored query + looser-with-author rule recover papers
    whose title changed between the presented working paper and publication (the
    main reason a strict title search misses real publications).
    """
    title = (title or "").strip()
    if not title:
        return None

    want_title = _normalize_title(title)
    want_surnames = _surnames(authors)

    # Two complementary queries: title-only, and author+title (the latter surfaces
    # the published version even when its title differs from the presented one).
    sel = "DOI,title,container-title,type,issued,author"
    queries = [{"query.bibliographic": title, "rows": "8", "select": sel}]
    if want_surnames:
        queries.append({"query.author": " ".join(sorted(want_surnames)[:3]),
                        "query.bibliographic": title, "rows": "8", "select": sel})

    items, reached = [], False
    for params in queries:
        if mailto:
            params["mailto"] = mailto
        url = CROSSREF_WORKS_URL + "?" + urllib.parse.urlencode(params)
        delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                _pace()
                items += (fetch(url).get("message", {}) or {}).get("items", []) or []
                reached = True
                break
            except Exception as exc:
                if attempt == max_retries:
                    break  # this query failed; try the other before giving up
                wait = delay
                if isinstance(exc, urllib.error.HTTPError) and exc.headers:
                    ra = exc.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = max(wait, min(float(ra), 60.0))
                time.sleep(wait)
                delay = min(delay * 2, 30.0)
    if not reached:
        raise OpenAlexUnavailable("Crossref unreachable for both queries")

    best: Optional[PublicationMatch] = None
    seen = set()
    for item in items:
        doi = item.get("DOI", "")
        if doi and doi in seen:
            continue
        seen.add(doi)
        titles = item.get("title") or []
        cand_title = titles[0] if titles else ""
        if not cand_title:
            continue
        sim = SequenceMatcher(None, want_title, _normalize_title(cand_title)).ratio()
        overlap = bool(want_surnames & _cr_surnames(item))
        # Accept a strong title match (>=0.90) on its own, or a looser one (>=0.72)
        # backed by an author-surname overlap (recovers renamed papers). Validated
        # against the HSU benchmark: this lands ~2pts under it without over-counting.
        if not (sim >= _TITLE_MATCH_THRESHOLD or (sim >= 0.72 and overlap)):
            continue
        wy = _cr_year(item)
        if year and wy and wy < year - 1:
            continue
        container = (item.get("container-title") or [None])[0]
        work_type = item.get("type", "")
        is_journal = _is_real_journal(container, work_type)
        match = PublicationMatch(
            openalex_id=("https://doi.org/" + doi) if doi else "",
            matched_title=cand_title,
            work_type=work_type,
            source_name=container,
            source_type="journal" if is_journal else (work_type or "other"),
            publication_year=wy,
            title_similarity=round(sim, 3),
        )
        # Prefer a real journal over a preprint; within the same kind, higher title
        # similarity. So the published JFE version beats an SSRN copy of equal title.
        better = (
            best is None
            or (match.is_journal_article and not best.is_journal_article)
            or (match.is_journal_article == best.is_journal_article
                and match.title_similarity > best.title_similarity)
        )
        if better:
            best = match
    return best
