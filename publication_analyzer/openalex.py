"""Look up where a paper was published, via the OpenAlex API.

OpenAlex (https://openalex.org) is a free, open bibliographic database with no
API key required. Given a paper's title (and optionally its authors and year),
we search OpenAlex for the matching work and report the journal it was published
in — which the analysis layer then checks against the top-tier list.

Matching is conservative: a candidate is only accepted when its title is a close
normalized match and, when authors are supplied, at least one author surname
overlaps. This avoids crediting a paper with an unrelated same-words hit. Only
works hosted in a *journal* count as "published in a journal" — working papers on
SSRN / NBER / RePEc (OpenAlex source type ``repository``) deliberately do not.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, List, Optional

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class OpenAlexUnavailable(Exception):
    """Raised when OpenAlex can't be reached (e.g. sustained rate-limiting).

    Distinct from a successful lookup that simply found no match, so callers can
    retry later instead of caching the failure as "not published".
    """

# Minimum seconds between OpenAlex requests. OpenAlex throttles bursts (HTTP 429);
# pacing sequential lookups keeps a large analysis under the limit so matches
# aren't silently lost. Set the contact email (mailto) to use the faster pool.
_MIN_REQUEST_INTERVAL = 0.12
_last_request = 0.0


def _pace() -> None:
    global _last_request
    wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()

# Accept a candidate only when the normalized titles are at least this similar.
_TITLE_MATCH_THRESHOLD = 0.90
# At/above this similarity a title match is accepted even without an author-surname
# overlap. Program scrapes often misattribute authors (e.g. a discussant listed in
# place of the author), so a near-exact title on a specific multi-word paper is a
# stronger signal than the scraped author list.
_TITLE_NEAR_EXACT = 0.95

_THE_PREFIX = re.compile(r"^the\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass
class PublicationMatch:
    """The OpenAlex work we matched a program paper to."""

    openalex_id: str
    matched_title: str
    work_type: str                 # e.g. "article"
    source_name: Optional[str]     # host venue display name (journal, etc.)
    source_type: Optional[str]     # e.g. "journal", "conference", "repository"
    publication_year: Optional[int]
    title_similarity: float

    @property
    def is_journal_article(self) -> bool:
        """True when the work is an article hosted in a journal."""
        return self.source_type == "journal" and bool(self.source_name)


def _normalize_title(title: str) -> str:
    # NFKD folds typographic ligatures (e.g. "ﬁ"/"ﬂ" from PDF text) back to ASCII
    # so "Inﬂation" and "Inflation" compare equal.
    s = unicodedata.normalize("NFKD", title or "").strip().lower()
    s = _THE_PREFIX.sub("", s)
    s = _NON_ALNUM.sub(" ", s).strip()
    return re.sub(r"\s+", " ", s)


def _search_term(title: str) -> str:
    """A punctuation-free term safe for OpenAlex's ``title.search`` filter.

    Commas and colons are structural in the filter syntax (a comma separates
    filters), so a raw title containing them yields HTTP 400. Folding ligatures
    and stripping punctuation also makes the full-text search more forgiving.
    """
    s = unicodedata.normalize("NFKD", title or "")
    s = re.sub(r"[^0-9A-Za-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _surnames(authors: Optional[List[str]]) -> set:
    """Lowercased last tokens of each author string ("Jane Q. Smith" -> "smith")."""
    out = set()
    for a in authors or []:
        toks = re.sub(r"[^A-Za-z\s]", " ", a).split()
        if toks:
            out.add(toks[-1].lower())
    return out


def _http_get_json(url: str, *, timeout: float = 30.0) -> dict:
    """Default HTTP fetch. Isolated so tests can inject a fake fetcher."""
    req = urllib.request.Request(url, headers={"User-Agent": "publication-analyzer"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _author_names(work: dict) -> List[str]:
    names = []
    for au in work.get("authorships", []) or []:
        name = (au.get("author") or {}).get("display_name")
        if name:
            names.append(name)
    return names


def lookup_publication(
    title: str,
    *,
    year: Optional[int] = None,
    authors: Optional[List[str]] = None,
    mailto: str = "",
    fetch: Callable[[str], dict] = _http_get_json,
    max_retries: int = 5,
) -> Optional[PublicationMatch]:
    """Find the OpenAlex work best matching ``title`` and report its venue.

    Returns ``None`` when no candidate clears the title-similarity (and, if
    given, author-overlap) bar. ``fetch`` is injectable so the network call can
    be stubbed in tests; ``mailto`` joins OpenAlex's polite pool when set.

    Transient fetch failures (notably HTTP 429 rate-limiting) are retried with
    backoff that honors a ``Retry-After`` header, so a throttled lookup isn't
    mistaken for "no publication found" — which would silently undercount. If
    every retry fails, ``OpenAlexUnavailable`` is raised rather than returning
    ``None``, so the caller can tell "couldn't reach OpenAlex" from "no match".
    """
    title = (title or "").strip()
    if not title:
        return None

    search_term = _search_term(title)
    if not search_term:
        return None
    params = {
        "filter": f"title.search:{search_term}",
        "per-page": "10",
        "select": "id,title,type,publication_year,primary_location,authorships",
    }
    if mailto:
        params["mailto"] = mailto
    url = OPENALEX_WORKS_URL + "?" + urllib.parse.urlencode(params)

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
            # Respect a server-supplied Retry-After on 429/503, but cap it: a
            # hard-throttled IP can return a multi-hour Retry-After, and we must
            # fail fast rather than hang the whole run on one sleep.
            if isinstance(exc, urllib.error.HTTPError) and exc.headers:
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, min(float(retry_after), 60.0))
            time.sleep(wait)
            delay = min(delay * 2, 30.0)

    want_title = _normalize_title(title)
    want_surnames = _surnames(authors)

    best: Optional[PublicationMatch] = None
    for work in data.get("results", []) or []:
        cand_title = work.get("title") or work.get("display_name") or ""
        sim = SequenceMatcher(None, want_title, _normalize_title(cand_title)).ratio()
        if sim < _TITLE_MATCH_THRESHOLD:
            continue
        # Require an author-surname overlap, except for near-exact matches on a
        # specific (multi-word) title, where the title alone is a strong signal and
        # the scraped author list is often unreliable.
        near_exact = sim >= _TITLE_NEAR_EXACT and len(want_title.split()) >= 4
        if want_surnames and not near_exact and not (
            want_surnames & _surnames(_author_names(work))
        ):
            continue
        # When we know the year, ignore works published before it (a conference
        # paper is published in the same year or later).
        wy = work.get("publication_year")
        if year and wy and wy < year - 1:
            continue
        source = (work.get("primary_location") or {}).get("source") or {}
        match = PublicationMatch(
            openalex_id=work.get("id", ""),
            matched_title=cand_title,
            work_type=work.get("type", ""),
            source_name=source.get("display_name"),
            source_type=source.get("type"),
            publication_year=wy,
            title_similarity=round(sim, 3),
        )
        # Prefer the closest title; break ties toward an actual journal article.
        if (
            best is None
            or match.title_similarity > best.title_similarity
            or (match.title_similarity == best.title_similarity
                and match.is_journal_article and not best.is_journal_article)
        ):
            best = match
    return best
