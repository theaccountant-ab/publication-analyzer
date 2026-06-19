"""Compute a conference's top-tier journal rate from its program.

Given the papers presented at a conference (the *program* — the denominator),
this matches each paper to its publication in **OpenAlex** and checks the journal
against the top-tier list (FT50 ∪ UTD24 by default). The headline figure is the
fraction of *presented* papers that were published in a top-tier journal.

How complete that figure is depends entirely on the program it's given: an
authoritative program CSV yields a rigorous rate; the best-effort search-based
program (see ``programs.discover_program_via_search``) yields an indicative one.
The publication side, via OpenAlex, is authoritative either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, List, Optional

from .openalex import PublicationMatch, lookup_publication
from .programs import Paper

# Default top-tier set: the Financial Times FT50 (2026 refresh) unioned with the
# UT Dallas UTD24. UTD24 is a near-subset of the FT50; the only UTD24 journal not
# already in the FT50 is INFORMS Journal on Computing (added at the end). Names
# are matched after normalization (see `is_top_tier_journal`), so "The Journal of
# Finance", "Manufacturing & Service Operations Management", and trailing
# citation noise all resolve correctly. Override per-run via config
# (`top_tier_journals`).
DEFAULT_TOP_TIER_JOURNALS = [
    # --- Financial Times FT50 (current, 2026 refresh) ---
    "Academy of Management Annals",
    "Academy of Management Journal",
    "Academy of Management Review",
    "Accounting, Organizations and Society",
    "The Accounting Review",
    "Administrative Science Quarterly",
    "American Economic Review",
    "American Sociological Review",
    "Contemporary Accounting Research",
    "Econometrica",
    "Entrepreneurship Theory and Practice",
    "Harvard Business Review",
    "Human Resource Management",
    "Information Systems Research",
    "Journal of Accounting and Economics",
    "Journal of Accounting Research",
    "Journal of Applied Psychology",
    "Journal of Business Venturing",
    "Journal of Consumer Psychology",
    "Journal of Consumer Research",
    "Journal of Finance",
    "Journal of Financial and Quantitative Analysis",
    "Journal of Financial Economics",
    "Journal of International Business Studies",
    "Journal of Management",
    "Journal of Management Information Systems",
    "Journal of Management Studies",
    "Journal of Marketing",
    "Journal of Marketing Research",
    "Journal of Operations Management",
    "Journal of Political Economy",
    "Journal of the Academy of Marketing Science",
    "Management Science",
    "Manufacturing and Service Operations Management",
    "Marketing Science",
    "MIS Quarterly",
    "Operations Research",
    "Organization Science",
    "Organizational Behavior and Human Decision Processes",
    "Production and Operations Management",
    "Quarterly Journal of Economics",
    "Research Policy",
    "Review of Accounting Studies",
    "Review of Economic Studies",
    "Review of Finance",
    "Review of Financial Studies",
    "MIT Sloan Management Review",
    "Psychological Science",
    "Strategic Entrepreneurship Journal",
    "Strategic Management Journal",
    # --- UTD24-only (not in the FT50) ---
    "INFORMS Journal on Computing",
]


def recent_years(n: int = 3, today: Optional[date] = None) -> List[int]:
    """The ``n`` completed calendar years before this one (most recent first).

    Papers need time to clear journal review, so we look at *completed* years:
    in 2026 with ``n=3`` this is ``[2025, 2024, 2023]``.
    """
    today = today or date.today()
    return [today.year - i for i in range(1, n + 1)]


def _normalize_journal(name: str) -> str:
    """Canonicalize a journal name for comparison.

    Lowercases; drops a parenthetical and any trailing citation after a comma
    (so "Journal of Finance (forthcoming)" and "Journal of Finance, 79(3)" both
    reduce to "journal of finance"); maps "&" to "and"; strips a leading "the";
    and removes remaining punctuation. Deliberately NOT a substring match, so
    "Operations Research" does not swallow "Annals of Operations Research" and
    "American Economic Review" does not swallow "American Economic Review:
    Insights".
    """
    s = (name or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\(.*?\)", " ", s)   # drop parentheticals: "(forthcoming)"
    s = s.split(",")[0]               # drop trailing citation: ", 79(3), 1-20"
    s = s.replace("&", " and ")
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def is_top_tier_journal(journal: Optional[str], top_tier: List[str]) -> bool:
    """True if ``journal`` matches a list entry after normalization (exact).

    Normalization (see ``_normalize_journal``) makes the match robust to "The"
    prefixes, "&"/"and" and casing differences, and trailing citation noise,
    while exact (rather than substring) comparison avoids false positives where
    a short top-tier name is contained in a different, non-top-tier journal.
    """
    if not journal:
        return False
    j = _normalize_journal(journal)
    if not j:
        return False
    return any(_normalize_journal(t) == j for t in top_tier if t.strip())


@dataclass
class PaperOutcome:
    """One presented paper and its resolved publication outcome."""

    title: str
    year: Optional[int]
    matched: bool                    # a confident OpenAlex match was found
    source_type: Optional[str]       # OpenAlex venue type: journal/repository/...
    journal: Optional[str]           # journal name, set only if published in one
    is_top_tier: bool


@dataclass
class PublicationAnalysis:
    """The publication outcome for one conference over the analyzed years."""

    conference: str
    years: List[int]
    outcomes: List[PaperOutcome] = field(default_factory=list)
    error: str = ""

    @property
    def total_presented(self) -> int:
        """Size of the program (the denominator)."""
        return len(self.outcomes)

    @property
    def matched_in_openalex(self) -> int:
        return sum(1 for o in self.outcomes if o.matched)

    @property
    def published_in_journal(self) -> int:
        return sum(1 for o in self.outcomes if o.journal)

    @property
    def top_tier_papers(self) -> int:
        return sum(1 for o in self.outcomes if o.is_top_tier)

    @property
    def top_tier_fraction(self) -> Optional[float]:
        """Top-tier papers / papers presented. None when the program is empty."""
        if not self.outcomes:
            return None
        return self.top_tier_papers / len(self.outcomes)

    @property
    def published_fraction(self) -> Optional[float]:
        """Papers published in any journal / papers presented."""
        if not self.outcomes:
            return None
        return self.published_in_journal / len(self.outcomes)


def analyze_program(
    papers: List[Paper],
    *,
    conference: str,
    years: List[int],
    top_tier: Optional[List[str]] = None,
    mailto: str = "",
    lookup: Callable[..., Optional[PublicationMatch]] = lookup_publication,
) -> PublicationAnalysis:
    """Resolve each program paper's publication via OpenAlex and tally the rate.

    ``lookup`` is injectable so the OpenAlex network call can be stubbed in
    tests. A paper counts as top-tier only when OpenAlex matches it to a journal
    article whose journal is on ``top_tier``.
    """
    top_tier = DEFAULT_TOP_TIER_JOURNALS if top_tier is None else top_tier
    analysis = PublicationAnalysis(conference=conference, years=list(years))
    for paper in papers:
        match = lookup(paper.title, year=paper.year, authors=paper.authors, mailto=mailto)
        journal = match.source_name if (match and match.is_journal_article) else None
        analysis.outcomes.append(
            PaperOutcome(
                title=paper.title,
                year=paper.year,
                matched=match is not None,
                source_type=match.source_type if match else None,
                journal=journal,
                is_top_tier=is_top_tier_journal(journal, top_tier),
            )
        )
    return analysis
