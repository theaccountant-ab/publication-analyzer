from datetime import date

import pytest

from publication_analyzer.analysis import (
    DEFAULT_TOP_TIER_JOURNALS,
    PublicationAnalysis,
    analyze_program,
    is_top_tier_journal,
    recent_years,
)
from publication_analyzer.config import load_config
from publication_analyzer.openalex import PublicationMatch, lookup_publication
from publication_analyzer.programs import Paper, read_program_csv


# --------------------------------------------------------------------------- #
# Year selection
# --------------------------------------------------------------------------- #
def test_recent_years_are_completed_years_most_recent_first():
    assert recent_years(3, today=date(2026, 6, 19)) == [2025, 2024, 2023]
    assert recent_years(1, today=date(2026, 1, 1)) == [2025]


# --------------------------------------------------------------------------- #
# Top-tier journal matching
# --------------------------------------------------------------------------- #
def test_is_top_tier_journal_normalizes_the_case_and_noise():
    tier = ["Journal of Finance", "American Economic Review",
            "Manufacturing and Service Operations Management"]
    assert is_top_tier_journal("The Journal of Finance", tier)
    assert is_top_tier_journal("AMERICAN ECONOMIC REVIEW", tier)
    assert is_top_tier_journal("Manufacturing & Service Operations Management", tier)
    assert is_top_tier_journal("Journal of Finance (forthcoming)", tier)
    assert is_top_tier_journal("Journal of Finance, 79(3), 1-40", tier)
    assert not is_top_tier_journal("Journal of Banking & Finance", tier)
    assert not is_top_tier_journal(None, tier)
    assert not is_top_tier_journal("", tier)


def test_is_top_tier_journal_is_exact_not_substring():
    assert not is_top_tier_journal(
        "Annals of Operations Research", ["Operations Research"]
    )
    assert not is_top_tier_journal(
        "American Economic Review: Insights", ["American Economic Review"]
    )
    assert is_top_tier_journal("Operations Research", ["Operations Research"])


def test_default_list_is_ft50_union_utd24():
    assert len(DEFAULT_TOP_TIER_JOURNALS) == 51
    assert len(set(DEFAULT_TOP_TIER_JOURNALS)) == 51
    assert is_top_tier_journal("INFORMS Journal on Computing", DEFAULT_TOP_TIER_JOURNALS)
    assert is_top_tier_journal("The Journal of Finance", DEFAULT_TOP_TIER_JOURNALS)


# --------------------------------------------------------------------------- #
# OpenAlex matching (network call stubbed via `fetch`)
# --------------------------------------------------------------------------- #
def _work(title, source_name, source_type, year=2024, authors=None):
    return {
        "id": "https://openalex.org/W1",
        "title": title,
        "type": "article",
        "publication_year": year,
        "primary_location": {"source": {"display_name": source_name, "type": source_type}},
        "authorships": [{"author": {"display_name": a}} for a in (authors or [])],
    }


def test_lookup_matches_journal_article():
    payload = {"results": [_work("Asset Pricing with Frictions", "Journal of Finance", "journal")]}
    match = lookup_publication(
        "Asset Pricing with Frictions", year=2024, fetch=lambda url: payload
    )
    assert match is not None
    assert match.is_journal_article
    assert match.source_name == "Journal of Finance"


def test_lookup_rejects_title_mismatch():
    payload = {"results": [_work("A Completely Different Paper", "Journal of Finance", "journal")]}
    assert lookup_publication("Asset Pricing with Frictions", fetch=lambda url: payload) is None


def test_lookup_working_paper_is_not_a_journal_article():
    payload = {"results": [_work("Asset Pricing with Frictions", "SSRN", "repository")]}
    match = lookup_publication("Asset Pricing with Frictions", fetch=lambda url: payload)
    assert match is not None
    assert not match.is_journal_article


def test_lookup_uses_authors_to_disambiguate():
    payload = {"results": [
        _work("Common Title", "Journal of Banking", "journal", authors=["Alice Brown"]),
        _work("Common Title", "Journal of Finance", "journal", authors=["John Smith"]),
    ]}
    match = lookup_publication(
        "Common Title", authors=["J. Smith"], fetch=lambda url: payload
    )
    assert match is not None
    assert match.source_name == "Journal of Finance"


def test_lookup_returns_none_on_fetch_error():
    def boom(url):
        raise RuntimeError("network down")

    assert lookup_publication("Anything", fetch=boom, max_retries=0) is None


# --------------------------------------------------------------------------- #
# Program CSV parsing
# --------------------------------------------------------------------------- #
def test_read_program_csv(tmp_path):
    csv_path = tmp_path / "programs.csv"
    csv_path.write_text(
        "conference,year,title,authors\n"
        "AFA,2024,Paper One,Jane Smith; John Doe\n"
        "AFA,2024,Paper Two,\n"
        "WFA,2023,Paper Three,A. Lee\n"
        ",,skipme,\n",  # missing conference+title -> skipped
        encoding="utf-8",
    )
    programs = read_program_csv(str(csv_path))
    assert set(programs) == {"AFA", "WFA"}
    assert len(programs["AFA"]) == 2
    assert programs["AFA"][0].authors == ["Jane Smith", "John Doe"]
    assert programs["WFA"][0].year == 2023


def test_read_program_csv_requires_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("name,year\nAFA,2024\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_program_csv(str(bad))


# --------------------------------------------------------------------------- #
# End-to-end analysis over a program (OpenAlex stubbed)
# --------------------------------------------------------------------------- #
def test_analyze_program_counts_and_fraction():
    papers = [
        Paper("Top paper", year=2024),
        Paper("Other journal paper", year=2024),
        Paper("Working paper only", year=2024),
        Paper("Never found", year=2024),
    ]

    def fake_lookup(title, *, year=None, authors=None, mailto=""):
        table = {
            "Top paper": PublicationMatch("W1", title, "article", "Journal of Finance", "journal", 2025, 1.0),
            "Other journal paper": PublicationMatch("W2", title, "article", "Journal of Banking & Finance", "journal", 2025, 1.0),
            "Working paper only": PublicationMatch("W3", title, "article", "SSRN Electronic Journal", "repository", 2024, 1.0),
        }
        return table.get(title)

    analysis = analyze_program(
        papers, conference="Test Conf", years=[2024], lookup=fake_lookup
    )
    assert analysis.total_presented == 4
    assert analysis.matched_in_openalex == 3
    assert analysis.published_in_journal == 2      # JoF + JBF (both journals)
    assert analysis.top_tier_papers == 1           # only JoF is FT50/UTD24
    assert analysis.top_tier_fraction == 0.25      # 1 of 4 presented


def test_fraction_is_none_for_empty_program():
    analysis = PublicationAnalysis(conference="Empty", years=[2025])
    assert analysis.total_presented == 0
    assert analysis.top_tier_fraction is None


def test_reported_years_come_from_program_when_known():
    # A program supplies its own years; the reported years should reflect those,
    # not the search look-back window passed in.
    papers = [Paper("P1", year=2014), Paper("P2", year=2018), Paper("P3", year=2018)]
    analysis = analyze_program(
        papers, conference="C", years=[2025, 2024, 2023],
        lookup=lambda *a, **k: None,
    )
    assert analysis.years == [2018, 2014]   # distinct program years, most recent first


def test_reported_years_fall_back_to_window_when_program_has_none():
    papers = [Paper("P1"), Paper("P2")]     # no years (e.g. search discovery)
    analysis = analyze_program(
        papers, conference="C", years=[2025, 2024, 2023],
        lookup=lambda *a, **k: None,
    )
    assert analysis.years == [2025, 2024, 2023]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_defaults_without_file():
    config = load_config(None)
    assert config.model == "gemini-2.5-flash"
    assert config.top_tier_journals == []
    assert config.mailto == ""
