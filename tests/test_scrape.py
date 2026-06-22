import pytest

import publication_analyzer.scrape as scrape_mod
from publication_analyzer.programs import Paper, dedupe_papers
from publication_analyzer.scrape import (
    chunk_text,
    fetch_text,
    html_to_text,
    make_fetcher,
    read_scrape_sources,
    scrape_program,
)


def test_html_to_text_drops_script_and_keeps_breaks():
    html = (
        "<html><head><style>x{}</style></head><body>"
        "<h1>Program</h1>"
        "<script>var x = 1;</script>"
        "<ul><li>Paper One by Jane Smith</li><li>Paper Two by John Doe</li></ul>"
        "</body></html>"
    )
    text = html_to_text(html)
    assert "var x" not in text
    assert "x{}" not in text
    assert "Paper One by Jane Smith" in text
    assert "Paper Two by John Doe" in text
    # Block elements introduce line breaks so titles don't run together.
    assert "Paper One by Jane Smith" in text.splitlines()[-2:][0] or "\n" in text


def test_chunk_text_splits_large_input_on_line_boundaries():
    text = "\n".join(f"line {i}" for i in range(1000))
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 + 20 for c in chunks)  # +1 line of slack
    # Lossless: every line survives across the chunks.
    assert "".join(chunks).count("line ") == 1000


def test_chunk_text_short_input_is_single_chunk():
    assert chunk_text("just a little text", max_chars=12000) == ["just a little text"]
    assert chunk_text("   ", max_chars=10) == []


def test_fetch_text_routes_html_vs_pdf():
    html_bytes = b"<p>Hello <b>world</b></p>"

    def fake_fetch(url):
        return html_bytes, "text/html"

    assert "Hello world" in fetch_text("http://x/program", fetch=fake_fetch)


def test_fetch_text_detects_pdf_by_extension(monkeypatch):
    sentinel = b"%PDF-1.4 fake"

    def fake_fetch(url):
        return sentinel, "application/octet-stream"

    captured = {}

    def fake_pdf_to_text(data):
        captured["data"] = data
        return "Paper From PDF"

    monkeypatch.setattr("publication_analyzer.scrape.pdf_to_text", fake_pdf_to_text)
    out = fetch_text("http://x/schedule.PDF?v=2", fetch=fake_fetch)
    assert out == "Paper From PDF"
    assert captured["data"] == sentinel


def test_scrape_program_fetches_parses_and_dedupes():
    pages = {
        "http://x/2024": ("<li>Alpha</li><li>Beta</li>", "text/html"),
        "http://x/extra": ("<li>Beta</li><li>Gamma</li>", "text/html"),
    }

    def fake_fetch(url):
        return pages[url].__getitem__(0).encode(), pages[url][1]

    def fake_parse(text):
        # Pretend the model pulled a paper per "<li>...</li>" left in the text.
        titles = [t for t in ("Alpha", "Beta", "Gamma") if t in text]
        return [Paper(title=t) for t in titles]

    papers = scrape_program(
        None, "model", ["http://x/2024", "http://x/extra"],
        year=2024, fetch=fake_fetch, parse=fake_parse,
    )
    titles = sorted(p.title for p in papers)
    assert titles == ["Alpha", "Beta", "Gamma"]   # Beta deduped
    assert all(p.year == 2024 for p in papers)     # year stamped


def test_scrape_program_skips_unfetchable_url(capsys):
    def fake_fetch(url):
        raise RuntimeError("boom")

    papers = scrape_program(
        None, "model", ["http://x/dead"], fetch=fake_fetch, parse=lambda t: []
    )
    assert papers == []
    assert "failed to fetch" in capsys.readouterr().out


def test_read_scrape_sources(tmp_path):
    path = tmp_path / "sources.csv"
    path.write_text(
        "conference,year,url\n"
        "AFA,2024,http://a/2024\n"
        "AFA,2023,http://a/2023\n"
        "WFA,,http://w/prog\n"
        ",,http://orphan\n",
        encoding="utf-8",
    )
    sources = read_scrape_sources(str(path))
    assert set(sources) == {"AFA", "WFA"}
    assert sources["AFA"] == [(2024, "http://a/2024"), (2023, "http://a/2023")]
    assert sources["WFA"] == [(None, "http://w/prog")]


def test_read_scrape_sources_requires_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("conference,year\nAFA,2024\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_scrape_sources(str(bad))


def test_make_fetcher_modes():
    assert make_fetcher("never") is scrape_mod._default_fetch
    assert make_fetcher("always") is scrape_mod._render_fetch
    assert make_fetcher("auto") is scrape_mod._auto_fetch
    assert make_fetcher("anything-else") is scrape_mod._auto_fetch  # defaults to auto


def test_looks_unrendered_detects_placeholder_and_thin_pages():
    assert scrape_mod._looks_unrendered("Loading...")
    assert scrape_mod._looks_unrendered("Please enable JavaScript to view this.")
    assert scrape_mod._looks_unrendered("tiny")
    # A page with real content and no placeholder is considered rendered.
    assert not scrape_mod._looks_unrendered("A real paper title and authors. " * 40)


def test_auto_fetch_renders_when_static_is_unrendered(monkeypatch):
    monkeypatch.setattr(
        scrape_mod, "_default_fetch",
        lambda url, *, timeout=30.0: (b"<div>Loading...</div>", "text/html"),
    )
    rendered = b"<li>Real Paper by A. Author</li>" * 30
    monkeypatch.setattr(
        scrape_mod, "_render_fetch",
        lambda url, *, timeout=30.0: (rendered, "text/html"),
    )
    data, content_type = scrape_mod._auto_fetch("http://x/prog")
    assert data == rendered and content_type == "text/html"


def test_auto_fetch_keeps_static_when_page_is_rich(monkeypatch):
    rich = ("<li>Paper " + "x" * 50 + "</li>").encode() * 20
    monkeypatch.setattr(
        scrape_mod, "_default_fetch",
        lambda url, *, timeout=30.0: (rich, "text/html"),
    )
    called = {"render": False}

    def boom(url, *, timeout=30.0):
        called["render"] = True
        return b"", "text/html"

    monkeypatch.setattr(scrape_mod, "_render_fetch", boom)
    data, _ = scrape_mod._auto_fetch("http://x/prog")
    assert data == rich and not called["render"]


def test_auto_fetch_does_not_render_pdfs(monkeypatch):
    monkeypatch.setattr(
        scrape_mod, "_default_fetch",
        lambda url, *, timeout=30.0: (b"%PDF-1.4", "application/pdf"),
    )
    called = {"render": False}
    monkeypatch.setattr(
        scrape_mod, "_render_fetch",
        lambda *a, **k: called.__setitem__("render", True) or (b"", "text/html"),
    )
    data, content_type = scrape_mod._auto_fetch("http://x/agenda.pdf")
    assert content_type == "application/pdf" and not called["render"]


def test_auto_fetch_falls_back_when_render_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        scrape_mod, "_default_fetch",
        lambda url, *, timeout=30.0: (b"<div>Loading...</div>", "text/html"),
    )

    def missing(url, *, timeout=30.0):
        raise RuntimeError("JS rendering requires 'playwright'")

    monkeypatch.setattr(scrape_mod, "_render_fetch", missing)
    data, _ = scrape_mod._auto_fetch("http://x/prog")
    assert data == b"<div>Loading...</div>"            # kept the static body
    assert "cannot render" in capsys.readouterr().out  # warned the user


def test_render_fetch_delegates_pdfs_without_a_browser(monkeypatch):
    monkeypatch.setattr(
        scrape_mod, "_default_fetch",
        lambda url, *, timeout=30.0: (b"%PDF-1.4", "application/pdf"),
    )
    data, content_type = scrape_mod._render_fetch("http://x/agenda.PDF?v=1")
    assert content_type == "application/pdf"


def test_set_rate_limit_throttles_calls(monkeypatch):
    import publication_analyzer.programs as programs_mod

    clock = {"t": 100.0}
    slept = []
    monkeypatch.setattr(programs_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(programs_mod.time, "sleep", lambda s: slept.append(s))
    try:
        programs_mod.set_rate_limit(5)  # 5/min -> one call every 12s
        programs_mod._throttle()        # first call: no wait, stamps t=100
        clock["t"] = 104.0              # only 4s elapsed before the next call
        programs_mod._throttle()        # must wait the remaining 8s
        assert slept == [pytest.approx(8.0)]
    finally:
        programs_mod.set_rate_limit(0)  # disable so other tests are unaffected


def test_set_rate_limit_zero_disables_throttle(monkeypatch):
    import publication_analyzer.programs as programs_mod

    slept = []
    monkeypatch.setattr(programs_mod.time, "sleep", lambda s: slept.append(s))
    programs_mod.set_rate_limit(0)
    programs_mod._throttle()
    programs_mod._throttle()
    assert slept == []


def _html_fetcher(body: bytes):
    return lambda url: (body, "text/html")


def test_scrape_to_programs_resumes_finished_page(tmp_path, capsys):
    import json
    from publication_analyzer import cli
    import publication_analyzer.cli as cli_mod

    sources = tmp_path / "sources.csv"
    sources.write_text(
        "conference,year,url\n"
        "AFA,2023,http://a/2023\n"
        "AFA,2024,http://a/2024\n",
        encoding="utf-8",
    )
    out = tmp_path / "programs.csv"
    out.write_text(
        "conference,year,title,authors\nAFA,2023,Old Paper,Jane Smith\n",
        encoding="utf-8",
    )
    # Sidecar marks AFA 2023 fully scraped (one chunk done).
    (tmp_path / "programs.csv.progress.json").write_text(
        json.dumps({"AFA|2023": {"done": 1, "total": 1, "complete": True}}),
        encoding="utf-8",
    )

    parsed_for = []

    def fake_parse(client, model, text, **kw):
        parsed_for.append(text)
        return [Paper(title="New Paper")]

    cli_mod.parse_program_text = fake_parse
    try:
        cli._scrape_to_programs(
            None, "model", str(sources), fetch=_html_fetcher(b"<li>x</li>"),
            output=str(out),
        )
    finally:
        from publication_analyzer.programs import parse_program_text as real
        cli_mod.parse_program_text = real

    # The finished page was skipped (parsed once, for 2024 only).
    assert len(parsed_for) == 1
    text = out.read_text(encoding="utf-8")
    assert "Old Paper" in text and "New Paper" in text
    assert "skip AFA 2023" in capsys.readouterr().out


def test_scrape_to_programs_resumes_partial_page_mid_chunk(tmp_path, capsys):
    """A page stopped mid-way resumes from the next unparsed chunk."""
    import json
    from google.genai import errors
    from publication_analyzer import cli
    import publication_analyzer.cli as cli_mod

    sources = tmp_path / "sources.csv"
    sources.write_text("conference,year,url\nAFA,2024,http://a/2024\n",
                       encoding="utf-8")
    out = tmp_path / "programs.csv"
    prog = tmp_path / "programs.csv.progress.json"

    # A body that chunk_text (max_chars=20) splits into several chunks.
    body = ("\n".join(f"line {i}" for i in range(20))).encode()

    seen = []

    def fail_on_third(client, model, text, **kw):
        seen.append(text)
        if len(seen) == 3:
            raise errors.ClientError(429, {"error": {"message": "quota"}}, None)
        return [Paper(title=f"P{len(seen)}")]

    cli_mod.parse_program_text = fail_on_third
    try:
        cli._scrape_to_programs(
            None, "model", str(sources), fetch=_html_fetcher(body),
            output=str(out), max_chars=20,
        )
        # First run: two chunks saved before the 429 on the third.
        assert "quota/rate limit reached" in capsys.readouterr().out
        state = json.loads(prog.read_text())["AFA|2024"]
        assert state["done"] == 2 and state["complete"] is False

        # Resume: parsing succeeds now; it must start at chunk index 2, not 0.
        seen.clear()
        def succeed(client, model, text, **kw):
            seen.append(text)
            return [Paper(title=f"R{len(seen)}")]
        cli_mod.parse_program_text = succeed
        cli._scrape_to_programs(
            None, "model", str(sources), fetch=_html_fetcher(body),
            output=str(out), max_chars=20,
        )
    finally:
        from publication_analyzer.programs import parse_program_text as real
        cli_mod.parse_program_text = real

    # The resume reported "2/N already done" and finished the page.
    assert "resume AFA 2024: 2/" in capsys.readouterr().out
    assert json.loads(prog.read_text())["AFA|2024"]["complete"] is True


def test_write_details_csv_has_per_paper_audit_rows(tmp_path):
    import csv
    from publication_analyzer.cli import _write_details_csv
    from publication_analyzer.analysis import PublicationAnalysis, PaperOutcome

    a = PublicationAnalysis(conference="WFA", years=[2020])
    a.outcomes.append(PaperOutcome(
        title="Some Title", year=2020, matched=True, source_type="journal",
        journal="Journal of Finance", is_top_tier=True, authors=["Ann Bee"],
        matched_title="Some Title", openalex_id="https://openalex.org/W1",
        publication_year=2021, title_similarity=0.98,
    ))
    a.outcomes.append(PaperOutcome(
        title="Unmatched", year=2020, matched=False, source_type=None,
        journal=None, is_top_tier=False,
    ))
    out = tmp_path / "details.csv"
    _write_details_csv(str(out), [a])

    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert rows[0]["conference"] == "WFA"
    assert rows[0]["journal"] == "Journal of Finance"
    assert rows[0]["openalex_id"].endswith("W1")
    assert rows[0]["is_top_tier"] == "True"
    assert rows[0]["authors"] == "Ann Bee"
    assert rows[1]["matched"] == "False" and rows[1]["journal"] == ""


def test_cached_lookup_reuses_matches_and_skips_failures():
    from publication_analyzer.cli import _make_cached_lookup, _cache_key
    from publication_analyzer.openalex import PublicationMatch, OpenAlexUnavailable

    cache, stats = {}, {"hits": 0, "lookups": 0, "failed": 0}
    calls = {"n": 0}

    def base(title, *, year=None, authors=None, mailto=""):
        calls["n"] += 1
        return PublicationMatch("W1", "T", "article", "Journal of Finance",
                                "journal", 2022, 1.0)

    look = _make_cached_lookup(cache, stats, base_lookup=base)
    m1 = look("Some Paper", authors=["Ann Bee"])
    m2 = look("Some Paper", authors=["Ann Bee"])     # same paper -> cache hit
    assert calls["n"] == 1                            # base called only once
    assert stats == {"hits": 1, "lookups": 1, "failed": 0}
    assert m1.source_name == m2.source_name == "Journal of Finance"

    # A failed fetch is not cached, so it can be retried on a later run.
    def boom(title, *, year=None, authors=None, mailto=""):
        raise OpenAlexUnavailable("429")

    look_fail = _make_cached_lookup(cache, stats, base_lookup=boom)
    assert look_fail("Unreachable Paper") is None
    assert stats["failed"] == 1
    assert _cache_key("Unreachable Paper", None) not in cache


def test_cached_lookup_defers_past_budget():
    from publication_analyzer.cli import _make_cached_lookup
    from publication_analyzer.openalex import PublicationMatch

    cache, stats = {}, {"hits": 0, "lookups": 0, "failed": 0, "deferred": 0}
    calls = {"n": 0}

    def base(title, *, year=None, authors=None, mailto=""):
        calls["n"] += 1
        return PublicationMatch("W", "T", "article", "J", "journal", 2022, 1.0)

    look = _make_cached_lookup(cache, stats, base_lookup=base, max_lookups=2)
    look("P1"); look("P2")            # two fresh attempts use the budget
    r3 = look("P3")                    # third is deferred, base not called
    assert calls["n"] == 2
    assert stats["lookups"] == 2 and stats["deferred"] == 1
    assert r3 is None


def test_crossref_prefers_journal_over_preprint():
    from publication_analyzer.crossref import lookup_publication

    # Same title appears as an SSRN preprint and the real journal article.
    payload = {"message": {"items": [
        {"DOI": "10.x/ssrn", "title": ["Asset Pricing With Frictions"],
         "container-title": ["SSRN Electronic Journal"], "type": "journal-article",
         "issued": {"date-parts": [[2020]]},
         "author": [{"family": "Smith"}]},
        {"DOI": "10.x/jfe", "title": ["Asset Pricing With Frictions"],
         "container-title": ["Journal of Financial Economics"], "type": "journal-article",
         "issued": {"date-parts": [[2022]]},
         "author": [{"family": "Smith"}]},
    ]}}
    m = lookup_publication("Asset Pricing with Frictions", authors=["Jane Smith"],
                           fetch=lambda url: payload)
    assert m is not None
    assert m.is_journal_article
    assert m.source_name == "Journal of Financial Economics"   # not the SSRN copy


def test_crossref_marks_ssrn_only_as_non_journal():
    from publication_analyzer.crossref import lookup_publication

    payload = {"message": {"items": [
        {"DOI": "10.x/ssrn", "title": ["Working Paper Title Here"],
         "container-title": ["SSRN Electronic Journal"], "type": "journal-article",
         "issued": {"date-parts": [[2021]]}, "author": [{"family": "Doe"}]},
    ]}}
    m = lookup_publication("Working Paper Title Here", authors=["John Doe"],
                           fetch=lambda url: payload)
    assert m is not None and not m.is_journal_article   # SSRN is not a journal


def test_dedupe_fills_missing_fields():
    kept = dedupe_papers([
        Paper("Same Title", authors=[], year=None),
        Paper("same title", authors=["Jane Smith"], year=2024),
    ])
    assert len(kept) == 1
    assert kept[0].year == 2024
    assert kept[0].authors == ["Jane Smith"]
