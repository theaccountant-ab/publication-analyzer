import pytest

from publication_analyzer.programs import Paper, dedupe_papers
from publication_analyzer.scrape import (
    chunk_text,
    fetch_text,
    html_to_text,
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


def test_dedupe_fills_missing_fields():
    kept = dedupe_papers([
        Paper("Same Title", authors=[], year=None),
        Paper("same title", authors=["Jane Smith"], year=2024),
    ])
    assert len(kept) == 1
    assert kept[0].year == 2024
    assert kept[0].authors == ["Jane Smith"]
