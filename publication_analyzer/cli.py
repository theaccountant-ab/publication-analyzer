"""Command-line interface for the publication analyzer.

    analyze ...    For each conference, report the fraction of presented papers
                   that were published in a top-tier journal.

Two ways to provide the program (the denominator):

    analyze --programs programs.csv         # authoritative: columns
                                            # conference,year,title,authors
    analyze names.txt                       # best-effort: discover the program
                                            # via Gemini + Google Search

Either way, each paper's publication is resolved authoritatively via OpenAlex.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

from .analysis import analyze_program, recent_years
from .config import Config, load_config
from .programs import (
    Paper,
    dedupe_papers,
    discover_program_via_search,
    parse_program_text,
    read_program_csv,
    set_rate_limit,
)
from .scrape import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    Fetcher,
    chunk_text,
    fetch_text,
    make_fetcher,
    read_scrape_sources,
    scrape_program,
)


def _client(config: Config):
    from google import genai

    # Install the client-side throttle so every Gemini call respects the
    # configured per-minute cap (important on the rate-limited free tier).
    set_rate_limit(config.rate_limit_rpm)
    if not config.gemini_api_key:
        # The SDK also reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
        return genai.Client()
    return genai.Client(api_key=config.gemini_api_key)


def _read_name_list(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def _append_program_csv(path: str, conference: str, papers: List[Paper]) -> None:
    """Append a conference's papers to the output CSV, writing a header if new."""
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["conference", "year", "title", "authors"])
        for p in papers:
            writer.writerow(
                [conference, p.year or "", p.title, "; ".join(p.authors)]
            )


def _progress_path(output: Optional[str]) -> Optional[str]:
    return f"{output}.progress.json" if output else None


def _load_progress(path: Optional[str]) -> Dict[str, dict]:
    """Load per-page chunk progress ``{conference|year: {done, total, complete}}``."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_progress(path: str, progress: Dict[str, dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, indent=0, sort_keys=True)


def _scrape_to_programs(
    client, model: str, path: str, *, fetch: Fetcher,
    output: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Dict[str, List[Paper]]:
    """Scrape every page in a sources CSV into ``{conference: [Paper, ...]}``.

    When ``output`` is given the scrape is *resumable at chunk granularity*: each
    parsed chunk's papers are appended to the CSV immediately and a sidecar
    ``<output>.progress.json`` records how many of a page's chunks are done. If
    the run stops (notably on the free tier's hard daily request cap), re-running
    skips finished pages and resumes a partial page mid-way — so a page larger
    than a day's quota still completes over several days instead of restarting
    (and wasting quota) every time.
    """
    from google.genai import errors

    sources = read_scrape_sources(path)
    programs: Dict[str, List[Paper]] = {}
    prog_path = _progress_path(output)
    progress = _load_progress(prog_path)
    for conference, items in sources.items():
        papers: List[Paper] = []
        for year, url in items:
            key = f"{conference}|{year if year is not None else ''}"
            state = progress.get(key, {})
            if state.get("complete"):
                print(f"  skip {conference} {year or ''}: already scraped.")
                continue
            try:
                chunks = chunk_text(
                    fetch_text(url, fetch=fetch), max_chars=max_chars
                )
            except Exception as exc:  # network / decode / PDF errors
                print(f"  ! failed to fetch {url}: {exc}")
                continue
            start = min(int(state.get("done", 0)), len(chunks))
            if start:
                print(
                    f"  resume {conference} {year or ''}: "
                    f"{start}/{len(chunks)} chunk(s) already done."
                )
            try:
                for i in range(start, len(chunks)):
                    found = parse_program_text(
                        client, model, chunks[i],
                        max_output_tokens=max_output_tokens,
                    )
                    if year is not None:
                        for p in found:
                            if p.year is None:
                                p.year = year
                    found = dedupe_papers(found)
                    papers.extend(found)
                    if output:
                        _append_program_csv(output, conference, found)
                    if prog_path:
                        progress[key] = {
                            "done": i + 1, "total": len(chunks),
                            "complete": False,
                        }
                        _save_progress(prog_path, progress)
            except errors.APIError as exc:
                # A quota / rate-limit error (notably the free tier's daily cap)
                # can't be waited out. Stop cleanly; finished chunks are saved.
                if getattr(exc, "code", None) == 429:
                    print(
                        f"\n  ! Gemini quota/rate limit reached while scraping "
                        f"{conference} {year or ''} (at chunk {i + 1}/"
                        f"{len(chunks)}). Progress saved; re-run the same command "
                        f"(e.g. tomorrow, once the daily quota resets) to resume."
                    )
                    return programs
                raise
            if prog_path:  # page fully parsed
                progress[key] = {
                    "done": len(chunks), "total": len(chunks), "complete": True,
                }
                _save_progress(prog_path, progress)
        programs[conference] = papers
        print(
            f"  scraped {conference}: {len(papers)} new paper(s) "
            f"from {len(items)} page(s)."
        )
    return programs


def cmd_scrape(config: Config, args: argparse.Namespace) -> int:
    print(f"Scraping conference programs listed in {args.file} ...")
    if os.path.exists(args.output):
        print(
            f"  (resuming: {args.output} exists; finished pages are skipped and "
            f"a partial page continues mid-way. Delete it and its .progress.json "
            f"for a fresh scrape.)"
        )
    client = _client(config)
    fetch = make_fetcher(args.render or config.render)
    programs = _scrape_to_programs(
        client, config.model, args.file, fetch=fetch, output=args.output
    )
    progress = _load_progress(_progress_path(args.output))
    complete = sum(1 for s in progress.values() if s.get("complete"))
    n = sum(len(p) for p in programs.values())
    print(
        f"\nWrote {n} newly-scraped paper(s) to {args.output} "
        f"({complete} page(s) fully scraped so far).\nReview it, then run: "
        f"publication-analyzer analyze --programs {args.output}"
    )
    return 0


def cmd_analyze(config: Config, args: argparse.Namespace) -> int:
    from google.genai import errors

    years = recent_years(args.years)
    top_tier = config.top_tier_journals or None  # None -> module default list

    # Build {conference: [Paper, ...]}. Authoritative CSV or scraped pages when
    # given, else discover each named conference's program via search.
    programs: Dict[str, List[Paper]] = {}
    client = None
    if args.programs:
        programs = read_program_csv(args.programs)
        # Optionally narrow to the conferences named in a list file.
        if args.file:
            wanted = set(_read_name_list(args.file))
            programs = {k: v for k, v in programs.items() if k in wanted}
        print(
            f"Loaded program for {len(programs)} conference(s) from "
            f"{args.programs}.\n"
        )
    elif args.scrape:
        print(f"Scraping conference programs listed in {args.scrape} ...")
        client = _client(config)
        fetch = make_fetcher(args.render or config.render)
        programs = _scrape_to_programs(
            client, config.model, args.scrape, fetch=fetch
        )
        print()
    else:
        if not args.file:
            print("error: provide a names file, or --programs CSV.", file=sys.stderr)
            return 2
        names = _read_name_list(args.file)
        print(
            f"Discovering programs for {len(names)} conference(s) across years "
            f"{', '.join(str(y) for y in years)} via search "
            "(best-effort; supply --programs for a rigorous denominator) ...\n"
        )
        client = _client(config)
        for name in names:
            try:
                programs[name] = discover_program_via_search(
                    client, config.model, name, years
                )
            except errors.APIError as exc:
                print(f"  ! {name}: program discovery failed: {exc}")
                programs[name] = []

    analyses = []
    tot_presented = tot_top = tot_pub = 0
    for conference, papers in programs.items():
        analysis = analyze_program(
            papers, conference=conference, years=years,
            top_tier=top_tier, mailto=config.mailto,
        )
        analyses.append(analysis)
        frac = analysis.top_tier_fraction
        frac_str = "n/a (empty program)" if frac is None else f"{frac:.1%}"
        print(
            f"  {conference}: {analysis.top_tier_papers}/"
            f"{analysis.total_presented} presented papers reached a top-tier "
            f"journal ({frac_str}); {analysis.published_in_journal} published in "
            f"any journal; {analysis.matched_in_openalex} matched in OpenAlex."
        )
        tot_presented += analysis.total_presented
        tot_top += analysis.top_tier_papers
        tot_pub += analysis.published_in_journal

    overall = (tot_top / tot_presented) if tot_presented else None
    overall_str = "n/a" if overall is None else f"{overall:.1%}"
    print(
        f"\nOverall: {tot_top}/{tot_presented} presented papers across all "
        f"conferences reached a top-tier journal ({overall_str}); "
        f"{tot_pub} published in any journal."
    )
    if not args.programs and not args.scrape:
        print(
            "Note: programs were discovered via search and are likely "
            "incomplete, so the denominator understates reality. Supply "
            "--programs or --scrape for a rigorous rate. Publication outcomes "
            "are from OpenAlex."
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["conference", "years", "total_presented", "matched_in_openalex",
                 "published_in_journal", "top_tier_papers", "top_tier_fraction",
                 "error"]
            )
            for a in analyses:
                frac = a.top_tier_fraction
                writer.writerow([
                    a.conference,
                    " ".join(str(y) for y in a.years),
                    a.total_presented,
                    a.matched_in_openalex,
                    a.published_in_journal,
                    a.top_tier_papers,
                    "" if frac is None else f"{frac:.4f}",
                    a.error,
                ])
        print(f"\nWrote per-conference results to {args.output}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="publication-analyzer",
        description=(
            "Report the fraction of a conference's presented papers that were "
            "later published in a top-tier journal (FT50 ∪ UTD24 by default)."
        ),
    )
    parser.add_argument(
        "-c", "--config", default=None, help="Path to a YAML config file."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "analyze",
        help="Report each conference's top-tier journal publication rate.",
    )
    p.add_argument(
        "file", nargs="?", default=None,
        help="Text file with one conference name per line. Required unless "
        "--programs is given (then it optionally narrows which conferences).",
    )
    p.add_argument(
        "--programs", default=None,
        help="Authoritative program CSV (columns: conference, year, title, "
        "authors). When given, this is the denominator instead of search.",
    )
    p.add_argument(
        "--scrape", default=None,
        help="Sources CSV (columns: conference, year, url) of program pages to "
        "scrape for the denominator instead of search.",
    )
    p.add_argument(
        "--years", type=int, default=3,
        help="Number of completed years to look back over (default: 3). "
        "Used for search-based discovery; ignored when --programs supplies years.",
    )
    p.add_argument(
        "--render", choices=("auto", "always", "never"), default=None,
        help="How to fetch pages with --scrape: 'auto' (default) renders via a "
        "headless browser only when a page looks JS-rendered, 'always' renders "
        "every page, 'never' uses plain HTTP. Needs the optional 'playwright'.",
    )
    p.add_argument(
        "--output", default=None,
        help="Optional CSV path to write per-conference results to.",
    )

    s = sub.add_parser(
        "scrape",
        help="Scrape conference program pages into a reusable program CSV.",
    )
    s.add_argument(
        "file",
        help="Sources CSV with columns: conference, url, and optional year "
        "(one row per program page).",
    )
    s.add_argument(
        "-o", "--output", default="programs.csv",
        help="Program CSV to write (default: programs.csv).",
    )
    s.add_argument(
        "--render", choices=("auto", "always", "never"), default=None,
        help="How to fetch pages: 'auto' (default) renders via a headless "
        "browser only when a page looks JS-rendered, 'always' renders every "
        "page, 'never' uses plain HTTP. Needs the optional 'playwright'.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    handlers = {"analyze": cmd_analyze, "scrape": cmd_scrape}
    handler = handlers[args.command]
    return handler(config, args)


if __name__ == "__main__":
    sys.exit(main())
