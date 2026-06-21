# Publication Analyzer

A tool that answers one question: **of the papers presented at a conference over
the past few years, what fraction were later published in a top-tier journal?**
It's a rough quality signal for a conference.

It computes the rate over a conference's **program** (the papers presented — the
denominator) by resolving each paper's publication in **[OpenAlex](https://openalex.org)**
(free, no API key) and checking the journal against a top-tier list (the
**Financial Times FT50 ∪ UT Dallas UTD24** by default).

## Install

```bash
pip install -r requirements.txt
```

## How it works

```
program (papers presented)  ─►  match each paper in OpenAlex  ─►  journal in FT50/UTD24?  ─►  fraction
```

1. **Get the program** — the list of papers presented, per year. This is the
   denominator, and how complete it is determines how rigorous the rate is. Two
   sources are supported (see below).
2. **Resolve each paper in OpenAlex** by title (plus authors/year when known),
   conservatively: a candidate is accepted only on a close title match and, when
   authors are known, an author-surname overlap. Only works hosted in a *journal*
   count as published — working papers on SSRN/NBER/RePEc do not.
3. **Classify the journal** against the top-tier list and report the fraction of
   *presented* papers that reached a top-tier journal.

### Where the program comes from

The denominator is the part with no universal data source, so there are three
ways to supply it (in decreasing order of rigor):

- **Authoritative program CSV.** You supply the program as a CSV with columns
  `conference, year, title, authors` (authors separated by `;`). Exact.
- **Scrape the program pages (recommended).** Point the tool at each conference's
  own program page (HTML or PDF) in a sources CSV (`conference, year, url`); it
  fetches the page, converts it to text, and uses Gemini to extract the paper
  list. Because it reads the conference's *actual* program, the denominator is as
  complete as that page — and it's far less manual than hand-building the CSV.
- **Best-effort search discovery (indicative).** With neither of the above, the
  tool asks Gemini — grounded in Google Search — to list the program. A search
  can't reliably enumerate a full program, so this **under-counts the
  denominator**; treat it as indicative.

In all three, OpenAlex resolves publications authoritatively.

### Scraping programs

```bash
# 1. Scrape program pages into a reusable program CSV (review/edit it after).
#    sources.csv columns: conference, url, and optional year
python -m publication_analyzer scrape sources.csv -o programs.csv

# 2. Analyze the scraped program.
python -m publication_analyzer analyze --programs programs.csv --output rates.csv

# Or do both in one shot:
python -m publication_analyzer analyze --scrape sources.csv --output rates.csv
```

The two-step form is recommended: scraping is the lossy part, so writing
`programs.csv` first lets you eyeball and fix it before computing the rate. PDF
programs need the optional `pypdf` dependency (in `requirements.txt`); HTML needs
nothing extra.

#### Rate limits and resuming (free tier)

Each program page is parsed by Gemini, and large programs are split into several
chunks — so a multi-conference scrape makes many API calls. The free Gemini tier
allows only ~5 requests/minute, which a burst will blow through (you'll see HTTP
429s and the run aborts). Two settings make a free-tier scrape practical:

- **Throttle** with `rate_limit_rpm` (config) or `PA_RATE_LIMIT_RPM` (env) to cap
  calls per minute — e.g. `4` to stay under the free tier's 5/min.
- **Resume** (per chunk): `scrape` appends papers to the output CSV as each chunk
  is parsed and records progress in a sidecar `<output>.progress.json`. If a run
  stops (rate limit, network), just re-run the same command: finished pages are
  skipped and a partially-parsed page continues from its next chunk — so a page
  bigger than a day's quota still completes over several runs instead of
  restarting and wasting calls. Delete the output CSV *and* its `.progress.json`
  for a fresh scrape.

On the free tier's hard daily cap (~20 requests/day for `gemini-2.5-flash`), a
large source list naturally fills in over several days of repeated runs;
`scripts/daily_update.sh` runs one such pass and commits the new rows.

```bash
PA_RATE_LIMIT_RPM=4 python -m publication_analyzer scrape sources.csv -o programs.csv
# ... if it stops on a rate limit, re-run the exact same line to resume:
PA_RATE_LIMIT_RPM=4 python -m publication_analyzer scrape sources.csv -o programs.csv
```

#### JavaScript-rendered program pages

Many conference sites build the program list with client-side JavaScript, so a
plain fetch sees only a `Loading…` shell. To handle these, the scraper can render
the page in a headless browser first. It's an optional dependency:

```bash
pip install -r requirements-render.txt
playwright install chromium   # one-time browser download
```

Rendering is controlled by `--render` (on both `scrape` and `analyze --scrape`),
or the `render:` config key / `PA_RENDER` env var:

- `auto` *(default)* — fetch statically, and re-fetch through the browser only
  when the page looks unrendered (a `Loading…`/"enable JavaScript" placeholder or
  suspiciously little text). If `playwright` isn't installed, it prints a hint and
  falls back to the static fetch.
- `always` — render every page (slower; use when `auto` guesses wrong).
- `never` — plain HTTP only (the original behavior).

```bash
python -m publication_analyzer scrape sources.csv -o programs.csv --render auto
```

PDFs are never rendered (there's nothing to run), so mixing PDF and JS pages in
one `sources.csv` is fine.

## Use

```bash
# Scrape program pages into a program CSV (see "Scraping programs" above).
python -m publication_analyzer scrape sources.csv -o programs.csv

# Rigorous: supply the program. Columns: conference,year,title,authors
python -m publication_analyzer analyze --programs programs.csv --output rates.csv

# Narrow a big program CSV to just the conferences named in a file:
python -m publication_analyzer analyze names.txt --programs programs.csv

# Best-effort: discover programs via search (one conference name per line).
python -m publication_analyzer analyze names.txt --years 3 --output rates.csv
```

It prints, per conference, how many presented papers reached a top-tier journal,
how many were published in any journal, and how many matched in OpenAlex — plus
an overall line. With `--output` it writes a per-conference CSV
(`total_presented`, `matched_in_openalex`, `published_in_journal`,
`top_tier_papers`, `top_tier_fraction`).

With `--details PATH` it also writes a **per-paper audit trail** — one row per
presented paper showing exactly what it matched in OpenAlex, so results can be
checked by hand: `conference, year, title, authors, matched, title_similarity,
matched_title, openalex_id, source_type, journal, publication_year,
is_top_tier`. Open an `openalex_id` URL to see the matched work directly.

```bash
python -m publication_analyzer analyze --programs programs.csv \
  --output rates.csv --details paper_details.csv
```

## Configure

A Gemini API key is needed for **scraping** and **search-based discovery** (both
use Gemini to extract papers) but **not** for the `--programs` path. Set it in
the environment (`GEMINI_API_KEY` / `GOOGLE_API_KEY`); get a free key at
[aistudio.google.com](https://aistudio.google.com/app/apikey). Other settings go
in a YAML file passed with `-c` (copy `config.example.yaml` to `config.yaml`):

```yaml
model: gemini-2.5-flash

# Contact email for OpenAlex's faster "polite pool" (optional). Also PA_MAILTO.
mailto: you@example.com

# How to fetch program pages when scraping: auto (default) | always | never.
# "auto" renders JS-heavy pages in a headless browser only when needed. Also
# PA_RENDER or --render. See "JavaScript-rendered program pages" above.
render: auto

# Journals counted as "top-tier". Names are matched after normalization
# (case-insensitive; "The" prefix, "&"/"and", and trailing citation noise
# ignored) using an EXACT match, so "Operations Research" will not wrongly match
# "Annals of Operations Research". Omit this key to use the built-in default.
top_tier_journals:
  - Journal of Finance
  - Journal of Financial Economics
  - Review of Financial Studies
  - American Economic Review
  - Econometrica
```

**Default list.** When `top_tier_journals` is omitted, the built-in default is
the **Financial Times FT50** (2026 refresh) unioned with the **UT Dallas UTD24**
— 51 journals (UTD24 is a near-subset of the FT50; the only addition is *INFORMS
Journal on Computing*).

> **What's rigorous and what isn't.** The publication side (OpenAlex) is
> authoritative. The denominator is only as complete as the program you give it:
> the `--programs` CSV yields a rigorous rate; search-based discovery yields an
> indicative one that understates the denominator. For finance/economics/business
> conferences there is no machine-readable program API, so the CSV is the
> reliable path.

## Develop

```bash
pip install -r requirements-dev.txt
pytest
```

Tests stub the network (OpenAlex `fetch` and the scraper's `fetch`/`parse` are
injected; no Gemini calls), covering year selection, journal matching, OpenAlex
match acceptance/rejection, program CSV parsing, HTML/PDF text extraction,
chunking, scrape dedup, and the end-to-end rate math.
