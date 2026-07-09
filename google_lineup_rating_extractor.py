"""
Best-effort Google Sports lineup rating extractor.

Google's World Cup sports module is rendered client-side. The useful player
ratings are not available through plain requests/BeautifulSoup. The route that
worked on July 2 was:

    direct match search -> More about this game -> LINEUPS -> rendered DOM text

For standalone reruns, install Playwright and run this script from the project
root:

    python -m pip install playwright
    python -m playwright install chromium
    python google_lineup_rating_extractor.py

The Anaconda environment used for the notebook did not include Playwright at the
time this was written, so the July 2 extraction was run through Codex's rendered
browser tool, then normalized with build_google_lineup_rating_import.py.
"""

from pathlib import Path
import csv
import re
import sys
from urllib.parse import quote_plus


ROOT = Path(__file__).resolve().parent
RAW_OUT = ROOT / "google_lineup_player_ratings_raw.csv"

COMPLETED_R32_MATCHES = [
    {"team_a": "South Africa", "team_b": "Canada", "label": "South Africa 0-1 Canada"},
    {"team_a": "Netherlands", "team_b": "Morocco", "label": "Netherlands 1-1 Morocco"},
    {"team_a": "Germany", "team_b": "Paraguay", "label": "Germany 1-1 Paraguay"},
    {"team_a": "France", "team_b": "Sweden", "label": "France 3-0 Sweden"},
    {"team_a": "Brazil", "team_b": "Japan", "label": "Brazil 2-1 Japan"},
    {
        "team_a": "Cote d'Ivoire",
        "team_b": "Norway",
        "label": "Cote d'Ivoire 1-2 Norway",
        "direct_url": "https://www.google.com/search?q=world+cup#sie=m;/g/11xl329t7w;2;/m/030q7;ln;fp;1;;;;-1",
    },
    {"team_a": "Mexico", "team_b": "Ecuador", "label": "Mexico 2-0 Ecuador"},
    {"team_a": "England", "team_b": "DR Congo", "label": "England 2-1 DR Congo"},
]


def parse_google_lineup_segment(segment, match_label):
    lines = [line.strip() for line in segment.splitlines() if line.strip()]
    rows = []

    def is_rating(value):
        return bool(re.match(r"^(?:[0-9](?:\.[0-9])?|10(?:\.0)?)$", value))

    ignored = {"Performance", "Age", "Club", "Bench"}
    for idx, line in enumerate(lines[:-1]):
        if not is_rating(line):
            continue
        next_idx = idx + 1
        while next_idx < len(lines) and (
            lines[next_idx] in ignored
            or lines[next_idx] == r"\t"
            or re.match(r"^\d+$", lines[next_idx])
            or re.match(r"^\d+-\d+-\d+$", lines[next_idx])
        ):
            next_idx += 1
        if next_idx >= len(lines):
            continue
        player = lines[next_idx].replace("\u00a0", " ").strip()
        player = re.sub(r"^\d{1,2}(?=\S)", "", player)
        player = re.sub(r"\s+\d+'$", "", player).strip()
        if not player or player in ignored or not re.search(r"[A-Za-z]", player):
            continue
        rows.append({
            "match": match_label,
            "player_display": player,
            "rating": float(line),
            "raw_player_line": lines[next_idx],
        })
    return rows


def extract_match(page, match):
    label = match["label"]
    if match.get("direct_url"):
        page.goto(match["direct_url"], wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
    else:
        query = f"{match['team_a']} {match['team_b']} World Cup 2026 lineups ratings"
        page.goto("https://www.google.com/search?q=" + quote_plus(query), wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        more = page.get_by_text("More about this game", exact=True)
        if more.count() == 0:
            print(f"no more-about-game control: {label}")
            return []
        more.first.click()
        page.wait_for_timeout(1800)

    lineups = page.get_by_text("LINEUPS", exact=True)
    if lineups.count() == 0:
        print(f"no lineups tab: {label}")
        return []
    lineups.first.click()
    page.wait_for_timeout(1500)

    text = page.locator("body").inner_text()
    start = text.find("Performance\nAge\nClub")
    end = text.find("Player ratings")
    if start < 0 or end <= start:
        print(f"no ratings block: {label}")
        return []

    rows = parse_google_lineup_segment(text[start:end], label)
    for row in rows:
        row["source_url"] = page.url
        row["extraction_status"] = "ok"
    print(f"{label}: {len(rows)} rows")
    return rows


def main():
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("Playwright is not installed in this Python environment.", file=sys.stderr)
        print("Install Playwright as shown in this file docstring, or use the Codex rendered-browser workflow.", file=sys.stderr)
        return 2

    extracted = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        for match in COMPLETED_R32_MATCHES:
            extracted.extend(extract_match(page, match))
        browser.close()

    with RAW_OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["match", "player_display", "rating", "raw_player_line", "source_url", "extraction_status"],
        )
        writer.writeheader()
        writer.writerows(extracted)

    print(f"wrote {len(extracted)} rows to {RAW_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
