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
    python scripts/google_lineup_rating_extractor.py

The Anaconda environment used for the notebook did not include Playwright at the
time this was written, so the July 2 extraction was run through Codex's rendered
browser tool, then normalized with build_google_lineup_rating_import.py.

By default this script writes sidecar refresh files instead of overwriting the
checked-in rating extracts. After inspection, merge the refresh file into the
canonical CSVs with the normal import/normalization workflow.
"""

from pathlib import Path
import argparse
import csv
import json
import re
import sys
from urllib.parse import quote_plus


ROOT = Path(__file__).resolve().parents[1]
GOOGLE_DATA_DIR = ROOT / "data" / "google"
TABLES_DIR = ROOT / "outputs" / "tables"
MATCH_LIST_PATH = GOOGLE_DATA_DIR / "google_rating_match_list.json"
RAW_CANONICAL = GOOGLE_DATA_DIR / "google_lineup_player_ratings_raw.csv"
ALL_PLAYER_CANONICAL = GOOGLE_DATA_DIR / "google_worldcup_all_player_ratings.csv"
SUMMARY_CANONICAL = GOOGLE_DATA_DIR / "google_lineup_rating_extraction_summary.csv"
RAW_REFRESH_OUT = GOOGLE_DATA_DIR / "google_lineup_player_ratings_raw_refresh.csv"
SUMMARY_REFRESH_OUT = GOOGLE_DATA_DIR / "google_lineup_rating_extraction_summary_refresh.csv"

RAW_FIELDNAMES = [
    "match_id",
    "match_no",
    "stage",
    "group",
    "date",
    "team_a",
    "team_b",
    "goals_a",
    "goals_b",
    "match",
    "player_display",
    "team",
    "rating",
    "raw_player_line",
    "source_url",
    "extraction_status",
    "lineup_order",
    "team_assignment_rule",
]

SUMMARY_FIELDNAMES = [
    "match_id",
    "match_no",
    "stage",
    "group",
    "date",
    "team_a",
    "team_b",
    "goals_a",
    "goals_b",
    "match",
    "status",
    "raw_rows",
    "source_url",
    "tried_queries",
]

GOOGLE_TEAM_ALIASES = {
    "Korea Republic": "South Korea",
    "Bosnia and Herzegovina": "Bosnia Herzegovina",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "Congo DR",
    "Turkiye": "Turkiye",
}


def google_team_name(team):
    return GOOGLE_TEAM_ALIASES.get(str(team), str(team))


def latest_knockout_schedule_path():
    candidates = sorted(TABLES_DIR.glob("worldcup_2026_projected_knockout_schedule*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_match_id(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if text.upper().startswith("M") or text.upper().startswith("G"):
        return text
    return f"M{int(float(text))}" if re.match(r"^\d+(?:\.0)?$", text) else text


def parse_goals_from_note(row):
    note = str(row.get("result_note") or "").strip()
    if not note:
        return None, None
    score_part = re.sub(r"\([^)]*\)", "", note.split(";")[0]).strip()
    score_match = re.search(r"(\d+)\s*[-\u2013]\s*(\d+)", score_part)
    if not score_match:
        return None, None
    first = int(score_match.group(1))
    second = int(score_match.group(2))
    left = score_part[: score_match.start()].strip().lower()
    right = score_part[score_match.end() :].strip().lower()
    team_a = str(row.get("team_a") or "").lower()
    team_b = str(row.get("team_b") or "").lower()
    if team_b and team_b in left and team_a and team_a in right:
        return second, first
    return first, second


def load_group_matches():
    if not MATCH_LIST_PATH.exists():
        return []
    with MATCH_LIST_PATH.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    matches = []
    for row in rows:
        goals_a = row.get("goals_a")
        goals_b = row.get("goals_b")
        label = row.get("label") or f"{row['team_a']} {goals_a}-{goals_b} {row['team_b']}"
        matches.append({
            "match_id": row.get("match_id"),
            "match_no": "",
            "stage": row.get("stage") or "Group Stage",
            "group": row.get("group") or "",
            "date": row.get("date") or "",
            "team_a": row.get("team_a"),
            "team_b": row.get("team_b"),
            "goals_a": goals_a,
            "goals_b": goals_b,
            "label": label,
            "direct_url": row.get("direct_url", ""),
        })
    return matches


def load_knockout_matches():
    path = latest_knockout_schedule_path()
    if path is None:
        return []
    matches = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).strip().lower() != "actual":
                continue
            goals_a, goals_b = parse_goals_from_note(row)
            match_no = row.get("match")
            label = (
                f"{row.get('team_a')} {goals_a}-{goals_b} {row.get('team_b')}"
                if goals_a is not None and goals_b is not None
                else row.get("result_note") or f"{row.get('team_a')} vs {row.get('team_b')}"
            )
            matches.append({
                "match_id": normalize_match_id(match_no),
                "match_no": match_no,
                "stage": row.get("round") or "Knockout",
                "group": "",
                "date": row.get("date") or "",
                "team_a": row.get("team_a"),
                "team_b": row.get("team_b"),
                "goals_a": goals_a,
                "goals_b": goals_b,
                "label": label,
                "direct_url": row.get("direct_url", ""),
            })
    return matches


def completed_matches():
    deduped = {}
    for match in [*load_group_matches(), *load_knockout_matches()]:
        match_id = match.get("match_id") or match.get("label")
        if match_id:
            deduped[match_id] = match
    return list(deduped.values())


def successful_existing_match_ids():
    ids = set()
    for path in [ALL_PLAYER_CANONICAL, RAW_CANONICAL]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("extraction_status", "ok")).lower() == "ok" and row.get("match_id"):
                    ids.add(row["match_id"])
    if SUMMARY_CANONICAL.exists():
        with SUMMARY_CANONICAL.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw_rows = int(float(row.get("raw_rows") or 0))
                if row.get("status") == "ok" and raw_rows > 0 and row.get("match_id"):
                    ids.add(row["match_id"])
    return ids


def query_variants(match):
    team_a = google_team_name(match["team_a"])
    team_b = google_team_name(match["team_b"])
    label = str(match.get("label") or "")
    variants = [
        f"{team_a} vs {team_b} FIFA World Cup 2026",
        f"{team_b} vs {team_a} FIFA World Cup 2026",
        f"{team_a} {team_b} FIFA World Cup 2026 full time",
        f"{team_a} {team_b} World Cup 2026 lineups ratings",
    ]
    if label:
        variants.append(f"{label} FIFA World Cup 2026 lineups")
    seen = set()
    clean = []
    for query in variants:
        if query not in seen:
            clean.append(query)
            seen.add(query)
    return clean


def assign_team(match, lineup_order):
    if lineup_order <= 11:
        return match["team_a"]
    if lineup_order <= 22:
        return match["team_b"]
    return match["team_a"] if (lineup_order - 23) % 2 == 0 else match["team_b"]


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
        lineup_order = len(rows) + 1
        rows.append({
            "match": match_label,
            "player_display": player,
            "rating": float(line),
            "raw_player_line": lines[next_idx],
            "lineup_order": lineup_order,
        })
    return rows


def click_first(locator):
    if locator.count() == 0:
        return False
    locator.first.click()
    return True


def try_open_lineups(page):
    lineups = page.get_by_text(re.compile(r"^LINEUPS$", re.I))
    if click_first(lineups):
        page.wait_for_timeout(1500)
        return True
    return False


def try_click_more_about(page):
    exact = page.get_by_text("More about this game", exact=True)
    if click_first(exact):
        page.wait_for_timeout(1800)
        return True
    loose = page.get_by_text(re.compile(r"More about (this )?game", re.I))
    if click_first(loose):
        page.wait_for_timeout(1800)
        return True
    return False


def parse_current_page(page, match):
    label = match["label"]
    if not try_open_lineups(page):
        return "no_lineups_tab", []

    text = page.locator("body").inner_text()
    start = text.find("Performance\nAge\nClub")
    if start < 0:
        start = text.find("Performance")
    end = text.find("Player ratings", start + 1)
    if end < 0:
        end = text.find("Match stats", start + 1)
    if start < 0 or end <= start:
        return "no_ratings_block", []

    rows = parse_google_lineup_segment(text[start:end], label)
    if not rows:
        return "no_ratings_block", []

    for row in rows:
        lineup_order = int(row["lineup_order"])
        row.update({
            "match_id": match.get("match_id", ""),
            "match_no": match.get("match_no", ""),
            "stage": match.get("stage", ""),
            "group": match.get("group", ""),
            "date": match.get("date", ""),
            "team_a": match.get("team_a", ""),
            "team_b": match.get("team_b", ""),
            "goals_a": match.get("goals_a", ""),
            "goals_b": match.get("goals_b", ""),
            "team": assign_team(match, lineup_order),
            "source_url": page.url,
            "extraction_status": "ok",
            "team_assignment_rule": "1-11 team_a, 12-22 team_b, substitutes alternate",
        })
    return "ok", rows


def extract_match(page, match):
    label = match["label"]
    tried_queries = []
    last_status = "not_attempted"
    if match.get("direct_url"):
        try:
            page.goto(match["direct_url"], wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(2500)
            status, rows = parse_current_page(page, match)
            if status == "ok":
                print(f"{label}: {len(rows)} rows via direct URL")
                return rows, {
                    "status": "ok",
                    "raw_rows": len(rows),
                    "source_url": page.url,
                    "tried_queries": "direct_url",
                }
            last_status = status
        except Exception as exc:
            last_status = f"direct_error:{type(exc).__name__}"

    for query in query_variants(match):
        tried_queries.append(query)
        try:
            page.goto("https://www.google.com/search?q=" + quote_plus(query), wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(2500)

            status, rows = parse_current_page(page, match)
            if status == "ok":
                print(f"{label}: {len(rows)} rows via {query}")
                return rows, {
                    "status": "ok",
                    "raw_rows": len(rows),
                    "source_url": page.url,
                    "tried_queries": " | ".join(tried_queries),
                }

            if try_click_more_about(page):
                status, rows = parse_current_page(page, match)
                if status == "ok":
                    print(f"{label}: {len(rows)} rows via {query}")
                    return rows, {
                        "status": "ok",
                        "raw_rows": len(rows),
                        "source_url": page.url,
                        "tried_queries": " | ".join(tried_queries),
                    }
            else:
                status = "no_more_about"
            last_status = status
        except Exception as exc:
            last_status = f"extraction_error:{type(exc).__name__}"

    print(f"{label}: {last_status}")
    return [], {
        "status": last_status,
        "raw_rows": 0,
        "source_url": page.url,
        "tried_queries": " | ".join(tried_queries),
    }


def summary_row(match, result):
    return {
        "match_id": match.get("match_id", ""),
        "match_no": match.get("match_no", ""),
        "stage": match.get("stage", ""),
        "group": match.get("group", ""),
        "date": match.get("date", ""),
        "team_a": match.get("team_a", ""),
        "team_b": match.get("team_b", ""),
        "goals_a": match.get("goals_a", ""),
        "goals_b": match.get("goals_b", ""),
        "match": match.get("label", ""),
        "status": result.get("status", ""),
        "raw_rows": result.get("raw_rows", 0),
        "source_url": result.get("source_url", ""),
        "tried_queries": result.get("tried_queries", ""),
    }


def filter_matches(matches, args):
    if args.missing_only:
        existing = successful_existing_match_ids()
        matches = [match for match in matches if match.get("match_id") not in existing]
    if args.match:
        needles = [needle.lower() for needle in args.match]
        matches = [
            match for match in matches
            if any(
                needle in " ".join([
                    str(match.get("match_id", "")),
                    str(match.get("match_no", "")),
                    str(match.get("label", "")),
                    str(match.get("team_a", "")),
                    str(match.get("team_b", "")),
                ]).lower()
                for needle in needles
            )
        ]
    if args.limit is not None:
        matches = matches[: max(0, args.limit)]
    return matches


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Refresh Google Sports rendered lineup ratings.")
    parser.add_argument("--missing-only", action="store_true", help="Skip matches that already have successful local rows.")
    parser.add_argument("--match", action="append", help="Restrict to a match id, number, team, or label substring. Repeatable.")
    parser.add_argument("--limit", type=int, help="Stop after this many target matches.")
    parser.add_argument("--list-targets", action="store_true", help="Print the target matches and exit before launching Playwright.")
    parser.add_argument("--headful", action="store_true", help="Launch Chromium with a visible browser window.")
    parser.add_argument("--output", type=Path, default=RAW_REFRESH_OUT, help="Raw rating rows output CSV.")
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_REFRESH_OUT, help="Extraction summary output CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    matches = filter_matches(completed_matches(), args)
    print(f"targets={len(matches)}")
    if args.list_targets:
        for match in matches:
            print(f"{match.get('match_id')} {match.get('date')} {match.get('stage')}: {match.get('label')}")
        return 0
    if not matches:
        return 0

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("Playwright is not installed in this Python environment.", file=sys.stderr)
        print("Install Playwright as shown in this file docstring, or ask Codex to install it with approval.", file=sys.stderr)
        return 2

    extracted = []
    summaries = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        for match in matches:
            rows, result = extract_match(page, match)
            extracted.extend(rows)
            summaries.append(summary_row(match, result))
        browser.close()

    write_csv(args.output, RAW_FIELDNAMES, extracted)
    write_csv(args.summary_output, SUMMARY_FIELDNAMES, summaries)

    print(f"wrote {len(extracted)} rows to {args.output}")
    print(f"wrote {len(summaries)} summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
