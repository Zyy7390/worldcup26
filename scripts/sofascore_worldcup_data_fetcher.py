"""
Fetch SofaScore World Cup event ids, player ratings, team stats, and shot maps.

This script intentionally uses only the Python standard library plus the system
curl executable. The local investigation found that PowerShell Invoke-RestMethod
is blocked with 403 for SofaScore's JSON endpoints, while curl succeeds.

Examples:
    python scripts/sofascore_worldcup_data_fetcher.py --probe-event 15186744
    python scripts/sofascore_worldcup_data_fetcher.py --discover --limit 5
    python scripts/sofascore_worldcup_data_fetcher.py --fetch --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sofascore"
MATCH_MAP_OUT = DATA_DIR / "sofascore_match_map.csv"
PLAYER_RATINGS_OUT = DATA_DIR / "sofascore_player_ratings.csv"
TEAM_STATS_OUT = DATA_DIR / "sofascore_team_match_stats.csv"
SHOTMAP_OUT = DATA_DIR / "sofascore_shotmap.csv"
FETCH_SUMMARY_OUT = DATA_DIR / "sofascore_fetch_summary.csv"
SOFASCORE_BASE = "https://www.sofascore.com/api/v1"
WORLD_CHAMPIONSHIP_ID = 16


TEAM_ALIASES = {
    "Bosnia and Herzegovina": ["Bosnia & Herzegovina", "Bosnia Herzegovina"],
    "Cabo Verde": ["Cape Verde"],
    "Cote d'Ivoire": ["Côte d'Ivoire", "Côte d’Ivoire", "Ivory Coast", "Cote d'Ivoire", "Cote d Ivoire"],
    "Curacao": ["Curacao", "Curaçao"],
    "Czechia": ["Czech Republic"],
    "DR Congo": ["Congo DR", "Democratic Republic of Congo", "DR Congo"],
    "Korea Republic": ["South Korea", "Republic of Korea"],
    "Turkiye": ["Türkiye", "Turkiye", "Turkey"],
    "USA": ["USA", "United States", "United States of America"],
}


def load_completed_matches():
    scripts_dir = ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from google_lineup_rating_extractor import completed_matches
    finally:
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))
    return completed_matches()


def clean_name(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def aliases_for(team: object) -> set[str]:
    team_text = str(team or "")
    variants = {team_text, *TEAM_ALIASES.get(team_text, [])}
    return {clean_name(value) for value in variants if str(value or "").strip()}


def display_alias(team: object) -> str:
    team_text = str(team or "")
    return TEAM_ALIASES.get(team_text, [team_text])[0]


def search_aliases(team: object) -> list[str]:
    team_text = str(team or "")
    return list(dict.fromkeys([team_text, *TEAM_ALIASES.get(team_text, [])]))


def curl_json(url: str) -> dict:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl executable not found")
    command = [
        curl,
        "-L",
        "-sS",
        "--fail",
        "-H",
        "Accept: application/json, text/plain, */*",
        "-H",
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def event_date(entity: dict) -> str:
    timestamp = entity.get("startTimestamp")
    if timestamp is None:
        return ""
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).date().isoformat()
    except (OSError, OverflowError, TypeError, ValueError):
        return ""


def date_close(local_date: str, event_utc_date: str) -> bool:
    if not local_date or not event_utc_date:
        return True
    try:
        local = datetime.fromisoformat(local_date).date()
        event = datetime.fromisoformat(event_utc_date).date()
    except ValueError:
        return True
    return abs((event - local).days) <= 1


def score_pair(entity: dict) -> tuple[int | None, int | None]:
    home = entity.get("homeScore", {}).get("display")
    away = entity.get("awayScore", {}).get("display")
    try:
        return int(home), int(away)
    except (TypeError, ValueError):
        return None, None


def event_matches_local_match(entity: dict, match: dict) -> tuple[bool, str]:
    tournament = entity.get("tournament", {}).get("uniqueTournament", {})
    if tournament.get("id") != WORLD_CHAMPIONSHIP_ID:
        return False, "not_world_championship"
    if entity.get("status", {}).get("type") != "finished":
        return False, "not_finished"

    home_name = clean_name(entity.get("homeTeam", {}).get("name"))
    away_name = clean_name(entity.get("awayTeam", {}).get("name"))
    a_aliases = aliases_for(match.get("team_a"))
    b_aliases = aliases_for(match.get("team_b"))
    orientation = None
    if home_name in a_aliases and away_name in b_aliases:
        orientation = "same"
    elif home_name in b_aliases and away_name in a_aliases:
        orientation = "reversed"
    else:
        return False, "team_mismatch"

    if not date_close(str(match.get("date") or ""), event_date(entity)):
        return False, "date_mismatch"

    home_score, away_score = score_pair(entity)
    goals_a = match.get("goals_a")
    goals_b = match.get("goals_b")
    if goals_a is not None and goals_b is not None and home_score is not None and away_score is not None:
        try:
            expected = (int(goals_a), int(goals_b)) if orientation == "same" else (int(goals_b), int(goals_a))
        except (TypeError, ValueError):
            expected = None
        if expected and (home_score, away_score) != expected:
            return False, "score_mismatch"
    return True, orientation or "matched"


def search_event(match: dict) -> dict:
    aliases_a = search_aliases(match.get("team_a"))
    aliases_b = search_aliases(match.get("team_b"))
    queries = []
    for alias_a in aliases_a:
        for alias_b in aliases_b:
            queries.append(f"{alias_a} {alias_b}")
            queries.append(f"{alias_b} {alias_a}")
    queries.append(f"{display_alias(match.get('team_a'))} {display_alias(match.get('team_b'))} World Championship")
    tried = []
    candidates = []
    for query in dict.fromkeys(queries):
        url = f"{SOFASCORE_BASE}/search/events?q={quote_plus(query)}"
        tried.append(query)
        payload = curl_json(url)
        for result in payload.get("results", []):
            if result.get("type") != "event":
                continue
            entity = result.get("entity", {})
            ok, reason = event_matches_local_match(entity, match)
            if ok:
                candidates.append((float(result.get("score") or 0), reason, entity, query))
    if not candidates:
        return {"status": "not_found", "tried_queries": " | ".join(tried)}
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, reason, entity, query = candidates[0]
    home_score, away_score = score_pair(entity)
    return {
        "status": "matched",
        "event_id": entity.get("id"),
        "custom_id": entity.get("customId", ""),
        "slug": entity.get("slug", ""),
        "sofascore_name": entity.get("name", ""),
        "sofascore_date_utc": event_date(entity),
        "sofascore_score": f"{home_score}-{away_score}",
        "orientation": reason,
        "search_score": round(score, 4),
        "matched_query": query,
        "tried_queries": " | ".join(tried),
    }


def discover_matches(matches: list[dict]) -> list[dict]:
    rows = []
    for match in matches:
        try:
            found = search_event(match)
        except Exception as exc:  # noqa: BLE001 - keep row-level diagnostics.
            found = {"status": "error", "error": str(exc)}
        rows.append({
            "match_id": match.get("match_id", ""),
            "match_no": match.get("match_no", ""),
            "stage": match.get("stage", ""),
            "date": match.get("date", ""),
            "team_a": match.get("team_a", ""),
            "team_b": match.get("team_b", ""),
            "goals_a": match.get("goals_a", ""),
            "goals_b": match.get("goals_b", ""),
            "label": match.get("label", ""),
            **found,
        })
        print(f"{rows[-1]['match_id']}: {rows[-1].get('status')} {rows[-1].get('event_id', '')}")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def player_rows_for_event(match_row: dict, payload: dict) -> list[dict]:
    rows = []
    for side in ["home", "away"]:
        team = match_row["team_a"] if (side == "home") == (match_row.get("orientation") == "same") else match_row["team_b"]
        for lineup_row in payload.get(side, {}).get("players", []) or []:
            player = lineup_row.get("player", {})
            stats = lineup_row.get("statistics", {}) or {}
            rows.append({
                "match_id": match_row.get("match_id"),
                "match_no": match_row.get("match_no"),
                "event_id": match_row.get("event_id"),
                "stage": match_row.get("stage"),
                "date": match_row.get("date"),
                "team": team,
                "side": side,
                "player": player.get("name") or player.get("shortName") or "",
                "player_id": player.get("id", ""),
                "position": lineup_row.get("position", ""),
                "substitute": lineup_row.get("substitute", False),
                "minutes_played": stats.get("minutesPlayed", ""),
                "rating": stats.get("rating", ""),
                "goals": stats.get("goals", ""),
                "assists": stats.get("goalAssist", ""),
                "saves": stats.get("saves", ""),
                "expected_goals": stats.get("expectedGoals", ""),
                "expected_assists": stats.get("expectedAssists", ""),
                "tackles": stats.get("totalTackle", ""),
                "shots_on_target": stats.get("onTargetScoringAttempt", ""),
                "passes": stats.get("totalPass", ""),
                "accurate_passes": stats.get("accuratePass", ""),
            })
    return rows


def stat_rows_for_event(match_row: dict, payload: dict) -> list[dict]:
    rows = []
    for period in payload.get("statistics", []) or []:
        period_name = period.get("period", "")
        for group in period.get("groups", []) or []:
            group_name = group.get("groupName", "")
            for item in group.get("statisticsItems", []) or []:
                rows.append({
                    "match_id": match_row.get("match_id"),
                    "match_no": match_row.get("match_no"),
                    "event_id": match_row.get("event_id"),
                    "stage": match_row.get("stage"),
                    "date": match_row.get("date"),
                    "team_a": match_row.get("team_a"),
                    "team_b": match_row.get("team_b"),
                    "period": period_name,
                    "group": group_name,
                    "stat_key": item.get("key", ""),
                    "stat_name": item.get("name", ""),
                    "home_value": item.get("home", ""),
                    "away_value": item.get("away", ""),
                })
    return rows


def shot_rows_for_event(match_row: dict, payload: dict) -> list[dict]:
    rows = []
    for shot in payload.get("shotmap", []) or []:
        player = shot.get("player", {}) or {}
        if shot.get("isHome") is True:
            team = match_row.get("team_a") if match_row.get("orientation") == "same" else match_row.get("team_b")
        elif shot.get("isHome") is False:
            team = match_row.get("team_b") if match_row.get("orientation") == "same" else match_row.get("team_a")
        else:
            team = (shot.get("team", {}) or {}).get("name", "")
        goal = shot.get("isGoal")
        if goal is None:
            goal = shot.get("shotType") == "goal"
        rows.append({
            "match_id": match_row.get("match_id"),
            "match_no": match_row.get("match_no"),
            "event_id": match_row.get("event_id"),
            "stage": match_row.get("stage"),
            "date": match_row.get("date"),
            "minute": shot.get("time", ""),
            "added_time": shot.get("addedTime", ""),
            "team": team,
            "player": player.get("name", ""),
            "player_id": player.get("id", ""),
            "shot_type": shot.get("shotType", ""),
            "situation": shot.get("situation", ""),
            "body_part": shot.get("bodyPart", ""),
            "xg": shot.get("xg", ""),
            "xgot": shot.get("xgot", ""),
            "goal": goal,
            "x": (shot.get("playerCoordinates") or {}).get("x", ""),
            "y": (shot.get("playerCoordinates") or {}).get("y", ""),
        })
    return rows


def fetch_payloads(match_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    players = []
    stats = []
    shots = []
    summary = []
    for row in match_rows:
        if str(row.get("status")) != "matched" or not row.get("event_id"):
            summary.append({**row, "lineups_rows": 0, "stat_rows": 0, "shot_rows": 0, "fetch_status": "skipped"})
            continue
        event_id = row["event_id"]
        try:
            lineups = curl_json(f"{SOFASCORE_BASE}/event/{event_id}/lineups")
            statistics = curl_json(f"{SOFASCORE_BASE}/event/{event_id}/statistics")
            shotmap = curl_json(f"{SOFASCORE_BASE}/event/{event_id}/shotmap")
            event_players = player_rows_for_event(row, lineups)
            event_stats = stat_rows_for_event(row, statistics)
            event_shots = shot_rows_for_event(row, shotmap)
            players.extend(event_players)
            stats.extend(event_stats)
            shots.extend(event_shots)
            fetch_status = "ok"
        except Exception as exc:  # noqa: BLE001 - row-level diagnostics.
            event_players = []
            event_stats = []
            event_shots = []
            fetch_status = f"error: {exc}"
        summary.append({
            **row,
            "lineups_rows": len(event_players),
            "stat_rows": len(event_stats),
            "shot_rows": len(event_shots),
            "fetch_status": fetch_status,
        })
        print(f"{row.get('match_id')}: {fetch_status}")
    return players, stats, shots, summary


def probe_event(event_id: str) -> None:
    lineups = curl_json(f"{SOFASCORE_BASE}/event/{event_id}/lineups")
    statistics = curl_json(f"{SOFASCORE_BASE}/event/{event_id}/statistics")
    shotmap = curl_json(f"{SOFASCORE_BASE}/event/{event_id}/shotmap")
    player_count = sum(len((lineups.get(side, {}) or {}).get("players", []) or []) for side in ["home", "away"])
    stat_count = sum(
        len(group.get("statisticsItems", []) or [])
        for period in statistics.get("statistics", []) or []
        for group in period.get("groups", []) or []
    )
    shot_count = len(shotmap.get("shotmap", []) or [])
    print(json.dumps({
        "event_id": event_id,
        "player_rows": player_count,
        "stat_items": stat_count,
        "shot_rows": shot_count,
        "lineup_keys": sorted(lineups.keys()),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-event", help="Fetch one SofaScore event id and print row counts.")
    parser.add_argument("--discover", action="store_true", help="Search event ids for local completed matches.")
    parser.add_argument("--fetch", action="store_true", help="Fetch ratings/stats/shotmaps after discovery.")
    parser.add_argument("--limit", type=int, default=0, help="Limit matches for a dry run.")
    args = parser.parse_args()

    if args.probe_event:
        probe_event(args.probe_event)
        return

    matches = load_completed_matches()
    if args.limit:
        matches = matches[: args.limit]

    if not args.discover and not args.fetch:
        print(f"{len(matches)} completed match targets loaded. Use --discover or --fetch.")
        return

    match_rows = discover_matches(matches)
    write_csv(MATCH_MAP_OUT, match_rows, [
        "match_id", "match_no", "stage", "date", "team_a", "team_b", "goals_a", "goals_b",
        "label", "status", "event_id", "custom_id", "slug", "sofascore_name", "sofascore_date_utc",
        "sofascore_score", "orientation", "search_score", "matched_query", "tried_queries", "error",
    ])

    if args.fetch:
        players, stats, shots, summary = fetch_payloads(match_rows)
        write_csv(PLAYER_RATINGS_OUT, players, [
            "match_id", "match_no", "event_id", "stage", "date", "team", "side", "player", "player_id",
            "position", "substitute", "minutes_played", "rating", "goals", "assists", "saves",
            "expected_goals", "expected_assists", "tackles", "shots_on_target", "passes",
            "accurate_passes",
        ])
        write_csv(TEAM_STATS_OUT, stats, [
            "match_id", "match_no", "event_id", "stage", "date", "team_a", "team_b", "period",
            "group", "stat_key", "stat_name", "home_value", "away_value",
        ])
        write_csv(SHOTMAP_OUT, shots, [
            "match_id", "match_no", "event_id", "stage", "date", "minute", "added_time", "team",
            "player", "player_id", "shot_type", "situation", "body_part", "xg", "xgot", "goal", "x", "y",
        ])
        write_csv(FETCH_SUMMARY_OUT, summary, [
            "match_id", "match_no", "stage", "date", "team_a", "team_b", "goals_a", "goals_b",
            "label", "status", "event_id", "fetch_status", "lineups_rows", "stat_rows", "shot_rows",
            "error",
        ])


if __name__ == "__main__":
    main()
