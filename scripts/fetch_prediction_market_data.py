"""Fetch and normalize current World Cup prediction-market snapshots.

The collector uses public, unauthenticated Polymarket and Kalshi endpoints. It
stores the raw responses for auditability and a compact normalized CSV for the
notebook. No trading credentials or third-party Python packages are required.

Example:
    python scripts/fetch_prediction_market_data.py --as-of 20260712
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = ROOT / "data" / "markets"

POLYMARKET_EVENTS = {
    "winner": "https://gamma-api.polymarket.com/events/30615",
    "final_qualifier": "https://gamma-api.polymarket.com/events/414457",
}
KALSHI_ENDPOINTS = {
    "winner": (
        "https://external-api.kalshi.com/trade-api/v2/events/"
        "KXMENWORLDCUP-26?with_nested_markets=true"
    ),
    "final_qualifier": (
        "https://external-api.kalshi.com/trade-api/v2/events?"
        "series_ticker=KXWCROUND&status=open&limit=200&with_nested_markets=true"
    ),
    "match_advance": (
        "https://external-api.kalshi.com/trade-api/v2/events?"
        "series_ticker=KXWCADVANCE&status=open&limit=200&with_nested_markets=true"
    ),
}
SEMIFINALISTS = {"Argentina", "England", "France", "Spain"}
MATCHUPS = {
    "France vs Spain": {"France", "Spain"},
    "England vs Argentina": {"England", "Argentina"},
}
MATCHUP_BY_TEAM = {
    team: matchup
    for matchup, teams in MATCHUPS.items()
    for team in teams
}


def curl_json(url: str) -> object:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl executable not found")
    result = subprocess.run(
        [
            curl,
            "-L",
            "-sS",
            "--fail",
            "-H",
            "Accept: application/json, text/plain, */*",
            "-H",
            (
                "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            url,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def jsonish_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def numeric(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def midpoint(bid: object, ask: object, last: object) -> float | None:
    bid_value = numeric(bid)
    ask_value = numeric(ask)
    if bid_value is not None and ask_value is not None and ask_value >= bid_value:
        return (bid_value + ask_value) / 2
    return numeric(last)


def team_from_question(question: str, phrase: str) -> str:
    prefix = "Will "
    if not question.startswith(prefix) or phrase not in question:
        return ""
    return question[len(prefix) : question.index(phrase)].strip()


def polymarket_rows(payload: object, market_type: str, fetched_at: str) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    phrase = (
        " win the 2026 FIFA World Cup?"
        if market_type == "winner"
        else " reach the 2026 FIFA World Cup final?"
    )
    rows = []
    for market in payload.get("markets", []) or []:
        if market.get("active") is not True or market.get("closed") is True:
            continue
        question = str(market.get("question") or "")
        team = team_from_question(question, phrase)
        if team not in SEMIFINALISTS:
            continue
        outcomes = jsonish_list(market.get("outcomes"))
        prices = [numeric(value) for value in jsonish_list(market.get("outcomePrices"))]
        if "Yes" not in outcomes or len(outcomes) != len(prices):
            continue
        probability = prices[outcomes.index("Yes")]
        rows.append({
            "source": "Polymarket",
            "market_type": market_type,
            "event": payload.get("title", ""),
            "matchup": MATCHUP_BY_TEAM.get(team, "") if market_type == "final_qualifier" else "",
            "team": team,
            "probability": probability,
            "normalized_probability": "",
            "yes_bid": "",
            "yes_ask": "",
            "last_price": probability,
            "liquidity": numeric(market.get("liquidity")),
            "volume": numeric(market.get("volume")),
            "ticker": market.get("slug", ""),
            "question": question,
            "status": "active",
            "fetched_at_utc": fetched_at,
            "is_complete_distribution": False,
        })
    return rows


def kalshi_events(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("event"), dict):
        return [payload["event"]]
    return [event for event in payload.get("events", []) or [] if isinstance(event, dict)]


def clean_kalshi_team(value: object) -> str:
    text = str(value or "").strip()
    return text.removesuffix(" advances").strip()


def kalshi_rows(payload: object, market_type: str, fetched_at: str) -> list[dict]:
    rows = []
    for event in kalshi_events(payload):
        event_title = str(event.get("title") or "")
        for market in event.get("markets", []) or []:
            if str(market.get("status") or "") != "active":
                continue
            team = clean_kalshi_team(market.get("yes_sub_title"))
            if team not in SEMIFINALISTS:
                continue
            matchup = (
                event_title
                if market_type == "match_advance"
                else MATCHUP_BY_TEAM.get(team, "")
                if market_type == "final_qualifier"
                else ""
            )
            probability = midpoint(
                market.get("yes_bid_dollars"),
                market.get("yes_ask_dollars"),
                market.get("last_price_dollars"),
            )
            rows.append({
                "source": "Kalshi",
                "market_type": market_type,
                "event": event_title,
                "matchup": matchup,
                "team": team,
                "probability": probability,
                "normalized_probability": "",
                "yes_bid": numeric(market.get("yes_bid_dollars")),
                "yes_ask": numeric(market.get("yes_ask_dollars")),
                "last_price": numeric(market.get("last_price_dollars")),
                "liquidity": numeric(market.get("liquidity_dollars")),
                "volume": numeric(market.get("volume_fp")),
                "ticker": market.get("ticker", ""),
                "question": market.get("title") or event_title,
                "status": market.get("status", ""),
                "fetched_at_utc": fetched_at,
                "is_complete_distribution": False,
            })
    return rows


def normalize_groups(rows: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        group_name = (
            row["matchup"]
            if row["market_type"] in {"final_qualifier", "match_advance"}
            else row["market_type"]
        )
        groups.setdefault((row["source"], row["market_type"], group_name), []).append(row)

    for (_, market_type, group_name), group_rows in groups.items():
        observed = {row["team"] for row in group_rows}
        expected = MATCHUPS.get(group_name, SEMIFINALISTS)
        complete = observed == expected
        total = sum(float(row["probability"] or 0) for row in group_rows)
        for row in group_rows:
            row["is_complete_distribution"] = complete
            if complete and total > 0:
                row["normalized_probability"] = float(row["probability"]) / total


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source", "market_type", "event", "matchup", "team", "probability",
        "normalized_probability", "yes_bid", "yes_ask", "last_price", "liquidity",
        "volume", "ticker", "question", "status", "fetched_at_utc",
        "is_complete_distribution",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict]) -> None:
    for source in ["Polymarket", "Kalshi"]:
        for market_type in ["winner", "final_qualifier"]:
            observed = {
                row["team"]
                for row in rows
                if row["source"] == source and row["market_type"] == market_type
            }
            if observed != SEMIFINALISTS:
                raise RuntimeError(
                    f"{source} {market_type} is incomplete: expected {sorted(SEMIFINALISTS)}, "
                    f"found {sorted(observed)}"
                )
    for matchup, expected in MATCHUPS.items():
        observed = {
            row["team"]
            for row in rows
            if row["source"] == "Kalshi"
            and row["market_type"] == "match_advance"
            and row["matchup"] == matchup
        }
        if observed != expected:
            raise RuntimeError(
                f"Kalshi {matchup} is incomplete: expected {sorted(expected)}, found {sorted(observed)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        default=datetime.now(tz=timezone.utc).strftime("%Y%m%d"),
        help="Snapshot date used in output filenames, formatted YYYYMMDD.",
    )
    args = parser.parse_args()
    if len(args.as_of) != 8 or not args.as_of.isdigit():
        raise ValueError("--as-of must use YYYYMMDD")

    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
    rows = []

    for market_type, url in POLYMARKET_EVENTS.items():
        payload = curl_json(url)
        write_json(MARKET_DIR / f"market_fetch_polymarket_{market_type}_{args.as_of}.json", payload)
        rows.extend(polymarket_rows(payload, market_type, fetched_at))

    for market_type, url in KALSHI_ENDPOINTS.items():
        payload = curl_json(url)
        write_json(MARKET_DIR / f"market_fetch_kalshi_{market_type}_{args.as_of}.json", payload)
        rows.extend(kalshi_rows(payload, market_type, fetched_at))

    normalize_groups(rows)
    validate(rows)
    normalized_path = MARKET_DIR / f"prediction_market_snapshot_{args.as_of}.csv"
    write_csv(normalized_path, rows)

    print(f"Wrote {normalized_path}")
    for source in ["Polymarket", "Kalshi"]:
        source_rows = [row for row in rows if row["source"] == source]
        print(f"{source}: {len(source_rows)} normalized rows")


if __name__ == "__main__":
    main()
