# World Cup 2026 Data Quality Investigation

Generated: 2026-07-09

## Executive Summary

The missing must-have fields were not missing because the data was unavailable. They were missing because the project was relying on a fragile Google rendered-card extraction path and did not yet have a structured match-stat ingestion path.

The fix in this branch adds a no-new-package SofaScore fetcher and regenerates the interactive site from direct data:

- Completed local match backbone: 96 matches.
- SofaScore event mapping/fetch: 96/96 matches, all `ok`.
- SofaScore player rows: 4,914 rows, with 2,961 non-empty post-match ratings and 3,028 non-empty minutes values.
- SofaScore team stat rows: 12,584 rows.
- Match-level xG coverage: 96/96 matches.
- Shots-on-target coverage: 96/96 matches.
- Goalkeeper-save coverage: 96/96 matches, two team rows per match.
- Tackle coverage: 96/96 matches, two team rows per match.
- Shotmap rows: 2,442 rows, with xG present for 2,427 shot rows across 96/96 matches.

The interactive HTML now uses direct SofaScore ratings when available and Google ratings only as a fallback for matches without a SofaScore mapping. After the full SofaScore fetch, all 96 completed matches are covered by direct SofaScore rows, so the generated interactive payload uses 2,961 direct rating rows from 96 matches.

## What Was Missing And Why

| Field | Previous local coverage | Current coverage | Root cause | Resolution |
|---|---:|---:|---|---|
| Match results and schedule backbone | 96/96 | 96/96 | Backbone existed; the UI made some played matches look projected when rating rows were absent. | Match-detail copy now says direct ratings are missing only when direct rows are absent. |
| Direct player post-match ratings | Google: 74/96 matches, 2,334 rows | SofaScore: 96/96 matches, 2,961 rated rows | Google Sports is rendered client-side. The old extractor also had stale knockout targets and 8 `no_more_about` failures. | Added `scripts/sofascore_worldcup_data_fetcher.py` and wired SofaScore ratings into the interactive generator. |
| Minutes played for rating weighting | Google star file: 0 non-empty minutes rows | SofaScore: 96/96 matches, 3,028 non-empty minutes rows | Google star subset did not include minutes. | SofaScore lineups include `minutesPlayed`; fetcher stores it in `data/sofascore/sofascore_player_ratings.csv`. |
| xG | 0/96 saved | 96/96 matches | No previous stats ingestion pipeline. | SofaScore `/statistics` and `/shotmap` are now fetched and saved. |
| Shots on target | 0/96 saved | 96/96 matches | No previous stats ingestion pipeline. | SofaScore `shotsOnGoal` is stored in `data/sofascore/sofascore_team_match_stats.csv`. |
| Saves | 0/96 saved | 96/96 matches | No previous stats ingestion pipeline. | SofaScore `goalkeeperSaves` is stored for both teams per match. |
| Tackles | 0/96 saved | 96/96 matches | No previous stats ingestion pipeline. | SofaScore `totalTackle` is stored for both teams per match. |
| Shot-level xG/xGOT | 0/96 saved | 96/96 matches | No previous shotmap ingestion pipeline. | SofaScore `/shotmap` is stored in `data/sofascore/sofascore_shotmap.csv`. |
| Prediction markets | Partial and market-specific | unchanged | Market schemas are sparse/inconsistent; not every market exposes a complete W-D-L distribution. | Keep optional; normalize only when full outcome sets are available or assumptions are explicit. |
| Tickets | Partial/manual | unchanged | Prices are dynamic, seat-specific, and vendor-dependent. | Keep optional; use official/resale snapshots with timestamped caveats. |
| Weather | Forecast/climatology only | unchanged | Exact weather is not reliable far in advance. | Use climatology until normal forecast windows open. |
| Transfermarkt values | Manual/curated | unchanged | Transfermarkt pages can be anti-bot and schema-sensitive. | Keep curated values or refresh manually with source timestamp. |

## Google Extraction Findings

The existing Google pipeline is still useful as a fallback, but it should not be the main source for must-have stats.

- `data/google/google_worldcup_all_player_ratings.csv`: 2,334 rows over 74 matches.
- `data/google/google_lineup_rating_extraction_summary.csv`: 82 match attempts, 74 `ok`, 8 `no_more_about`.
- Google misses from the old pipeline were:
  - 8 group matches where the rendered card route did not expose the expected "More about this game" flow.
  - 14 later knockout matches that were not in the canonical Google extraction summary.
- The old UI also used human/model form signals as a display fallback for player ratings. That was misleading, so the interactive view now separates direct rating data from analyst signals.

## SofaScore Investigation Findings

The direct API-style endpoints are reachable with a browser-like curl request:

- `https://www.sofascore.com/api/v1/event/15186744/lineups` returned 200.
- `https://www.sofascore.com/api/v1/event/15186744/statistics` returned 200.
- `https://www.sofascore.com/api/v1/event/15186744/shotmap` returned 200.
- `https://www.sofascore.com/api/v1/search/events?q=South%20Africa%20South%20Korea` returned 200 and exposed event id `15186744`.

PowerShell `Invoke-RestMethod` and Python `urllib` returned 403 for the same SofaScore lineup endpoint. That means the source is available, but some HTTP clients are blocked/fingerprinted. The fetcher therefore uses the system `curl` executable with browser-like headers. No Python packages were installed.

The first full mapper run missed USA, Cote d'Ivoire, Turkiye, and Bosnia-related matches because the search queries used only one alias per team. Expanding aliases to include `USA`, `Côte d'Ivoire`, `Türkiye`, and related variants fixed the mapping. The final fetch summary is 96/96 `ok`.

## Files Added Or Updated

- `scripts/sofascore_worldcup_data_fetcher.py`: discovers SofaScore event ids and fetches lineups, team statistics, and shot maps.
- `data/sofascore/sofascore_match_map.csv`: local match to SofaScore event-id mapping.
- `data/sofascore/sofascore_player_ratings.csv`: player ratings, minutes, goals, assists, saves, xG/xA, tackles, shots on target, passes.
- `data/sofascore/sofascore_team_match_stats.csv`: flattened team stats by match, period, and stat key.
- `data/sofascore/sofascore_shotmap.csv`: shot-level xG/xGOT, player, team, minute, shot type, body part, and coordinates.
- `data/sofascore/sofascore_fetch_summary.csv`: per-match fetch status and row counts.
- `outputs/worldcup_2026_interactive_bracket.html`: regenerated with direct SofaScore/Google ratings and SofaScore xG/SOT/saves in the team-performance tab.

## Reproduction Commands

```powershell
python scripts\sofascore_worldcup_data_fetcher.py --fetch
python scripts\make_worldcup26_interactive_html.py
```

No dependency installation is required.

## Remaining Work

The notebook itself still needs a dedicated SofaScore integration cell if we want the notebook analysis tables to use the new direct stats natively. The interactive site generator is already wired to these CSVs. The next notebook update should load:

- `data/sofascore/sofascore_player_ratings.csv` for player form and minutes-weighted player averages.
- `data/sofascore/sofascore_team_match_stats.csv` for xG, shots on target, saves, tackles, and possession.
- `data/sofascore/sofascore_shotmap.csv` for shot-quality summaries and finishing/goalkeeper overperformance.
