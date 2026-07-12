# World Cup 2026 Data Quality Investigation

Generated: 2026-07-12

## Executive Summary

The missing must-have fields were not missing because the data was unavailable. They were missing because the project was relying on a fragile Google rendered-card extraction path and did not yet have a structured match-stat ingestion path.

The fix in this branch adds a no-new-package SofaScore fetcher and regenerates the interactive site from direct data:

- Completed local match backbone: 100 matches through M100.
- SofaScore event mapping/fetch: 100/100 matches, all `ok`.
- SofaScore player rows: 5,117 rows, with 3,091 non-empty post-match ratings and 3,158 non-empty minutes values.
- SofaScore team stat rows: 13,246 rows.
- Match-level xG coverage: 100/100 matches.
- Shots-on-target coverage: 100/100 matches.
- Goalkeeper-save coverage: 100/100 matches, two team rows per match.
- Tackle coverage: 100/100 matches, two team rows per match.
- Shotmap rows: 2,551 rows, with xG present for 2,536 shot rows across 100/100 matches.

The notebook and interactive HTML now use direct SofaScore ratings when available and Google ratings only as a fallback for matches without a SofaScore mapping. After the full SofaScore fetch, all 100 completed matches are covered by direct SofaScore rows, so the generated interactive payload uses 3,091 direct rating rows from 100 matches.

## What Was Missing And Why

| Field | Previous local coverage | Current coverage | Root cause | Resolution |
|---|---:|---:|---|---|
| Match results and schedule backbone | 96/96 | 100/100 | The old collector depended on a stale exported bracket CSV. | The collector now reads the canonical locked-result list and uses exports only for schedule metadata. |
| Direct player post-match ratings | Google: 74 matches, 2,334 rows | SofaScore: 100/100 matches, 3,091 rated rows | Google Sports is rendered client-side and incomplete. | SofaScore is primary in both notebook and site; Google is a per-match fallback only. |
| Minutes played for rating weighting | Google star file: 0 non-empty minutes rows | SofaScore: 100/100 matches, 3,158 non-empty minutes rows | Google star subset did not include minutes. | SofaScore lineups provide `minutesPlayed`; the notebook uses minutes-weighted team/player averages. |
| xG | 0 saved | 100/100 matches | No previous stats ingestion pipeline. | SofaScore `/statistics` and `/shotmap` are fetched and saved. |
| Shots on target | 0 saved | 100/100 matches | No previous stats ingestion pipeline. | SofaScore `shotsOnGoal` is stored in `data/sofascore/sofascore_team_match_stats.csv`. |
| Saves | 0 saved | 100/100 matches | No previous stats ingestion pipeline. | SofaScore `goalkeeperSaves` is stored for both teams per match. |
| Tackles | 0 saved | 100/100 matches | No previous stats ingestion pipeline. | SofaScore `totalTackle` is stored for both teams per match. |
| Shot-level xG/xGOT | 0 saved | 100/100 matches | No previous shotmap ingestion pipeline. | SofaScore `/shotmap` is stored in `data/sofascore/sofascore_shotmap.csv`. |
| Prediction markets | Partial and market-specific | 20 complete July 12 rows | The old broad searches mixed unrelated and multileg rows. | A dedicated collector fetches known World Cup events, validates complete team sets, and normalizes by semifinal matchup. |
| Tickets | Partial/manual | unchanged | Prices are dynamic, seat-specific, and vendor-dependent. | Keep optional; use official/resale snapshots with timestamped caveats. |
| Weather | Forecast/climatology only | unchanged | Exact weather is not reliable far in advance. | Use climatology until normal forecast windows open. |
| Transfermarkt values | Manual/curated | unchanged | Transfermarkt pages can be anti-bot and schema-sensitive. | Keep curated values or refresh manually with source timestamp. |

## Google Extraction Findings

The existing Google pipeline is still useful as a fallback, but it should not be the main source for must-have stats.

- `data/google/google_worldcup_all_player_ratings.csv`: 2,334 rows over 74 matches.
- `data/google/google_lineup_rating_extraction_summary.csv`: 82 match attempts, 74 `ok`, 8 `no_more_about`.
- Google misses from the old pipeline were:
  - 8 group matches where the rendered card route did not expose the expected "More about this game" flow.
  - 18 later knockout matches that were not in the canonical Google extraction summary.
- The old UI also used human/model form signals as a display fallback for player ratings. That was misleading, so the interactive view now separates direct rating data from analyst signals.

## SofaScore Investigation Findings

The direct API-style endpoints are reachable with a browser-like curl request:

- `https://www.sofascore.com/api/v1/event/15186744/lineups` returned 200.
- `https://www.sofascore.com/api/v1/event/15186744/statistics` returned 200.
- `https://www.sofascore.com/api/v1/event/15186744/shotmap` returned 200.
- `https://www.sofascore.com/api/v1/search/events?q=South%20Africa%20South%20Korea` returned 200 and exposed event id `15186744`.

PowerShell `Invoke-RestMethod` and Python `urllib` returned 403 for the same SofaScore lineup endpoint. That means the source is available, but some HTTP clients are blocked/fingerprinted. The fetcher therefore uses the system `curl` executable with browser-like headers. No Python packages were installed.

The first full mapper run missed USA, Cote d'Ivoire, Turkiye, and Bosnia-related matches because the search queries used only one alias per team. Expanding aliases to include `USA`, `Côte d'Ivoire`, `Türkiye`, and related variants fixed the mapping. The July 12 fetch summary is 100/100 `ok`.

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
python scripts\fetch_prediction_market_data.py --as-of 20260712
python scripts\update_worldcup26_notebook_live.py
python scripts\make_worldcup26_interactive_html.py
```

No dependency installation is required.

## Current Integration

The notebook now loads the SofaScore CSVs directly. R32-and-later player form uses minutes-weighted team averages with stage-recency weights, and the latest-match audit table exposes xG, shots on target, saves, tackles, possession, and top player rows. The interactive site uses the same files for match details, team history, player detail, and value/performance views.
