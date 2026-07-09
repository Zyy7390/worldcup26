from __future__ import annotations

from pathlib import Path
import ast
import html
import io
import json
import math
import re

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "worldcup_2026_prediction_bracket.ipynb"
GOOGLE_DATA_DIR = ROOT / "data" / "google"
OUTPUT_DIR = ROOT / "outputs"
OUT = OUTPUT_DIR / "worldcup_2026_interactive_bracket.html"
MODEL_SOURCE = ROOT / "scripts" / "update_worldcup26_notebook_live.py"


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df


def html_tables_from_cell(cell) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        table_html = data.get("text/html")
        if not table_html:
            continue
        try:
            tables.extend(clean_frame(df) for df in pd.read_html(io.StringIO(table_html)))
        except ValueError:
            continue
    return tables


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for record in df.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def load_notebook_tables():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    live_parts = []
    market_table = pd.DataFrame()
    recommendations = pd.DataFrame()
    summary = pd.DataFrame()

    for cell in nb.cells:
        source = cell.get("source", "")
        tables = html_tables_from_cell(cell)
        if "live_result_view" in source or "live_knockout_table" in source:
            for table in tables:
                required = {"round", "match", "date", "weekday", "team_a", "team_b", "winner", "winner_probability"}
                if required.issubset(table.columns):
                    live_parts.append(table)
        if "market_ko_view" in source:
            for table in tables:
                required = {"round", "match", "date", "weekday", "team_a", "team_b", "winner", "winner_probability"}
                if required.issubset(table.columns) and len(table) >= 30:
                    market_table = table
        if "recommendations_ranked" in source:
            for table in tables:
                if {"rank", "match", "possible_matchup", "recommendation_score"}.issubset(table.columns):
                    recommendations = table
        if "live_summary" in source:
            for table in tables:
                if {"result", "team"}.issubset(table.columns):
                    summary = table

    if not live_parts:
        raise RuntimeError("Could not find live knockout table in executed notebook outputs.")
    live_table = pd.concat(live_parts, ignore_index=True)
    live_table = live_table.drop_duplicates(["round", "match"], keep="last").sort_values("match")
    if market_table.empty:
        market_table = live_table.copy()
    return live_table, market_table.sort_values("match"), recommendations, summary


def load_rating_data():
    all_path = GOOGLE_DATA_DIR / "google_worldcup_all_player_ratings.csv"
    summary_path = GOOGLE_DATA_DIR / "google_lineup_rating_extraction_summary.csv"
    star_path = GOOGLE_DATA_DIR / "google_lineup_player_ratings.csv"

    ratings = pd.read_csv(all_path) if all_path.exists() else pd.DataFrame()
    extraction_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    stars = pd.read_csv(star_path) if star_path.exists() else pd.DataFrame()

    ratings_by_match = {}
    team_avgs_by_match = {}
    top_knockout_players = []
    knockout_team_form = []
    if not ratings.empty:
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        knockout_stages = {"Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Third-place Match", "Final"}
        knockout = ratings[ratings["stage"].isin(knockout_stages) & ratings["rating"].notna()].copy()
        for match_no, group in knockout.groupby("match_no"):
            match_key = str(int(match_no))
            top = (
                group.sort_values("rating", ascending=False)
                [["team", "player_display", "rating", "match"]]
                .head(12)
                .rename(columns={"player_display": "player"})
            )
            ratings_by_match[match_key] = dataframe_to_records(top)
            team_avg = (
                group.groupby("team", as_index=False)
                .agg(avg_rating=("rating", "mean"), players_rated=("player_display", "nunique"))
                .sort_values("avg_rating", ascending=False)
            )
            team_avgs_by_match[match_key] = dataframe_to_records(team_avg.round(3))

        top_knockout_players = dataframe_to_records(
            knockout.sort_values("rating", ascending=False)
            [["match_no", "match", "team", "player_display", "rating"]]
            .head(24)
            .rename(columns={"player_display": "player"})
        )
        knockout_team_form = dataframe_to_records(
            knockout.groupby(["match_no", "match", "team"], as_index=False)
            .agg(avg_rating=("rating", "mean"), players_rated=("player_display", "nunique"))
            .sort_values("avg_rating", ascending=False)
            .round(3)
        )

    coverage = []
    if not extraction_summary.empty:
        coverage = dataframe_to_records(
            extraction_summary.groupby(["stage", "status"], dropna=False, as_index=False)
            .agg(matches=("match_id", "count"), player_rows=("raw_rows", "sum"))
            .sort_values(["stage", "status"])
        )

    star_summary = []
    if not stars.empty:
        stars["rating"] = pd.to_numeric(stars["rating"], errors="coerce")
        star_summary = dataframe_to_records(
            stars.groupby(["team", "player"], as_index=False)
            .agg(avg_rating=("rating", "mean"), matches=("match", "nunique"))
            .sort_values("avg_rating", ascending=False)
            .head(30)
            .round(3)
        )

    return {
        "ratingsByMatch": ratings_by_match,
        "teamAveragesByMatch": team_avgs_by_match,
        "topR32Players": top_knockout_players,
        "r32TeamForm": knockout_team_form,
        "coverage": coverage,
        "allRows": int(len(ratings)),
        "scrapedMatches": int(ratings["match"].nunique()) if not ratings.empty else 0,
        "starRows": int(len(stars)),
        "starSummary": star_summary,
    }


def load_player_value_data() -> list[dict]:
    if not MODEL_SOURCE.exists():
        return []
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"transfermarkt_values_data\s*=\s*\[(.*?)\]\s*star_player_values",
        source,
        flags=re.S,
    )
    if not match:
        return []
    values = ast.literal_eval("[" + match.group(1) + "]")
    value_df = pd.DataFrame(values, columns=["team", "player", "transfermarkt_value_eur_m"])

    ratings_path = GOOGLE_DATA_DIR / "google_lineup_player_ratings.csv"
    if not ratings_path.exists():
        return []
    ratings = pd.read_csv(ratings_path)
    required = {"team", "player", "match", "rating", "minutes_played"}
    if not required.issubset(ratings.columns):
        return []
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings["minutes_played"] = pd.to_numeric(ratings["minutes_played"], errors="coerce")
    ratings = ratings[ratings["rating"].notna()].copy()

    rows = []
    for (team, player), sub in ratings.groupby(["team", "player"], dropna=False):
        has_minutes = sub["minutes_played"].notna().all() and sub["minutes_played"].sum() > 0
        if has_minutes:
            world_cup_rating = (sub["rating"] * sub["minutes_played"]).sum() / sub["minutes_played"].sum()
            rating_method = "minutes-weighted"
        else:
            world_cup_rating = sub["rating"].mean()
            rating_method = "simple average"
        rows.append({
            "team": team,
            "player": player,
            "world_cup_rating": world_cup_rating,
            "rated_matches": sub["match"].nunique(),
            "rating_method": rating_method,
        })

    performance = pd.DataFrame(rows)
    merged = value_df.merge(performance, on=["team", "player"], how="inner")
    if merged.empty:
        return []
    merged["value_tier"] = merged["transfermarkt_value_eur_m"].ge(75).map({True: "high value", False: "lower value"})
    merged["performance_tier"] = merged["world_cup_rating"].ge(7.0).map({True: "high performance", False: "lower performance"})
    return dataframe_to_records(
        merged.sort_values(["team", "transfermarkt_value_eur_m"], ascending=[True, False]).round(3)
    )


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    champion = html.escape(data.get("summaryMap", {}).get("Champion", "TBD"))
    runner_up = html.escape(data.get("summaryMap", {}).get("Runner-up", "TBD"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup 2026 Interactive Bracket</title>
  <style>
    :root {{
      --bg: #091013;
      --panel: #111b20;
      --panel-2: #17252b;
      --ink: #edf4f1;
      --muted: #9fb0ad;
      --gold: #d8b45f;
      --teal: #26c6b8;
      --red: #e35d5b;
      --blue: #6da8ff;
      --line: rgba(255,255,255,.12);
      --shadow: 0 18px 48px rgba(0,0,0,.34);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(115deg, rgba(216,180,95,.13), transparent 34%),
        linear-gradient(245deg, rgba(38,198,184,.10), transparent 36%),
        repeating-linear-gradient(90deg, rgba(255,255,255,.032) 0 1px, transparent 1px 116px),
        var(--bg);
      min-height: 100vh;
    }}
    header {{
      padding: 30px clamp(18px, 4vw, 56px) 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(9,16,19,.72);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 4;
    }}
    .eyebrow {{
      color: var(--gold);
      font-size: 12px;
      letter-spacing: .14em;
      text-transform: uppercase;
      font-weight: 800;
    }}
    h1 {{
      margin: 8px 0 14px;
      font-size: clamp(30px, 5vw, 58px);
      line-height: 1.02;
      letter-spacing: 0;
    }}
    .lead {{
      max-width: 880px;
      color: var(--muted);
      line-height: 1.55;
      margin: 0;
    }}
    .scoreboard {{
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .metric {{
      background: linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.035));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      min-height: 76px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .metric strong {{
      font-size: 20px;
      letter-spacing: 0;
    }}
    main {{ padding: 24px clamp(14px, 3vw, 42px) 54px; }}
    .toolbar {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .toolbar .group {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      padding: 6px;
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      background: rgba(255,255,255,.025);
    }}
    .toggle {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      padding: 8px 10px;
    }}
    .zoom-label {{
      color: var(--muted);
      font-size: 13px;
      min-width: 46px;
      text-align: center;
    }}
    button, select, input {{
      border: 1px solid var(--line);
      background: #0f191e;
      color: var(--ink);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
      transition: transform .16s ease, border-color .16s ease, background .16s ease;
    }}
    button:hover {{ transform: translateY(-1px); border-color: rgba(216,180,95,.55); }}
    button.active {{ background: rgba(216,180,95,.18); border-color: var(--gold); }}
    input {{ min-width: min(320px, 100%); }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }}
    .bracket {{
      overflow-x: auto;
      overflow-y: hidden;
      padding-bottom: 12px;
      border-radius: 8px;
    }}
    .rounds {{
      display: grid;
      grid-template-columns: repeat(5, minmax(230px, 1fr));
      gap: 12px;
      min-width: 1180px;
      transform-origin: top left;
      transition: transform .18s ease;
    }}
    .round-col {{
      background: rgba(17,27,32,.72);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .round-col h2 {{
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .12em;
      margin: 0 0 12px;
      color: var(--teal);
    }}
    .match-card {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 10px;
      min-height: 150px;
      position: relative;
      box-shadow: 0 9px 22px rgba(0,0,0,.20);
    }}
    .match-card.actual {{ border-left: 4px solid var(--teal); }}
    .match-card.projected {{ border-left: 4px solid var(--gold); }}
    .match-card.selected {{ outline: 2px solid rgba(216,180,95,.75); }}
    .match-card.team-hit {{
      border-color: rgba(38,198,184,.78);
      background: linear-gradient(180deg, rgba(38,198,184,.16), rgba(255,255,255,.035));
    }}
    .match-card.dimmed {{
      opacity: .28;
      filter: grayscale(.55);
    }}
    .team-token {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      margin: 1px 0;
    }}
    .team-token.highlight {{
      background: rgba(38,198,184,.18);
      color: #dffdf8;
      box-shadow: inset 0 0 0 1px rgba(38,198,184,.55);
    }}
    .match-top {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
    }}
    .teams {{ font-weight: 800; font-size: 16px; line-height: 1.32; }}
    .pick {{
      margin-top: 10px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
    }}
    .pick strong {{ color: var(--gold); }}
    .location {{
      margin-top: 9px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    aside {{
      position: sticky;
      top: 180px;
      background: rgba(17,27,32,.88);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      max-height: calc(100vh - 198px);
      overflow: auto;
    }}
    aside h2, section h2 {{
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .detail-line {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin: 8px 0;
    }}
    .rating-row {{
      display: grid;
      grid-template-columns: 1fr 46px;
      gap: 10px;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,.07);
    }}
    .bar {{
      height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,.10);
      overflow: hidden;
      margin-top: 5px;
    }}
    .bar > i {{
      display: block;
      height: 100%;
      width: var(--w);
      background: linear-gradient(90deg, var(--teal), var(--gold));
    }}
    .sections {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr);
      gap: 18px;
      margin-top: 24px;
    }}
    section {{
      background: rgba(17,27,32,.78);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 12px 34px rgba(0,0,0,.22);
    }}
    .rec-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 12px;
    }}
    .rec-card {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.045);
      border-radius: 8px;
      padding: 12px;
    }}
    .rec-card h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .rec-card p {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin: 7px 0;
    }}
    .pill {{
      display: inline-block;
      border: 1px solid rgba(216,180,95,.42);
      color: var(--gold);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      margin-right: 5px;
      margin-bottom: 5px;
    }}
    .wide-section {{
      margin-top: 18px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .chart-tools {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .chart-wrap {{
      width: 100%;
      min-height: 460px;
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      background: rgba(255,255,255,.025);
      overflow: hidden;
    }}
    .value-svg {{
      width: 100%;
      height: min(64vw, 560px);
      min-height: 440px;
      display: block;
    }}
    .axis text {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .axis line, .axis path {{
      stroke: rgba(255,255,255,.20);
    }}
    .threshold-line {{
      stroke: var(--gold);
      stroke-width: 1.5;
      stroke-dasharray: 6 6;
      opacity: .9;
    }}
    .player-dot {{
      cursor: pointer;
      transition: opacity .16s ease, r .16s ease, stroke-width .16s ease;
    }}
    .player-dot.dimmed {{
      opacity: .16;
    }}
    .player-dot.active {{
      stroke: var(--gold);
      stroke-width: 3;
    }}
    .player-label {{
      fill: var(--ink);
      font-size: 11px;
      paint-order: stroke;
      stroke: rgba(9,16,19,.88);
      stroke-width: 3px;
      stroke-linejoin: round;
      pointer-events: none;
    }}
    .chart-note {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-top: 10px;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      margin-top: 22px;
    }}
    @media (max-width: 980px) {{
      .scoreboard {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .layout, .sections {{ grid-template-columns: 1fr; }}
      aside {{ position: static; max-height: none; }}
      header {{ position: static; }}
    }}
    @media (prefers-reduced-motion: no-preference) {{
      .match-card {{ transition: transform .16s ease, border-color .16s ease, background .16s ease; }}
      .match-card:hover {{ transform: translateY(-2px); border-color: rgba(216,180,95,.55); }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">World Cup 2026 Forecast Lab</div>
    <h1>Interactive Knockout Bracket</h1>
    <p class="lead">Latest notebook projection with actual R32 results, rendered Google lineup ratings, human tactical priors, ticket-watch recommendations, and a market-prior diagnostic view.</p>
    <div class="scoreboard">
      <div class="metric"><span>Projected Champion</span><strong>{champion}</strong></div>
      <div class="metric"><span>Projected Runner-up</span><strong>{runner_up}</strong></div>
      <div class="metric"><span>Google Rows</span><strong>{data["ratings"]["allRows"]:,}</strong></div>
      <div class="metric"><span>Scraped Matches</span><strong>{data["ratings"]["scrapedMatches"]} / 82</strong></div>
      <div class="metric"><span>Star Matches</span><strong>{data["ratings"]["starRows"]}</strong></div>
    </div>
  </header>
  <main>
    <div class="toolbar">
      <div class="group">
        <button class="active" data-dataset="live">Live Model</button>
        <button data-dataset="market">Market Diagnostic</button>
      </div>
      <div class="group">
        <select id="roundFilter" aria-label="Round filter">
          <option value="All">All rounds</option>
        </select>
        <select id="statusFilter" aria-label="Status filter">
          <option value="All">Actual + projected</option>
          <option value="actual">Actual only</option>
          <option value="projected">Projected only</option>
        </select>
      </div>
      <div class="group">
        <select id="teamHighlight" aria-label="Highlight country">
          <option value="">Highlight country</option>
        </select>
        <label class="toggle"><input id="focusToggle" type="checkbox"> Focus</label>
        <button id="clearHighlight" type="button">Clear</button>
      </div>
      <div class="group">
        <button id="bracketZoomOut" type="button">-</button>
        <span id="bracketZoomLabel" class="zoom-label">100%</span>
        <button id="bracketZoomIn" type="button">+</button>
        <button id="bracketZoomReset" type="button">Reset</button>
      </div>
      <input id="searchBox" placeholder="Search team, city, stadium, match number" aria-label="Search">
    </div>
    <div class="layout">
      <div class="bracket"><div id="rounds" class="rounds"></div></div>
      <aside id="details"></aside>
    </div>
    <div class="sections">
      <section>
        <h2>Games To Watch</h2>
        <div id="recommendations" class="rec-grid"></div>
      </section>
      <section>
        <h2>Knockout Player Form</h2>
        <div id="topPlayers"></div>
      </section>
    </div>
    <section style="margin-top:18px">
      <h2>Knockout Team Rating Averages</h2>
      <div id="teamForm"></div>
    </section>
    <section class="wide-section">
      <div class="section-head">
        <h2>Player Value vs Transfermarkt Value</h2>
        <div class="chart-tools">
          <button id="valueZoomOut" type="button">Zoom out</button>
          <span id="valueZoomLabel" class="zoom-label">100%</span>
          <button id="valueZoomIn" type="button">Zoom in</button>
          <button id="valueZoomReset" type="button">Reset</button>
        </div>
      </div>
      <div id="valueChart" class="chart-wrap"></div>
      <div id="valueDetails" class="chart-note"></div>
    </section>
    <footer>
      Data note: Google ratings are from rendered match-card lineup panels stored locally after the July 2 sweep. Knockout team averages are used as forward signals where available; group-stage ratings are stored for analysis. The July 7 Polymarket outright snapshot is blended lightly into the notebook model, while Kalshi rows remain diagnostic.
    </footer>
  </main>
  <script>
    const DATA = {payload};
    const roundOrder = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Third-place Match", "Final"];
    const bracketRounds = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"];
    const state = {{
      dataset: "live",
      round: "All",
      status: "All",
      search: "",
      selected: null,
      highlightTeam: "",
      focusTeam: false,
      bracketZoom: 1,
      valueZoom: 1,
      selectedPlayer: null
    }};

    const pctNumber = value => Number(String(value || "0").replace("%", "")) || 0;
    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    const currentRows = () => state.dataset === "market" ? DATA.marketBracket : DATA.liveBracket;
    const valueRows = () => DATA.playerValue || [];

    function teamInRow(row, team) {{
      if (!team) return false;
      const target = String(team).toLowerCase();
      return [row.team_a, row.team_b, row.winner].some(value => String(value ?? "").toLowerCase() === target);
    }}

    function teamToken(name) {{
      const isHit = state.highlightTeam && String(name ?? "").toLowerCase() === state.highlightTeam.toLowerCase();
      return `<span class="team-token${{isHit ? " highlight" : ""}}">${{escapeHtml(name)}}</span>`;
    }}

    function collectTeams() {{
      const teams = new Set();
      [...DATA.liveBracket, ...DATA.marketBracket].forEach(row => {{
        [row.team_a, row.team_b, row.winner].forEach(team => {{
          if (team) teams.add(String(team));
        }});
      }});
      valueRows().forEach(row => {{
        if (row.team) teams.add(String(row.team));
      }});
      return [...teams].sort((a, b) => a.localeCompare(b));
    }}

    function setHighlight(team) {{
      state.highlightTeam = team || "";
      const select = document.querySelector("#teamHighlight");
      if (select && select.value !== state.highlightTeam) select.value = state.highlightTeam;
    }}

    function setZoom(field, value) {{
      state[field] = Math.max(0.75, Math.min(2.25, Number(value.toFixed(2))));
      renderAll(false);
    }}

    function initControls() {{
      const roundFilter = document.querySelector("#roundFilter");
      roundOrder.forEach(round => {{
        const opt = document.createElement("option");
        opt.value = round;
        opt.textContent = round;
        roundFilter.appendChild(opt);
      }});
      const teamHighlight = document.querySelector("#teamHighlight");
      collectTeams().forEach(team => {{
        const opt = document.createElement("option");
        opt.value = team;
        opt.textContent = team;
        teamHighlight.appendChild(opt);
      }});
      document.querySelectorAll("[data-dataset]").forEach(btn => {{
        btn.addEventListener("click", () => {{
          state.dataset = btn.dataset.dataset;
          document.querySelectorAll("[data-dataset]").forEach(b => b.classList.toggle("active", b === btn));
          renderAll();
        }});
      }});
      roundFilter.addEventListener("change", e => {{ state.round = e.target.value; renderAll(); }});
      document.querySelector("#statusFilter").addEventListener("change", e => {{ state.status = e.target.value; renderAll(); }});
      document.querySelector("#searchBox").addEventListener("input", e => {{ state.search = e.target.value.trim().toLowerCase(); renderAll(); }});
      teamHighlight.addEventListener("change", e => {{
        setHighlight(e.target.value);
        state.selectedPlayer = null;
        renderAll(false);
      }});
      document.querySelector("#focusToggle").addEventListener("change", e => {{
        state.focusTeam = e.target.checked;
        renderAll(false);
      }});
      document.querySelector("#clearHighlight").addEventListener("click", () => {{
        setHighlight("");
        state.focusTeam = false;
        state.selectedPlayer = null;
        document.querySelector("#focusToggle").checked = false;
        renderAll(false);
      }});
      document.querySelector("#bracketZoomOut").addEventListener("click", () => setZoom("bracketZoom", state.bracketZoom - 0.1));
      document.querySelector("#bracketZoomIn").addEventListener("click", () => setZoom("bracketZoom", state.bracketZoom + 0.1));
      document.querySelector("#bracketZoomReset").addEventListener("click", () => setZoom("bracketZoom", 1));
      document.querySelector("#valueZoomOut").addEventListener("click", () => setZoom("valueZoom", state.valueZoom - 0.15));
      document.querySelector("#valueZoomIn").addEventListener("click", () => setZoom("valueZoom", state.valueZoom + 0.15));
      document.querySelector("#valueZoomReset").addEventListener("click", () => setZoom("valueZoom", 1));
    }}

    function filteredRows() {{
      return currentRows().filter(row => {{
        if (state.round !== "All" && row.round !== state.round) return false;
        if (state.status !== "All" && row.status !== state.status) return false;
        if (state.search) {{
          const haystack = [row.match, row.round, row.team_a, row.team_b, row.winner, row.location, row.date, row.weekday].join(" ").toLowerCase();
          if (!haystack.includes(state.search)) return false;
        }}
        return true;
      }});
    }}

    function matchCard(row) {{
      const status = row.status || "projected";
      const prob = pctNumber(row.winner_probability);
      const selected = Number(state.selected) === Number(row.match) ? " selected" : "";
      const hit = state.highlightTeam && teamInRow(row, state.highlightTeam) ? " team-hit" : "";
      const dimmed = state.highlightTeam && state.focusTeam && !teamInRow(row, state.highlightTeam) ? " dimmed" : "";
      const note = row.result_note ? `<div class="location">${{escapeHtml(row.result_note)}}</div>` : "";
      return `<article class="match-card ${{status}}${{selected}}${{hit}}${{dimmed}}" data-match="${{row.match}}">
        <div class="match-top"><span>M${{row.match}} · ${{escapeHtml(row.weekday || "")}}</span><span>${{escapeHtml(status)}}</span></div>
        <div class="teams">${{teamToken(row.team_a)}}<br>vs<br>${{teamToken(row.team_b)}}</div>
        <div class="pick"><span>Pick: <strong>${{escapeHtml(row.winner)}}</strong></span><span>${{prob.toFixed(1)}}%</span></div>
        <div class="bar"><i style="--w:${{Math.max(4, Math.min(100, prob))}}%"></i></div>
        <div class="location">${{escapeHtml(row.date)}} · ${{escapeHtml(row.local_kickoff || "")}}<br>${{escapeHtml(row.location || "")}}</div>
        ${{note}}
      </article>`;
    }}

    function renderBracket() {{
      const rows = filteredRows();
      const host = document.querySelector("#rounds");
      host.style.transform = `scale(${{state.bracketZoom}})`;
      host.style.width = `${{Math.round(100 / state.bracketZoom)}}%`;
      document.querySelector("#bracketZoomLabel").textContent = `${{Math.round(state.bracketZoom * 100)}}%`;
      host.innerHTML = bracketRounds.map(round => {{
        const cards = rows.filter(r => r.round === round).sort((a, b) => Number(a.match) - Number(b.match));
        return `<section class="round-col"><h2>${{escapeHtml(round)}}</h2>${{cards.length ? cards.map(matchCard).join("") : `<div class="detail-line">No matches shown by current filters.</div>`}}</section>`;
      }}).join("");
      host.querySelectorAll(".match-card").forEach(card => {{
        card.addEventListener("click", () => {{
          state.selected = Number(card.dataset.match);
          renderAll(false);
        }});
      }});
      if (!state.selected && rows.length) state.selected = Number(rows[0].match);
    }}

    function renderDetails() {{
      const rows = currentRows();
      const row = rows.find(r => Number(r.match) === Number(state.selected)) || rows[0];
      const ratings = DATA.ratings.ratingsByMatch[String(row?.match)] || [];
      const teamAvgs = DATA.ratings.teamAveragesByMatch[String(row?.match)] || [];
      const ratingHtml = ratings.length ? ratings.map(r => `
        <div class="rating-row">
          <div><strong>${{escapeHtml(r.player)}}</strong><div class="detail-line">${{escapeHtml(r.team)}} · ${{escapeHtml(r.match)}}</div></div>
          <div>${{Number(r.rating).toFixed(1)}}</div>
        </div>`).join("") : `<div class="detail-line">No Google lineup rating rows for this match yet.</div>`;
      const avgHtml = teamAvgs.length ? teamAvgs.map(r => `
        <div class="rating-row">
          <div><strong>${{escapeHtml(r.team)}}</strong><div class="bar"><i style="--w:${{Math.max(1, Math.min(100, Number(r.avg_rating) * 10))}}%"></i></div></div>
          <div>${{Number(r.avg_rating).toFixed(2)}}</div>
        </div>`).join("") : "";
      document.querySelector("#details").innerHTML = `<h2>M${{escapeHtml(row?.match)}} Details</h2>
        <div class="detail-line">${{escapeHtml(row?.round)}} · ${{escapeHtml(row?.date)}} · ${{escapeHtml(row?.weekday)}} · ${{escapeHtml(row?.local_kickoff || "")}}</div>
        <div class="teams">${{escapeHtml(row?.team_a)}} vs ${{escapeHtml(row?.team_b)}}</div>
        <div class="detail-line">Location: ${{escapeHtml(row?.location || "")}}</div>
        <div class="detail-line">Winner: <strong>${{escapeHtml(row?.winner)}}</strong> (${{escapeHtml(row?.winner_probability)}}) · ${{escapeHtml(row?.status || "")}}</div>
        ${{row?.result_note ? `<div class="detail-line">Result: ${{escapeHtml(row.result_note)}}</div>` : ""}}
        <h2 style="margin-top:18px">Team Averages</h2>${{avgHtml || `<div class="detail-line">Projected match; ratings will appear after the match is played and scraped.</div>`}}
        <h2 style="margin-top:18px">Top Rated Players</h2>${{ratingHtml}}`;
    }}

    function renderRecommendations() {{
      const host = document.querySelector("#recommendations");
      const recs = DATA.recommendations.slice(0, 6);
      host.innerHTML = recs.map(r => `<article class="rec-card">
        <span class="pill">Rank ${{r.rank}}</span><span class="pill">M${{r.match}}</span><span class="pill">${{escapeHtml(r.weekday)}}</span>
        <h3>${{escapeHtml(r.possible_matchup)}}</h3>
        <p><strong>${{escapeHtml(r.round)}}</strong> · ${{escapeHtml(r.city)}} · ${{escapeHtml(r.stadium)}}</p>
        <p>${{escapeHtml(r.good_seat_price_estimate_usd)}}</p>
        <p>${{escapeHtml(r.reason)}}</p>
      </article>`).join("");
    }}

    function renderTopPlayers() {{
      const host = document.querySelector("#topPlayers");
      host.innerHTML = DATA.ratings.topR32Players.slice(0, 14).map(r => `
        <div class="rating-row">
          <div><strong>${{escapeHtml(r.player)}}</strong><div class="detail-line">M${{Number(r.match_no).toFixed(0)}} · ${{escapeHtml(r.team)}} · ${{escapeHtml(r.match)}}</div></div>
          <div>${{Number(r.rating).toFixed(1)}}</div>
        </div>`).join("");
    }}

    function renderTeamForm() {{
      const host = document.querySelector("#teamForm");
      host.innerHTML = DATA.ratings.r32TeamForm.slice(0, 26).map(r => {{
        const width = Math.max(5, Math.min(100, Number(r.avg_rating) * 10));
        return `<div class="rating-row">
          <div><strong>${{escapeHtml(r.team)}}</strong><div class="detail-line">M${{Number(r.match_no).toFixed(0)}} · ${{escapeHtml(r.match)}} · ${{r.players_rated}} players</div><div class="bar"><i style="--w:${{width}}%"></i></div></div>
          <div>${{Number(r.avg_rating).toFixed(2)}}</div>
        </div>`;
      }}).join("");
    }}

    function playerColor(row) {{
      const highValue = Number(row.transfermarkt_value_eur_m) >= 75;
      const highRating = Number(row.world_cup_rating) >= 7;
      if (state.highlightTeam && row.team === state.highlightTeam) return "#d8b45f";
      if (highValue && highRating) return "#26c6b8";
      if (highValue) return "#6da8ff";
      if (highRating) return "#e35d5b";
      return "#9fb0ad";
    }}

    function renderValueChart() {{
      const host = document.querySelector("#valueChart");
      const details = document.querySelector("#valueDetails");
      const rows = valueRows().filter(row => Number.isFinite(Number(row.transfermarkt_value_eur_m)) && Number.isFinite(Number(row.world_cup_rating)));
      document.querySelector("#valueZoomLabel").textContent = `${{Math.round(state.valueZoom * 100)}}%`;
      if (!rows.length) {{
        host.innerHTML = `<div class="detail-line" style="padding:18px">No player-value rows found. Regenerate the notebook after rating extraction to populate this chart.</div>`;
        details.textContent = "";
        return;
      }}

      const width = 1120;
      const height = 560;
      const margin = {{ top: 34, right: 46, bottom: 72, left: 74 }};
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const maxValue = Math.max(100, ...rows.map(row => Number(row.transfermarkt_value_eur_m))) * 1.08;
      const minRating = Math.max(5, Math.floor((Math.min(...rows.map(row => Number(row.world_cup_rating))) - 0.25) * 2) / 2);
      const maxRating = Math.min(9, Math.ceil((Math.max(...rows.map(row => Number(row.world_cup_rating))) + 0.25) * 2) / 2);
      const x = value => margin.left + (Number(value) / maxValue) * plotW;
      const y = value => margin.top + ((maxRating - Number(value)) / (maxRating - minRating)) * plotH;
      const xTicks = [0, 25, 50, 75, 100, 125, 150, 175, 200].filter(value => value <= maxValue);
      const yTicks = [];
      for (let tick = minRating; tick <= maxRating + 0.001; tick += 0.5) yTicks.push(Number(tick.toFixed(1)));
      const selectedKey = state.selectedPlayer;

      const grid = [
        ...xTicks.map(value => `<g class="axis"><line x1="${{x(value)}}" x2="${{x(value)}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}" opacity=".35"></line><text x="${{x(value)}}" y="${{height - margin.bottom + 24}}" text-anchor="middle">${{value}}</text></g>`),
        ...yTicks.map(value => `<g class="axis"><line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y(value)}}" y2="${{y(value)}}" opacity=".35"></line><text x="${{margin.left - 12}}" y="${{y(value) + 4}}" text-anchor="end">${{value.toFixed(1)}}</text></g>`)
      ].join("");
      const labels = rows.filter(row => {{
        const key = `${{row.team}}|${{row.player}}`;
        return key === selectedKey || row.team === state.highlightTeam || Number(row.transfermarkt_value_eur_m) >= 120 || Number(row.world_cup_rating) >= 7.55;
      }}).map(row => `<text class="player-label" x="${{x(row.transfermarkt_value_eur_m) + 8}}" y="${{y(row.world_cup_rating) - 8}}">${{escapeHtml(row.player)}}</text>`).join("");
      const points = rows.map((row, index) => {{
        const key = `${{row.team}}|${{row.player}}`;
        const active = key === selectedKey || row.team === state.highlightTeam;
        const dimmed = state.highlightTeam && state.focusTeam && row.team !== state.highlightTeam;
        const radius = active ? 7 : 5;
        return `<circle class="player-dot${{dimmed ? " dimmed" : ""}}${{active ? " active" : ""}}" data-index="${{index}}" cx="${{x(row.transfermarkt_value_eur_m)}}" cy="${{y(row.world_cup_rating)}}" r="${{radius}}" fill="${{playerColor(row)}}"><title>${{escapeHtml(row.player)}} &middot; ${{escapeHtml(row.team)}} &middot; EUR ${{Number(row.transfermarkt_value_eur_m).toFixed(0)}}m &middot; ${{Number(row.world_cup_rating).toFixed(2)}}</title></circle>`;
      }}).join("");

      host.innerHTML = `<svg class="value-svg" style="width:${{Math.round(state.valueZoom * 100)}}%; min-width:100%" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Player rating versus Transfermarkt value scatter plot">
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="transparent"></rect>
        <text x="${{margin.left}}" y="22" fill="#edf4f1" font-size="15" font-weight="700">World Cup rating vs market value</text>
        <text x="${{width - margin.right}}" y="22" fill="#9fb0ad" font-size="12" text-anchor="end">Click a dot to highlight its country across the page</text>
        ${{grid}}
        <line class="threshold-line" x1="${{x(75)}}" x2="${{x(75)}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}"></line>
        <line class="threshold-line" x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y(7)}}" y2="${{y(7)}}"></line>
        <text x="${{x(75) + 8}}" y="${{margin.top + 16}}" fill="#d8b45f" font-size="12">EUR 75m</text>
        <text x="${{width - margin.right - 8}}" y="${{y(7) - 8}}" fill="#d8b45f" font-size="12" text-anchor="end">7.0 rating</text>
        ${{points}}
        ${{labels}}
        <text x="${{width / 2}}" y="${{height - 22}}" fill="#9fb0ad" font-size="13" text-anchor="middle">Transfermarkt value, EUR millions</text>
        <text x="18" y="${{height / 2}}" fill="#9fb0ad" font-size="13" text-anchor="middle" transform="rotate(-90 18 ${{height / 2}})">Google lineup post-match rating</text>
      </svg>`;

      host.querySelectorAll(".player-dot").forEach(dot => {{
        dot.addEventListener("click", () => {{
          const row = rows[Number(dot.dataset.index)];
          state.selectedPlayer = `${{row.team}}|${{row.player}}`;
          setHighlight(row.team);
          renderAll(false);
        }});
      }});

      const selected = rows.find(row => `${{row.team}}|${{row.player}}` === state.selectedPlayer);
      if (selected) {{
        details.innerHTML = `<strong>${{escapeHtml(selected.player)}}</strong>, ${{escapeHtml(selected.team)}}: EUR ${{Number(selected.transfermarkt_value_eur_m).toFixed(0)}}m, World Cup rating ${{Number(selected.world_cup_rating).toFixed(2)}} across ${{selected.rated_matches}} rated match(es), ${{escapeHtml(selected.rating_method)}}.`;
      }} else if (state.highlightTeam) {{
        const teamRows = rows.filter(row => row.team === state.highlightTeam);
        const avgRating = teamRows.reduce((sum, row) => sum + Number(row.world_cup_rating), 0) / Math.max(1, teamRows.length);
        const totalValue = teamRows.reduce((sum, row) => sum + Number(row.transfermarkt_value_eur_m), 0);
        details.innerHTML = `<strong>${{escapeHtml(state.highlightTeam)}}</strong>: ${{teamRows.length}} tracked star player(s), EUR ${{totalValue.toFixed(0)}}m combined snapshot value, ${{avgRating.toFixed(2)}} average World Cup rating.`;
      }} else {{
        details.innerHTML = `Thresholds: high market value is EUR 75m or above; high performance is 7.0 or above. Use the country selector or click a dot to link the chart back to the bracket.`;
      }}
    }}

    function renderAll(keepSelection = true) {{
      if (!keepSelection) state.selected = state.selected;
      renderBracket();
      renderDetails();
      renderRecommendations();
      renderTopPlayers();
      renderTeamForm();
      renderValueChart();
    }}

    initControls();
    renderAll();
  </script>
</body>
</html>"""


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    live_table, market_table, recommendations, summary = load_notebook_tables()
    summary_map = dict(zip(summary.get("result", []), summary.get("team", [])))
    data = {
        "summaryMap": summary_map,
        "liveBracket": dataframe_to_records(live_table),
        "marketBracket": dataframe_to_records(market_table),
        "recommendations": dataframe_to_records(recommendations),
        "ratings": load_rating_data(),
        "playerValue": load_player_value_data(),
    }
    OUT.write_text(render_html(data), encoding="utf-8")
    print(OUT)
    print(f"live_matches={len(data['liveBracket'])}")
    print(f"recommendations={len(data['recommendations'])}")
    print(f"ratings_rows={data['ratings']['allRows']}")


if __name__ == "__main__":
    main()
