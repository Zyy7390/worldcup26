from __future__ import annotations

from pathlib import Path
import ast
import html
import io
import json
import re
import unicodedata

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "worldcup_2026_prediction_bracket.ipynb"
GOOGLE_DATA_DIR = ROOT / "data" / "google"
SOFASCORE_DATA_DIR = ROOT / "data" / "sofascore"
OUTPUT_DIR = ROOT / "outputs"
OUT = OUTPUT_DIR / "worldcup_2026_interactive_bracket.html"
MODEL_SOURCE = ROOT / "scripts" / "update_worldcup26_notebook_live.py"


MAJOR_CLUB_BY_PLAYER = {
    "Kylian Mbappe": "Real Madrid",
    "Jude Bellingham": "Real Madrid",
    "Vinicius Junior": "Real Madrid",
    "Rodrygo": "Real Madrid",
    "Federico Valverde": "Real Madrid",
    "Brahim Diaz": "Real Madrid",
    "Lamine Yamal": "Barcelona",
    "Pedri": "Barcelona",
    "Erling Haaland": "Manchester City",
    "Rodri": "Manchester City",
    "Jeremy Doku": "Manchester City",
    "Josko Gvardiol": "Manchester City",
    "Bukayo Saka": "Arsenal",
    "Martin Odegaard": "Arsenal",
    "Harry Kane": "Bayern Munich",
    "Jamal Musiala": "Bayern Munich",
    "Michael Olise": "Bayern Munich",
    "Ousmane Dembele": "Paris Saint-Germain",
    "Achraf Hakimi": "Paris Saint-Germain",
    "Lionel Messi": "Inter Miami",
    "Lautaro Martinez": "Inter Milan",
    "Hakan Calhanoglu": "Inter Milan",
    "Mohamed Salah": "Liverpool",
    "Virgil van Dijk": "Liverpool",
    "Cody Gakpo": "Liverpool",
    "Luis Diaz": "Liverpool",
    "Darwin Nunez": "Liverpool",
    "Julian Alvarez": "Atletico Madrid",
    "Alexander Sorloth": "Atletico Madrid",
    "Moises Caicedo": "Chelsea",
    "Nicolas Jackson": "Chelsea",
    "Christian Pulisic": "AC Milan",
    "Bruno Fernandes": "Manchester United",
    "Cristiano Ronaldo": "Al Nassr",
    "Florian Wirtz": "Bayer Leverkusen",
    "Granit Xhaka": "Bayer Leverkusen",
    "Patrik Schick": "Bayer Leverkusen",
    "Jonathan David": "Lille",
    "Alphonso Davies": "Bayern Munich",
}


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
            try:
                is_missing = pd.isna(value)
            except (TypeError, ValueError):
                is_missing = False
            if isinstance(is_missing, bool) and is_missing:
                clean[key] = None
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def json_safe(value):
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, bool) and is_missing:
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def normalize_match_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def result_match_signature(value: object) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value or "").split(";")[0]).strip()
    match = re.search(r"(.+?)\s+(\d+)\s*[-\u2013]\s*(\d+)\s+(.+)", text)
    if not match:
        return ""
    left = normalize_match_name(match.group(1))
    right = normalize_match_name(match.group(4))
    goals_left = int(match.group(2))
    goals_right = int(match.group(3))
    if left <= right:
        return f"{left}|{goals_left}-{goals_right}|{right}"
    return f"{right}|{goals_right}-{goals_left}|{left}"


def result_lookup_keys(value: object) -> list[str]:
    keys = [f"name:{normalize_match_name(value)}"]
    signature = result_match_signature(value)
    if signature:
        keys.append(f"score:{signature}")
    return keys


def load_player_form_signals() -> pd.DataFrame:
    if not MODEL_SOURCE.exists():
        return pd.DataFrame()
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"player_form_signals_data\s*=\s*\[(.*?)\]\s*player_form_signals\s*=",
        source,
        flags=re.S,
    )
    if not match:
        return pd.DataFrame()
    rows = ast.literal_eval("[" + match.group(1) + "]")
    signals = pd.DataFrame(
        rows,
        columns=["match", "team", "player", "rating_proxy", "importance_weight", "source_type", "note"],
    )
    signals["rating_proxy"] = pd.to_numeric(signals["rating_proxy"], errors="coerce")
    signals["importance_weight"] = pd.to_numeric(signals["importance_weight"], errors="coerce").fillna(0)
    signals["match_key"] = signals["match"].map(normalize_match_name)
    signals["match_signature"] = signals["match"].map(result_match_signature)
    return signals[signals["rating_proxy"].notna()].copy()


def load_group_result_backbone() -> pd.DataFrame:
    if not MODEL_SOURCE.exists():
        return pd.DataFrame()
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"played_results_data\s*=\s*\[(.*?)\]\s*played_results\s*=",
        source,
        flags=re.S,
    )
    if not match:
        return pd.DataFrame()
    rows = ast.literal_eval("[" + match.group(1) + "]")
    group_results = pd.DataFrame(
        rows,
        columns=["group", "team_a", "team_b", "goals_a", "goals_b", "date"],
    )
    group_results["stage"] = "Group Stage"
    group_results["status"] = "actual"
    group_results["match_no"] = None
    group_results["match_id"] = [
        f"G{row.group}-{idx + 1:02d}" for idx, row in enumerate(group_results.itertuples())
    ]
    group_results["match"] = group_results.apply(
        lambda row: f"{row['team_a']} {int(row['goals_a'])}-{int(row['goals_b'])} {row['team_b']}",
        axis=1,
    )
    group_results["winner"] = group_results.apply(
        lambda row: row["team_a"]
        if row["goals_a"] > row["goals_b"]
        else row["team_b"]
        if row["goals_b"] > row["goals_a"]
        else "Draw",
        axis=1,
    )
    for column in ["weekday", "local_kickoff", "location", "slot", "result_note", "winner_probability"]:
        group_results[column] = None
    return group_results


def parse_knockout_score(row: pd.Series) -> tuple[int | None, int | None]:
    note = str(row.get("result_note") or "").strip()
    if not note or note.lower() == "nan":
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


def build_actual_match_backbone(live_table: pd.DataFrame) -> pd.DataFrame:
    group_results = load_group_result_backbone()
    live_actual = live_table[live_table.get("status", "").eq("actual")].copy()
    if not live_actual.empty:
        live_actual["stage"] = live_actual["round"]
        live_actual["goals_a"], live_actual["goals_b"] = zip(
            *live_actual.apply(parse_knockout_score, axis=1)
        )
        live_actual["match_id"] = live_actual["match"].apply(lambda value: f"M{int(value)}")
        live_actual["match_no"] = live_actual["match"]
        live_actual["match"] = live_actual.apply(
            lambda row: f"{row['team_a']} {int(row['goals_a'])}-{int(row['goals_b'])} {row['team_b']}"
            if pd.notna(row["goals_a"]) and pd.notna(row["goals_b"])
            else str(row.get("result_note") or f"{row['team_a']} vs {row['team_b']}"),
            axis=1,
        )
    columns = [
        "match_id",
        "match_no",
        "stage",
        "date",
        "weekday",
        "local_kickoff",
        "location",
        "slot",
        "status",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "match",
        "winner",
        "winner_probability",
    ]
    parts = []
    for frame in [group_results, live_actual]:
        if not frame.empty:
            parts.append(frame.reindex(columns=columns))
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "match_id"])


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


def load_direct_player_ratings(match_backbone: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one direct player-rating source per match, preferring SofaScore."""
    all_path = GOOGLE_DATA_DIR / "google_worldcup_all_player_ratings.csv"
    sofa_path = SOFASCORE_DATA_DIR / "sofascore_player_ratings.csv"
    frames = []

    match_labels = pd.DataFrame()
    if match_backbone is not None and not match_backbone.empty:
        match_labels = match_backbone[["match_id", "match"]].drop_duplicates()

    if all_path.exists():
        google = pd.read_csv(all_path)
        if not google.empty:
            google["source"] = "Google"
            google["player_display"] = google.get("player_display", google.get("player", ""))
            google["rating"] = pd.to_numeric(google["rating"], errors="coerce")
            frames.append(google)

    sofa_match_ids: set[str] = set()
    if sofa_path.exists():
        sofa = pd.read_csv(sofa_path)
        if not sofa.empty:
            sofa["source"] = "SofaScore"
            sofa["player_display"] = sofa["player"]
            sofa["rating"] = pd.to_numeric(sofa["rating"], errors="coerce")
            sofa["match_no"] = pd.to_numeric(sofa["match_no"], errors="coerce")
            if not match_labels.empty:
                sofa = sofa.merge(match_labels, on="match_id", how="left")
            if "match" not in sofa.columns:
                sofa["match"] = sofa["match_id"]
            sofa_match_ids = set(sofa.loc[sofa["rating"].notna(), "match_id"].dropna().astype(str))
            frames.append(sofa)

    if not frames:
        return pd.DataFrame()

    ratings = pd.concat(frames, ignore_index=True, sort=False)
    ratings["match_id"] = ratings["match_id"].astype(str)
    if sofa_match_ids:
        ratings = ratings[(ratings["source"].eq("SofaScore")) | (~ratings["match_id"].isin(sofa_match_ids))]
    return ratings[ratings["rating"].notna()].copy()


def load_sofascore_team_stats() -> dict[tuple[str, str], dict]:
    stats_path = SOFASCORE_DATA_DIR / "sofascore_team_match_stats.csv"
    map_path = SOFASCORE_DATA_DIR / "sofascore_match_map.csv"
    if not stats_path.exists() or not map_path.exists():
        return {}

    stats = pd.read_csv(stats_path)
    match_map = pd.read_csv(map_path)
    if stats.empty or match_map.empty:
        return {}

    match_map = match_map[["match_id", "team_a", "team_b", "orientation"]].drop_duplicates("match_id")
    stats = stats.merge(match_map, on="match_id", how="left", suffixes=("", "_map"))
    metrics = {
        "expectedGoals": "xg",
        "shotsOnGoal": "shots_on_target",
        "goalkeeperSaves": "saves",
        "totalTackle": "tackles",
    }
    team_stats: dict[tuple[str, str], dict] = {}
    all_stats = stats[stats["period"].eq("ALL") & stats["stat_key"].isin(metrics)]
    for _, row in all_stats.iterrows():
        stat_name = metrics[row["stat_key"]]
        teams = [row.get("team_a_map") or row.get("team_a"), row.get("team_b_map") or row.get("team_b")]
        if row.get("orientation") == "reversed":
            values = [row.get("away_value"), row.get("home_value")]
        else:
            values = [row.get("home_value"), row.get("away_value")]
        for team, value in zip(teams, values):
            key = (str(row["match_id"]), team)
            team_stats.setdefault(key, {})[stat_name] = pd.to_numeric(value, errors="coerce")
    return team_stats


def load_rating_data(live_table: pd.DataFrame):
    summary_path = GOOGLE_DATA_DIR / "google_lineup_rating_extraction_summary.csv"
    star_path = GOOGLE_DATA_DIR / "google_lineup_player_ratings.csv"

    ratings = load_direct_player_ratings(build_actual_match_backbone(live_table))
    extraction_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    stars = pd.read_csv(star_path) if star_path.exists() else pd.DataFrame()

    ratings_by_match = {}
    team_avgs_by_match = {}
    form_signals_by_match = {}
    form_team_avgs_by_match = {}
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
                [["team", "player_display", "rating", "match", "source"]]
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
            [["match_no", "match", "team", "player_display", "rating", "source"]]
            .head(24)
            .rename(columns={"player_display": "player"})
        )
        knockout_team_form = dataframe_to_records(
            knockout.groupby(["match_no", "match", "team"], as_index=False)
            .agg(avg_rating=("rating", "mean"), players_rated=("player_display", "nunique"))
            .sort_values("avg_rating", ascending=False)
            .round(3)
        )

    form_signals = load_player_form_signals()
    if not form_signals.empty:
        backbone = build_actual_match_backbone(live_table)
        if not backbone.empty:
            match_lookup = backbone[pd.notna(backbone["match_no"])].copy()
            match_lookup["match_key"] = match_lookup["match"].map(normalize_match_name)
            match_lookup["match_signature"] = match_lookup["match"].map(result_match_signature)
            signals_with_id = form_signals.reset_index().rename(columns={"index": "_signal_id"})
            by_name = signals_with_id.merge(
                match_lookup[["match_key", "match_no"]],
                on="match_key",
                how="inner",
            )
            matched_signal_ids = set(by_name["_signal_id"])
            remaining = signals_with_id[~signals_with_id["_signal_id"].isin(matched_signal_ids)]
            by_score = pd.DataFrame()
            if not remaining.empty:
                by_score = remaining[remaining["match_signature"].ne("")].merge(
                    match_lookup[["match_signature", "match_no"]][match_lookup["match_signature"].ne("")],
                    on="match_signature",
                    how="inner",
                )
            form_for_ko = pd.concat([by_name, by_score], ignore_index=True)
            for match_no, group in form_for_ko.groupby("match_no"):
                match_key = str(int(match_no))
                rows = (
                    group.sort_values(["rating_proxy", "importance_weight"], ascending=False)
                    [["team", "player", "rating_proxy", "source_type", "note", "match"]]
                    .head(12)
                    .rename(columns={"rating_proxy": "rating", "source_type": "source"})
                    .round(2)
                )
                form_signals_by_match[match_key] = dataframe_to_records(rows)

                team_rows = []
                for team, team_group in group.groupby("team"):
                    weights = team_group["importance_weight"].fillna(0)
                    if weights.sum() > 0:
                        avg_rating = (team_group["rating_proxy"] * weights).sum() / weights.sum()
                    else:
                        avg_rating = team_group["rating_proxy"].mean()
                    team_rows.append({
                        "team": team,
                        "avg_rating": round(float(avg_rating), 3),
                        "signals": int(len(team_group)),
                    })
                form_team_avgs_by_match[match_key] = sorted(
                    team_rows,
                    key=lambda item: item["avg_rating"],
                    reverse=True,
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
        "formSignalsByMatch": form_signals_by_match,
        "formTeamAveragesByMatch": form_team_avgs_by_match,
        "topR32Players": top_knockout_players,
        "r32TeamForm": knockout_team_form,
        "coverage": coverage,
        "allRows": int(len(ratings)),
        "scrapedMatches": int(ratings["match_id"].nunique()) if not ratings.empty else 0,
        "starRows": int(len(stars)),
        "starSummary": star_summary,
    }


def normalize_person_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def load_transfermarkt_value_frame() -> pd.DataFrame:
    if not MODEL_SOURCE.exists():
        return pd.DataFrame()
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"transfermarkt_values_data\s*=\s*\[(.*?)\]\s*star_player_values",
        source,
        flags=re.S,
    )
    if not match:
        return pd.DataFrame()
    values = ast.literal_eval("[" + match.group(1) + "]")
    return pd.DataFrame(values, columns=["team", "player", "transfermarkt_value_eur_m"])


def build_player_match_info(match_backbone: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if match_backbone is None or match_backbone.empty:
        return pd.DataFrame()
    for _, row in match_backbone.iterrows():
        for team, opponent, goals_for, goals_against in [
            (row["team_a"], row["team_b"], row["goals_a"], row["goals_b"]),
            (row["team_b"], row["team_a"], row["goals_b"], row["goals_a"]),
        ]:
            rows.append({
                "match_id": str(row["match_id"]),
                "team": team,
                "opponent": opponent,
                "match": row["match"],
                "stage": row["stage"],
                "date": row["date"],
                "match_no": row.get("match_no"),
                "score": f"{int(goals_for)}-{int(goals_against)}"
                if pd.notna(goals_for) and pd.notna(goals_against)
                else "",
            })
    return pd.DataFrame(rows)


def load_player_value_data(match_backbone: pd.DataFrame) -> tuple[list[dict], dict]:
    value_df = load_transfermarkt_value_frame()
    sofa_path = SOFASCORE_DATA_DIR / "sofascore_player_ratings.csv"
    if value_df.empty or not sofa_path.exists():
        return [], {}

    ratings = pd.read_csv(sofa_path)
    required = {"match_id", "team", "player", "rating", "minutes_played", "goals", "assists"}
    if ratings.empty or not required.issubset(ratings.columns):
        return [], {}

    value_df["player_key"] = value_df["player"].map(normalize_person_name)
    ratings["player_key"] = ratings["player"].map(normalize_person_name)
    ratings["match_id"] = ratings["match_id"].astype(str)
    numeric_columns = [
        "rating",
        "minutes_played",
        "goals",
        "assists",
        "expected_goals",
        "expected_assists",
        "shots_on_target",
        "tackles",
        "saves",
    ]
    for column in numeric_columns:
        if column in ratings.columns:
            ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
        else:
            ratings[column] = None

    tracked = ratings.merge(
        value_df.rename(columns={"player": "value_player"}),
        on=["team", "player_key"],
        how="inner",
    )
    if tracked.empty:
        return [], {}

    match_info = build_player_match_info(match_backbone)
    if not match_info.empty:
        tracked = tracked.merge(match_info, on=["match_id", "team"], how="left", suffixes=("", "_info"))
    if "match" not in tracked.columns:
        tracked["match"] = tracked["match_id"]

    rows = []
    details = {}
    for (team, player), sub in tracked.groupby(["team", "value_player"], dropna=False):
        rated = sub[sub["rating"].notna()].copy()
        weighted = rated[rated["minutes_played"].fillna(0).gt(0)]
        if not weighted.empty and weighted["minutes_played"].sum() > 0:
            world_cup_rating = (weighted["rating"] * weighted["minutes_played"]).sum() / weighted["minutes_played"].sum()
            rating_method = "minutes-weighted"
        elif not rated.empty:
            world_cup_rating = rated["rating"].mean()
            rating_method = "simple average"
        else:
            world_cup_rating = None
            rating_method = "simple average"

        match_rows = []
        ordered = sub.sort_values(["date", "match_no", "match_id"], na_position="last")
        for _, match_row in ordered.iterrows():
            match_rows.append({
                "match_id": match_row.get("match_id"),
                "match_no": match_row.get("match_no"),
                "stage": match_row.get("stage"),
                "date": match_row.get("date"),
                "match": match_row.get("match") or match_row.get("match_id"),
                "opponent": match_row.get("opponent"),
                "score": match_row.get("score"),
                "player": match_row.get("player"),
                "minutes_played": match_row.get("minutes_played"),
                "rating": match_row.get("rating"),
                "goals": match_row.get("goals"),
                "assists": match_row.get("assists"),
                "expected_goals": match_row.get("expected_goals"),
                "expected_assists": match_row.get("expected_assists"),
                "shots_on_target": match_row.get("shots_on_target"),
                "tackles": match_row.get("tackles"),
                "saves": match_row.get("saves"),
            })

        summary = {
            "team": team,
            "player": player,
            "world_cup_rating": world_cup_rating,
            "rated_matches": int(rated["match_id"].nunique()),
            "appearances": int(sub.loc[sub["minutes_played"].fillna(0).gt(0) | sub["rating"].notna(), "match_id"].nunique()),
            "rating_method": rating_method,
            "total_minutes": sub["minutes_played"].fillna(0).sum(),
            "goals": sub["goals"].fillna(0).sum(),
            "assists": sub["assists"].fillna(0).sum(),
            "expected_goals": sub["expected_goals"].fillna(0).sum(),
            "expected_assists": sub["expected_assists"].fillna(0).sum(),
            "transfermarkt_value_eur_m": sub["transfermarkt_value_eur_m"].iloc[0],
            "sofascore_player": sub["player"].dropna().iloc[0] if sub["player"].notna().any() else player,
        }
        summary["club"] = MAJOR_CLUB_BY_PLAYER.get(player, "Other / untagged")
        summary["value_tier"] = "high value" if summary["transfermarkt_value_eur_m"] >= 75 else "lower value"
        summary["performance_tier"] = "high performance" if world_cup_rating is not None and world_cup_rating >= 7.0 else "lower performance"
        summary["detail_key"] = f"{team}|{player}"
        rows.append(summary)
        details[summary["detail_key"]] = {**summary, "matches": match_rows}

    performance = pd.DataFrame(rows)
    performance = performance[performance["world_cup_rating"].notna()].copy()
    if performance.empty:
        return [], {}
    performance["club"] = performance["player"].map(MAJOR_CLUB_BY_PLAYER).fillna("Other / untagged")
    performance["value_tier"] = performance["transfermarkt_value_eur_m"].ge(75).map({True: "high value", False: "lower value"})
    performance["performance_tier"] = performance["world_cup_rating"].ge(7.0).map({True: "high performance", False: "lower performance"})

    rows_out = dataframe_to_records(
        performance.sort_values(["team", "transfermarkt_value_eur_m"], ascending=[True, False]).round(3)
    )
    return rows_out, json_safe(details)


def load_team_performance_data(live_table: pd.DataFrame) -> dict:
    match_backbone = build_actual_match_backbone(live_table)
    star_path = GOOGLE_DATA_DIR / "google_lineup_player_ratings.csv"
    if match_backbone.empty:
        return {"summary": [], "matches": [], "note": "No actual match backbone found."}

    ratings = load_direct_player_ratings(match_backbone)
    sofa_stats = load_sofascore_team_stats()
    team_rating_metrics: dict[tuple[str, str], dict] = {}
    top_by_team_match: dict[tuple[str, str], list[dict]] = {}
    if not ratings.empty:
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        for (match_id, team), group in ratings[ratings["rating"].notna()].groupby(["match_id", "team"]):
            players_rated = group["player_display"].nunique()
            team_rating_metrics[(str(match_id), team)] = {
                "team_avg_rating": round(float(group["rating"].mean()), 3),
                "players_rated": int(players_rated),
            }
            top_by_team_match[(str(match_id), team)] = dataframe_to_records(
                group.sort_values("rating", ascending=False)
                [["player_display", "rating", "source"]]
                .head(3)
                .rename(columns={"player_display": "player"})
                .round(2)
            )

    stars_by_team_match: dict[tuple[str, str], list[dict]] = {}
    star_summary = pd.DataFrame()
    if star_path.exists():
        stars = pd.read_csv(star_path)
        if not stars.empty:
            stars["rating"] = pd.to_numeric(stars["rating"], errors="coerce")
            stars["match_key"] = stars["match"].map(normalize_match_name)
            for (match_key, team), group in stars[stars["rating"].notna()].groupby(["match_key", "team"]):
                stars_by_team_match[(match_key, team)] = dataframe_to_records(
                    group.sort_values("rating", ascending=False)
                    [["player", "rating"]]
                    .head(4)
                    .round(2)
                )
            star_summary = (
                stars.groupby("team", as_index=False)
                .agg(star_avg_rating=("rating", "mean"), star_rows=("rating", "count"))
                .round(3)
            )

    form_signals = load_player_form_signals()
    proxy_by_team_match: dict[tuple[str, str], list[dict]] = {}
    proxy_metrics: dict[tuple[str, str], dict] = {}
    if not form_signals.empty:
        def add_proxy_rows(lookup_key: str, team: str, group: pd.DataFrame) -> None:
            ordered = group.sort_values(["rating_proxy", "importance_weight"], ascending=False)
            proxy_by_team_match[(lookup_key, team)] = dataframe_to_records(
                ordered[["player", "rating_proxy", "source_type", "note"]]
                .rename(columns={"rating_proxy": "rating", "source_type": "source"})
                .head(5)
                .round(2)
            )
            weights = group["importance_weight"].fillna(0)
            if weights.sum() > 0:
                proxy_avg = (group["rating_proxy"] * weights).sum() / weights.sum()
            else:
                proxy_avg = group["rating_proxy"].mean()
            proxy_metrics[(lookup_key, team)] = {
                "proxy_avg_rating": round(float(proxy_avg), 3),
                "proxy_signals": int(len(group)),
            }

        for (match_key, team), group in form_signals.groupby(["match_key", "team"]):
            add_proxy_rows(f"name:{match_key}", team, group)
        signature_signals = form_signals[form_signals["match_signature"].ne("")]
        for (match_signature, team), group in signature_signals.groupby(["match_signature", "team"]):
            add_proxy_rows(f"score:{match_signature}", team, group)

    match_rows = []
    for _, row in match_backbone.iterrows():
        for team, opponent, goals_for, goals_against in [
            (row["team_a"], row["team_b"], row["goals_a"], row["goals_b"]),
            (row["team_b"], row["team_a"], row["goals_b"], row["goals_a"]),
        ]:
            match_key = normalize_match_name(row["match"])
            rating_key = (str(row["match_id"]), team)
            opponent_key = (str(row["match_id"]), opponent)
            metrics = team_rating_metrics.get(rating_key, {})
            stats_for = sofa_stats.get(rating_key, {})
            stats_against = sofa_stats.get(opponent_key, {})
            lookup_keys = result_lookup_keys(row["match"])
            proxy = next(
                (proxy_metrics[(lookup_key, team)] for lookup_key in lookup_keys if (lookup_key, team) in proxy_metrics),
                {},
            )
            top_players = top_by_team_match.get(rating_key, [])
            star_players = stars_by_team_match.get((match_key, team), [])
            proxy_players = next(
                (
                    proxy_by_team_match[(lookup_key, team)]
                    for lookup_key in lookup_keys
                    if (lookup_key, team) in proxy_by_team_match
                ),
                [],
            )
            goals_for_clean = int(goals_for) if pd.notna(goals_for) else None
            goals_against_clean = int(goals_against) if pd.notna(goals_against) else None
            match_rows.append({
                "team": team,
                "opponent": opponent,
                "match": row["match"],
                "match_id": row["match_id"],
                "match_no": None if pd.isna(row.get("match_no")) else int(row["match_no"]),
                "stage": row["stage"],
                "date": row["date"],
                "weekday": row.get("weekday"),
                "local_kickoff": row.get("local_kickoff"),
                "location": row.get("location"),
                "status": row.get("status"),
                "goals_for": goals_for_clean,
                "goals_against": goals_against_clean,
                "score": f"{goals_for_clean}-{goals_against_clean}"
                if goals_for_clean is not None and goals_against_clean is not None
                else "",
                "team_avg_rating": metrics.get("team_avg_rating"),
                "players_rated": metrics.get("players_rated"),
                "proxy_avg_rating": proxy.get("proxy_avg_rating"),
                "proxy_signals": proxy.get("proxy_signals", 0),
                "xg_for": stats_for.get("xg"),
                "xg_against": stats_against.get("xg"),
                "shots_on_target_for": stats_for.get("shots_on_target"),
                "shots_on_target_against": stats_against.get("shots_on_target"),
                "saves_for": stats_for.get("saves"),
                "saves_against": stats_against.get("saves"),
                "tackles_for": stats_for.get("tackles"),
                "tackles_against": stats_against.get("tackles"),
                "top_players": top_players,
                "star_players": star_players,
                "proxy_players": proxy_players,
                "has_player_ratings": bool(top_players or star_players),
                "has_form_signals": bool(proxy_players),
            })

    match_team = pd.DataFrame(match_rows)
    summary = (
        match_team.groupby("team", as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            avg_team_rating=("team_avg_rating", "mean"),
            rated_matches=("has_player_ratings", "sum"),
            signal_matches=("has_form_signals", "sum"),
            avg_signal_rating=("proxy_avg_rating", "mean"),
            players_rated=("players_rated", lambda values: int(pd.Series(values).dropna().sum())),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            shots_on_target_for=("shots_on_target_for", "sum"),
            shots_on_target_against=("shots_on_target_against", "sum"),
        )
        .round(3)
    )
    summary["goal_diff"] = summary["goals_for"] - summary["goals_against"]

    if not star_summary.empty:
        summary = summary.merge(star_summary, on="team", how="left")
    if "star_avg_rating" not in summary.columns:
        summary["star_avg_rating"] = None
        summary["star_rows"] = 0
    summary = summary.sort_values(
        ["matches", "avg_team_rating", "goal_diff", "goals_for"],
        ascending=False,
        na_position="last",
    )

    return {
        "summary": dataframe_to_records(summary),
        "matches": match_rows,
        "note": "Match counts use the full actual group-stage plus knockout result backbone. Player ratings use direct SofaScore rows where available and Google rows only where SofaScore has no match mapping; xG, shots on target, saves, and tackles come from SofaScore where mapped.",
    }


def render_html(data: dict) -> str:
    payload = json.dumps(json_safe(data), ensure_ascii=True)
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
    .app-tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 18px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
    }}
    .tab-button {{
      background: transparent;
      border-color: transparent;
      color: var(--muted);
      font-weight: 800;
    }}
    .tab-button.active {{
      color: var(--ink);
      background: rgba(38,198,184,.12);
      border-color: rgba(38,198,184,.48);
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.active {{
      display: block;
    }}
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
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .team-summary-card {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.045);
      border-radius: 8px;
      padding: 12px;
      cursor: pointer;
      transition: border-color .18s ease, background .18s ease, transform .18s ease;
    }}
    .team-summary-card:hover,
    .team-summary-card.active {{
      border-color: rgba(38,198,184,.55);
      background: rgba(38,198,184,.09);
      transform: translateY(-1px);
    }}
    .team-summary-card h3 {{
      margin: 0 0 10px;
      font-size: 17px;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .stat {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .stat strong {{
      display: block;
      color: var(--ink);
      font-size: 18px;
      margin-top: 2px;
    }}
    .player-profile {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(280px, .85fr);
      gap: 16px;
      align-items: start;
    }}
    .player-summary h3 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      background: rgba(255,255,255,.025);
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
      font-size: 13px;
    }}
    .data-table th,
    .data-table td {{
      padding: 9px 10px;
      border-bottom: 1px solid rgba(255,255,255,.07);
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      color: var(--gold);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      background: rgba(255,255,255,.035);
    }}
    .data-table td {{
      color: var(--ink);
    }}
    .data-table .muted-cell {{
      color: var(--muted);
    }}
    .match-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .team-match-card {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.025));
      border-radius: 8px;
      padding: 12px;
    }}
    .team-match-card h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .mini-list {{
      margin: 9px 0 0;
      padding: 0;
      list-style: none;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .mini-list li {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      border-top: 1px solid rgba(255,255,255,.07);
      padding: 6px 0;
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
      .player-profile {{ grid-template-columns: 1fr; }}
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
    <p class="lead">Latest notebook projection with actual knockout results, direct SofaScore player ratings and match stats, human tactical priors, ticket-watch recommendations, and a market-prior diagnostic view.</p>
    <div class="scoreboard">
      <div class="metric"><span>Projected Champion</span><strong>{champion}</strong></div>
      <div class="metric"><span>Projected Runner-up</span><strong>{runner_up}</strong></div>
      <div class="metric"><span>Direct Rating Rows</span><strong>{data["ratings"]["allRows"]:,}</strong></div>
      <div class="metric"><span>Rated Matches</span><strong>{data["ratings"]["scrapedMatches"]} / 96</strong></div>
      <div class="metric"><span>Value Players</span><strong>{len(data["playerValue"])}</strong></div>
    </div>
  </header>
  <main>
    <nav class="app-tabs" aria-label="Analysis tabs">
      <button class="tab-button active" data-tab="forecast">Forecast</button>
      <button class="tab-button" data-tab="value">Player Value</button>
      <button class="tab-button" data-tab="player">Player Detail</button>
      <button class="tab-button" data-tab="team">Team Performance</button>
    </nav>

    <div id="forecastTab" class="tab-panel active">
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
    </div>

    <div id="valueTab" class="tab-panel">
      <section class="wide-section">
        <div class="section-head">
          <h2>Player Value vs Transfermarkt Value</h2>
          <div class="chart-tools">
            <select id="valueCountryFilter" aria-label="Highlight country in value chart">
              <option value="">All countries</option>
            </select>
            <select id="valueClubFilter" aria-label="Highlight major club in value chart">
              <option value="">All major clubs</option>
            </select>
            <button id="valueClearFilters" type="button">Clear filters</button>
            <button id="valueZoomOut" type="button">Zoom out</button>
            <span id="valueZoomLabel" class="zoom-label">100%</span>
            <button id="valueZoomIn" type="button">Zoom in</button>
            <button id="valueZoomReset" type="button">Reset</button>
          </div>
        </div>
        <div id="valueChart" class="chart-wrap"></div>
        <div id="valueDetails" class="chart-note"></div>
      </section>
    </div>

    <div id="playerTab" class="tab-panel">
      <section class="wide-section">
        <div class="section-head">
          <h2>Player Detail</h2>
          <div class="chart-tools">
            <button id="playerBackToChart" type="button">Back to value chart</button>
          </div>
        </div>
        <div id="playerDetail"></div>
      </section>
    </div>

    <div id="teamTab" class="tab-panel">
      <section class="wide-section">
        <div class="section-head">
          <h2>Team Performance So Far</h2>
          <div class="chart-tools">
            <select id="teamPerformanceSelect" aria-label="Team performance country">
              <option value="">All teams</option>
            </select>
          </div>
        </div>
        <div id="teamPerformanceNote" class="chart-note"></div>
        <div id="teamSummaryGrid" class="summary-grid"></div>
        <div id="teamPerformanceDetails"></div>
      </section>
    </div>
    <footer>
      Data note: direct player ratings, minutes, goals, assists, xG, shots on target, saves, tackles, and shot maps are fetched from SofaScore where mapped. Google lineup extracts remain as a fallback/audit trail. The July 7 Polymarket outright snapshot is blended lightly into the notebook model, while Kalshi rows remain diagnostic.
    </footer>
  </main>
  <script>
    const DATA = {payload};
    const roundOrder = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Third-place Match", "Final"];
    const bracketRounds = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"];
    const state = {{
      activeTab: "forecast",
      dataset: "live",
      round: "All",
      status: "All",
      search: "",
      selected: null,
      highlightTeam: "",
      focusTeam: false,
      bracketZoom: 1,
      valueZoom: 1,
      selectedPlayer: null,
      valueCountry: "",
      valueClub: "",
      teamPerformanceTeam: ""
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

    function collectMajorClubs() {{
      const clubs = new Set();
      valueRows().forEach(row => {{
        if (row.club && row.club !== "Other / untagged") clubs.add(String(row.club));
      }});
      return [...clubs].sort((a, b) => a.localeCompare(b));
    }}

    function syncValueControls() {{
      const country = document.querySelector("#valueCountryFilter");
      const club = document.querySelector("#valueClubFilter");
      if (country && country.value !== state.valueCountry) country.value = state.valueCountry;
      if (club && club.value !== state.valueClub) club.value = state.valueClub;
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

    function setActiveTab(tabName) {{
      state.activeTab = tabName;
      document.querySelectorAll("[data-tab]").forEach(tab => tab.classList.toggle("active", tab.dataset.tab === tabName));
      document.querySelectorAll(".tab-panel").forEach(panel => {{
        panel.classList.toggle("active", panel.id === `${{tabName}}Tab`);
      }});
    }}

    function playerKey(row) {{
      return row?.detail_key || `${{row?.team}}|${{row?.player}}`;
    }}

    function initControls() {{
      document.querySelectorAll("[data-tab]").forEach(btn => {{
        btn.addEventListener("click", () => {{
          setActiveTab(btn.dataset.tab);
        }});
      }});
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
      const valueCountry = document.querySelector("#valueCountryFilter");
      collectTeams().forEach(team => {{
        const opt = document.createElement("option");
        opt.value = team;
        opt.textContent = team;
        valueCountry.appendChild(opt);
      }});
      const valueClub = document.querySelector("#valueClubFilter");
      collectMajorClubs().forEach(club => {{
        const opt = document.createElement("option");
        opt.value = club;
        opt.textContent = club;
        valueClub.appendChild(opt);
      }});
      const teamPerformanceSelect = document.querySelector("#teamPerformanceSelect");
      (DATA.teamPerformance?.summary || []).map(row => row.team).sort((a, b) => a.localeCompare(b)).forEach(team => {{
        const opt = document.createElement("option");
        opt.value = team;
        opt.textContent = team;
        teamPerformanceSelect.appendChild(opt);
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
      valueCountry.addEventListener("change", e => {{
        state.valueCountry = e.target.value;
        setHighlight(e.target.value);
        state.selectedPlayer = null;
        renderAll(false);
      }});
      valueClub.addEventListener("change", e => {{
        state.valueClub = e.target.value;
        state.selectedPlayer = null;
        renderAll(false);
      }});
      document.querySelector("#valueClearFilters").addEventListener("click", () => {{
        state.valueCountry = "";
        state.valueClub = "";
        state.selectedPlayer = null;
        syncValueControls();
        renderAll(false);
      }});
      document.querySelector("#playerBackToChart").addEventListener("click", () => {{
        setActiveTab("value");
        renderAll(false);
      }});
      teamPerformanceSelect.addEventListener("change", e => {{
        state.teamPerformanceTeam = e.target.value;
        renderTeamPerformance();
      }});
      document.querySelector("#focusToggle").addEventListener("change", e => {{
        state.focusTeam = e.target.checked;
        renderAll(false);
      }});
      document.querySelector("#clearHighlight").addEventListener("click", () => {{
        setHighlight("");
        state.valueCountry = "";
        state.focusTeam = false;
        state.selectedPlayer = null;
        document.querySelector("#focusToggle").checked = false;
        syncValueControls();
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
      const formSignals = [];
      const formTeamAvgs = [];
      const noAvgMessage = row?.status === "actual"
        ? "Actual match; direct player-rating rows have not been fetched or matched yet."
        : "Projected match; direct player ratings will appear after the match is played and fetched.";
      const ratingHtml = ratings.length ? ratings.map(r => `
        <div class="rating-row">
          <div><strong>${{escapeHtml(r.player)}}</strong><div class="detail-line">${{escapeHtml(r.team)}} · ${{escapeHtml(r.match)}}</div></div>
          <div>${{Number(r.rating).toFixed(1)}}</div>
        </div>`).join("") : formSignals.length ? formSignals.map(r => `
        <div class="rating-row">
          <div><strong>${{escapeHtml(r.player)}}</strong><div class="detail-line">${{escapeHtml(r.team)}} Â· ${{escapeHtml(r.source)}} Â· ${{escapeHtml(r.note || "")}}</div></div>
          <div>${{Number(r.rating).toFixed(1)}}</div>
        </div>`).join("") : `<div class="detail-line">No direct player-rating rows for this match yet.</div>`;
      const avgHtml = teamAvgs.length ? teamAvgs.map(r => `
        <div class="rating-row">
          <div><strong>${{escapeHtml(r.team)}}</strong><div class="bar"><i style="--w:${{Math.max(1, Math.min(100, Number(r.avg_rating) * 10))}}%"></i></div></div>
          <div>${{Number(r.avg_rating).toFixed(2)}}</div>
        </div>`).join("") : formTeamAvgs.length ? formTeamAvgs.map(r => `
        <div class="rating-row">
          <div><strong>${{escapeHtml(r.team)}}</strong><div class="detail-line">${{r.signals || 0}} model signal(s)</div><div class="bar"><i style="--w:${{Math.max(1, Math.min(100, Number(r.avg_rating) * 10))}}%"></i></div></div>
          <div>${{Number(r.avg_rating).toFixed(2)}}</div>
        </div>`).join("") : "";
      const avgTitle = "Team Averages";
      const ratingsTitle = "Top Rated Players";
      document.querySelector("#details").innerHTML = `<h2>M${{escapeHtml(row?.match)}} Details</h2>
        <div class="detail-line">${{escapeHtml(row?.round)}} · ${{escapeHtml(row?.date)}} · ${{escapeHtml(row?.weekday)}} · ${{escapeHtml(row?.local_kickoff || "")}}</div>
        <div class="teams">${{escapeHtml(row?.team_a)}} vs ${{escapeHtml(row?.team_b)}}</div>
        <div class="detail-line">Location: ${{escapeHtml(row?.location || "")}}</div>
        <div class="detail-line">Winner: <strong>${{escapeHtml(row?.winner)}}</strong> (${{escapeHtml(row?.winner_probability)}}) · ${{escapeHtml(row?.status || "")}}</div>
        ${{row?.result_note ? `<div class="detail-line">Result: ${{escapeHtml(row.result_note)}}</div>` : ""}}
        <h2 style="margin-top:18px">${{avgTitle}}</h2>${{avgHtml || `<div class="detail-line">${{noAvgMessage}}</div>`}}
        <h2 style="margin-top:18px">${{ratingsTitle}}</h2>${{ratingHtml}}`;
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
      if (!host) return;
      host.innerHTML = DATA.ratings.r32TeamForm.slice(0, 26).map(r => {{
        const width = Math.max(5, Math.min(100, Number(r.avg_rating) * 10));
        return `<div class="rating-row">
          <div><strong>${{escapeHtml(r.team)}}</strong><div class="detail-line">M${{Number(r.match_no).toFixed(0)}} · ${{escapeHtml(r.match)}} · ${{r.players_rated}} players</div><div class="bar"><i style="--w:${{width}}%"></i></div></div>
          <div>${{Number(r.avg_rating).toFixed(2)}}</div>
        </div>`;
      }}).join("");
    }}

    function valueFocusMatches(row) {{
      const countryHit = state.valueCountry && row.team === state.valueCountry;
      const clubHit = state.valueClub && row.club === state.valueClub;
      const globalHit = state.highlightTeam && row.team === state.highlightTeam;
      return countryHit || clubHit || globalHit;
    }}

    function formatNullable(value, digits = 1) {{
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : "n/a";
    }}

    function renderPlayerMiniList(players, fallbackLabel) {{
      if (!players || !players.length) return `<div class="detail-line">${{fallbackLabel}}</div>`;
      return `<ul class="mini-list">${{players.map(player => {{
        const source = player.source ? ` · ${{escapeHtml(player.source)}}` : "";
        return `<li><span>${{escapeHtml(player.player)}}${{source}}</span><strong>${{formatNullable(player.rating, 1)}}</strong></li>`;
      }}).join("")}}</ul>`;
    }}

    function playerColor(row) {{
      const highValue = Number(row.transfermarkt_value_eur_m) >= 75;
      const highRating = Number(row.world_cup_rating) >= 7;
      if (state.valueClub && row.club === state.valueClub) return "#d8b45f";
      if ((state.valueCountry && row.team === state.valueCountry) || (state.highlightTeam && row.team === state.highlightTeam)) return "#26c6b8";
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
        const key = playerKey(row);
        return key === selectedKey || valueFocusMatches(row) || Number(row.transfermarkt_value_eur_m) >= 120 || Number(row.world_cup_rating) >= 7.55;
      }}).map(row => `<text class="player-label" x="${{x(row.transfermarkt_value_eur_m) + 8}}" y="${{y(row.world_cup_rating) - 8}}">${{escapeHtml(row.player)}}</text>`).join("");
      const points = rows.map((row, index) => {{
        const key = playerKey(row);
        const active = key === selectedKey || valueFocusMatches(row);
        const hasFocus = state.valueCountry || state.valueClub || (state.highlightTeam && state.focusTeam);
        const dimmed = hasFocus && !active;
        const radius = active ? 7 : 5;
        return `<circle class="player-dot${{dimmed ? " dimmed" : ""}}${{active ? " active" : ""}}" data-index="${{index}}" cx="${{x(row.transfermarkt_value_eur_m)}}" cy="${{y(row.world_cup_rating)}}" r="${{radius}}" fill="${{playerColor(row)}}"><title>${{escapeHtml(row.player)}} &middot; ${{escapeHtml(row.team)}} &middot; ${{escapeHtml(row.club || "No major-club tag")}} &middot; EUR ${{Number(row.transfermarkt_value_eur_m).toFixed(0)}}m &middot; ${{Number(row.world_cup_rating).toFixed(2)}}</title></circle>`;
      }}).join("");

      host.innerHTML = `<svg class="value-svg" style="width:${{Math.round(state.valueZoom * 100)}}%; min-width:100%" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Player rating versus Transfermarkt value scatter plot">
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="transparent"></rect>
        <text x="${{margin.left}}" y="22" fill="#edf4f1" font-size="15" font-weight="700">World Cup rating vs market value</text>
        <text x="${{width - margin.right}}" y="22" fill="#9fb0ad" font-size="12" text-anchor="end">Click a dot to open its tournament match log</text>
        ${{grid}}
        <line class="threshold-line" x1="${{x(75)}}" x2="${{x(75)}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}"></line>
        <line class="threshold-line" x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y(7)}}" y2="${{y(7)}}"></line>
        <text x="${{x(75) + 8}}" y="${{margin.top + 16}}" fill="#d8b45f" font-size="12">EUR 75m</text>
        <text x="${{width - margin.right - 8}}" y="${{y(7) - 8}}" fill="#d8b45f" font-size="12" text-anchor="end">7.0 rating</text>
        ${{points}}
        ${{labels}}
        <text x="${{width / 2}}" y="${{height - 22}}" fill="#9fb0ad" font-size="13" text-anchor="middle">Transfermarkt value, EUR millions</text>
        <text x="18" y="${{height / 2}}" fill="#9fb0ad" font-size="13" text-anchor="middle" transform="rotate(-90 18 ${{height / 2}})">Direct post-match rating</text>
      </svg>`;

      host.querySelectorAll(".player-dot").forEach(dot => {{
        dot.addEventListener("click", () => {{
          const row = rows[Number(dot.dataset.index)];
          state.selectedPlayer = playerKey(row);
          state.valueCountry = row.team;
          setHighlight(row.team);
          syncValueControls();
          setActiveTab("player");
          renderAll(false);
        }});
      }});

      const selected = rows.find(row => playerKey(row) === state.selectedPlayer);
      if (selected) {{
        details.innerHTML = `<strong>${{escapeHtml(selected.player)}}</strong>, ${{escapeHtml(selected.team)}} / ${{escapeHtml(selected.club || "No major-club tag")}}: EUR ${{Number(selected.transfermarkt_value_eur_m).toFixed(0)}}m, World Cup rating ${{Number(selected.world_cup_rating).toFixed(2)}} across ${{selected.rated_matches}} rated match(es), ${{escapeHtml(selected.rating_method)}}.`;
      }} else if (state.valueClub) {{
        const clubRows = rows.filter(row => row.club === state.valueClub);
        const avgRating = clubRows.reduce((sum, row) => sum + Number(row.world_cup_rating), 0) / Math.max(1, clubRows.length);
        const totalValue = clubRows.reduce((sum, row) => sum + Number(row.transfermarkt_value_eur_m), 0);
        details.innerHTML = `<strong>${{escapeHtml(state.valueClub)}}</strong>: ${{clubRows.length}} tracked World Cup star player(s), EUR ${{totalValue.toFixed(0)}}m combined snapshot value, ${{avgRating.toFixed(2)}} average World Cup rating.`;
      }} else if (state.valueCountry || state.highlightTeam) {{
        const teamName = state.valueCountry || state.highlightTeam;
        const teamRows = rows.filter(row => row.team === teamName);
        const avgRating = teamRows.reduce((sum, row) => sum + Number(row.world_cup_rating), 0) / Math.max(1, teamRows.length);
        const totalValue = teamRows.reduce((sum, row) => sum + Number(row.transfermarkt_value_eur_m), 0);
        details.innerHTML = `<strong>${{escapeHtml(teamName)}}</strong>: ${{teamRows.length}} tracked star player(s), EUR ${{totalValue.toFixed(0)}}m combined snapshot value, ${{avgRating.toFixed(2)}} average World Cup rating.`;
      }} else {{
        details.innerHTML = `Thresholds: high market value is EUR 75m or above; high performance is 7.0 or above. Use country or major-club selectors, or click a dot to open a player match log.`;
      }}
    }}

    function renderPlayerDetail() {{
      const host = document.querySelector("#playerDetail");
      if (!host) return;
      const detail = (DATA.playerDetails || {{}})[state.selectedPlayer];
      if (!detail) {{
        host.innerHTML = `<div class="detail-line">Select a player dot in the Player Value tab to open a tournament match log.</div>`;
        return;
      }}
      const matches = detail.matches || [];
      const totalGoals = Number(detail.goals || 0);
      const totalAssists = Number(detail.assists || 0);
      const totalMinutes = Number(detail.total_minutes || 0);
      const rows = matches.map(row => {{
        const matchLabel = row.match_no ? `M${{Number(row.match_no).toFixed(0)}}` : escapeHtml(row.match_id || "");
        return `<tr>
          <td><strong>${{escapeHtml(matchLabel)}}</strong><div class="muted-cell">${{escapeHtml(row.date || "")}}</div></td>
          <td>${{escapeHtml(row.stage || "")}}</td>
          <td>${{escapeHtml(row.match || "")}}</td>
          <td>${{escapeHtml(row.opponent || "")}}<div class="muted-cell">${{escapeHtml(row.score || "")}}</div></td>
          <td>${{formatNullable(row.minutes_played, 0)}}</td>
          <td><strong>${{formatNullable(row.rating, 1)}}</strong></td>
          <td>${{formatNullable(row.goals || 0, 0)}}</td>
          <td>${{formatNullable(row.assists || 0, 0)}}</td>
          <td>${{formatNullable(row.expected_goals, 2)}}</td>
          <td>${{formatNullable(row.expected_assists, 2)}}</td>
        </tr>`;
      }}).join("");
      host.innerHTML = `<div class="player-profile">
        <div class="player-summary">
          <h3>${{escapeHtml(detail.player)}}</h3>
          <div class="detail-line">${{escapeHtml(detail.team)}} / ${{escapeHtml(detail.club || "Other / untagged")}} / SofaScore player: ${{escapeHtml(detail.sofascore_player || detail.player)}}</div>
          <div class="stat-grid">
            <div class="stat">Overall rating<strong>${{formatNullable(detail.world_cup_rating, 2)}}</strong></div>
            <div class="stat">Rated matches<strong>${{detail.rated_matches || 0}}</strong></div>
            <div class="stat">Appearances<strong>${{detail.appearances || 0}}</strong></div>
            <div class="stat">Minutes<strong>${{formatNullable(totalMinutes, 0)}}</strong></div>
            <div class="stat">Goals<strong>${{formatNullable(totalGoals, 0)}}</strong></div>
            <div class="stat">Assists<strong>${{formatNullable(totalAssists, 0)}}</strong></div>
            <div class="stat">xG<strong>${{formatNullable(detail.expected_goals, 2)}}</strong></div>
            <div class="stat">xA<strong>${{formatNullable(detail.expected_assists, 2)}}</strong></div>
          </div>
        </div>
        <div>
          <div class="stat-grid">
            <div class="stat">Transfermarkt value<strong>EUR ${{formatNullable(detail.transfermarkt_value_eur_m, 0)}}m</strong></div>
            <div class="stat">Rating method<strong>${{escapeHtml(detail.rating_method || "")}}</strong></div>
            <div class="stat">Performance tier<strong>${{escapeHtml(detail.performance_tier || "")}}</strong></div>
            <div class="stat">Value tier<strong>${{escapeHtml(detail.value_tier || "")}}</strong></div>
          </div>
          <div class="detail-line">Rows come from SofaScore lineup payloads matched to the local World Cup result backbone. Blank ratings remain n/a instead of being proxied.</div>
        </div>
      </div>
      <div class="table-wrap" style="margin-top:16px">
        <table class="data-table">
          <thead><tr><th>Match</th><th>Round</th><th>Fixture</th><th>Opponent</th><th>Min</th><th>Rating</th><th>G</th><th>A</th><th>xG</th><th>xA</th></tr></thead>
          <tbody>${{rows || `<tr><td colspan="10" class="muted-cell">No match rows found for this player.</td></tr>`}}</tbody>
        </table>
      </div>`;
    }}

    function renderTeamPerformance() {{
      const perf = DATA.teamPerformance || {{ summary: [], matches: [], note: "" }};
      const note = document.querySelector("#teamPerformanceNote");
      const grid = document.querySelector("#teamSummaryGrid");
      const details = document.querySelector("#teamPerformanceDetails");
      if (!perf.summary.length) {{
        note.textContent = perf.note || "No team performance data found.";
        grid.innerHTML = "";
        details.innerHTML = "";
        return;
      }}

      note.textContent = perf.note || "";
      const selectedTeam = state.teamPerformanceTeam || perf.summary[0].team;
      const visibleSummary = state.teamPerformanceTeam
        ? perf.summary.filter(row => row.team === selectedTeam)
        : perf.summary;
      grid.innerHTML = visibleSummary.map(row => `
        <article class="team-summary-card${{row.team === selectedTeam ? " active" : ""}}" data-team="${{escapeHtml(row.team)}}" tabindex="0" role="button" aria-label="Show ${{escapeHtml(row.team)}} match history">
          <h3>${{escapeHtml(row.team)}}</h3>
          <div class="stat-grid">
            <div class="stat">Matches<strong>${{row.matches}}</strong></div>
            <div class="stat">Goals<strong>${{row.goals_for}}-${{row.goals_against}}</strong></div>
            <div class="stat">Goal diff<strong>${{Number(row.goal_diff) > 0 ? "+" : ""}}${{row.goal_diff}}</strong></div>
            <div class="stat">Direct avg<strong>${{formatNullable(row.avg_team_rating, 2)}}</strong></div>
            <div class="stat">Rated matches<strong>${{row.rated_matches || 0}}/${{row.matches}}</strong></div>
            <div class="stat">Rated players<strong>${{row.players_rated || 0}}</strong></div>
          </div>
        </article>`).join("");
      grid.querySelectorAll(".team-summary-card[data-team]").forEach(card => {{
        const chooseTeam = () => {{
          state.teamPerformanceTeam = card.dataset.team;
          const select = document.querySelector("#teamPerformanceSelect");
          if (select) select.value = state.teamPerformanceTeam;
          renderTeamPerformance();
        }};
        card.addEventListener("click", chooseTeam);
        card.addEventListener("keydown", event => {{
          if (event.key === "Enter" || event.key === " ") {{
            event.preventDefault();
            chooseTeam();
          }}
        }});
      }});

      const teamMatches = perf.matches.filter(row => row.team === selectedTeam);
      details.innerHTML = `<div class="section-head"><h2>${{escapeHtml(selectedTeam)}} Match Log</h2><span class="pill">${{teamMatches.length}} match(es)</span></div>
        <div class="match-list">${{teamMatches.map(row => {{
          const players = row.star_players?.length
            ? row.star_players
            : row.top_players?.length
              ? row.top_players
              : [];
          const label = row.star_players?.length
            ? "Star player ratings"
            : row.top_players?.length
              ? "Top direct ratings"
              : "Player ratings";
          return `<article class="team-match-card">
            <span class="pill">${{escapeHtml(row.stage)}}</span>${{row.match_no ? `<span class="pill">M${{row.match_no}}</span>` : ""}}
            <h3>${{escapeHtml(row.team)}} ${{escapeHtml(row.score)}} vs ${{escapeHtml(row.opponent)}}</h3>
            <div class="detail-line">${{escapeHtml(row.date)}}${{row.weekday ? ` &middot; ${{escapeHtml(row.weekday)}}` : ""}}${{row.location ? ` &middot; ${{escapeHtml(row.location)}}` : ""}}</div>
            <div class="detail-line">${{escapeHtml(row.match)}}</div>
            <div class="stat-grid">
              <div class="stat">Direct rating<strong>${{formatNullable(row.team_avg_rating, 2)}}</strong></div>
              <div class="stat">xG<strong>${{formatNullable(row.xg_for, 2)}}</strong></div>
              <div class="stat">Shots on target<strong>${{formatNullable(row.shots_on_target_for, 0)}}</strong></div>
              <div class="stat">Saves<strong>${{formatNullable(row.saves_for, 0)}}</strong></div>
            </div>
            <div class="detail-line">${{label}}</div>
            ${{renderPlayerMiniList(players, "No direct player-rating rows for this match yet.")}}
          </article>`;
        }}).join("")}}</div>`;
    }}

    function renderAll(keepSelection = true) {{
      if (!keepSelection) state.selected = state.selected;
      renderBracket();
      renderDetails();
      renderRecommendations();
      renderTopPlayers();
      renderTeamForm();
      renderValueChart();
      renderPlayerDetail();
      renderTeamPerformance();
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
    match_backbone = build_actual_match_backbone(live_table)
    player_value, player_details = load_player_value_data(match_backbone)
    data = {
        "summaryMap": summary_map,
        "liveBracket": dataframe_to_records(live_table),
        "marketBracket": dataframe_to_records(market_table),
        "recommendations": dataframe_to_records(recommendations),
        "ratings": load_rating_data(live_table),
        "playerValue": player_value,
        "playerDetails": player_details,
        "teamPerformance": load_team_performance_data(live_table),
    }
    OUT.write_text(render_html(data), encoding="utf-8")
    print(OUT)
    print(f"live_matches={len(data['liveBracket'])}")
    print(f"recommendations={len(data['recommendations'])}")
    print(f"ratings_rows={data['ratings']['allRows']}")


if __name__ == "__main__":
    main()
