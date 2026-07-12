from __future__ import annotations

from pathlib import Path
import ast
import re

import matplotlib.pyplot as plt
import pandas as pd

from make_worldcup26_interactive_html import (
    build_actual_match_backbone,
    load_notebook_tables,
    load_player_value_data,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets" / "readme"
GOOGLE_RATINGS = ROOT / "data" / "google" / "google_lineup_player_ratings.csv"
MODEL_SOURCE = ROOT / "scripts" / "update_worldcup26_notebook_live.py"
OUT = ASSET_DIR / "player_value_vs_transfermarkt.png"


ACTIVE_TEAMS = {
    "Argentina",
    "England",
    "France",
    "Spain",
}
LABEL_PLAYERS = {
    "Kylian Mbappe",
    "Lionel Messi",
    "Erling Haaland",
    "Harry Kane",
    "Jude Bellingham",
    "Lamine Yamal",
    "Rodri",
    "Kevin De Bruyne",
    "Cristiano Ronaldo",
    "Vinicius Junior",
    "Yassine Bounou",
    "Achraf Hakimi",
}


def load_transfermarkt_values() -> pd.DataFrame:
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"transfermarkt_values_data\s*=\s*\[(.*?)\]\s*star_player_values",
        source,
        flags=re.S,
    )
    if not match:
        raise RuntimeError("Could not locate transfermarkt_values_data in model source.")
    values = ast.literal_eval("[" + match.group(1) + "]")
    return pd.DataFrame(values, columns=["team", "player", "transfermarkt_value_eur_m"])


def load_player_ratings() -> pd.DataFrame:
    ratings = pd.read_csv(GOOGLE_RATINGS)
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
    return pd.DataFrame(rows)


def make_plot() -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    live_table, _, _, _ = load_notebook_tables()
    match_backbone = build_actual_match_backbone(live_table)
    player_rows, _ = load_player_value_data(match_backbone)
    df = pd.DataFrame(player_rows)
    df = df[df["world_cup_rating"].notna()].copy()
    df["is_active_team"] = df["team"].isin(ACTIVE_TEAMS)

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=160)
    inactive = df[~df["is_active_team"]]
    active = df[df["is_active_team"]]
    ax.scatter(
        inactive["transfermarkt_value_eur_m"],
        inactive["world_cup_rating"],
        s=42,
        color="#9aa6b2",
        alpha=0.55,
        label="Other star players",
    )
    ax.scatter(
        active["transfermarkt_value_eur_m"],
        active["world_cup_rating"],
        s=74,
        color="#26c6b8",
        edgecolor="#0b3538",
        linewidth=0.7,
        alpha=0.95,
        label="Current semifinalist stars",
    )

    ax.axvline(75, color="#d8b45f", linestyle="--", linewidth=1.2)
    ax.axhline(7.0, color="#d8b45f", linestyle="--", linewidth=1.2)
    ax.text(77, 6.05, "EUR 75m value threshold", color="#7c6526", fontsize=8)
    ax.text(3, 7.08, "7.0 rating threshold", color="#7c6526", fontsize=8)

    for _, row in df[df["player"].isin(LABEL_PLAYERS)].iterrows():
        ax.annotate(
            row["player"],
            (row["transfermarkt_value_eur_m"], row["world_cup_rating"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.8,
            color="#1a2428",
        )

    ax.set_title("World Cup Player Rating vs Transfermarkt Value", fontsize=15, weight="bold")
    ax.set_xlabel("Transfermarkt value, EUR millions")
    ax.set_ylabel("World Cup post-match rating")
    ax.set_xlim(-3, max(210, df["transfermarkt_value_eur_m"].max() + 12))
    ax.set_ylim(4.4, 10.0)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.text(
        0.01,
        0.01,
        "Ratings use direct SofaScore rows with minutes weighting; values are the model's Transfermarkt snapshot.",
        fontsize=7.5,
        color="#59666d",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(make_plot())
