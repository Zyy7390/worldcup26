from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GOOGLE_DATA_DIR = ROOT / "data" / "google"
RATING_FILES = [
    GOOGLE_DATA_DIR / "google_worldcup_all_player_ratings.csv",
    GOOGLE_DATA_DIR / "google_lineup_player_ratings_raw.csv",
]


def assign_team_from_lineup_order(group):
    group = group.copy()
    team_a = group["team_a"].iloc[0]
    team_b = group["team_b"].iloc[0]
    orders = range(1, len(group) + 1)
    assigned = []
    for order in orders:
        if order <= 11:
            assigned.append(team_a)
        elif order <= 22:
            assigned.append(team_b)
        elif (order - 23) % 2 == 0:
            assigned.append(team_a)
        else:
            assigned.append(team_b)
    group["lineup_order"] = list(orders)
    group["team"] = assigned
    group["team_assignment_rule"] = "1-11 team_a, 12-22 team_b, substitutes alternate"
    return group


def repair_file(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"match_id", "match", "team_a", "team_b", "player_display", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    repaired = (
        df.groupby(["match_id", "match"], group_keys=False, sort=False)
        .apply(assign_team_from_lineup_order)
        .reset_index(drop=True)
    )
    repaired.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "file": path.name,
        "rows": len(repaired),
        "matches": repaired["match_id"].nunique(),
    }


def main():
    summaries = [summary for path in RATING_FILES if (summary := repair_file(path))]
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
