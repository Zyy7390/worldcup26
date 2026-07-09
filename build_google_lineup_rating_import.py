from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "google_lineup_player_ratings_raw.csv"
OUT_PATH = ROOT / "google_lineup_player_ratings.csv"
UNMATCHED_PATH = ROOT / "google_lineup_player_ratings_unmatched.csv"


STAR_PLAYERS = [
    ("Mexico", "Santiago Gimenez"), ("Mexico", "Edson Alvarez"), ("Mexico", "Hirving Lozano"),
    ("Korea Republic", "Son Heung-min"), ("Korea Republic", "Kim Min-jae"), ("Korea Republic", "Lee Kang-in"),
    ("Czechia", "Patrik Schick"), ("Czechia", "Tomas Soucek"), ("Czechia", "Adam Hlozek"),
    ("Switzerland", "Granit Xhaka"), ("Switzerland", "Manuel Akanji"), ("Switzerland", "Breel Embolo"),
    ("Canada", "Alphonso Davies"), ("Canada", "Jonathan David"), ("Canada", "Tajon Buchanan"),
    ("Brazil", "Vinicius Junior"), ("Brazil", "Rodrygo"), ("Brazil", "Alisson"),
    ("Morocco", "Achraf Hakimi"), ("Morocco", "Yassine Bounou"), ("Morocco", "Brahim Diaz"),
    ("Scotland", "Andy Robertson"), ("Scotland", "Scott McTominay"), ("Scotland", "John McGinn"),
    ("USA", "Christian Pulisic"), ("USA", "Weston McKennie"), ("USA", "Antonee Robinson"),
    ("Turkiye", "Hakan Calhanoglu"), ("Turkiye", "Arda Guler"), ("Turkiye", "Kenan Yildiz"),
    ("Australia", "Mathew Ryan"), ("Australia", "Harry Souttar"), ("Australia", "Jackson Irvine"),
    ("Germany", "Jamal Musiala"), ("Germany", "Florian Wirtz"), ("Germany", "Joshua Kimmich"),
    ("Ecuador", "Moises Caicedo"), ("Ecuador", "Piero Hincapie"), ("Ecuador", "Willian Pacho"),
    ("Cote d'Ivoire", "Simon Adingra"), ("Cote d'Ivoire", "Franck Kessie"), ("Cote d'Ivoire", "Sebastien Haller"),
    ("Netherlands", "Virgil van Dijk"), ("Netherlands", "Cody Gakpo"), ("Netherlands", "Xavi Simons"),
    ("Japan", "Takefusa Kubo"), ("Japan", "Kaoru Mitoma"), ("Japan", "Wataru Endo"),
    ("Belgium", "Kevin De Bruyne"), ("Belgium", "Jeremy Doku"), ("Belgium", "Romelu Lukaku"),
    ("Iran", "Mehdi Taremi"), ("Iran", "Sardar Azmoun"), ("Iran", "Alireza Jahanbakhsh"),
    ("Egypt", "Mohamed Salah"), ("Egypt", "Omar Marmoush"), ("Egypt", "Mostafa Mohamed"),
    ("Spain", "Lamine Yamal"), ("Spain", "Pedri"), ("Spain", "Rodri"),
    ("Uruguay", "Federico Valverde"), ("Uruguay", "Darwin Nunez"), ("Uruguay", "Ronald Araujo"),
    ("France", "Kylian Mbappe"), ("France", "Ousmane Dembele"), ("France", "Michael Olise"),
    ("Senegal", "Sadio Mane"), ("Senegal", "Nicolas Jackson"), ("Senegal", "Kalidou Koulibaly"),
    ("Norway", "Erling Haaland"), ("Norway", "Martin Odegaard"), ("Norway", "Alexander Sorloth"),
    ("Argentina", "Lionel Messi"), ("Argentina", "Lautaro Martinez"), ("Argentina", "Julian Alvarez"),
    ("Austria", "David Alaba"), ("Austria", "Marcel Sabitzer"), ("Austria", "Christoph Baumgartner"),
    ("Algeria", "Riyad Mahrez"), ("Algeria", "Ismael Bennacer"), ("Algeria", "Amine Gouiri"),
    ("Portugal", "Cristiano Ronaldo"), ("Portugal", "Bruno Fernandes"), ("Portugal", "Rafael Leao"),
    ("Colombia", "Luis Diaz"), ("Colombia", "James Rodriguez"), ("Colombia", "Daniel Munoz"),
    ("England", "Harry Kane"), ("England", "Jude Bellingham"), ("England", "Bukayo Saka"),
    ("Croatia", "Luka Modric"), ("Croatia", "Josko Gvardiol"), ("Croatia", "Mateo Kovacic"),
    ("Panama", "Adalberto Carrasquilla"), ("Panama", "Michael Murillo"), ("Panama", "Jose Fajardo"),
]


def norm(value):
    text = str(value).replace("ø", "o").replace("Ø", "O").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def display_keys(player):
    parts = player.split()
    keys = {norm(player)}
    if len(parts) >= 2:
        keys.add(norm(parts[0][0] + " " + " ".join(parts[1:])))
        keys.add(norm(parts[0][0] + ". " + " ".join(parts[1:])))
        keys.add(norm(parts[0][0] + " " + parts[-1]))
        keys.add(norm(parts[0][0] + ". " + parts[-1]))
        keys.add(norm(parts[-1]))
    return keys


def build_lookup():
    lookup = {}
    for team, player in STAR_PLAYERS:
        for key in display_keys(player):
            lookup.setdefault(key, []).append((team, player))

    aliases = {
        norm("A. Becker"): ("Brazil", "Alisson"),
        norm("V. Junior"): ("Brazil", "Vinicius Junior"),
        norm("K. Mbappe"): ("France", "Kylian Mbappe"),
        norm("O. Dembele"): ("France", "Ousmane Dembele"),
        norm("M. Odegaard"): ("Norway", "Martin Odegaard"),
        norm("A. Sorloth"): ("Norway", "Alexander Sorloth"),
        norm("F. Kessie"): ("Cote d'Ivoire", "Franck Kessie"),
        norm("M. Caicedo"): ("Ecuador", "Moises Caicedo"),
        norm("P. Hincapie"): ("Ecuador", "Piero Hincapie"),
        norm("W. Pacho"): ("Ecuador", "Willian Pacho"),
        norm("Santiago Gimenez"): ("Mexico", "Santiago Gimenez"),
        norm("H. Kane"): ("England", "Harry Kane"),
        norm("J. Bellingham"): ("England", "Jude Bellingham"),
        norm("Bukayo Saka"): ("England", "Bukayo Saka"),
        norm("M. Olise"): ("France", "Michael Olise"),
    }
    return lookup, aliases


def main():
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing raw Google lineup scrape file: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH)
    lookup, aliases = build_lookup()
    matched_rows = []
    unmatched_rows = []

    for _, row in raw.iterrows():
        key = norm(row["player_display"])
        hit = aliases.get(key)
        if hit is None:
            candidates = lookup.get(key, [])
            hit = candidates[0] if len(candidates) == 1 else None

        if hit is None:
            unmatched_rows.append(row.to_dict())
            continue

        team, player = hit
        matched_rows.append({
            "player": player,
            "team": team,
            "match": row["match"],
            "rating": float(row["rating"]),
            "minutes_played": "",
            "rating_source": "Google rendered lineup rating scrape",
            "note": f"Matched from Google display '{row['player_display']}'. Source URL: {row.get('source_url', '')}",
        })

    matched = (
        pd.DataFrame(matched_rows)
        .drop_duplicates(["player", "team", "match", "rating_source"], keep="last")
        .sort_values(["match", "team", "player"])
    )
    unmatched = pd.DataFrame(unmatched_rows)

    matched.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    unmatched.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    print(f"raw_rows={len(raw)}")
    print(f"matched_rows={len(matched)}")
    print(f"unmatched_rows={len(unmatched)}")
    print(OUT_PATH)
    print(UNMATCHED_PATH)


if __name__ == "__main__":
    main()
