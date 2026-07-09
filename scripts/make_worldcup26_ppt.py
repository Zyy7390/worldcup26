from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets" / "ppt"
ASSET_DIR.mkdir(exist_ok=True)
PPTX_PATH = ROOT / "outputs" / "presentations" / "worldcup_2026_prediction_bracket_deck.pptx"
PPTX_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_prediction():
    rows = [
        ("A", "Mexico", 15, 3, 1.0), ("A", "South Africa", 60, 0, 0.0), ("A", "Korea Republic", 25, 0, 0.6), ("A", "Czechia", 41, 0, 0.3),
        ("B", "Canada", 30, 2, 0.5), ("B", "Bosnia and Herzegovina", 65, 0, 0.2), ("B", "Qatar", 55, 0, 0.0), ("B", "Switzerland", 19, 0, 0.8),
        ("C", "Brazil", 6, 0, 2.5), ("C", "Morocco", 8, 0, 2.0), ("C", "Haiti", 83, 0, 0.0), ("C", "Scotland", 43, 0, 0.3),
        ("D", "USA", 16, 3, 0.0), ("D", "Paraguay", 40, 0, 0.5), ("D", "Australia", 27, 0, 0.4), ("D", "Turkiye", 22, 0, 1.5),
        ("E", "Germany", 10, 0, 2.5), ("E", "Curacao", 82, 0, 0.0), ("E", "Cote d'Ivoire", 34, 0, 1.0), ("E", "Ecuador", 23, 0, 1.0),
        ("F", "Netherlands", 7, 0, 2.0), ("F", "Japan", 18, 0, 1.5), ("F", "Sweden", 38, 0, 1.2), ("F", "Tunisia", 44, 0, 0.1),
        ("G", "Belgium", 9, 0, 1.5), ("G", "Egypt", 29, 0, 1.0), ("G", "Iran", 21, 0, 0.1), ("G", "New Zealand", 85, 0, 0.0),
        ("H", "Spain", 2, 0, 4.0), ("H", "Cabo Verde", 69, 0, 0.0), ("H", "Saudi Arabia", 61, 0, 0.0), ("H", "Uruguay", 17, 0, 2.0),
        ("I", "France", 1, 0, 4.0), ("I", "Senegal", 14, 0, 1.0), ("I", "Iraq", 57, 0, 0.2), ("I", "Norway", 31, 0, 2.0),
        ("J", "Argentina", 3, 0, 3.5), ("J", "Algeria", 28, 0, 0.8), ("J", "Austria", 24, 0, 1.5), ("J", "Jordan", 63, 0, 0.0),
        ("K", "Portugal", 5, 0, 3.0), ("K", "DR Congo", 46, 0, 0.2), ("K", "Uzbekistan", 50, 0, 0.1), ("K", "Colombia", 13, 0, 2.0),
        ("L", "England", 4, 0, 3.5), ("L", "Croatia", 11, 0, 1.0), ("L", "Ghana", 74, 0, 0.2), ("L", "Panama", 33, 0, 0.2),
    ]
    teams = pd.DataFrame(rows, columns=["group", "team", "fifa_rank", "host_bonus", "form_modifier"])

    def depth(rank):
        if rank <= 10:
            return 2.0
        if rank <= 20:
            return 1.2
        if rank <= 30:
            return 0.6
        if rank <= 45:
            return 0.2
        return 0.0

    def logistic(x):
        return 1 / (1 + math.exp(-x))

    teams["depth_modifier"] = teams["fifa_rank"].map(depth)
    teams["rating"] = 100 - 0.8 * teams["fifa_rank"] + teams["host_bonus"] + teams["form_modifier"] + teams["depth_modifier"]

    def group_probs(ra, rb):
        diff = ra - rb
        draw = max(0.18, min(0.30, 0.30 - abs(diff) * 0.004))
        p_a = logistic(diff / 8)
        return (1 - draw) * p_a, draw, (1 - draw) * (1 - p_a)

    standings_rows = []
    for group, sub in teams.groupby("group", sort=True):
        stats = {team: {"xpts": 0.0, "xgf": 0.0, "xga": 0.0} for team in sub["team"]}
        recs = sub.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                p_aw, p_d, p_bw = group_probs(a["rating"], b["rating"])
                stats[a["team"]]["xpts"] += 3 * p_aw + p_d
                stats[b["team"]]["xpts"] += 3 * p_bw + p_d
                stats[a["team"]]["xgf"] += 1.15 + (a["rating"] - b["rating"]) / 30
                stats[a["team"]]["xga"] += 1.15 + (b["rating"] - a["rating"]) / 30
                stats[b["team"]]["xgf"] += 1.15 + (b["rating"] - a["rating"]) / 30
                stats[b["team"]]["xga"] += 1.15 + (a["rating"] - b["rating"]) / 30
        for team, stat in stats.items():
            base = teams.loc[teams["team"] == team].iloc[0]
            standings_rows.append({
                "group": group,
                "team": team,
                "fifa_rank": int(base["fifa_rank"]),
                "rating": base["rating"],
                "xpts": stat["xpts"],
                "xgd": stat["xgf"] - stat["xga"],
            })

    standings = pd.DataFrame(standings_rows).sort_values(
        ["group", "xpts", "xgd", "rating"], ascending=[True, False, False, False]
    )
    standings["group_position"] = standings.groupby("group").cumcount() + 1
    best_thirds = standings[standings.group_position == 3].sort_values(
        ["xpts", "xgd", "rating"], ascending=False
    ).head(8).copy()
    best_thirds["third_rank"] = range(1, 9)

    winners = standings[standings.group_position == 1].set_index("group")["team"].to_dict()
    runners = standings[standings.group_position == 2].set_index("group")["team"].to_dict()
    thirds = best_thirds.set_index("group")["team"].to_dict()
    ratings = teams.set_index("team")["rating"].to_dict()

    third_map = {"A": "E", "B": "G", "D": "J", "E": "C", "G": "A", "I": "D", "K": "L", "L": "I"}
    r32_pairs = OrderedDict([
        (73, (runners["A"], runners["B"], "2A vs 2B")),
        (74, (winners["E"], thirds[third_map["E"]], "1E vs 3C")),
        (75, (winners["F"], runners["C"], "1F vs 2C")),
        (76, (winners["C"], runners["F"], "1C vs 2F")),
        (77, (winners["I"], thirds[third_map["I"]], "1I vs 3D")),
        (78, (runners["E"], runners["I"], "2E vs 2I")),
        (79, (winners["A"], thirds[third_map["A"]], "1A vs 3E")),
        (80, (winners["L"], thirds[third_map["L"]], "1L vs 3I")),
        (81, (winners["D"], thirds[third_map["D"]], "1D vs 3J")),
        (82, (winners["G"], thirds[third_map["G"]], "1G vs 3A")),
        (83, (runners["K"], runners["L"], "2K vs 2L")),
        (84, (winners["H"], runners["J"], "1H vs 2J")),
        (85, (winners["B"], thirds[third_map["B"]], "1B vs 3G")),
        (86, (winners["J"], runners["H"], "1J vs 2H")),
        (87, (winners["K"], thirds[third_map["K"]], "1K vs 3L")),
        (88, (runners["D"], runners["G"], "2D vs 2G")),
    ])

    def ko_prob(a, b):
        return logistic((ratings[a] - ratings[b]) / 8)

    def ko_result(match, round_name, a, b, slot):
        p = ko_prob(a, b)
        winner = a if p >= 0.5 else b
        loser = b if winner == a else a
        return {
            "match": match,
            "round": round_name,
            "slot": slot,
            "team_a": a,
            "team_b": b,
            "winner": winner,
            "loser": loser,
            "winner_probability": max(p, 1 - p),
        }

    ko = {}
    for match, (a, b, slot) in r32_pairs.items():
        ko[match] = ko_result(match, "Round of 32", a, b, slot)

    def W(match):
        return ko[match]["winner"]

    for match, (a, b, slot) in OrderedDict([
        (89, (W(74), W(77), "W74 vs W77")), (90, (W(73), W(75), "W73 vs W75")),
        (91, (W(76), W(78), "W76 vs W78")), (92, (W(79), W(80), "W79 vs W80")),
        (93, (W(83), W(84), "W83 vs W84")), (94, (W(81), W(82), "W81 vs W82")),
        (95, (W(86), W(88), "W86 vs W88")), (96, (W(85), W(87), "W85 vs W87")),
    ]).items():
        ko[match] = ko_result(match, "Round of 16", a, b, slot)
    for match, (a, b, slot) in OrderedDict([
        (97, (W(89), W(90), "W89 vs W90")), (98, (W(93), W(94), "W93 vs W94")),
        (99, (W(91), W(92), "W91 vs W92")), (100, (W(95), W(96), "W95 vs W96")),
    ]).items():
        ko[match] = ko_result(match, "Quarterfinal", a, b, slot)
    for match, (a, b, slot) in OrderedDict([
        (101, (W(97), W(98), "W97 vs W98")), (102, (W(99), W(100), "W99 vs W100")),
    ]).items():
        ko[match] = ko_result(match, "Semifinal", a, b, slot)
    ko[103] = ko_result(103, "Third-place Match", ko[101]["loser"], ko[102]["loser"], "L101 vs L102")
    ko[104] = ko_result(104, "Final", W(101), W(102), "W101 vs W102")
    return teams, standings, best_thirds, pd.DataFrame(ko.values()).sort_values("match"), ko


def make_charts(teams, standings, best_thirds, ko_table, ko):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 170
    paths = {}

    contenders = teams.sort_values("rating", ascending=False).head(12).sort_values("rating")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(contenders.team, contenders.rating, color=["#d8a23a" if t == "France" else "#245b7a" for t in contenders.team])
    ax.set_xlabel("Model rating")
    ax.set_title("Top Title Contenders By Model Rating")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(88, 108)
    for i, v in enumerate(contenders.rating):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    paths["contenders"] = ASSET_DIR / "title_contenders.png"
    fig.savefig(paths["contenders"], facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(12.4, 7.0))
    axes = axes.ravel()
    third_set = set(best_thirds.team)
    for ax, (group, sub) in zip(axes, standings.groupby("group")):
        ax.axis("off")
        ax.set_title(f"Group {group}", loc="left", fontsize=12, fontweight="bold", color="#17384d")
        y = 0.86
        for _, row in sub.iterrows():
            fill = "#dfeee1" if row.group_position <= 2 else "#fff0d4" if row.team in third_set else "#f1f1f1"
            ax.add_patch(plt.Rectangle((0.0, y - 0.085), 0.98, 0.08, color=fill, ec="#ffffff"))
            ax.text(0.02, y - 0.045, f"{int(row.group_position)}. {row.team}", fontsize=9, va="center", ha="left")
            ax.text(0.78, y - 0.045, f"{row.xpts:.1f} pts", fontsize=8, va="center", ha="right")
            y -= 0.13
    fig.suptitle("Projected Group Finish", fontsize=18, fontweight="bold", color="#17384d", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    paths["groups"] = ASSET_DIR / "group_projection.png"
    fig.savefig(paths["groups"], facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.axis("off")
    cols = ["Rank", "Group", "Team", "Exp. Pts", "Exp. GD"]
    cell_text = [[int(r.third_rank), r.group, r.team, f"{r.xpts:.2f}", f"{r.xgd:.2f}"] for _, r in best_thirds.iterrows()]
    tbl = ax.table(cellText=cell_text, colLabels=cols, cellLoc="left", colLoc="left", loc="center", colColours=["#17384d"] * len(cols))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.45)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 else "#ffffff")
            if col == 2:
                cell.get_text().set_weight("bold")
    ax.set_title("Projected Best Third-Place Qualifiers", fontsize=16, fontweight="bold", color="#17384d", pad=16)
    paths["thirds"] = ASSET_DIR / "best_thirds.png"
    fig.savefig(paths["thirds"], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    r32 = ko_table[ko_table["round"] == "Round of 32"]
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.axis("off")
    ax.text(0.02, 0.95, "Round of 32 - Upper Half", fontsize=14, fontweight="bold", color="#17384d", transform=ax.transAxes)
    ax.text(0.52, 0.95, "Round of 32 - Lower Half", fontsize=14, fontweight="bold", color="#17384d", transform=ax.transAxes)

    def draw_list(data, x0):
        y = 0.88
        for _, r in data.iterrows():
            ax.add_patch(plt.Rectangle((x0, y - 0.055), 0.45, 0.055, color="#f4f6f7", ec="#d1d8dd", transform=ax.transAxes))
            ax.text(x0 + 0.012, y - 0.028, f"M{int(r.match)}: {r.team_a} vs {r.team_b}", fontsize=9.2, va="center", transform=ax.transAxes)
            ax.text(x0 + 0.33, y - 0.028, f"=> {r.winner}", fontsize=9.2, va="center", fontweight="bold", color="#245b7a", transform=ax.transAxes)
            y -= 0.095

    draw_list(r32.iloc[:8], 0.02)
    draw_list(r32.iloc[8:], 0.52)
    paths["r32"] = ASSET_DIR / "round_of_32.png"
    fig.savefig(paths["r32"], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    round_order = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"]
    round_x = {name: i for i, name in enumerate(round_order)}
    y_pos = {}
    for rn in round_order:
        matches = ko_table[ko_table["round"] == rn]["match"].tolist()
        for i, m in enumerate(matches):
            y_pos[m] = len(matches) - i
    parents = {89: [74, 77], 90: [73, 75], 91: [76, 78], 92: [79, 80], 93: [83, 84], 94: [81, 82], 95: [86, 88], 96: [85, 87], 97: [89, 90], 98: [93, 94], 99: [91, 92], 100: [95, 96], 101: [97, 98], 102: [99, 100], 104: [101, 102]}
    fig, ax = plt.subplots(figsize=(13.8, 8.4))
    for _, r in ko_table[ko_table["round"].isin(round_order)].iterrows():
        x, y = round_x[r["round"]], y_pos[r["match"]]
        label = f"M{int(r.match)}  {r.team_a} vs {r.team_b}\n{r.winner} ({r.winner_probability * 100:.0f}%)"
        ax.text(x, y, label, ha="center", va="center", fontsize=7.3, bbox=dict(boxstyle="round,pad=0.25", fc="#fde7a1" if r["round"] == "Final" else "#f7f7f7", ec="#777777", lw=0.8))
    for child, parent_matches in parents.items():
        cx, cy = round_x[ko[child]["round"]], y_pos[child]
        for parent in parent_matches:
            px, py = round_x[ko[parent]["round"]], y_pos[parent]
            ax.plot([px + 0.39, cx - 0.39], [py, cy], color="#888888", lw=0.8)
    ax.text(5.08, y_pos[104], f"Champion\n{ko[104]['winner']}", ha="center", va="center", fontsize=12, weight="bold", bbox=dict(boxstyle="round,pad=0.38", fc="#d8a23a", ec="#8c6b1f"))
    ax.plot([4.39, 4.74], [y_pos[104], y_pos[104]], color="#888888", lw=0.8)
    ax.set_xlim(-0.6, 5.55)
    ax.set_ylim(0, 17)
    ax.set_yticks([])
    ax.set_xticks(list(round_x.values()) + [5])
    ax.set_xticklabels(round_order + ["Champion"], fontsize=10)
    ax.set_title("Predicted Knockout Bracket", fontsize=18, fontweight="bold", color="#17384d")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    paths["bracket"] = ASSET_DIR / "knockout_bracket.png"
    fig.savefig(paths["bracket"], facecolor="white")
    plt.close(fig)

    final_four = pd.DataFrame([
        ["Semifinal 1", ko[101]["team_a"], ko[101]["team_b"], ko[101]["winner"], f"{ko[101]['winner_probability'] * 100:.0f}%"],
        ["Semifinal 2", ko[102]["team_a"], ko[102]["team_b"], ko[102]["winner"], f"{ko[102]['winner_probability'] * 100:.0f}%"],
        ["Third place", ko[103]["team_a"], ko[103]["team_b"], ko[103]["winner"], f"{ko[103]['winner_probability'] * 100:.0f}%"],
        ["Final", ko[104]["team_a"], ko[104]["team_b"], ko[104]["winner"], f"{ko[104]['winner_probability'] * 100:.0f}%"],
    ], columns=["Stage", "Team A", "Team B", "Pick", "Pick Prob."])
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=final_four.values, colLabels=final_four.columns, cellLoc="left", colLoc="left", loc="center", colColours=["#17384d"] * 5)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.55)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 else "#ffffff")
            if col == 3:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color("#245b7a")
    ax.set_title("Final Four Projection", fontsize=17, fontweight="bold", color="#17384d", pad=14)
    paths["finalfour"] = ASSET_DIR / "final_four.png"
    fig.savefig(paths["finalfour"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


EMU = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000


def emu(inches):
    return int(inches * EMU)


def solid(hexval):
    return f'<a:solidFill><a:srgbClr val="{hexval}"/></a:solidFill>'


def rect(shape_id, x, y, w, h, fill="FFFFFF", line="FFFFFF"):
    line_xml = f"<a:ln>{solid(line)}</a:ln>" if line else "<a:ln><a:noFill/></a:ln>"
    return f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Rect {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{solid(fill)}{line_xml}</p:spPr></p:sp>'


def textbox(shape_id, x, y, w, h, text, size=18, color="17384D", bold=False, fill=None, line=None):
    paras = []
    for para in str(text).split("\n"):
        b = ' b="1"' if bold else ""
        paras.append(f'<a:p><a:pPr algn="l"/><a:r><a:rPr lang="en-US" sz="{int(size * 100)}"{b}>{solid(color)}<a:latin typeface="Aptos"/></a:rPr><a:t>{escape(para)}</a:t></a:r><a:endParaRPr lang="en-US" sz="{int(size * 100)}"/></a:p>')
    fill_xml = solid(fill) if fill else "<a:noFill/>"
    line_xml = f"<a:ln>{solid(line)}</a:ln>" if line else "<a:ln><a:noFill/></a:ln>"
    return f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr><p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720"/><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>'


def picture(shape_id, rid, x, y, w, h, name="Image"):
    return f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{escape(name)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'


def slide_xml(shapes):
    body = "".join(shapes)
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:bg><p:bgPr>{solid("FFFFFF")}<a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{body}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'


def slide_rels(image_targets):
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
    for idx, target in enumerate(image_targets, start=2):
        rels.append(f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>')
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(rels)}</Relationships>'


def build_ppt(paths, ko):
    media = []
    slides = []

    def media_target(path):
        target = f"image{len(media) + 1}.png"
        media.append((Path(path), target))
        return target

    def add(shapes, images=None):
        slides.append((shapes, images or []))

    add([
        rect(2, 0, 0, 13.333, 0.18, "D8A23A", "D8A23A"),
        textbox(3, 0.65, 0.65, 10.5, 0.8, "2026 FIFA World Cup Prediction", 34, "17384D", True),
        textbox(4, 0.68, 1.52, 9.5, 0.45, "Bracket visualization and things to watch", 18, "5C6B73"),
        textbox(5, 0.72, 2.45, 4.1, 1.6, "Champion Pick\nFrance", 34, "FFFFFF", True, "17384D", "17384D"),
        textbox(6, 5.15, 2.45, 3.2, 1.6, "Runner-up\nArgentina", 26, "17384D", True, "F7F7F7", "D1D8DD"),
        textbox(7, 8.65, 2.45, 3.2, 1.6, "Third Place\nSpain", 26, "17384D", True, "F7F7F7", "D1D8DD"),
        textbox(8, 0.72, 4.65, 11.7, 0.95, "Built from the current notebook model: FIFA rank, host bonus, squad-depth bucket, and player-form modifier. No new API fetches were used.", 16, "17384D"),
        textbox(9, 0.72, 6.6, 5.8, 0.3, "Generated 2026-04-30", 10, "7A858B"),
    ])

    img = media_target(paths["contenders"])
    add([
        rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"),
        textbox(3, 0.55, 0.32, 6.6, 0.45, "Model Snapshot", 24, "17384D", True),
        textbox(4, 0.58, 0.92, 4.25, 4.6, "Rating formula\n100 - 0.8 x FIFA rank\n+ host bonus\n+ player-form modifier\n+ squad-depth modifier\n\nKnockout probability\nlogistic((rating A - rating B) / 8)\n\nTransparent scenario model, not a betting market.", 14, "17384D"),
        picture(5, "rId2", 5.0, 1.0, 7.45, 4.15, "Title contenders"),
        textbox(6, 0.58, 6.25, 11.5, 0.45, "Top-tier calls are close: France over Spain and Argentina over England are both narrow semifinal picks.", 14, "5C6B73"),
    ], [img])

    img = media_target(paths["groups"])
    add([rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"), textbox(3, 0.55, 0.28, 6.8, 0.45, "Group Stage Projection", 24, "17384D", True), picture(4, "rId2", 0.43, 0.82, 12.45, 6.18, "Projected groups")], [img])

    img = media_target(paths["thirds"])
    add([
        rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"),
        textbox(3, 0.55, 0.32, 7.4, 0.45, "Best Third-Place Qualifiers", 24, "17384D", True),
        picture(4, "rId2", 0.7, 1.02, 7.25, 3.6, "Best thirds"),
        textbox(5, 8.35, 1.08, 4.05, 3.6, "Why this matters\n- Eight of twelve third-place teams advance.\n- The exact groups that advance change the official Annex C bracket mapping.\n- This model's third-place set is A, C, D, E, G, I, J, L.\n- Egypt, Algeria, Cote d'Ivoire, Australia, Norway, and Panama are the hinge teams.", 14, "17384D"),
        textbox(6, 0.72, 5.3, 11.8, 0.9, "The third-place pool is the biggest bracket uncertainty: one extra group-stage goal can redraw an entire route.", 16, "5C6B73", True),
    ], [img])

    img = media_target(paths["r32"])
    add([rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"), textbox(3, 0.55, 0.3, 7.6, 0.45, "Round Of 32 Matchups", 24, "17384D", True), picture(4, "rId2", 0.52, 0.86, 12.3, 5.9, "Round of 32")], [img])

    img = media_target(paths["bracket"])
    add([rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"), picture(3, "rId2", 0.22, 0.32, 12.9, 6.85, "Bracket")], [img])

    img = media_target(paths["finalfour"])
    add([
        rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"),
        textbox(3, 0.55, 0.32, 6.5, 0.45, "Final Four", 24, "17384D", True),
        picture(4, "rId2", 1.2, 1.08, 10.6, 3.7, "Final four"),
        textbox(5, 0.85, 5.35, 11.7, 0.8, "Projected route: France beats Spain, Argentina beats England, then France beats Argentina in the final. Spain edges England for third place.", 17, "17384D", True),
    ], [img])

    add([
        rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"),
        textbox(3, 0.6, 0.35, 7.4, 0.45, "Things To Watch", 24, "17384D", True),
        textbox(4, 0.8, 1.18, 5.65, 4.9, "Tournament variables\n- Final squads and injuries\n- June 10 FIFA ranking update\n- Host-continent travel and climate\n- Third-place mapping volatility\n- Penalty shootout variance\n- Refereeing and game-state effects", 17, "17384D", False, "F7F7F7", "D1D8DD"),
        textbox(5, 6.9, 1.18, 5.65, 4.9, "Player-form variables\n- Rodri/Spain availability and minutes\n- Messi/Argentina workload\n- Mbappe and France attacking form\n- Kane/Bellingham/Saka finishing output\n- Haaland/Odegaard effect on Norway's upset risk\n- Goalkeeper form in knockout games", 17, "17384D", False, "F7F7F7", "D1D8DD"),
    ])

    add([
        rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"),
        textbox(3, 0.6, 0.35, 8.4, 0.45, "How To Refresh The Scenario", 24, "17384D", True),
        textbox(4, 0.8, 1.15, 11.7, 3.0, "1. Update FIFA rankings when the June 2026 list is published.\n2. Refresh the player-form modifier after final squads, injuries, and late club matches.\n3. Re-run the notebook cells from top to bottom.\n4. Rebuild this deck from the current notebook outputs if the bracket path changes.", 18, "17384D"),
        textbox(5, 0.8, 4.55, 11.65, 1.1, "Best next enhancement: add a player-form input table for recent ratings, minutes, goals/assists, defensive actions, goalkeeper saves, and injury status.", 17, "FFFFFF", True, "17384D", "17384D"),
    ])

    add([
        rect(2, 0, 0, 13.333, 0.14, "D8A23A", "D8A23A"),
        textbox(3, 0.6, 0.35, 5.0, 0.45, "Sources And Caveats", 24, "17384D", True),
        textbox(4, 0.8, 1.05, 11.9, 4.8, "Sources used in the notebook\n- FIFA men's ranking page, April 2026 update\n- ESPN April 2026 FIFA ranking table\n- FIFA qualified-teams article and official schedule article\n- Official bracket template summarized via the 2026 knockout-stage reference\n- DR Congo playoff placement reporting\n\nCaveats\n- Player-form modifiers are transparent subjective inputs.\n- Exact FIFA points were not used for every team.\n- Third-place qualification and mapping are highly sensitive.\n- Final squads and injuries may materially change the bracket.", 15, "17384D"),
    ])

    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ] + [f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, len(slides) + 1)]
    content_types = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>{"".join(overrides)}</Types>'
    root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>'
    sld_ids = "".join([f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, len(slides) + 1)])
    pres = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle></p:presentation>'
    pres_rels_items = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'] + [f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, len(slides) + 1)]
    pres_rels = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(pres_rels_items)}</Relationships>'
    master = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    master_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>'
    layout = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    layout_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>'
    theme = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="WorldCup26"><a:themeElements><a:clrScheme name="WorldCup26"><a:dk1><a:srgbClr val="17384D"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="245B7A"/></a:dk2><a:lt2><a:srgbClr val="F7F7F7"/></a:lt2><a:accent1><a:srgbClr val="17384D"/></a:accent1><a:accent2><a:srgbClr val="D8A23A"/></a:accent2><a:accent3><a:srgbClr val="5B8C5A"/></a:accent3><a:accent4><a:srgbClr val="D98859"/></a:accent4><a:accent5><a:srgbClr val="7AA6C2"/></a:accent5><a:accent6><a:srgbClr val="5C6B73"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="WorldCup26"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>'

    with ZipFile(PPTX_PATH, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("ppt/presentation.xml", pres)
        z.writestr("ppt/_rels/presentation.xml.rels", pres_rels)
        z.writestr("ppt/slideMasters/slideMaster1.xml", master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels)
        z.writestr("ppt/theme/theme1.xml", theme)
        for src, target in media:
            z.write(src, f"ppt/media/{target}")
        for i, (shapes, imgs) in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(shapes))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels(imgs))
    return len(slides)


if __name__ == "__main__":
    teams, standings, best_thirds, ko_table, ko = build_prediction()
    paths = make_charts(teams, standings, best_thirds, ko_table, ko)
    slide_count = build_ppt(paths, ko)
    print(f"Created {PPTX_PATH}")
    print(f"Slides: {slide_count}")
    print(f"Assets: {ASSET_DIR}")
    print(f"Champion: {ko[104]['winner']}; runner-up: {ko[104]['loser']}; third: {ko[103]['winner']}")
