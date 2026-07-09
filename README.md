<div align="center">

# World Cup 2026 Prediction Lab

An evolving, auditable tournament forecast that blends results, player form,
human tactical reads, market priors, ticket planning, and bracket visualization.

[![Notebook](https://img.shields.io/badge/Jupyter-analysis-F37626?logo=jupyter&logoColor=white)](notebooks/worldcup_2026_prediction_bracket.ipynb)
[![Interactive bracket](https://img.shields.io/badge/interactive-bracket-26c6b8)](outputs/worldcup_2026_interactive_bracket.html)
[![Latest tables](https://img.shields.io/badge/latest-tables-6da8ff)](outputs/tables/worldcup_2026_updated_tables_20260707.xlsx)
[![PowerPoint](https://img.shields.io/badge/slides-PowerPoint-b7472a)](outputs/presentations/worldcup_2026_prediction_bracket_deck.pptx)

If GitHub Pages is enabled from the repo root, open the app at:

**https://zyy7390.github.io/worldcup26/**

</div>

## Current Snapshot

Last model refresh: **July 7, 2026, about 7:13 p.m. ET**

| Result | Team |
|---|---|
| Champion | France |
| Runner-up | Argentina |
| Third place | Spain |
| Fourth place | England |

![Projected knockout bracket](assets/ppt/knockout_bracket.png)

## Explore The Work

| Artifact | What it is for |
|---|---|
| [Prediction notebook](notebooks/worldcup_2026_prediction_bracket.ipynb) | Full analysis, formulas, model assumptions, plots, and executed results. |
| [Interactive bracket](outputs/worldcup_2026_interactive_bracket.html) | Filterable bracket view with match details, country highlighting, zoom controls, recommendations, ratings, and market diagnostic views. |
| [Updated tables workbook](outputs/tables/worldcup_2026_updated_tables_20260707.xlsx) | Shareable Excel workbook with live bracket, market-integrated bracket, recommendations, and summary. |
| [Projected knockout CSV](outputs/tables/worldcup_2026_projected_knockout_schedule_20260707.csv) | Machine-readable latest live bracket. |
| [Recommendations CSV](outputs/tables/worldcup_2026_match_recommendations_20260707.csv) | Ticket-focused match shortlist with matchup, price, date, weekday, and weather factors. |
| [Presentation deck](outputs/presentations/worldcup_2026_prediction_bracket_deck.pptx) | A concise visual slide deck for sharing the bracket and story. |
| [Notebook archive](notebooks/archive/worldcup_2026_prediction_bracket.pre_live_update_20260617.ipynb) | Sanitized pre-live-update notebook snapshot available in the current project files. |

## Projected Last Eight

| Round | Match | Date | Weekday | Matchup | Pick |
|---|---:|---|---|---|---|
| Quarterfinal | 97 | 2026-07-09 | Thursday | France vs Morocco | France, 72.3% |
| Quarterfinal | 98 | 2026-07-10 | Friday | Spain vs Belgium | Spain, 75.8% |
| Quarterfinal | 99 | 2026-07-11 | Saturday | Norway vs England | England, 94.6% |
| Quarterfinal | 100 | 2026-07-11 | Saturday | Argentina vs Switzerland | Argentina, 90.4% |
| Semifinal | 101 | 2026-07-14 | Tuesday | France vs Spain | France, 53.5% |
| Semifinal | 102 | 2026-07-15 | Wednesday | England vs Argentina | Argentina, 56.7% |
| Third-place Match | 103 | 2026-07-18 | Saturday | Spain vs England | Spain, 54.4% |
| Final | 104 | 2026-07-19 | Sunday | France vs Argentina | France, 51.2% |

```mermaid
flowchart LR
  QF97["M97 Quarterfinal<br/>France 72.3% over Morocco"] --> SF101["M101 Semifinal<br/>France 53.5% over Spain"]
  QF98["M98 Quarterfinal<br/>Spain 75.8% over Belgium"] --> SF101
  QF99["M99 Quarterfinal<br/>England 94.6% over Norway"] --> SF102["M102 Semifinal<br/>Argentina 56.7% over England"]
  QF100["M100 Quarterfinal<br/>Argentina 90.4% over Switzerland"] --> SF102
  SF101 --> Final["M104 Final<br/>France 51.2% over Argentina"]
  SF102 --> Final
  SF101 --> Third["M103 Third Place<br/>Spain 54.4% over England"]
  SF102 --> Third
```

## Player Value vs Performance

The notebook compares World Cup post-match ratings against Transfermarkt market
value. The chart below uses the saved Google lineup rating extract where
available and highlights star players from teams still alive in the quarterfinal
field.

![Player rating vs Transfermarkt value](assets/readme/player_value_vs_transfermarkt.png)

The interactive bracket app also includes this value/performance chart as a
clickable SVG: select a country to highlight it across the bracket and chart,
click a player dot to focus that country, and use zoom controls for closer
inspection.

## Games To Watch

The recommendation model prioritizes the user's team preferences first, then
price fit, then date and weather convenience.

| Rank | Match | Round | Weekday | Possible matchup | Score |
|---:|---:|---|---|---|---:|
| 1 | 100 | Quarterfinal | Saturday | Argentina vs Switzerland | 4.56 |
| 2 | 102 | Semifinal | Wednesday | England vs Argentina | 4.44 |
| 3 | 99 | Quarterfinal | Saturday | Norway vs England | 4.43 |
| 4 | 104 | Final | Sunday | France vs Argentina | 4.31 |
| 5 | 101 | Semifinal | Tuesday | France vs Spain | 4.16 |

<details>
<summary><strong>How the model works</strong></summary>

The live rating for each team is built from a transparent stack of signals:

```text
updated_rating =
  FIFA-rank base
+ host bonus
+ squad-depth modifier
+ live result shock
+ player-form signal
+ human tactical prior
+ prediction-market prior
```

The notebook separates these layers so each update can be reviewed. Human reads
are deliberately explicit rather than hidden inside a vague "form" number.
Prediction market data is blended lightly with `market_weight = 0.10`, so it can
correct the model without taking over the forecast.

</details>

<details open>
<summary><strong>Human analytics ledger</strong></summary>

These are the historical analyst reads captured during the project and folded
into either player-form signals, team tactical deltas, or matchup-specific
priors.

| Theme | Human read | Model treatment |
|---|---|---|
| Korea vs Canada | Canada's Qatar win mattered, but Korea's lineup quality still deserved respect even with Son Heung-min aging. | Korea was treated as a narrow 50-60% style favorite in that matchup prior. |
| Morocco vs Japan | Morocco's AFCON/2022 pedigree was strong, but Japan's teamwork and results made them dangerous. | Morocco edge compressed to roughly 60-70%, not a blowout. |
| Portugal vs Croatia | Portugal had elite names but poor team play; Ronaldo's low mobility, offside-line positioning, and chance selection were major drags. | Croatia-leaning analyst prior and Portugal tactical penalty. |
| Morocco vs Norway | Morocco were clearly stronger overall, but Norway's Haaland counterattacking path could punish tiny chances. | Morocco favored around 70-80%; Norway upset path kept alive. |
| Mexico vs England | England's talent/pedigree were superior, but Mexico's collective running, passing, dribbling, and ball progression looked much better than the base model credited. | England remained favored, but Mexico's tactical upgrade compressed the gap before England advanced. |
| Portugal vs Spain | Spain's structure and control looked materially better than Portugal's chance creation. | Spain received a strong head-to-head prior over Portugal. |
| Argentina vs Belgium | Belgium looked diminished from its peak; Lukaku struggled as a starter, and draws against Egypt/Iran reduced confidence. | Argentina received a very strong matchup prior over Belgium. |
| Cristiano Ronaldo aging curve | Ronaldo's post-match rating understated the problem: weak chance conversion, limited facilitation, little pressing, and low back-and-forth impact. | Ronaldo signals were discounted and Portugal team-play was penalized. |
| Portugal vs Uzbekistan revision | Portugal showed better cohesion and Ronaldo participation against Uzbekistan, but 90 minutes for Ronaldo still looked suboptimal and glory-driven. | Portugal's team signal improved slightly, but Ronaldo/role concerns remained capped. |
| Harry Kane clutch signal | Kane showed comprehensive striker play and superstar late-match deciding power. | England's late-game ceiling and Kane player-form signal were upgraded. |
| Mexico collective quality | Mexico looked almost Argentina-like in team movement and progression despite lacking a single global superstar. | Mexico received a team-cohesion tactical upgrade. |
| Portugal post-Uzbekistan caution | Portugal still did not create enough repeatable chances; Ronaldo's brace looked more like opponent error/luck than proof of high-level current impact. | Portugal retained a negative tactical delta. |
| Belgium psychology | Belgium lacked title-winning ferocity; the Senegal win looked lucky/even, and Lukaku seemed better as a bench option. | Belgium's title profile was downgraded despite later USA conversion. |
| USA quality | USA looked tenacious, physical, technical, and coordinated, better than player market value implied. | USA received a collective-quality upgrade before elimination. |
| France vs Paraguay | France showed next-level tenacity and attacking talent even in a narrow knockout win. | France received a title-ceiling tactical upgrade. |
| Argentina vs Cape Verde/Egypt | Argentina showed tenacity, but Messi overreliance and lower supporting-player running/creativity were worrying. | Argentina kept resilience credit but received a Messi-dependence penalty. |
| England vs Mexico | England showed collaboration, attacking talent, and strong 10-man defensive resilience. | England tactical delta increased after the R16. |
| Spain latest form | Spain remained structurally good, but Lamine Yamal and midfield chance creation were muted; Oyarzabal was not a traditional No. 9 finisher. | Spain kept control credit but received a finishing/chance-creation caution. |
| Switzerland vs Colombia | Switzerland's defense looked excellent, but open-play offense remained limited. | Switzerland received defensive credit and attacking ceiling caution. |
| Norway vs Brazil | Norway could catch elite opponents off guard; if Odegaard/Nusa can find Haaland, the chances are extremely dangerous. | Norway received a transition-threat upgrade, offset by possession/build-up limitations. |

</details>

<details>
<summary><strong>Data sources and caveats</strong></summary>

Core inputs include:

- Official-style match schedule and bracket structure.
- Latest locked match results through Switzerland 0-0 Colombia, Switzerland 4-3 on penalties.
- Saved Google lineup player-rating extracts from the July 2 sweep.
- Event and player-performance proxies where rendered ratings were unavailable.
- Polymarket July 7 outright market snapshot.
- Kalshi public-market snapshots kept as diagnostics when markets were not clean single-team or W-D-L priors.

Important caveat: the July 7 Google player-rating scrape hit Google's
"sorry/unusual traffic" page, so new Round of 16 player-rating cards were not
scraped in that run. The notebook preserves the existing 2,334 saved Google
player-rating rows and uses result proxies plus human reads for the latest R16
matches.

</details>

<details>
<summary><strong>Repository layout</strong></summary>

```text
assets/
  ppt/       bracket and deck images
  readme/    README showcase images
data/
  google/    scraped and normalized lineup ratings
  markets/   Polymarket and Kalshi snapshots
notebooks/   executed analysis notebook
outputs/
  tables/    CSV/XLSX summaries
  presentations/
scripts/     generators, extractors, repair utilities
```

</details>

<details>
<summary><strong>Reproduce the latest notebook run</strong></summary>

Use the same Python environment that has `pandas`, `matplotlib`, `nbformat`, and
`jupyter` installed. From the repository root:

```powershell
python scripts/update_worldcup26_notebook_live.py

$env:USERPROFILE = Join-Path (Get-Location) ".home"
$env:HOME = Join-Path (Get-Location) ".home"
$env:JUPYTER_CONFIG_DIR = Join-Path (Get-Location) ".jupyter-config"
$env:JUPYTER_DATA_DIR = Join-Path (Get-Location) ".jupyter-data"
$env:JUPYTER_RUNTIME_DIR = Join-Path (Get-Location) ".jupyter-runtime"
$env:IPYTHONDIR = Join-Path (Get-Location) ".ipython"
$env:WORLDCUP26_SKIP_WIDGETS = "1"

python -m jupyter nbconvert `
  --to notebook --execute --inplace notebooks/worldcup_2026_prediction_bracket.ipynb `
  --ExecutePreprocessor.timeout=900

python scripts/make_worldcup26_interactive_html.py
python scripts/make_readme_assets.py
```

</details>

<details>
<summary><strong>Git update workflow</strong></summary>

Use one commit per tournament update so model accuracy can be audited later:

```powershell
git status
git add .
git commit -m "Update model after latest World Cup matches"
git push
```

For comparison over time:

```powershell
git log --oneline
git diff HEAD~1 -- outputs/tables/worldcup_2026_projected_knockout_schedule_20260707.csv
```

</details>

## Why This Exists

The goal is not to pretend a model can remove uncertainty from football. The
goal is to make every belief visible: what came from the data, what came from
market prices, what came from match watching, and how those beliefs changed as
the tournament unfolded.
