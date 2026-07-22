# MLBTool / mlb — file manifest

**The durable backup is this repo itself.** Every file below lives on GitHub at
`https://raw.githubusercontent.com/CheeseDagg/MLBTool/main/mlb/<path>`. If a file goes
missing from your Desktop, pull it from there — you never need it re-sent. Claude can
re-pull any live file the same way. This manifest documents what each file is, how it's
produced, and how to deploy, so nothing depends on chat history.

Covers the files that matter for running/deploying. Not every data blob is listed.

---

## Scripts (you run / Actions run)

| File | What it does | Run |
|---|---|---|
| `mlb_hr.py` | HR board engine. Builds the day's HR% board from Baseball-Reference stats + Marcel talent + park/heat/platoon/bullpen/barrel. Writes `data/hr_board.json`, appends `data/hr_predictions.csv`. Includes the **spot** readability tags (today vs a bat's own recent median) and the Ben-Rice dedupe. | `python mlb_hr.py` |
| `mlb_bvp.py` | **BvP leaderboard.** Batters with the most career HR off the arm they face today (StatsAPI schedule + vsPlayer). Writes `data/bvp_board.json`. History/color only — not a model input. Min 6 career PA, top 15, ranked HR→SLG→PA. Fails soft. | `python mlb_bvp.py` (or `--probe` / `--selftest`) |
| `mlb_kprops.py` | Pitcher strikeout-props model (pure Poisson w/ regression). | via workflow |
| `mlb_kprops_run.py` | K-props Actions runner (whiff-pull hardening: raw REST primary + wrapper fallback). | via workflow |
| `mlb_publish.py` *(referenced by the dashboard's error text; not inspected in detail)* | Assembles `data/slate.json` — the master file the dashboard reads — by merging the HR board, games, ratings, edges, futures, calibration, etc. | your daily pipeline |

> **Run scripts from inside `mlb/`** so the `data/` folder sits beside them. If you run from
> anywhere else (e.g. `Desktop\New folder`), the output falls back to *next to the script*
> instead of `mlb/data/`, and the site won't see it.

## Dashboard

| File | What it does |
|---|---|
| `index.html` | The tool UI (homers / edges / parlays / slate / ratings / K-props / futures tabs). Fetches `data/slate.json` + `data/bvp_board.json` + `data/lineshop.json` + `data/kprops.json`. The homers tab shows the HR board, the **spot** tags, and the **Best HR history vs today's starter** (BvP) panel. |

## Data (`mlb/data/*` — produced by the scripts, consumed by the dashboard)

| File | Produced by | Used for |
|---|---|---|
| `slate.json` | `mlb_publish.py` | master file the dashboard reads (games, hr_board, hr_cal, backtest, ratings, edges, futures) |
| `hr_board.json` | `mlb_hr.py` | the HR board rows |
| `bvp_board.json` | `mlb_bvp.py` | the BvP panel |
| `hr_predictions.csv` | `mlb_hr.py` | dated daily board log (the spot baseline reads this) |
| `hr_graded.csv` | grader (next-morning) | settled outcomes for calibration |
| `hr_backtest.csv` | season replay | 25k-prediction walk-forward backtest |
| `hr_backtest_panel.json` | season replay | (was the changelog panel; panel now removed from UI) |
| `marcel_talent.json` | talent build | per-batter Marcel HR/PA baselines |
| `kprops.json` | K-props runner | K-props tab |
| `lineshop.json` | HR-shop workflow | line-shop odds (needs `ODDS_API_KEY`) |
| `futures.json` | futures build | World Series futures tab |

## Workflows (`.github/workflows/`)

| File | What |
|---|---|
| `mlb-kprops.yml` | K-props cron. Commit step rebases-before-push with a retry loop (fixes the push race). |
| *HR line-shop + daily-slate workflows* | daily board/odds/publish. **Note:** `.github/` is easy to drop during drag-and-drop uploads — check it survived on every deploy. |

---

## Deploy (any changed file)
1. Put the file in `mlb/` (overwrite), keeping `data/` beside the scripts.
2. Run the relevant script from inside `mlb/` if it produces data.
3. Commit + push the file **and** any `data/*.json` it wrote.
4. Hard-refresh the tool.

## Pending changes from our sessions — confirm these are pushed
- **BvP** (`index.html` + `mlb_bvp.py` + `data/bvp_board.json`) — in progress.
- Earlier fixes staged in their own `PUT-IN-*` folders: **K-props** `mlb-kprops.yml` push-race fix, **UFC** results home-runner. Confirm they got deployed.
- Housekeeping: rotate the exposed Odds API key; the quota was burned (0/500) — fresh account or monthly reset.

## Other repos in the suite (same durable-backup rule — pull from raw)
- `CheeseDagg/world-cup-2026-` — WC tool (`index.html` on `main`)
- `CheeseDagg/UFC-ODDS` — UFC results/odds (`Github/` subfolder)
- `CheeseDagg/TheTool` — hub at cheesedagg.github.io/TheTool
