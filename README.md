# MLB Model & Value

- `mlb/index.html` — dashboard (slate, edges, ratings, method). Reads `mlb/data/slate.json`.
- `.github/workflows/mlb-daily.yml` — refreshes the slate every morning automatically.

## One-time setup
1. Repo Settings -> Secrets and variables -> Actions -> New repository secret:
   name `ODDS_API_KEY`, value = your the-odds-api.com key.
2. Settings -> Pages -> deploy from branch `main`, folder `/ (root)`.
3. Actions tab -> "MLB daily slate" -> Run workflow (first manual run; daily after that).

Dashboard URL: `https://<user>.github.io/<repo>/mlb/`
