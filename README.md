# 🏆 Fantasy Vibes — Multi-League Football App

A Streamlit web app for exploring football league data:
fixtures, live results, tables, team stats and match predictions. It is designed around a
swappable, multi-source data layer so new leagues and providers can be added easily.

## ✨ Features

- 🏆 **Multi-league support** — Premier League, Bundesliga, Serie A, La Liga, Ligue 1.
- 📡 **Multiple free data sources** — football-data.org, thesportsdb.com and the public
  `openfootball/football.json` dataset.
- 🌙 **Night mode UI** — dark theme by default with custom cards and Plotly charts.
- 🔮 **Match predictions** — attack/defense ratings + Poisson model, with neutral-venue toggle.
- 📅 **Fixtures** — full schedule, filter by round/date, predict any match.
- 📊 **Table & stats** — standings, team form, head-to-head and league trend charts.

## 🚀 Quick Start

```powershell
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Add free API keys
Copy-Item .env.example .env
# Edit .env with your FOOTBALL_DATA_API_KEY and/or THESPORTSDB_API_KEY

# Run the app
streamlit run app/main.py
```

Open `http://localhost:8501`.

## 📂 Project Structure

```
epl-app/
├── app/
│   ├── main.py              # Entry point, navigation, league selector
│   ├── ui.py                # Dark theme, shared components and charts
│   └── pages/               # Home, Live, Fixtures, Table, Predictions, Stats, About
├── src/
│   └── data/
│       ├── league_service.py # High-level data + prediction service
│       ├── utils.py          # Cached HTTP client
│       └── sources/          # Football-Data, TheSportsDB, Openfootball, Composite
├── data/
│   ├── leagues.json          # League registry (easy to extend)
│   └── cache/                # On-disk HTTP cache
├── .streamlit/config.toml    # Dark theme config
├── .env.example              # API key template
└── requirements.txt
```

## 📡 Data Sources

| Source | API key | Best for |
|---|---|---|
| Football-Data.org | Required (free) | Current season live matches & standings |
| TheSportsDB | Optional (demo key `3`) | Tables & fixtures |
| Openfootball | None | Full season schedules & results, public domain |

If no API keys are configured the app automatically falls back to **Openfootball** data.

## ➕ Adding a New League

Edit `data/leagues.json` and add a new entry with `openfootball`, `football_data` and/or
`thesportsdb` configuration. For example:

```json
"PLE": {
  "name": "Primeira Liga",
  "openfootball": {"season": "2025-26", "path": "pt.1.json"},
  "football_data": {"code": "PPL", "season": "2025"}
}
```

## ⚠️ Disclaimer

Predictions and stats are for entertainment only. This app is not affiliated with any
football league or governing body. Data is sourced from free public APIs and datasets.
