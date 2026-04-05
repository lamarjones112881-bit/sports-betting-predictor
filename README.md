# Sports Betting Predictor

A Streamlit app for NBA and NFL betting analysis using real historical game data, live sportsbook markets, bankroll management, and persistent tracking.

## What it does

- Trains baseline winner, spread, and total models from real NBA and NFL game history when available
- Builds rolling team features such as recent scoring, win form, and rest days
- Pulls live sportsbook moneyline, spread, and totals markets from The Odds API
- Highlights moneyline value bets and shows expected value per wager
- Compares model spread and total projections against live market lines
- Suggests conservative bankroll sizing with a capped Kelly approach
- Generates AI summaries when `GEMINI_API_KEY` is configured
- Persists tracked bets and running profit/loss in `bet_history.csv`

## Run locally

```bash
cd ~/sports-betting-predictor
python3 -m streamlit run app.py
```

## Requirements

- Python 3.13
- streamlit
- pandas
- scikit-learn
- numpy
- requests
- nba_api
- nfl_data_py
- google-genai

## Environment variables

- `ODDS_API_KEY`: required for live sportsbook odds
- `GEMINI_API_KEY`: optional, enables AI betting summaries

## Notes

If real historical data cannot be loaded, the app falls back to synthetic data so the UI remains usable. Treat the output as decision support, not guaranteed betting advice.
