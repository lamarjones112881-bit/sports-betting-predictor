# Sports Betting Predictor

A Streamlit demo app for predicting NBA and NFL game outcomes.

## What it does

- Predicts game winner probability
- Predicts point spread
- Predicts total score
- Estimates parlay probabilities
- Factors in weather conditions (for NFL outdoor games)
- Displays up-to-date injury reports
- Uses synthetic baseline data for demo purposes

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

## Notes

This project currently uses synthetic data. Replace the dataset with real game data and betting odds for more realistic predictions. Weather is factored in for NFL games but set to neutral for NBA (indoor).
