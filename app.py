import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import requests
from nba_api.stats.endpoints import injuryreport
import nfl_data_py as nfl

st.set_page_config(page_title="Sports Betting Predictor", layout="wide")

st.title("Sports Betting Predictor")
st.write("A demo app for NBA and NFL predictions: game winner, point spread, and total score.")


@st.cache_data
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@st.cache_data
def generate_synthetic_data(sport: str, n_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    home_rating = np.random.randint(70, 101, size=n_samples)
    away_rating = np.random.randint(70, 101, size=n_samples)
    home_form = np.random.rand(n_samples)
    away_form = np.random.rand(n_samples)
    home_rest = np.random.randint(2, 8, size=n_samples)
    away_rest = np.random.randint(2, 8, size=n_samples)
    home_adv = np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8])

    if sport == "NBA":
        base_total = 215
        total_noise = np.random.normal(0, 12, size=n_samples)
        # NBA is indoor, so weather neutral
        temperature = np.zeros(n_samples)
        wind_speed = np.zeros(n_samples)
        precipitation = np.zeros(n_samples)
    else:
        base_total = 45
        total_noise = np.random.normal(0, 7, size=n_samples)
        # NFL outdoor weather
        temperature = np.random.normal(60, 20, size=n_samples)  # F
        wind_speed = np.random.normal(10, 5, size=n_samples)  # mph
        precipitation = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])  # 0 or 1

    rating_diff = home_rating - away_rating
    form_diff = home_form - away_form
    rest_diff = home_rest - away_rest

    # Weather impact (more for NFL)
    weather_impact = 0 if sport == "NBA" else (temperature - 60) * 0.01 + wind_speed * 0.02 + precipitation * 0.05

    margin = 0.5 * rating_diff + 8 * form_diff + 0.8 * rest_diff + 5 * home_adv + weather_impact * 10
    total = base_total + 1.2 * (home_rating + away_rating - 160) + 18 * (home_form + away_form - 1) + 1.1 * (home_rest + away_rest - 8) + total_noise + weather_impact * 5

    winner_prob = sigmoid(0.06 * rating_diff + 4 * form_diff + 0.4 * rest_diff + 1.5 * home_adv + weather_impact)
    winner = (winner_prob > 0.5).astype(int)

    data = pd.DataFrame(
        {
            "home_rating": home_rating,
            "away_rating": away_rating,
            "home_form": home_form,
            "away_form": away_form,
            "home_rest": home_rest,
            "away_rest": away_rest,
            "home_advantage": home_adv,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
            "rating_diff": rating_diff,
            "form_diff": form_diff,
            "rest_diff": rest_diff,
            "margin": margin + np.random.normal(0, 4, size=n_samples),
            "total_score": total,
            "home_win": winner,
        }
    )
    return data


@st.cache_data
def train_models(data: pd.DataFrame):
    features = ["rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"]
    X = data[features]

    winner_model = LogisticRegression(max_iter=200)
    winner_model.fit(X, data["home_win"])

    spread_model = LinearRegression()
    spread_model.fit(X, data["margin"])

    total_model = LinearRegression()
    total_features = ["home_rating", "away_rating", "home_form", "away_form", "home_rest", "away_rest", "temperature", "wind_speed", "precipitation"]
    total_model.fit(data[total_features], data["total_score"])

    return winner_model, spread_model, total_model


def predict_outcome(sport: str, target: str, inputs: dict, models: tuple):
    rating_diff = inputs["home_rating"] - inputs["away_rating"]
    form_diff = inputs["home_form"] - inputs["away_form"]
    rest_diff = inputs["home_rest"] - inputs["away_rest"]
    home_advantage = 1

    X = np.array([[rating_diff, form_diff, rest_diff, home_advantage, inputs["temperature"], inputs["wind_speed"], inputs["precipitation"]]])
    total_X = np.array(
        [
            [
                inputs["home_rating"],
                inputs["away_rating"],
                inputs["home_form"],
                inputs["away_form"],
                inputs["home_rest"],
                inputs["away_rest"],
                inputs["temperature"],
                inputs["wind_speed"],
                inputs["precipitation"],
            ]
        ]
    )

    winner_model, spread_model, total_model = models

    if target == "Game winner":
        prob = winner_model.predict_proba(X)[0, 1]
        return {
            "home_win_prob": prob,
            "away_win_prob": 1 - prob,
            "prediction": "Home team" if prob > 0.5 else "Away team",
        }
    if target == "Point spread":
        margin = spread_model.predict(X)[0]
        return {
            "predicted_margin": margin,
            "predicted_winner": "Home team" if margin > 0 else "Away team",
        }
    if target == "Total score":
        total = total_model.predict(total_X)[0]
        return {"predicted_total": total}

    return {}


def compute_parlay_prob(probability: float, legs: int) -> float:
    return probability ** legs


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_injury_reports(sport: str, team: str = None):
    try:
        if sport == "NBA":
            injuries = injuryreport.InjuryReport()
            df = injuries.get_data_frames()[0]
            if team:
                df = df[df['TEAM_NAME'].str.contains(team, case=False, na=False)]
            return df
        elif sport == "NFL":
            # Get current season injuries
            current_year = pd.Timestamp.now().year
            injuries = nfl.import_injuries([current_year])
            if team:
                injuries = injuries[injuries['team'].str.upper() == team.upper()]
            return injuries
    except Exception as e:
        st.error(f"Error fetching injury reports: {e}")
        return pd.DataFrame()


with st.sidebar:
    st.header("Prediction settings")
    sport = st.selectbox("Sport", ["NBA", "NFL"])
    target = st.selectbox(
        "Prediction target",
        ["Game winner", "Point spread", "Total score"],
    )
    st.markdown("---")
    st.subheader("Team inputs")
    home_rating = st.slider("Home team rating", 70, 100, 85)
    away_rating = st.slider("Away team rating", 70, 100, 82)
    home_form = st.slider("Home recent form", 0.0, 1.0, 0.6, 0.05)
    away_form = st.slider("Away recent form", 0.0, 1.0, 0.55, 0.05)
    home_rest = st.slider("Home rest days", 2, 7, 4)
    away_rest = st.slider("Away rest days", 2, 7, 3)
    st.markdown("---")
    if sport == "NFL":
        st.subheader("Weather conditions")
        temperature = st.slider("Temperature (°F)", 0, 100, 60)
        wind_speed = st.slider("Wind speed (mph)", 0, 30, 10)
        precipitation = st.selectbox("Precipitation", [0, 1], format_func=lambda x: "Yes" if x else "No")
    else:
        temperature = 0
        wind_speed = 0
        precipitation = 0
    st.markdown("---")
    st.subheader("Parlay builder")
    parlay_legs = st.slider("Number of parlay legs", 1, 5, 1)
    parlay_choice = st.selectbox("Pick for parlay", ["Home win", "Away win"])
    st.write("Use the parlay builder to estimate the chance of all legs hitting together.")
    st.markdown("---")
    st.subheader("Injury Reports")
    injury_sport = st.selectbox("Sport for injuries", ["NBA", "NFL"], key="injury_sport")
    if injury_sport == "NBA":
        nba_teams = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"]
        injury_team = st.selectbox("NBA Team", nba_teams, key="nba_injury_team")
    else:
        nfl_teams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]
        injury_team = st.selectbox("NFL Team", nfl_teams, key="nfl_injury_team")
    if st.button("Fetch Injury Reports"):
        injuries_df = get_injury_reports(injury_sport, injury_team)
        if not injuries_df.empty:
            st.write(f"### Injury Reports for {injury_team}")
            st.dataframe(injuries_df)
        else:
            st.write("No injury reports found or error occurred.")
    st.markdown("---")
    show_data = st.checkbox("Show sample training data", value=False)

inputs = {
    "home_rating": home_rating,
    "away_rating": away_rating,
    "home_form": home_form,
    "away_form": away_form,
    "home_rest": home_rest,
    "away_rest": away_rest,
    "temperature": temperature,
    "wind_speed": wind_speed,
    "precipitation": precipitation,
}

st.subheader("Model training and results")

data = generate_synthetic_data(sport)
models = train_models(data)

if show_data:
    st.write("### Sample training data")
    st.dataframe(data.head())
    st.write("### Training dataset statistics")
    st.write(data[["margin", "total_score", "home_win"]].describe())

result = predict_outcome(sport, target, inputs, models)

cols = st.columns(2)
with cols[0]:
    st.write("### Selected matchup")
    st.write(f"**Home rating:** {home_rating}")
    st.write(f"**Away rating:** {away_rating}")
    st.write(f"**Home form:** {home_form:.2f}")
    st.write(f"**Away form:** {away_form:.2f}")
    st.write(f"**Home rest:** {home_rest} days")
    st.write(f"**Away rest:** {away_rest} days")
    st.write(f"**Sport:** {sport}")
    st.write(f"**Prediction target:** {target}")
    if sport == "NFL":
        st.write(f"**Temperature:** {temperature}°F")
        st.write(f"**Wind speed:** {wind_speed} mph")
        st.write(f"**Precipitation:** {'Yes' if precipitation else 'No'}")

with cols[1]:
    st.write("### Prediction output")
    if target == "Game winner":
        st.metric("Home team win probability", f"{result['home_win_prob']:.1%}")
        st.metric("Away team win probability", f"{result['away_win_prob']:.1%}")
        st.success(result["prediction"])

        parlay_prob = compute_parlay_prob(
            result["home_win_prob"] if parlay_choice == "Home win" else result["away_win_prob"],
            parlay_legs,
        )
        st.write(f"**Parlay pick:** {parlay_choice}")
        st.write(f"**Parlay legs:** {parlay_legs}")
        st.write(f"**Parlay probability:** {parlay_prob:.2%}")
        if parlay_legs > 1:
            st.write(f"**Approx. parlay odds:** {1 / parlay_prob:.1f}:1")
    elif target == "Point spread":
        st.metric("Predicted margin", f"{result['predicted_margin']:.1f} points")
        st.success(result["predicted_winner"])
    else:
        st.metric("Predicted total score", f"{result['predicted_total']:.1f}")

st.markdown("---")

st.write("## Notes")
st.write(
    "This app uses a demo synthetic dataset and baseline models. "
    "For real betting predictions, replace the synthetic data with real historical game data, odds, and team statistics."
)

st.write("### Model accuracy")
features = ["rating_diff", "form_diff", "rest_diff", "home_advantage"]
X = data[features]
y = data["home_win"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

winner_model, spread_model, total_model = models

home_prob = winner_model.predict_proba(X_test)[:, 1]
y_pred = winner_model.predict(X_test)

st.write(f"Home-win classification accuracy: {accuracy_score(y_test, y_pred):.3f}")

spread_preds = spread_model.predict(X_test)
mae_spread = mean_absolute_error(data.loc[X_test.index, "margin"], spread_preds)
st.write(f"Point spread MAE: {mae_spread:.2f} points")

total_features = ["home_rating", "away_rating", "home_form", "away_form", "home_rest", "away_rest"]
X_total = data[total_features]
X_total_train, X_total_test, y_total_train, y_total_test = train_test_split(
    X_total, data["total_score"], test_size=0.25, random_state=42
)

total_preds = total_model.predict(X_total_test)
mae_total = mean_absolute_error(y_total_test, total_preds)
st.write(f"Total score MAE: {mae_total:.2f} points")

st.info(
    "Use real game data and betting odds for more accurate predictions. "
    "This demo shows how to wire the model and prediction logic into Streamlit."
)
