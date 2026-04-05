import os
import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import requests
import nfl_data_py as nfl

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

try:
    from google import genai
except Exception:
    genai = None

ODDS_API_KEY = os.getenv("ODDS_API_KEY") or st.secrets.get("ODDS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
BET_HISTORY_PATH = Path(__file__).with_name("bet_history.csv")
PREFERENCES_PATH = Path(__file__).with_name("user_preferences.json")
MODEL_HISTORY_PATH = Path(__file__).with_name("model_history.json")

NBA_LOGO_MAP = {
    "Atlanta Hawks": "atl", "Boston Celtics": "bos", "Brooklyn Nets": "bkn", "Charlotte Hornets": "cha",
    "Chicago Bulls": "chi", "Cleveland Cavaliers": "cle", "Dallas Mavericks": "dal", "Denver Nuggets": "den",
    "Detroit Pistons": "det", "Golden State Warriors": "gs", "Houston Rockets": "hou", "Indiana Pacers": "ind",
    "LA Clippers": "lac", "Los Angeles Lakers": "lal", "Memphis Grizzlies": "mem", "Miami Heat": "mia",
    "Milwaukee Bucks": "mil", "Minnesota Timberwolves": "min", "New Orleans Pelicans": "no", "New York Knicks": "ny",
    "Oklahoma City Thunder": "okc", "Orlando Magic": "orl", "Philadelphia 76ers": "phi", "Phoenix Suns": "phx",
    "Portland Trail Blazers": "por", "Sacramento Kings": "sac", "San Antonio Spurs": "sa", "Toronto Raptors": "tor",
    "Utah Jazz": "uta", "Washington Wizards": "wsh",
}

NFL_LOGO_MAP = {
    "Arizona Cardinals": "ari", "Atlanta Falcons": "atl", "Baltimore Ravens": "bal", "Buffalo Bills": "buf",
    "Carolina Panthers": "car", "Chicago Bears": "chi", "Cincinnati Bengals": "cin", "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal", "Denver Broncos": "den", "Detroit Lions": "det", "Green Bay Packers": "gb",
    "Houston Texans": "hou", "Indianapolis Colts": "ind", "Jacksonville Jaguars": "jax", "Kansas City Chiefs": "kc",
    "Los Angeles Chargers": "lac", "Los Angeles Rams": "lar", "Las Vegas Raiders": "lv", "Miami Dolphins": "mia",
    "Minnesota Vikings": "min", "New England Patriots": "ne", "New Orleans Saints": "no", "New York Giants": "nyg",
    "New York Jets": "nyj", "Philadelphia Eagles": "phi", "Pittsburgh Steelers": "pit", "Seattle Seahawks": "sea",
    "San Francisco 49ers": "sf", "Tampa Bay Buccaneers": "tb", "Tennessee Titans": "ten", "Washington Commanders": "was",
}

NFL_TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def canonical_team_name(sport: str, team_name: str | None) -> str | None:
    if team_name is None:
        return None
    if sport == "NFL":
        return NFL_TEAM_NAME_MAP.get(str(team_name).upper(), team_name)
    return team_name


def expected_value(prob, odds, stake=1.0):
    if odds is None or prob is None:
        return None
    payout = stake * (odds / 100) if odds > 0 else stake * (100 / abs(odds))
    return (prob * payout) - ((1 - prob) * stake)


def kelly_fraction(prob, odds):
    if odds is None or prob is None:
        return 0.0
    b = odds / 100 if odds > 0 else 100 / abs(odds)
    q = 1 - prob
    edge = (b * prob) - q
    if b <= 0:
        return 0.0
    return max(0.0, edge / b)


def format_percent(value):
    return "N/A" if value is None else f"{value:.1%}"


def format_number(value):
    return "N/A" if value is None else f"{value:.2f}"


def format_odds(value):
    if value is None:
        return "N/A"
    value = int(value)
    return f"+{value}" if value > 0 else str(value)


def american_to_decimal(odds):
    if odds is None or odds == 0:
        return None
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1


def closing_line_value(open_odds, close_odds):
    if open_odds is None or close_odds is None:
        return None
    open_dec = american_to_decimal(open_odds)
    close_dec = american_to_decimal(close_odds)
    if open_dec is None or close_dec is None:
        return None
    return close_dec - open_dec


def build_team_profiles(data: pd.DataFrame, sport: str) -> dict[str, dict]:
    if "home_team" not in data.columns or "away_team" not in data.columns:
        return {}

    date_col = "game_date" if "game_date" in data.columns else None
    if date_col is None:
        for candidate in ["GAME_DATE", "gameday"]:
            if candidate in data.columns:
                date_col = candidate
                break

    home = data[[c for c in [date_col, "home_team", "home_rating", "home_form", "home_rest"] if c is not None]].rename(
        columns={"home_team": "team", "home_rating": "rating", "home_form": "form", "home_rest": "rest"}
    )
    away = data[[c for c in [date_col, "away_team", "away_rating", "away_form", "away_rest"] if c is not None]].rename(
        columns={"away_team": "team", "away_rating": "rating", "away_form": "form", "away_rest": "rest"}
    )
    profiles = pd.concat([home, away], ignore_index=True).dropna(subset=["team", "rating", "form", "rest"])
    profiles["team"] = profiles["team"].apply(lambda team: canonical_team_name(sport, team))
    if date_col and date_col in profiles.columns:
        profiles = profiles.sort_values(date_col)
        profiles = profiles.groupby("team", as_index=False).tail(1)
    else:
        profiles = profiles.groupby("team", as_index=False).tail(1)
    return profiles.set_index("team")[["rating", "form", "rest"]].to_dict("index")


def matchup_label(snapshot: dict) -> str:
    return f"{snapshot['away_team']} at {snapshot['home_team']}"


def matchup_selector_label(snapshot: dict) -> str:
    return f"{snapshot['away_team']} at {snapshot['home_team']} | {sportsbook_summary(snapshot)}"


def apply_matchup_defaults(snapshot: dict | None, team_profiles: dict[str, dict], sport: str) -> None:
    if snapshot is None:
        return
    matchup_id = matchup_label(snapshot)
    if st.session_state.get("selected_matchup_applied") == matchup_id:
        return

    home_profile = team_profiles.get(snapshot["home_team"], team_profiles.get(canonical_team_name(sport, snapshot["home_team"]), {}))
    away_profile = team_profiles.get(snapshot["away_team"], team_profiles.get(canonical_team_name(sport, snapshot["away_team"]), {}))

    st.session_state["home_rating"] = int(round(home_profile.get("rating", st.session_state.get("home_rating", 85))))
    st.session_state["away_rating"] = int(round(away_profile.get("rating", st.session_state.get("away_rating", 82))))
    st.session_state["home_form"] = float(round(home_profile.get("form", st.session_state.get("home_form", 0.6)), 2))
    st.session_state["away_form"] = float(round(away_profile.get("form", st.session_state.get("away_form", 0.55)), 2))
    st.session_state["home_rest"] = int(round(home_profile.get("rest", st.session_state.get("home_rest", 3))))
    st.session_state["away_rest"] = int(round(away_profile.get("rest", st.session_state.get("away_rest", 3))))
    if sport == "NBA":
        st.session_state["temperature"] = 0
        st.session_state["wind_speed"] = 0
        st.session_state["precipitation"] = 0
    st.session_state["selected_matchup_applied"] = matchup_id


def team_logo_url(sport: str, team_name: str) -> str | None:
    if sport == "NBA":
        code = NBA_LOGO_MAP.get(team_name)
        return None if code is None else f"https://a.espncdn.com/i/teamlogos/nba/500/{code}.png"
    code = NFL_LOGO_MAP.get(team_name)
    return None if code is None else f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"


def sportsbook_summary(snapshot: dict | None) -> str:
    if snapshot is None:
        return "No sportsbook prices loaded"
    sources = []
    if snapshot["h2h"].get("home_source") or snapshot["h2h"].get("away_source"):
        sources.append(f"ML: {snapshot['h2h'].get('home_source') or snapshot['h2h'].get('away_source')}")
    if snapshot["spreads"].get("home_source") or snapshot["spreads"].get("away_source"):
        sources.append(f"Spread: {snapshot['spreads'].get('home_source') or snapshot['spreads'].get('away_source')}")
    if snapshot["totals"].get("over_source") or snapshot["totals"].get("under_source"):
        sources.append(f"Total: {snapshot['totals'].get('over_source') or snapshot['totals'].get('under_source')}")
    return " | ".join(sources) if sources else "No sportsbook prices loaded"


def apply_tracker_defaults(snapshot: dict | None, target: str, result: dict) -> None:
    if snapshot is None:
        return
    tracker_id = f"{matchup_label(snapshot)}::{target}"
    if st.session_state.get("tracker_defaults_applied") == tracker_id:
        return

    if target == "Game winner":
        st.session_state["bet_market"] = "Moneyline"
        if result.get("home_win_prob", 0.5) >= result.get("away_win_prob", 0.5):
            st.session_state["bet_side"] = "Home"
            st.session_state["bet_odds_value"] = float(snapshot["h2h"].get("home_price") or 100)
        else:
            st.session_state["bet_side"] = "Away"
            st.session_state["bet_odds_value"] = float(snapshot["h2h"].get("away_price") or 100)
    elif target == "Point spread":
        st.session_state["bet_market"] = "Spread"
        if result.get("predicted_margin", 0) >= 0:
            st.session_state["bet_side"] = "Home"
            st.session_state["bet_odds_value"] = float(snapshot["spreads"].get("home_price") or 100)
        else:
            st.session_state["bet_side"] = "Away"
            st.session_state["bet_odds_value"] = float(snapshot["spreads"].get("away_price") or 100)
    else:
        st.session_state["bet_market"] = "Total"
        if result.get("predicted_total", 0) >= (snapshot["totals"].get("line") or 0):
            st.session_state["bet_side"] = "Over"
            st.session_state["bet_odds_value"] = float(snapshot["totals"].get("over_price") or 100)
        else:
            st.session_state["bet_side"] = "Under"
            st.session_state["bet_odds_value"] = float(snapshot["totals"].get("under_price") or 100)
    st.session_state["tracker_defaults_applied"] = tracker_id


def calibration_table(probabilities: np.ndarray, outcomes: pd.Series, bins: int = 6) -> pd.DataFrame:
    table = pd.DataFrame({"prob": probabilities, "actual": outcomes.to_numpy()})
    table["bucket"] = pd.cut(table["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = table.groupby("bucket", observed=False).agg(
        predicted_prob=("prob", "mean"),
        actual_rate=("actual", "mean"),
        count=("actual", "size"),
    ).reset_index()
    grouped = grouped[grouped["count"] > 0]
    return grouped


def model_backtest_table(probabilities: np.ndarray, outcomes: pd.Series) -> pd.DataFrame:
    records = []
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        confidence = np.abs(probabilities - 0.5)
        mask = confidence >= threshold
        if not mask.any():
            records.append({"threshold": threshold, "bets": 0, "hit_rate": None, "units": 0.0})
            continue
        picks = (probabilities[mask] >= 0.5).astype(int)
        actual = outcomes.to_numpy()[mask]
        wins = (picks == actual)
        units = np.where(wins, 0.91, -1.0).sum()
        records.append({
            "threshold": threshold,
            "bets": int(mask.sum()),
            "hit_rate": float(wins.mean()),
            "units": float(units),
        })
    return pd.DataFrame(records)


def tracker_performance_summary(history_df: pd.DataFrame) -> dict:
    settled = history_df[history_df["result"].isin(["Win", "Loss", "Push"])].copy()
    if settled.empty:
        return {"bets": 0, "roi": None, "avg_clv": None}
    risk_bets = settled[settled["result"].isin(["Win", "Loss"])]
    total_staked = risk_bets["stake"].sum()
    roi = settled["profit"].sum() / total_staked if total_staked else None
    avg_clv = settled["clv"].dropna().mean() if "clv" in settled.columns else None
    return {"bets": len(settled), "roi": roi, "avg_clv": avg_clv}


def build_bet_recommendation(result: dict, target: str, market_snapshot: dict | None):
    if market_snapshot is None:
        return "No live odds available yet. The model can score the matchup, but it cannot rank bet value without sportsbook prices."

    home_team = market_snapshot["home_team"]
    away_team = market_snapshot["away_team"]

    if target == "Game winner" and "home_win_prob" in result:
        home_odds = market_snapshot["h2h"]["home_price"]
        away_odds = market_snapshot["h2h"]["away_price"]
        home_ev = expected_value(result["home_win_prob"], home_odds)
        away_ev = expected_value(result["away_win_prob"], away_odds)
        home_kelly = kelly_fraction(result["home_win_prob"], home_odds)
        away_kelly = kelly_fraction(result["away_win_prob"], away_odds)

        home_ev_val = home_ev if home_ev is not None else float("-inf")
        away_ev_val = away_ev if away_ev is not None else float("-inf")
        if home_ev_val >= away_ev_val:
            team = home_team
            ev = home_ev
            prob = result["home_win_prob"]
            odds = home_odds
            kelly = home_kelly
        else:
            team = away_team
            ev = away_ev
            prob = result["away_win_prob"]
            odds = away_odds
            kelly = away_kelly

        return (
            f"Best current side: {team} at {odds}. Model win probability is {prob:.1%}, "
            f"expected value is {ev:.2f} per $1 risked, and full Kelly suggests {kelly:.1%} of bankroll."
        )

    if target == "Point spread" and market_snapshot["spreads"]["home_point"] is not None:
        model_margin = result["predicted_margin"]
        home_line = market_snapshot["spreads"]["home_point"]
        away_line = market_snapshot["spreads"]["away_point"]
        home_edge = model_margin + home_line
        away_edge = -model_margin + away_line
        if home_edge >= away_edge:
            return f"Spread lean: {home_team} {home_line:+.1f}. Model margin is {model_margin:.1f}, which is {home_edge:.1f} points better than the market line."
        return f"Spread lean: {away_team} {away_line:+.1f}. Model margin is {model_margin:.1f}, which is {away_edge:.1f} points better than the market line."

    if target == "Total score" and market_snapshot["totals"]["line"] is not None:
        model_total = result["predicted_total"]
        total_line = market_snapshot["totals"]["line"]
        delta = model_total - total_line
        if delta >= 0:
            return f"Total lean: Over {total_line:.1f}. Model projects {model_total:.1f}, a {delta:.1f}-point edge versus the market."
        return f"Total lean: Under {total_line:.1f}. Model projects {model_total:.1f}, a {abs(delta):.1f}-point edge versus the market."

    return "The selected market does not have enough live pricing data yet to form a recommendation."


_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]


def maybe_generate_ai_summary(prompt: str):
    if not GEMINI_API_KEY or genai is None:
        return None
    client = genai.Client(api_key=GEMINI_API_KEY)
    for model in _GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                continue  # try next model
            break  # non-quota error — stop trying
    return None


def load_bet_history() -> list[dict]:
    if not BET_HISTORY_PATH.exists():
        return []
    try:
        return pd.read_csv(BET_HISTORY_PATH).to_dict("records")
    except Exception:
        return []


def save_bet_history(records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(BET_HISTORY_PATH, index=False)


def load_preferences() -> dict:
    if not PREFERENCES_PATH.exists():
        return {}
    try:
        return json.loads(PREFERENCES_PATH.read_text())
    except Exception:
        return {}


def save_preferences(preferences: dict) -> None:
    PREFERENCES_PATH.write_text(json.dumps(preferences, indent=2))


def load_model_history() -> list[dict]:
    if not MODEL_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(MODEL_HISTORY_PATH.read_text())
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def save_model_history(records: list[dict]) -> None:
    MODEL_HISTORY_PATH.write_text(json.dumps(records[-200:], indent=2))


def record_model_snapshot(
    sport: str,
    training_rows: int,
    learned_rows: int,
    latest_game_date: str | None,
    accuracy: float,
    spread_mae: float,
    total_mae: float,
) -> dict | None:
    snapshot = {
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "sport": sport,
        "training_rows": int(training_rows),
        "learned_rows": int(learned_rows),
        "latest_game_date": latest_game_date,
        "accuracy": float(accuracy),
        "spread_mae": float(spread_mae),
        "total_mae": float(total_mae),
    }
    signature = (
        sport,
        snapshot["training_rows"],
        snapshot["learned_rows"],
        latest_game_date,
        round(snapshot["accuracy"], 4),
        round(snapshot["spread_mae"], 4),
        round(snapshot["total_mae"], 4),
    )
    history = load_model_history()
    for existing in reversed(history):
        existing_signature = (
            existing.get("sport"),
            int(existing.get("training_rows", 0)),
            int(existing.get("learned_rows", 0)),
            existing.get("latest_game_date"),
            round(float(existing.get("accuracy", 0.0)), 4),
            round(float(existing.get("spread_mae", 0.0)), 4),
            round(float(existing.get("total_mae", 0.0)), 4),
        )
        if existing_signature == signature:
            return None
    history.append(snapshot)
    save_model_history(history)
    return snapshot


def model_history_frame(sport: str) -> pd.DataFrame:
    history = load_model_history()
    if not history:
        return pd.DataFrame()
    frame = pd.DataFrame(history)
    if "sport" in frame.columns:
        frame = frame[frame["sport"] == sport].copy()
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.sort_values("timestamp")

def todays_games(odds_data, sport):
    """Filter odds_data to games commencing today (local time) and return display-ready list."""
    if not odds_data:
        return []
    from datetime import datetime, timezone
    today = datetime.now().date()
    games = []
    for game in odds_data:
        ct = game.get("commence_time")
        if not ct:
            continue
        try:
            game_dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone()
            if game_dt.date() != today:
                continue
        except (ValueError, TypeError):
            continue
        snapshot = extract_market_snapshot(game)
        games.append({
            "time": game_dt.strftime("%-I:%M %p"),
            "home": snapshot["home_team"],
            "away": snapshot["away_team"],
            "home_odds": snapshot["h2h"]["home_price"],
            "away_odds": snapshot["h2h"]["away_price"],
            "spread_home": snapshot["spreads"]["home_point"],
            "spread_away": snapshot["spreads"]["away_point"],
            "total": snapshot["totals"]["line"],
            "snapshot": snapshot,
        })
    games.sort(key=lambda g: g["time"])
    return games


def fetch_odds(sport_key="basketball_nba", regions="us", markets="h2h,spreads,totals", odds_format="american"):
    """Fetch odds from The Odds API for NBA or NFL games."""
    if not ODDS_API_KEY:
        return None
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception as e:
        st.warning(f"Could not fetch odds: {e}")
        return None


def fetch_historical_event_odds(sport_key: str, event_id: str, date_iso: str, bookmakers: str | None = None, markets: str = "h2h,spreads,totals"):
    if not ODDS_API_KEY:
        return None
    url = f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "date": date_iso,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers and bookmakers != "best":
        params["bookmakers"] = bookmakers
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            payload = response.json()
            return payload.get("data")
    except Exception:
        return None
    return None


def fetch_recent_scores(sport_key: str, days_from: int = 3):
    if not ODDS_API_KEY:
        return None
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": days_from,
        "dateFormat": "iso",
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


@st.cache_data(ttl=600)
def cached_recent_scores(sport_key: str, days_from: int = 3):
    return fetch_recent_scores(sport_key=sport_key, days_from=days_from)

def implied_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def extract_market_snapshot(game: dict) -> dict:
    snapshot = {
        "event_id": game.get("id"),
        "commence_time": game.get("commence_time"),
        "home_team": game.get("home_team", ""),
        "away_team": game.get("away_team", ""),
        "h2h": {"home_price": None, "away_price": None, "home_source": None, "away_source": None},
        "spreads": {"home_price": None, "home_point": None, "away_price": None, "away_point": None, "home_source": None, "away_source": None},
        "totals": {"over_price": None, "under_price": None, "line": None, "over_source": None, "under_source": None},
    }

    for bookmaker in game.get("bookmakers", []):
        source = bookmaker.get("title") or bookmaker.get("key")
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            for outcome in market.get("outcomes", []):
                name = outcome.get("name")
                price = outcome.get("price")
                point = outcome.get("point")
                if key == "h2h":
                    if name == snapshot["home_team"] and (snapshot["h2h"]["home_price"] is None or price > snapshot["h2h"]["home_price"]):
                        snapshot["h2h"]["home_price"] = price
                        snapshot["h2h"]["home_source"] = source
                    if name == snapshot["away_team"] and (snapshot["h2h"]["away_price"] is None or price > snapshot["h2h"]["away_price"]):
                        snapshot["h2h"]["away_price"] = price
                        snapshot["h2h"]["away_source"] = source
                elif key == "spreads":
                    if name == snapshot["home_team"] and (snapshot["spreads"]["home_price"] is None or price > snapshot["spreads"]["home_price"]):
                        snapshot["spreads"]["home_price"] = price
                        snapshot["spreads"]["home_point"] = point
                        snapshot["spreads"]["home_source"] = source
                    if name == snapshot["away_team"] and (snapshot["spreads"]["away_price"] is None or price > snapshot["spreads"]["away_price"]):
                        snapshot["spreads"]["away_price"] = price
                        snapshot["spreads"]["away_point"] = point
                        snapshot["spreads"]["away_source"] = source
                elif key == "totals":
                    if name == "Over" and (snapshot["totals"]["over_price"] is None or price > snapshot["totals"]["over_price"]):
                        snapshot["totals"]["over_price"] = price
                        snapshot["totals"]["line"] = point
                        snapshot["totals"]["over_source"] = source
                    if name == "Under" and (snapshot["totals"]["under_price"] is None or price > snapshot["totals"]["under_price"]):
                        snapshot["totals"]["under_price"] = price
                        snapshot["totals"]["line"] = point
                        snapshot["totals"]["under_source"] = source
    return snapshot


def filter_game_by_bookmaker(game: dict, bookmaker_key: str | None) -> dict:
    if not bookmaker_key or bookmaker_key == "best":
        return game
    filtered = dict(game)
    filtered["bookmakers"] = [bookmaker for bookmaker in game.get("bookmakers", []) if bookmaker.get("key") == bookmaker_key]
    return filtered


def available_bookmakers(odds_data: list[dict] | None) -> list[tuple[str, str]]:
    if not odds_data:
        return [("best", "Best available")]
    seen = {"best": "Best available"}
    for game in odds_data:
        for bookmaker in game.get("bookmakers", []):
            seen[bookmaker.get("key", "unknown")] = bookmaker.get("title", bookmaker.get("key", "Unknown"))
    return list(seen.items())


def historical_close_for_snapshot(snapshot: dict | None, sport_key: str, bookmaker_key: str | None) -> dict | None:
    if snapshot is None or not snapshot.get("event_id") or not snapshot.get("commence_time"):
        return None
    return fetch_historical_event_odds(
        sport_key=sport_key,
        event_id=snapshot["event_id"],
        date_iso=snapshot["commence_time"],
        bookmakers=bookmaker_key,
    )


def settle_bet_from_score(record: dict, score_event: dict) -> str:
    scores = score_event.get("scores") or []
    if len(scores) < 2:
        return record.get("result", "Pending")
    score_map = {item["name"]: float(item["score"]) for item in scores if item.get("score") is not None}
    home_team = record.get("home_team")
    away_team = record.get("away_team")
    if home_team not in score_map or away_team not in score_map:
        return record.get("result", "Pending")

    home_score = score_map[home_team]
    away_score = score_map[away_team]
    market = record.get("market")
    side = record.get("side")

    if market == "Moneyline":
        if side == "Home":
            if home_score == away_score:
                return "Push"
            return "Win" if home_score > away_score else "Loss"
        if side == "Away":
            if home_score == away_score:
                return "Push"
            return "Win" if away_score > home_score else "Loss"

    if market == "Spread":
        line = record.get("line")
        if line is None:
            return record.get("result", "Pending")
        if side == "Home":
            adjusted = home_score + float(line) - away_score
            if adjusted == 0:
                return "Push"
            return "Win" if adjusted > 0 else "Loss"
        if side == "Away":
            adjusted = away_score + float(line) - home_score
            if adjusted == 0:
                return "Push"
            return "Win" if adjusted > 0 else "Loss"

    if market == "Total":
        line = record.get("line")
        if line is None:
            return record.get("result", "Pending")
        total_score = home_score + away_score
        if side == "Over":
            if total_score == float(line):
                return "Push"
            return "Win" if total_score > float(line) else "Loss"
        if side == "Under":
            if total_score == float(line):
                return "Push"
            return "Win" if total_score < float(line) else "Loss"

    return record.get("result", "Pending")


def final_score_fields(score_event: dict) -> dict:
    scores = score_event.get("scores") or []
    if len(scores) < 2:
        return {}
    score_map = {item["name"]: float(item["score"]) for item in scores if item.get("score") is not None}
    home_team = score_event.get("home_team")
    away_team = score_event.get("away_team")
    if home_team not in score_map or away_team not in score_map:
        return {}
    home_score = score_map[home_team]
    away_score = score_map[away_team]
    return {
        "home_score": home_score,
        "away_score": away_score,
        "margin": home_score - away_score,
        "total_score": home_score + away_score,
        "home_win": int(home_score > away_score),
    }


def auto_settle_bets(records: list[dict], sport_key: str):
    scores = cached_recent_scores(sport_key)
    if not scores:
        return records, 0
    completed = {
        (event.get("home_team"), event.get("away_team")): event
        for event in scores
        if event.get("completed")
    }
    updated = []
    changed = 0
    for record in records:
        event = completed.get((record.get("home_team"), record.get("away_team")))
        if event is not None:
            score_update = final_score_fields(event)
            needs_score_backfill = any(record.get(key) is None or pd.isna(record.get(key)) for key in ["home_score", "away_score", "margin", "total_score", "home_win"])
            if record.get("result") == "Pending":
                new_result = settle_bet_from_score(record, event)
                if new_result != record.get("result") or needs_score_backfill:
                    record = dict(record)
                    record["result"] = new_result
                    record.update(score_update)
                    changed += 1
            elif needs_score_backfill:
                record = dict(record)
                record.update(score_update)
                changed += 1
        updated.append(record)
    return updated, changed


def settle_history_for_sport(records: list[dict], sport: str) -> tuple[list[dict], int]:
    sport_key = "basketball_nba" if sport == "NBA" else "americanfootball_nfl"
    scoped_records = []
    changed = 0
    sport_records = [record for record in records if record.get("sport") == sport]
    updated_sport_records, changed = auto_settle_bets(sport_records, sport_key)
    updated_iter = iter(updated_sport_records)
    output = []
    for record in records:
        if record.get("sport") == sport:
            output.append(next(updated_iter))
        else:
            output.append(record)
    return output, changed


def learned_bet_rows(records: list[dict], sport: str) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("sport") != sport:
            continue
        if record.get("result") not in ["Win", "Loss", "Push"]:
            continue
        required = [
            "home_rating", "away_rating", "home_form", "away_form", "home_rest", "away_rest",
            "temperature", "wind_speed", "precipitation",
        ]
        if any(record.get(key) is None or pd.isna(record.get(key)) for key in required):
            continue
        margin = record.get("margin")
        total_score = record.get("total_score")
        home_win = record.get("home_win")
        if margin is None or pd.isna(margin) or total_score is None or pd.isna(total_score) or home_win is None or pd.isna(home_win):
            continue
        rows.append(
            {
                "home_team": record.get("home_team"),
                "away_team": record.get("away_team"),
                "game_date": pd.to_datetime(record.get("logged_at"), errors="coerce"),
                "home_rating": float(record.get("home_rating")),
                "away_rating": float(record.get("away_rating")),
                "home_form": float(record.get("home_form")),
                "away_form": float(record.get("away_form")),
                "home_rest": float(record.get("home_rest")),
                "away_rest": float(record.get("away_rest")),
                "temperature": float(record.get("temperature")),
                "wind_speed": float(record.get("wind_speed")),
                "precipitation": float(record.get("precipitation")),
                "home_advantage": 1,
                "rating_diff": float(record.get("home_rating")) - float(record.get("away_rating")),
                "form_diff": float(record.get("home_form")) - float(record.get("away_form")),
                "rest_diff": float(record.get("home_rest")) - float(record.get("away_rest")),
                "margin": float(margin),
                "total_score": float(total_score),
                "home_win": int(home_win),
                "training_weight": 0.35,
                "data_source": "bet_history",
            }
        )
    return pd.DataFrame(rows)


def augment_with_learning_data(base_data: pd.DataFrame, bet_records: list[dict], sport: str) -> tuple[pd.DataFrame, int]:
    learned = learned_bet_rows(bet_records, sport)
    base = base_data.copy()
    if "training_weight" not in base.columns:
        base["training_weight"] = 1.0
    if learned.empty:
        return base, 0
    combined = pd.concat([base, learned], ignore_index=True, sort=False)
    return combined, len(learned)


def install_auto_refresh(enabled: bool, interval_minutes: int) -> None:
    if not enabled:
        return
    interval_ms = max(1, interval_minutes) * 60 * 1000
    st.html(
        f"""
        <script>
        const interval = {interval_ms};
        if (!window.__bettingAppAutoRefreshInstalled) {{
            window.__bettingAppAutoRefreshInstalled = true;
            setTimeout(() => window.parent.location.reload(), interval);
        }}
        </script>
        """
    )


def fetch_historical_snapshot_for_game(sport_key: str, game_date: str, bookmaker_key: str | None = None):
    if not ODDS_API_KEY:
        return None
    url = f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": game_date,
    }
    if bookmaker_key and bookmaker_key != "best":
        params["bookmakers"] = bookmaker_key
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 200:
            return response.json().get("data", [])
    except Exception:
        return None
    return None


def historical_moneyline_backtest(data: pd.DataFrame, sport_key: str, bookmaker_key: str | None, sample_size: int = 3) -> pd.DataFrame:
    required = ["game_date", "home_team", "away_team", "home_win", "rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"]
    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    sample = data.sort_values("game_date").tail(sample_size).copy()
    sample["home_team"] = sample["home_team"].apply(lambda team: canonical_team_name("NFL" if sport_key == "americanfootball_nfl" else "NBA", team))
    sample["away_team"] = sample["away_team"].apply(lambda team: canonical_team_name("NFL" if sport_key == "americanfootball_nfl" else "NBA", team))
    rows = []
    for _, game in sample.iterrows():
        snapshot_games = fetch_historical_snapshot_for_game(sport_key, pd.to_datetime(game["game_date"]).isoformat(), bookmaker_key)
        if not snapshot_games:
            continue
        matched = next((item for item in snapshot_games if item.get("home_team") == game["home_team"] and item.get("away_team") == game["away_team"]), None)
        if matched is None:
            continue
        snapshot = extract_market_snapshot(matched)
        features = pd.DataFrame([[game["rating_diff"], game["form_diff"], game["rest_diff"], game["home_advantage"], game["temperature"], game["wind_speed"], game["precipitation"]]], columns=["rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"])
        model = LogisticRegression(max_iter=200)
        train_cols = ["rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"]
        model.fit(data[train_cols], data["home_win"])
        home_prob = model.predict_proba(features)[0, 1]
        side = "Home" if home_prob >= 0.5 else "Away"
        odds = snapshot["h2h"]["home_price"] if side == "Home" else snapshot["h2h"]["away_price"]
        won = (game["home_win"] == 1 and side == "Home") or (game["home_win"] == 0 and side == "Away")
        units = (odds / 100) if won and odds and odds > 0 else ((100 / abs(odds)) if won and odds else -1.0)
        rows.append(
            {
                "game_date": game["game_date"],
                "matchup": f"{game['away_team']} at {game['home_team']}",
                "pick": side,
                "home_prob": home_prob,
                "odds": odds,
                "result": "Win" if won else "Loss",
                "units": units,
            }
        )
    return pd.DataFrame(rows)


def add_rolling_team_features(frame: pd.DataFrame, team_col: str, date_col: str, points_col: str, opp_points_col: str, win_col: str) -> pd.DataFrame:
    data = frame.copy()
    data = data.sort_values([team_col, date_col])
    grouped = data.groupby(team_col, group_keys=False)
    data["rating"] = grouped[points_col].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    data["form"] = grouped[win_col].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    data["opp_rating"] = grouped[opp_points_col].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    data["rest_days"] = grouped[date_col].diff().dt.days.clip(lower=1, upper=10).fillna(3)
    return data


def recent_nba_seasons(lookback: int = 2) -> list[str]:
    now = pd.Timestamp.now()
    current_start_year = now.year if now.month >= 10 else now.year - 1
    seasons = []
    for offset in range(lookback):
        start_year = current_start_year - offset
        seasons.append(f"{start_year}-{str((start_year + 1) % 100).zfill(2)}")
    return seasons


def recent_nfl_seasons(lookback: int = 2) -> list[int]:
    now = pd.Timestamp.now()
    current_season = now.year if now.month >= 9 else now.year - 1
    return [current_season - offset for offset in range(lookback)]


def build_nba_training_data(seasons: list[str] | None = None) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder

    seasons = seasons or recent_nba_seasons(lookback=2)
    frames = []
    for season in seasons:
        season_games = leaguegamefinder.LeagueGameFinder(season_nullable=season, league_id_nullable="00").get_data_frames()[0]
        if not season_games.empty:
            frames.append(season_games)
    if not frames:
        return pd.DataFrame()

    games = pd.concat(frames, ignore_index=True)
    games = games[["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "MATCHUP", "WL", "PTS", "PLUS_MINUS"]].dropna()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["is_home"] = games["MATCHUP"].str.contains("vs.", regex=False)
    games["opp_points"] = games["PTS"] - games["PLUS_MINUS"]
    games["win"] = (games["WL"] == "W").astype(int)
    games = add_rolling_team_features(games, "TEAM_ID", "GAME_DATE", "PTS", "opp_points", "win")

    home = games[games["is_home"]].rename(columns={
        "TEAM_NAME": "home_team",
        "PTS": "home_points",
        "rating": "home_rating",
        "form": "home_form",
        "rest_days": "home_rest",
    })
    away = games[~games["is_home"]].rename(columns={
        "TEAM_NAME": "away_team",
        "PTS": "away_points",
        "rating": "away_rating",
        "form": "away_form",
        "rest_days": "away_rest",
    })
    merged = home[["GAME_ID", "GAME_DATE", "home_team", "home_points", "home_rating", "home_form", "home_rest"]].merge(
        away[["GAME_ID", "away_team", "away_points", "away_rating", "away_form", "away_rest"]], on="GAME_ID", how="inner"
    )
    merged["game_date"] = merged["GAME_DATE"]
    merged["home_advantage"] = 1
    merged["temperature"] = 0.0
    merged["wind_speed"] = 0.0
    merged["precipitation"] = 0
    merged["rating_diff"] = merged["home_rating"] - merged["away_rating"]
    merged["form_diff"] = merged["home_form"] - merged["away_form"]
    merged["rest_diff"] = merged["home_rest"] - merged["away_rest"]
    merged["margin"] = merged["home_points"] - merged["away_points"]
    merged["total_score"] = merged["home_points"] + merged["away_points"]
    merged["home_win"] = (merged["margin"] > 0).astype(int)
    return merged.dropna()


def build_nfl_training_data(seasons: list[int] | None = None) -> pd.DataFrame:
    seasons = seasons or recent_nfl_seasons(lookback=2)
    schedules = nfl.import_schedules(seasons).copy()
    if "game_type" in schedules.columns:
        schedules = schedules[schedules["game_type"] == "REG"]
    schedules = schedules.dropna(subset=["home_score", "away_score"])
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])
    if "game_id" not in schedules.columns:
        schedules["game_id"] = schedules.index.astype(str)

    home_rows = pd.DataFrame({
        "game_id": schedules["game_id"],
        "gameday": schedules["gameday"],
        "team": schedules["home_team"],
        "team_points": schedules["home_score"],
        "opp_points": schedules["away_score"],
        "win": (schedules["home_score"] > schedules["away_score"]).astype(int),
        "is_home": 1,
    })
    away_rows = pd.DataFrame({
        "game_id": schedules["game_id"],
        "gameday": schedules["gameday"],
        "team": schedules["away_team"],
        "team_points": schedules["away_score"],
        "opp_points": schedules["home_score"],
        "win": (schedules["away_score"] > schedules["home_score"]).astype(int),
        "is_home": 0,
    })
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games = add_rolling_team_features(team_games, "team", "gameday", "team_points", "opp_points", "win")

    home = team_games[team_games["is_home"] == 1].rename(columns={
        "team": "home_team",
        "team_points": "home_points",
        "rating": "home_rating",
        "form": "home_form",
        "rest_days": "home_rest",
    })
    away = team_games[team_games["is_home"] == 0].rename(columns={
        "team": "away_team",
        "team_points": "away_points",
        "rating": "away_rating",
        "form": "away_form",
        "rest_days": "away_rest",
    })
    merged = schedules[["game_id", "gameday"]].merge(
        home[["game_id", "home_team", "home_points", "home_rating", "home_form", "home_rest"]], on="game_id", how="inner"
    ).merge(
        away[["game_id", "away_team", "away_points", "away_rating", "away_form", "away_rest"]], on="game_id", how="inner"
    )
    merged["game_date"] = merged["gameday"]
    merged["home_advantage"] = 1
    weather_cols = ["game_id"]
    for _col in ["temp", "wind", "weather"]:
        if _col in schedules.columns:
            weather_cols.append(_col)
    merged = merged.merge(schedules[weather_cols].drop_duplicates("game_id"), on="game_id", how="left")
    if "temp" in merged.columns:
        merged["temperature"] = merged["temp"].fillna(60)
        merged.drop(columns=["temp"], inplace=True)
    else:
        merged["temperature"] = 60.0
    if "wind" in merged.columns:
        merged["wind_speed"] = merged["wind"].fillna(10)
        merged.drop(columns=["wind"], inplace=True)
    else:
        merged["wind_speed"] = 10.0
    if "weather" in merged.columns:
        merged["precipitation"] = merged["weather"].astype(str).str.contains("rain|snow|storm", case=False, regex=True).astype(int)
        merged.drop(columns=["weather"], inplace=True)
    else:
        merged["precipitation"] = 0
    merged["rating_diff"] = merged["home_rating"] - merged["away_rating"]
    merged["form_diff"] = merged["home_form"] - merged["away_form"]
    merged["rest_diff"] = merged["home_rest"] - merged["away_rest"]
    merged["margin"] = merged["home_points"] - merged["away_points"]
    merged["total_score"] = merged["home_points"] + merged["away_points"]
    merged["home_win"] = (merged["margin"] > 0).astype(int)
    return merged.dropna()

st.set_page_config(page_title="Sportsbook Tool", layout="centered")

# --- Mobile-responsive CSS ---
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
/* Stack columns vertically on small screens */
@media (max-width: 768px) {
    /* Make main content full-width */
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    /* Stack horizontal blocks vertically */
    [data-testid="column"] {
        width: 100% !important;
        flex: 100% !important;
        min-width: 100% !important;
    }
    /* Horizontal block: wrap to column */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.25rem !important;
    }
    /* Smaller title for phones */
    h1 {
        font-size: 1.5rem !important;
    }
    h2 {
        font-size: 1.2rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
    }
    /* Team logos smaller on mobile */
    [data-testid="stImage"] img {
        max-width: 48px !important;
    }
    /* Metric cards: tighter spacing */
    [data-testid="stMetric"] {
        padding: 0.25rem 0 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    /* Dataframes scroll horizontally */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }
    /* Sidebar: full-width overlay on mobile */
    [data-testid="stSidebar"] {
        min-width: 100% !important;
        max-width: 100% !important;
    }
    /* Charts fill width */
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        width: 100% !important;
    }
}

/* Medium screens: 2 columns instead of 4 */
@media (min-width: 769px) and (max-width: 1024px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="column"] {
        min-width: 45% !important;
    }
}

/* Touch-friendly: bigger tap targets */
@media (pointer: coarse) {
    button, [data-testid="stButton"] button {
        min-height: 44px !important;
        font-size: 0.95rem !important;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 0.25rem !important;
    }
    /* Slider thumb larger for touch */
    input[type="range"]::-webkit-slider-thumb {
        width: 24px !important;
        height: 24px !important;
    }
}
</style>
""", unsafe_allow_html=True)


st.title("Sportsbook Tool")
st.caption("Predict NBA and NFL outcomes: winner, spread, total, and parlays. Powered by real data.")

# Initialize widget defaults to prevent session-state / value conflicts
_widget_defaults = {
    "home_rating": 85, "away_rating": 82,
    "home_form": 0.6, "away_form": 0.55,
    "home_rest": 4, "away_rest": 3,
    "temperature": 60, "wind_speed": 10, "precipitation": 0,
    "bet_odds_value": 100.0, "closing_odds_value": 100.0,
}
for _key, _default in _widget_defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


@st.cache_data
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

# --- Real data loader ---
@st.cache_data(ttl=900)
def load_real_data(sport: str, lookback_seasons: int = 2, refresh_token: int = 0) -> pd.DataFrame:
    try:
        if sport == "NBA":
            return build_nba_training_data(seasons=recent_nba_seasons(lookback=lookback_seasons))
        elif sport == "NFL":
            return build_nfl_training_data(seasons=recent_nfl_seasons(lookback=lookback_seasons))
    except Exception as e:
        st.warning(f"Could not load real {sport} data: {e}. Using synthetic data.")
        return None



@st.cache_data(ttl=900)
def train_models(data: pd.DataFrame, refresh_token: int = 0):
    features = ["rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"]
    winner_training = data.dropna(subset=features + ["home_win"]).copy()
    X = winner_training[features]
    winner_weights = winner_training.get("training_weight", pd.Series(1.0, index=winner_training.index))

    winner_model = LogisticRegression(max_iter=200)
    winner_model.fit(X, winner_training["home_win"], sample_weight=winner_weights)

    spread_model = LinearRegression()
    spread_training = data.dropna(subset=features + ["margin"]).copy()
    spread_weights = spread_training.get("training_weight", pd.Series(1.0, index=spread_training.index))
    spread_model.fit(spread_training[features], spread_training["margin"], sample_weight=spread_weights)

    total_model = LinearRegression()
    total_features = ["home_rating", "away_rating", "home_form", "away_form", "home_rest", "away_rest", "temperature", "wind_speed", "precipitation"]
    total_training = data.dropna(subset=total_features + ["total_score"]).copy()
    total_weights = total_training.get("training_weight", pd.Series(1.0, index=total_training.index))
    total_model.fit(total_training[total_features], total_training["total_score"], sample_weight=total_weights)

    return winner_model, spread_model, total_model


def predict_outcome(sport: str, target: str, inputs: dict, models: tuple):
    rating_diff = inputs["home_rating"] - inputs["away_rating"]
    form_diff = inputs["home_form"] - inputs["away_form"]
    rest_diff = inputs["home_rest"] - inputs["away_rest"]
    home_advantage = 1

    X = pd.DataFrame(
        [[rating_diff, form_diff, rest_diff, home_advantage, inputs["temperature"], inputs["wind_speed"], inputs["precipitation"]]],
        columns=["rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"],
    )
    total_X = pd.DataFrame(
        [[
            inputs["home_rating"],
            inputs["away_rating"],
            inputs["home_form"],
            inputs["away_form"],
            inputs["home_rest"],
            inputs["away_rest"],
            inputs["temperature"],
            inputs["wind_speed"],
            inputs["precipitation"],
        ]],
        columns=["home_rating", "away_rating", "home_form", "away_form", "home_rest", "away_rest", "temperature", "wind_speed", "precipitation"],
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


def parlay_decimal_odds(legs: list[dict]) -> float | None:
    """Multiply decimal odds of all legs to get combined parlay decimal odds."""
    combined = 1.0
    for leg in legs:
        dec = american_to_decimal(leg.get("odds"))
        if dec is None:
            return None
        combined *= dec
    return combined


def parlay_payout(stake: float, legs: list[dict]) -> float | None:
    """Calculate total payout (stake * combined decimal odds)."""
    dec = parlay_decimal_odds(legs)
    if dec is None:
        return None
    return stake * dec


def parlay_implied_prob(legs: list[dict]) -> float | None:
    """Combined implied probability (product of individual implied probs, no-vig)."""
    prob = 1.0
    for leg in legs:
        odds = leg.get("odds")
        if odds is None or odds == 0:
            return None
        prob *= implied_prob(odds)
    return prob


def build_parlay_leg_options(matchup_snapshots: list[dict]) -> list[dict]:
    """Build selectable leg options from today's matchup snapshots."""
    options = []
    for snap in matchup_snapshots:
        home = snap["home_team"]
        away = snap["away_team"]
        tag = f"{away} @ {home}"
        # Moneyline
        if snap["h2h"]["home_price"] is not None:
            options.append({"label": f"{home} ML ({format_odds(snap['h2h']['home_price'])})", "matchup": tag, "market": "Moneyline", "pick": home, "odds": snap["h2h"]["home_price"]})
        if snap["h2h"]["away_price"] is not None:
            options.append({"label": f"{away} ML ({format_odds(snap['h2h']['away_price'])})", "matchup": tag, "market": "Moneyline", "pick": away, "odds": snap["h2h"]["away_price"]})
        # Spread
        if snap["spreads"]["home_point"] is not None and snap["spreads"]["home_price"] is not None:
            options.append({"label": f"{home} {snap['spreads']['home_point']:+.1f} ({format_odds(snap['spreads']['home_price'])})", "matchup": tag, "market": "Spread", "pick": home, "odds": snap["spreads"]["home_price"]})
        if snap["spreads"]["away_point"] is not None and snap["spreads"]["away_price"] is not None:
            options.append({"label": f"{away} {snap['spreads']['away_point']:+.1f} ({format_odds(snap['spreads']['away_price'])})", "matchup": tag, "market": "Spread", "pick": away, "odds": snap["spreads"]["away_price"]})
        # Totals
        if snap["totals"]["line"] is not None:
            if snap["totals"]["over_price"] is not None:
                options.append({"label": f"Over {snap['totals']['line']:.1f} ({format_odds(snap['totals']['over_price'])}) — {tag}", "matchup": tag, "market": "Total", "pick": "Over", "odds": snap["totals"]["over_price"]})
            if snap["totals"]["under_price"] is not None:
                options.append({"label": f"Under {snap['totals']['line']:.1f} ({format_odds(snap['totals']['under_price'])}) — {tag}", "matchup": tag, "market": "Total", "pick": "Under", "odds": snap["totals"]["under_price"]})
    return options


@st.cache_data(ttl=3600)
def get_injury_reports(sport: str, team: str = None):
    try:
        if sport == "NBA":
            return _fetch_nba_injuries(team)
        elif sport == "NFL":
            current_year = pd.Timestamp.now().year
            injuries = nfl.import_injuries([current_year])
            if team:
                injuries = injuries[injuries['team'].str.upper() == team.upper()]
            return injuries
    except Exception as e:
        st.error(f"Error fetching injury reports: {e}")
        return pd.DataFrame()


def _fetch_nba_injuries(team_name: str | None = None) -> pd.DataFrame:
    """Fetch NBA injury reports from the ESPN public API."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        payload = resp.json()
        rows = []
        for team_entry in payload.get("items", []):
            team_display = team_entry.get("team", {}).get("displayName", "")
            for athlete_entry in team_entry.get("injuries", []):
                athlete = athlete_entry.get("athlete", {})
                rows.append({
                    "team": team_display,
                    "player": athlete.get("displayName", ""),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "status": athlete_entry.get("status", ""),
                    "detail": athlete_entry.get("details", {}).get("detail", ""),
                })
        df = pd.DataFrame(rows)
        if team_name and not df.empty:
            df = df[df["team"].str.contains(team_name, case=False, na=False)]
        return df
    except Exception:
        return pd.DataFrame()


def detect_arbitrage(odds_data_raw: list[dict] | None) -> list[dict]:
    """Detect arbitrage opportunities across bookmakers for h2h, spreads, and totals."""
    if not odds_data_raw:
        return []
    arbs = []
    for game in odds_data_raw:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        best = {
            "h2h_home": (None, ""), "h2h_away": (None, ""),
            "spread_home": (None, None, ""), "spread_away": (None, None, ""),
            "total_over": (None, None, ""), "total_under": (None, None, ""),
        }
        for bk in game.get("bookmakers", []):
            bk_name = bk.get("title", bk.get("key", ""))
            for mkt in bk.get("markets", []):
                key = mkt.get("key")
                for oc in mkt.get("outcomes", []):
                    price = oc.get("price")
                    point = oc.get("point")
                    if price is None:
                        continue
                    dec = american_to_decimal(price)
                    if key == "h2h":
                        if oc["name"] == home_team and (best["h2h_home"][0] is None or dec > best["h2h_home"][0]):
                            best["h2h_home"] = (dec, bk_name)
                        elif oc["name"] == away_team and (best["h2h_away"][0] is None or dec > best["h2h_away"][0]):
                            best["h2h_away"] = (dec, bk_name)
                    elif key == "spreads":
                        if oc["name"] == home_team and (best["spread_home"][0] is None or dec > best["spread_home"][0]):
                            best["spread_home"] = (dec, point, bk_name)
                        elif oc["name"] == away_team and (best["spread_away"][0] is None or dec > best["spread_away"][0]):
                            best["spread_away"] = (dec, point, bk_name)
                    elif key == "totals":
                        if oc["name"] == "Over" and (best["total_over"][0] is None or dec > best["total_over"][0]):
                            best["total_over"] = (dec, point, bk_name)
                        elif oc["name"] == "Under" and (best["total_under"][0] is None or dec > best["total_under"][0]):
                            best["total_under"] = (dec, point, bk_name)

        matchup = f"{away_team} at {home_team}"
        # Check h2h arb
        if best["h2h_home"][0] and best["h2h_away"][0]:
            margin = (1 / best["h2h_home"][0]) + (1 / best["h2h_away"][0])
            if margin < 1.0:
                arbs.append({"matchup": matchup, "market": "Moneyline", "profit_pct": (1 - margin) * 100,
                             "side_a": f"{home_team} @ {best['h2h_home'][1]}", "side_b": f"{away_team} @ {best['h2h_away'][1]}"})
        # Check spreads arb (same line)
        if best["spread_home"][0] and best["spread_away"][0] and best["spread_home"][1] is not None and best["spread_away"][1] is not None:
            if best["spread_home"][1] + best["spread_away"][1] == 0:
                margin = (1 / best["spread_home"][0]) + (1 / best["spread_away"][0])
                if margin < 1.0:
                    arbs.append({"matchup": matchup, "market": f"Spread ({best['spread_home'][1]:+.1f})", "profit_pct": (1 - margin) * 100,
                                 "side_a": f"{home_team} @ {best['spread_home'][2]}", "side_b": f"{away_team} @ {best['spread_away'][2]}"})
        # Check totals arb (same line)
        if best["total_over"][0] and best["total_under"][0] and best["total_over"][1] == best["total_under"][1]:
            margin = (1 / best["total_over"][0]) + (1 / best["total_under"][0])
            if margin < 1.0:
                arbs.append({"matchup": matchup, "market": f"Total ({best['total_over'][1]})", "profit_pct": (1 - margin) * 100,
                             "side_a": f"Over @ {best['total_over'][2]}", "side_b": f"Under @ {best['total_under'][2]}"})
    return arbs


def track_line_movement(snapshots: list[dict]) -> list[dict]:
    """Track line movement from first-seen to current odds."""
    if "first_seen_odds" not in st.session_state:
        st.session_state["first_seen_odds"] = {}
    movements = []
    for snap in snapshots:
        key = f"{snap['away_team']}@{snap['home_team']}"
        current = {
            "h2h_home": snap["h2h"].get("home_price"),
            "h2h_away": snap["h2h"].get("away_price"),
            "spread_home": snap["spreads"].get("home_point"),
            "total_line": snap["totals"].get("line"),
        }
        if key not in st.session_state["first_seen_odds"]:
            st.session_state["first_seen_odds"][key] = dict(current)
            continue
        first = st.session_state["first_seen_odds"][key]
        if first.get("h2h_home") and current["h2h_home"]:
            move = current["h2h_home"] - first["h2h_home"]
            if abs(move) >= 5:
                movements.append({"matchup": f"{snap['away_team']} at {snap['home_team']}", "market": "Moneyline",
                                  "open": format_odds(first["h2h_home"]), "current": format_odds(current["h2h_home"]), "move": int(move)})
        if first.get("spread_home") is not None and current["spread_home"] is not None:
            move = current["spread_home"] - first["spread_home"]
            if abs(move) >= 0.5:
                movements.append({"matchup": f"{snap['away_team']} at {snap['home_team']}", "market": "Spread",
                                  "open": f"{first['spread_home']:+.1f}", "current": f"{current['spread_home']:+.1f}", "move": move})
        if first.get("total_line") is not None and current["total_line"] is not None:
            move = current["total_line"] - first["total_line"]
            if abs(move) >= 0.5:
                movements.append({"matchup": f"{snap['away_team']} at {snap['home_team']}", "market": "Total",
                                  "open": f"{first['total_line']:.1f}", "current": f"{current['total_line']:.1f}", "move": move})
    return movements


def h2h_matchup_history(train_data: pd.DataFrame, home_team: str, away_team: str, sport_name: str) -> pd.DataFrame:
    """Get head-to-head matchup history between two teams from training data."""
    if "home_team" not in train_data.columns or "away_team" not in train_data.columns:
        return pd.DataFrame()
    home_norm = canonical_team_name(sport_name, home_team)
    away_norm = canonical_team_name(sport_name, away_team)
    ht = train_data["home_team"].apply(lambda t: canonical_team_name(sport_name, t))
    at = train_data["away_team"].apply(lambda t: canonical_team_name(sport_name, t))
    mask = ((ht == home_norm) & (at == away_norm)) | ((ht == away_norm) & (at == home_norm))
    h2h = train_data.loc[mask].copy()
    if h2h.empty:
        return pd.DataFrame()
    cols = [c for c in ["game_date", "home_team", "away_team", "margin", "total_score", "home_win"] if c in h2h.columns]
    if "game_date" in h2h.columns:
        h2h = h2h.sort_values("game_date", ascending=False)
    return h2h[cols].head(10)


def grade_bet(record: dict) -> str:
    """Grade a settled bet A-F based on EV and CLV quality."""
    if record.get("result") not in ("Win", "Loss", "Push"):
        return ""
    odds = record.get("odds")
    model_prob = None
    if record.get("market") == "Moneyline":
        if record.get("side") == "Home":
            model_prob = record.get("model_home_prob")
        elif record.get("side") == "Away":
            model_prob = record.get("model_away_prob")
    ev = expected_value(model_prob, odds) if model_prob and odds else None
    clv_raw = record.get("clv")
    clv_val = float(clv_raw) if clv_raw is not None and not pd.isna(clv_raw) else None
    score = 0
    if ev is not None:
        if ev > 0.10:
            score += 5
        elif ev > 0.05:
            score += 4
        elif ev > 0:
            score += 3
        elif ev > -0.05:
            score += 2
    else:
        score += 2
    if clv_val is not None:
        if clv_val > 0.10:
            score += 5
        elif clv_val > 0.05:
            score += 4
        elif clv_val > 0:
            score += 3
        elif clv_val > -0.05:
            score += 2
    else:
        score += 2
    if score >= 8:
        return "A"
    if score >= 6:
        return "B"
    if score >= 4:
        return "C"
    if score >= 2:
        return "D"
    return "F"


def feature_importance_data(models_tuple: tuple, sport_name: str) -> pd.DataFrame:
    """Extract feature importance from trained models."""
    winner_model, spread_model, total_model = models_tuple
    diff_features = ["rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"]
    total_features = ["home_rating", "away_rating", "home_form", "away_form", "home_rest", "away_rest", "temperature", "wind_speed", "precipitation"]
    rows = []
    for feat, coef in zip(diff_features, winner_model.coef_[0]):
        rows.append({"feature": feat, "model": "Winner (log-odds)", "importance": coef})
    for feat, coef in zip(diff_features, spread_model.coef_):
        rows.append({"feature": feat, "model": "Spread", "importance": coef})
    for feat, coef in zip(total_features, total_model.coef_):
        rows.append({"feature": feat, "model": "Total", "importance": coef})
    return pd.DataFrame(rows)



with st.sidebar:
    st.header("1. Select Sport & Target")
    sport = st.selectbox("Sport", ["NBA", "NFL"])
    target = st.selectbox("Prediction target", ["Game winner", "Point spread", "Total score"])
    training_lookback = st.slider("Training seasons", min_value=1, max_value=4, value=2)
    auto_refresh_enabled = st.checkbox("Auto-refresh & retrain", value=True)
    auto_refresh_minutes = st.slider("Refresh interval (minutes)", min_value=5, max_value=60, value=15, step=5)
    if "refresh_token" not in st.session_state:
        st.session_state["refresh_token"] = 0
    if st.button("Refresh real data & retrain"):
        st.session_state["refresh_token"] += 1
        load_real_data.clear()
        train_models.clear()
        st.rerun()

install_auto_refresh(auto_refresh_enabled, auto_refresh_minutes)

if "bet_history" not in st.session_state:
    st.session_state.bet_history = load_bet_history()

updated_bet_history, auto_settled_count = settle_history_for_sport(st.session_state.bet_history, sport)
if auto_settled_count:
    st.session_state.bet_history = updated_bet_history
    save_bet_history(st.session_state.bet_history)

preferences = load_preferences()
if "sportsbook_filter" not in st.session_state and preferences.get("sportsbook_filter"):
    st.session_state["sportsbook_filter"] = preferences["sportsbook_filter"]

# Try to load real data, fallback to synthetic
data = load_real_data(sport, lookback_seasons=training_lookback, refresh_token=st.session_state.get("refresh_token", 0))
if data is None or data.empty:
    data = generate_synthetic_data(sport)
data, learned_rows = augment_with_learning_data(data, st.session_state.bet_history, sport)
models = train_models(data, refresh_token=st.session_state.get("refresh_token", 0))

sport_key = "basketball_nba" if sport == "NBA" else "americanfootball_nfl"
odds_data = fetch_odds(sport_key=sport_key)
bookmaker_options = available_bookmakers(odds_data)
matchup_snapshots = [extract_market_snapshot(game) for game in odds_data] if odds_data else []
team_profiles = build_team_profiles(data, sport)

with st.sidebar:
    st.divider()
    latest_game_date = pd.to_datetime(data["game_date"], errors="coerce").max() if "game_date" in data.columns else None
    st.caption(
        f"Training rows: {len(data):,} | Latest game: {latest_game_date.date().isoformat() if pd.notna(latest_game_date) else 'N/A'} | Cache TTL: 15 min"
    )
    if learned_rows:
        st.caption(f"Bet-history learning rows: {learned_rows}")
    if auto_settled_count:
        st.caption(f"Auto-settled this run: {auto_settled_count}")
    st.header("2. Team & Game Inputs")
    selected_bookmaker = st.selectbox(
        "Sportsbook view",
        options=[key for key, _label in bookmaker_options],
        format_func=lambda key: dict(bookmaker_options).get(key, key),
        key="sportsbook_filter",
    )
    if preferences.get("sportsbook_filter") != selected_bookmaker:
        preferences["sportsbook_filter"] = selected_bookmaker
        save_preferences(preferences)
    filtered_odds_data = [filter_game_by_bookmaker(game, selected_bookmaker) for game in odds_data] if odds_data else []
    matchup_snapshots = [extract_market_snapshot(game) for game in filtered_odds_data] if filtered_odds_data else []
    if matchup_snapshots:
        labels = [matchup_selector_label(snapshot) for snapshot in matchup_snapshots]
        selected_label = st.selectbox("Live matchup", labels, key="live_matchup")
        selected_market = matchup_snapshots[labels.index(selected_label)]
        apply_matchup_defaults(selected_market, team_profiles, sport)
        st.caption("Inputs below are auto-filled from the selected live matchup and recent team form.")
    else:
        selected_market = None
        st.caption("No live matchups loaded. Manual team inputs are active.")

    col1, col2 = st.columns(2)
    with col1:
        home_rating = st.slider("Home rating", 70, 140, key="home_rating")
        home_form = st.slider("Home form", 0.0, 1.0, step=0.05, key="home_form")
        home_rest = st.slider("Home rest days", 1, 10, key="home_rest")
    with col2:
        away_rating = st.slider("Away rating", 70, 140, key="away_rating")
        away_form = st.slider("Away form", 0.0, 1.0, step=0.05, key="away_form")
        away_rest = st.slider("Away rest days", 1, 10, key="away_rest")
    if sport == "NFL":
        st.subheader("Weather (NFL only)")
        temperature = st.slider("Temperature (°F)", 0, 100, key="temperature")
        wind_speed = st.slider("Wind (mph)", 0, 30, key="wind_speed")
        precipitation = st.selectbox("Precipitation", [0, 1], format_func=lambda x: "Yes" if x else "No", key="precipitation")
    else:
        temperature = 0
        wind_speed = 0
        precipitation = 0
        st.session_state["temperature"] = 0
        st.session_state["wind_speed"] = 0
        st.session_state["precipitation"] = 0
    st.divider()
    st.header("3. Parlay Calculator")
    parlay_legs = st.slider("# of parlay legs", 1, 5, 1)
    parlay_choice = st.selectbox("Parlay pick", ["Home win", "Away win"])
    st.caption("Estimate the chance of all legs hitting together.")
    st.divider()
    st.header("4. Bankroll")
    bankroll = st.number_input("Bankroll ($)", min_value=50.0, value=500.0, step=50.0)
    flat_bet = st.number_input("Flat bet size ($)", min_value=1.0, value=25.0, step=1.0)
    use_ai = st.checkbox("AI betting summary", value=True)
    if not GEMINI_API_KEY:
        st.caption("Set GEMINI_API_KEY to enable AI summaries.")
    st.divider()
    st.header("5. Injury Reports")
    injury_sport = st.selectbox("Injury sport", ["NBA", "NFL"], key="injury_sport")
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
            st.info("No injury reports found for this team right now.")
    st.divider()
    show_data = st.checkbox("Show training data", value=False)

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



# --- Calculate model probabilities before tabs ---
result = predict_outcome(sport, target, inputs, models)
apply_tracker_defaults(selected_market, target, result)

tab_today, tab_predictions, tab_odds, tab_parlay, tab_tracker, tab_analytics, tab_chat = st.tabs(
    ["🏀 Today's Games", "📊 Predictions", "📈 Odds & Markets", "🎯 Parlay Builder", "📝 Bet Tracker", "🔬 Analytics", "💬 Chat"]
)

# --- Today's Games ---
_todays = todays_games(odds_data, sport)
with tab_today:
    from datetime import datetime
    st.subheader(f"Today's {sport} Games — {datetime.now().strftime('%A, %B %-d')}")
    if _todays:
        st.caption(f"{len(_todays)} game{'s' if len(_todays) != 1 else ''} scheduled today")
        for g in _todays:
            with st.container():
                st.markdown(f"**{g['time']}**")
                gcols = st.columns([3, 1, 1])
                with gcols[0]:
                    st.markdown(f"**{g['away']}** at **{g['home']}**")
                with gcols[1]:
                    if g["home_odds"] is not None and g["away_odds"] is not None:
                        st.caption("Moneyline")
                        st.write(f"{format_odds(g['home_odds'])} / {format_odds(g['away_odds'])}")
                with gcols[2]:
                    if g["total"] is not None:
                        st.caption("Total")
                        st.write(f"O/U {g['total']:.1f}")
                if g["spread_home"] is not None:
                    st.caption(f"Spread: {g['home']} {g['spread_home']:+.1f} | {g['away']} {g['spread_away']:+.1f}")
                st.markdown("---")
    elif not ODDS_API_KEY:
        st.info("Set ODDS_API_KEY to load today's game schedule.")
    else:
        st.info(f"No {sport} games scheduled for today.")

# --- Parlay Builder ---
with tab_parlay:
    st.subheader("🎯 Parlay Builder (up to 15 legs)")
    st.caption("Select legs from today's available games. Odds, implied probability, and payout update live.")
    _leg_options = build_parlay_leg_options(matchup_snapshots)
    if _leg_options:
        _leg_labels = [opt["label"] for opt in _leg_options]
        if "parlay_legs_selected" not in st.session_state:
            st.session_state.parlay_legs_selected = []
        selected_labels = st.multiselect(
            "Pick your legs (max 15)",
            options=_leg_labels,
            default=st.session_state.parlay_legs_selected,
            max_selections=15,
            key="parlay_builder_picks",
        )
        st.session_state.parlay_legs_selected = selected_labels
        selected_legs = [opt for opt in _leg_options if opt["label"] in selected_labels]
        parlay_stake = st.number_input("Parlay stake ($)", min_value=1.0, value=10.0, step=5.0, key="parlay_stake")
        if selected_legs:
            st.markdown("---")
            st.write(f"### Parlay Slip — {len(selected_legs)} leg{'s' if len(selected_legs) != 1 else ''}")
            for i, leg in enumerate(selected_legs, 1):
                st.write(f"**Leg {i}:** {leg['label']}  \n"
                         f"&nbsp;&nbsp;&nbsp;&nbsp;{leg['market']} • {leg['matchup']}")
            st.markdown("---")
            combined_dec = parlay_decimal_odds(selected_legs)
            combined_prob = parlay_implied_prob(selected_legs)
            payout = parlay_payout(parlay_stake, selected_legs)
            profit = payout - parlay_stake if payout else None
            # Convert combined decimal back to American for display
            if combined_dec is not None and combined_dec > 1:
                if combined_dec >= 2:
                    combined_american = (combined_dec - 1) * 100
                else:
                    combined_american = -100 / (combined_dec - 1)
            else:
                combined_american = None
            pcols = st.columns(4)
            with pcols[0]:
                st.metric("Combined Odds", format_odds(combined_american) if combined_american is not None else "N/A")
            with pcols[1]:
                st.metric("Implied Prob", f"{combined_prob:.2%}" if combined_prob is not None else "N/A")
            with pcols[2]:
                st.metric("Payout", f"${payout:.2f}" if payout is not None else "N/A")
            with pcols[3]:
                st.metric("Profit", f"${profit:.2f}" if profit is not None else "N/A")
            # Risk assessment
            if combined_prob is not None:
                if combined_prob >= 0.25:
                    st.success("Reasonable probability. This parlay has a realistic shot.")
                elif combined_prob >= 0.10:
                    st.warning("Low probability. Treat this as a long shot — size accordingly.")
                elif combined_prob >= 0.03:
                    st.warning("Very low probability. This is a lottery ticket — bet only what you can lose.")
                else:
                    st.error("Extremely unlikely. Combined odds are stacked heavily against you.")
            if len(selected_legs) >= 6:
                st.caption("Parlays with 6+ legs have a very low hit rate historically. Consider correlated legs or smaller combos.")
        else:
            st.info("Select at least one leg above to build your parlay.")
    elif not ODDS_API_KEY:
        st.info("Set ODDS_API_KEY to enable the parlay builder.")
    else:
        st.info(f"No {sport} games with odds available right now to build a parlay.")

# --- Odds and value bet UI ---
with tab_odds:
    st.subheader("Live Sportsbook Odds & Value Bets")
    st.caption("Moneyline value bets include implied probability and EV. Spread and total rows compare your model against the live line.")
    if matchup_snapshots:
        for snapshot in matchup_snapshots:
            home = snapshot["home_team"]
            away = snapshot["away_team"]
            st.markdown(f"**{away} at {home}**")
            home_odds = snapshot["h2h"]["home_price"]
            away_odds = snapshot["h2h"]["away_price"]
            model_home_prob = result.get("home_win_prob") if target == "Game winner" else None
            model_away_prob = result.get("away_win_prob") if target == "Game winner" else None
            home_imp = implied_prob(home_odds) if home_odds else None
            away_imp = implied_prob(away_odds) if away_odds else None
            value_home = model_home_prob is not None and home_imp is not None and model_home_prob > home_imp
            value_away = model_away_prob is not None and away_imp is not None and model_away_prob > away_imp
            ev_home = expected_value(model_home_prob, home_odds, flat_bet)
            ev_away = expected_value(model_away_prob, away_odds, flat_bet)
            st.write(
                f"{home}: Odds {format_odds(home_odds)}, Implied Prob {format_percent(home_imp)}, "
                f"Model Prob {format_percent(model_home_prob)}, EV ${format_number(ev_home)}"
            )
            if value_home:
                st.success(f"Value bet: {home}")
            st.write(
                f"{away}: Odds {format_odds(away_odds)}, Implied Prob {format_percent(away_imp)}, "
                f"Model Prob {format_percent(model_away_prob)}, EV ${format_number(ev_away)}"
            )
            if value_away:
                st.success(f"Value bet: {away}")
            if snapshot["spreads"]["home_point"] is not None:
                predicted_margin = result.get("predicted_margin")
                spread_edge = predicted_margin + snapshot["spreads"]["home_point"] if predicted_margin is not None else None
                st.write(
                    f"Spread board: {home} {snapshot['spreads']['home_point']:+.1f} ({format_odds(snapshot['spreads']['home_price'])}), "
                    f"{away} {snapshot['spreads']['away_point']:+.1f} ({format_odds(snapshot['spreads']['away_price'])})"
                )
                if spread_edge is not None:
                    st.write(f"Model vs spread line: {spread_edge:+.1f} points to the home side.")
            if snapshot["totals"]["line"] is not None:
                predicted_total = result.get("predicted_total")
                total_edge = predicted_total - snapshot["totals"]["line"] if predicted_total is not None else None
                st.write(
                    f"Total board: Over {snapshot['totals']['line']:.1f} ({format_odds(snapshot['totals']['over_price'])}), "
                    f"Under {snapshot['totals']['line']:.1f} ({format_odds(snapshot['totals']['under_price'])})"
                )
                if total_edge is not None:
                    st.write(f"Model vs total line: {total_edge:+.1f} points.")
            st.markdown("---")
    elif not ODDS_API_KEY:
        st.info("Odds data is unavailable. Set ODDS_API_KEY to load live sportsbook markets.")
    else:
        st.info("No odds available for the selected sport/bookmaker right now.")

    # --- Market Intelligence: Arbitrage & Line Movement ---
    _arb_opps = detect_arbitrage(odds_data)
    _line_moves = track_line_movement(matchup_snapshots)
    if _arb_opps or _line_moves:
        st.subheader("Market Intelligence")
    if _arb_opps:
        st.write("#### Arbitrage Opportunities")
        st.caption("Cross-book pricing that guarantees profit regardless of outcome.")
        for arb in _arb_opps:
            st.success(f"**{arb['matchup']}** — {arb['market']}: {arb['profit_pct']:.2f}% guaranteed profit  \n"
                       f"Leg 1: {arb['side_a']} | Leg 2: {arb['side_b']}")
    if _line_moves:
        st.write("#### Line Movement")
        st.caption("Significant moves since your session started. Reverse movement (line moving against the public) can signal sharp action.")
        _move_df = pd.DataFrame(_line_moves)
        st.dataframe(_move_df, width="stretch")


with tab_predictions:
    cols = st.columns(2)
    with cols[0]:
        if selected_market is not None:
            logo_cols = st.columns([1, 3, 1])
            away_logo = team_logo_url(sport, selected_market["away_team"])
            home_logo = team_logo_url(sport, selected_market["home_team"])
            with logo_cols[0]:
                if away_logo:
                    st.image(away_logo, width=72)
            with logo_cols[1]:
                st.write("### Selected matchup")
                st.write(f"**{selected_market['away_team']} at {selected_market['home_team']}**")
                st.caption(sportsbook_summary(selected_market))
            with logo_cols[2]:
                if home_logo:
                    st.image(home_logo, width=72)
        else:
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

    # --- Head-to-Head History ---
    if selected_market is not None:
        _h2h_df = h2h_matchup_history(data, selected_market["home_team"], selected_market["away_team"], sport)
        if not _h2h_df.empty:
            st.write("### Head-to-Head History")
            st.caption(f"Last {len(_h2h_df)} meetings between these teams from training data.")
            if "home_win" in _h2h_df.columns:
                home_wins = int(_h2h_df["home_win"].sum())
                away_wins = len(_h2h_df) - home_wins
                h2h_cols = st.columns(3)
                with h2h_cols[0]:
                    st.metric(f"{selected_market['home_team']} wins (at home)", home_wins)
                with h2h_cols[1]:
                    st.metric(f"{selected_market['away_team']} wins (at home)", away_wins)
                with h2h_cols[2]:
                    avg_margin = _h2h_df["margin"].mean() if "margin" in _h2h_df.columns else None
                    st.metric("Avg margin (home perspective)", f"{avg_margin:+.1f}" if avg_margin is not None else "N/A")
            st.dataframe(_h2h_df, width="stretch")

    st.markdown("---")

    st.write("## Betting Edge")
    recommendation = build_bet_recommendation(result, target, selected_market)
    st.write(recommendation)

    if target == "Game winner":
        home_kelly = kelly_fraction(result["home_win_prob"], selected_market["h2h"]["home_price"] if selected_market else None)
        away_kelly = kelly_fraction(result["away_win_prob"], selected_market["h2h"]["away_price"] if selected_market else None)
        suggested_fraction = max(home_kelly, away_kelly)
        suggested_wager = bankroll * min(suggested_fraction * 0.25, 0.05)
        st.write(
            f"Conservative Kelly stake: ${suggested_wager:.2f} "
            f"({min(suggested_fraction * 0.25, 0.05):.1%} of bankroll, capped at 5%)."
        )
        if suggested_wager > flat_bet * 2:
            st.warning("Model edge is strong, but keep sizing disciplined. Avoid scaling too quickly after a short hot streak.")
        else:
            st.info("No major edge detected. Flat-bet sizing is safer than pressing this matchup.")

    if use_ai:
        ai_prompt = (
            f"Summarize this betting spot in 3 short sentences. Sport: {sport}. Target: {target}. "
            f"Home win probability: {result.get('home_win_prob')}. Away win probability: {result.get('away_win_prob')}. "
            f"Market snapshot: {selected_market}. Recommendation: {recommendation}"
        )
        ai_summary = maybe_generate_ai_summary(ai_prompt)
        st.write("### AI Summary")
        st.write(ai_summary or recommendation)
    elif not GEMINI_API_KEY:
        st.caption("AI summary is disabled because GEMINI_API_KEY is not set.")


with tab_tracker:
    st.write("### Bet tracker")
    tracker_filter_cols = st.columns(2)
    with tracker_filter_cols[0]:
        filter_sport = st.selectbox("Filter sport", ["All"] + sorted({record.get("sport", "") for record in st.session_state.bet_history if record.get("sport")}), key="filter_sport")
        filter_market = st.selectbox("Filter market", ["All"] + sorted({record.get("market", "") for record in st.session_state.bet_history if record.get("market")}), key="filter_market")
    with tracker_filter_cols[1]:
        filter_book = st.selectbox("Filter book", ["All"] + sorted({record.get("bookmaker", "") for record in st.session_state.bet_history if record.get("bookmaker")}), key="filter_book")
        filter_tag = st.selectbox("Filter tag", ["All"] + sorted({record.get("tag", "") for record in st.session_state.bet_history if record.get("tag")}), key="filter_tag")

    bet_market = st.selectbox("Log market", ["Moneyline", "Spread", "Total"], key="bet_market")
    bet_side = st.selectbox("Log side", ["Home", "Away", "Over", "Under"], key="bet_side")
    bet_odds = st.number_input("Odds to log", step=1.0, key="bet_odds_value")
    closing_odds = st.number_input("Closing odds (optional)", step=1.0, key="closing_odds_value")
    if selected_market is not None and st.button("Fetch historical close"):
        historical_event = historical_close_for_snapshot(selected_market, sport_key, selected_bookmaker)
        if historical_event is not None:
            historical_snapshot = extract_market_snapshot(historical_event)
            if bet_market == "Moneyline":
                if bet_side == "Home":
                    st.session_state["closing_odds_value"] = float(historical_snapshot["h2h"]["home_price"] or closing_odds)
                else:
                    st.session_state["closing_odds_value"] = float(historical_snapshot["h2h"]["away_price"] or closing_odds)
            elif bet_market == "Spread":
                if bet_side == "Home":
                    st.session_state["closing_odds_value"] = float(historical_snapshot["spreads"]["home_price"] or closing_odds)
                else:
                    st.session_state["closing_odds_value"] = float(historical_snapshot["spreads"]["away_price"] or closing_odds)
            else:
                if bet_side == "Over":
                    st.session_state["closing_odds_value"] = float(historical_snapshot["totals"]["over_price"] or closing_odds)
                else:
                    st.session_state["closing_odds_value"] = float(historical_snapshot["totals"]["under_price"] or closing_odds)
            st.success("Historical closing odds loaded from snapshot nearest game start.")
        else:
            st.info("Historical close not available for this event or your Odds API plan may not include historical data.")
    bet_stake = st.number_input("Stake to log ($)", min_value=1.0, value=float(flat_bet), step=1.0)
    bet_result = st.selectbox("Result", ["Pending", "Win", "Loss", "Push"], key="bet_result")
    bet_tag = st.text_input("Tag", placeholder="NBA model, revenge spot, weather edge", key="bet_tag")
    bet_notes = st.text_area("Notes", placeholder="Why this bet exists, what would invalidate it, and what to review later.", key="bet_notes")
    if st.button("Add bet to tracker"):
        clv = closing_line_value(bet_odds, closing_odds)
        line_value = None
        score_fields = {}
        if selected_market is not None:
            if bet_market == "Spread":
                line_value = selected_market["spreads"]["home_point"] if bet_side == "Home" else selected_market["spreads"]["away_point"]
            elif bet_market == "Total":
                line_value = selected_market["totals"]["line"]
        st.session_state.bet_history.append(
            {
                "logged_at": pd.Timestamp.now("UTC").isoformat(),
                "sport": sport,
                "target": target,
                "market": bet_market,
                "side": bet_side,
                "bookmaker": dict(bookmaker_options).get(selected_bookmaker, selected_bookmaker),
                "event_id": selected_market.get("event_id") if selected_market else None,
                "commence_time": selected_market.get("commence_time") if selected_market else None,
                "home_team": selected_market.get("home_team") if selected_market else None,
                "away_team": selected_market.get("away_team") if selected_market else None,
                "matchup": matchup_label(selected_market) if selected_market else None,
                "home_rating": home_rating,
                "away_rating": away_rating,
                "home_form": home_form,
                "away_form": away_form,
                "home_rest": home_rest,
                "away_rest": away_rest,
                "temperature": temperature,
                "wind_speed": wind_speed,
                "precipitation": precipitation,
                "model_home_prob": result.get("home_win_prob"),
                "model_away_prob": result.get("away_win_prob"),
                "predicted_margin": result.get("predicted_margin"),
                "predicted_total": result.get("predicted_total"),
                "line": line_value,
                "odds": bet_odds,
                "closing_odds": closing_odds,
                "clv": clv,
                "stake": bet_stake,
                "result": bet_result,
                "tag": bet_tag,
                "notes": bet_notes,
                **score_fields,
            }
        )
        save_bet_history(st.session_state.bet_history)

    settle_cols = st.columns([1, 1, 2])
    with settle_cols[0]:
        auto_settle_requested = st.button("Auto-settle recent bets")
    with settle_cols[1]:
        settlement_sport = st.selectbox("Settlement sport", [sport], key="settlement_sport")

    if auto_settle_requested and st.session_state.bet_history:
        settlement_key = "basketball_nba" if settlement_sport == "NBA" else "americanfootball_nfl"
        updated_history, settled_count = auto_settle_bets(st.session_state.bet_history, settlement_key)
        st.session_state.bet_history = updated_history
        save_bet_history(st.session_state.bet_history)
        if settled_count:
            st.success(f"Auto-settled {settled_count} pending bets using recent final scores.")
        else:
            st.info("No pending bets matched recent completed games. The scores endpoint only covers recent events.")

    if st.session_state.bet_history and st.button("Clear tracked bets"):
        st.session_state.bet_history = []
        save_bet_history(st.session_state.bet_history)

    if st.session_state.bet_history:
        history_df = pd.DataFrame(st.session_state.bet_history)
        if "logged_at" in history_df.columns:
            history_df["logged_at"] = pd.to_datetime(history_df["logged_at"], errors="coerce")
        history_df["profit"] = history_df.apply(
            lambda row: row["stake"] * (row["odds"] / 100) if row["result"] == "Win" and row["odds"] > 0
            else (row["stake"] * (100 / abs(row["odds"])) if row["result"] == "Win" else (-row["stake"] if row["result"] == "Loss" else 0)),
            axis=1,
        )
        filtered_history = history_df.copy()
        if filter_sport != "All":
            filtered_history = filtered_history[filtered_history["sport"] == filter_sport]
        if filter_market != "All":
            filtered_history = filtered_history[filtered_history["market"] == filter_market]
        if filter_book != "All":
            filtered_history = filtered_history[filtered_history["bookmaker"] == filter_book]
        if filter_tag != "All" and "tag" in filtered_history.columns:
            filtered_history = filtered_history[filtered_history["tag"] == filter_tag]

        export_df = filtered_history.copy()
        if "logged_at" in export_df.columns:
            export_df["logged_at"] = export_df["logged_at"].astype(str)
        st.download_button(
            "Download filtered tracker CSV",
            data=export_df.to_csv(index=False),
            file_name="bet_tracker_export.csv",
            mime="text/csv",
        )
        # Add bet grades for settled bets
        filtered_history["grade"] = filtered_history.apply(lambda row: grade_bet(row.to_dict()), axis=1)
        st.dataframe(filtered_history, width="stretch")
        st.metric("Tracked profit/loss", f"${filtered_history['profit'].sum():.2f}")
        perf = tracker_performance_summary(filtered_history)
        perf_cols = st.columns(3)
        with perf_cols[0]:
            st.metric("Settled bets", perf["bets"])
        with perf_cols[1]:
            st.metric("ROI", format_percent(perf["roi"]))
        with perf_cols[2]:
            st.metric("Avg CLV", format_number(perf["avg_clv"]))

        if not filtered_history.empty:
            chart_df = filtered_history.reset_index(drop=True).copy()
            chart_df["bet_number"] = np.arange(1, len(chart_df) + 1)
            chart_df["cumulative_profit"] = chart_df["profit"].cumsum()
            st.write("### Tracker charts")
            st.line_chart(chart_df.set_index("bet_number")[["cumulative_profit"]])
            if "tag" in filtered_history.columns:
                grouped_profit = filtered_history.groupby(filtered_history["tag"].fillna("").replace("", "untagged"))["profit"].sum().sort_values(ascending=False)
                if not grouped_profit.empty:
                    st.bar_chart(grouped_profit)
            if "logged_at" in filtered_history.columns and filtered_history["logged_at"].notna().any():
                review_df = filtered_history.dropna(subset=["logged_at"]).copy()
                review_df["week"] = review_df["logged_at"].dt.to_period("W").astype(str)
                weekly_profit = review_df.groupby("week")["profit"].sum()
                if not weekly_profit.empty:
                    st.write("### Weekly review")
                    st.bar_chart(weekly_profit)
                market_review = review_df.groupby("market")["profit"].agg(["sum", "count"]).rename(columns={"sum": "profit", "count": "bets"})
                if not market_review.empty:
                    st.dataframe(market_review, width="stretch")
    else:
        st.caption("No bets logged yet. Add wagers here to track profit and loss over time.")


with tab_analytics:
    if show_data:
        st.write("### Sample training data")
        st.dataframe(data.head())
        st.write("### Training dataset statistics")
        st.write(data[["margin", "total_score", "home_win"]].describe())


    st.write("### Model accuracy")
    try:
        latest_game_date_str = latest_game_date.date().isoformat() if pd.notna(latest_game_date) else None
        features = ["rating_diff", "form_diff", "rest_diff", "home_advantage", "temperature", "wind_speed", "precipitation"]
        if all(f in data.columns for f in features):
            X = data[features]
        else:
            X = data[[c for c in features if c in data.columns]]
        y = data["home_win"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        y_pred = models[0].predict(X_test)
        home_win_accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Home-win classification accuracy: {home_win_accuracy:.3f}")

        spread_preds = models[1].predict(X_test)
        mae_spread = mean_absolute_error(data.loc[X_test.index, "margin"], spread_preds)
        st.write(f"Point spread MAE: {mae_spread:.2f} points")

        total_features = ["home_rating", "away_rating", "home_form", "away_form", "home_rest", "away_rest", "temperature", "wind_speed", "precipitation"]
        if all(f in data.columns for f in total_features):
            X_total = data[total_features]
        else:
            X_total = data[[c for c in total_features if c in data.columns]]
        X_total_train, X_total_test, y_total_train, y_total_test = train_test_split(
            X_total, data["total_score"], test_size=0.25, random_state=42
        )
        total_preds = models[2].predict(X_total_test)
        mae_total = mean_absolute_error(y_total_test, total_preds)
        st.write(f"Total score MAE: {mae_total:.2f} points")

        record_model_snapshot(
            sport=sport,
            training_rows=len(data),
            learned_rows=learned_rows,
            latest_game_date=latest_game_date_str,
            accuracy=home_win_accuracy,
            spread_mae=mae_spread,
            total_mae=mae_total,
        )

        home_win_probs = models[0].predict_proba(X_test)[:, 1]
        with st.expander("Calibration & backtest", expanded=False):
            calib = calibration_table(home_win_probs, y_test)
            if not calib.empty:
                st.dataframe(calib, width="stretch")
                chart_data = calib[["predicted_prob", "actual_rate"]].rename(columns={"predicted_prob": "Predicted", "actual_rate": "Actual"})
                st.line_chart(chart_data)
            backtest = model_backtest_table(home_win_probs, y_test)
            st.dataframe(backtest, width="stretch")
            st.caption("Backtest uses a simple holdout proxy: bet the model side only when confidence exceeds the threshold, scored at -110 equivalent pricing.")

        with st.expander("Historical closing-line backtest", expanded=False):
            historical_sample_size = st.slider("Historical closing-line sample", min_value=1, max_value=5, value=3)
            if st.button("Run historical closing-line backtest"):
                historical_backtest = historical_moneyline_backtest(data, sport_key, selected_bookmaker, historical_sample_size)
                if not historical_backtest.empty:
                    st.dataframe(historical_backtest, width="stretch")
                    st.metric("Historical backtest units", f"{historical_backtest['units'].sum():.2f}u")
                    st.metric("Historical win rate", format_percent((historical_backtest["result"] == "Win").mean()))
                    st.caption("This uses archived Odds API moneyline snapshots matched by teams and game time. Historical endpoints require a paid plan and consume quota.")
                else:
                    st.info("No archived odds snapshots were returned for the sampled games. Historical odds may be unavailable on your plan or for the chosen bookmaker.")

        history_frame = model_history_frame(sport)
        if not history_frame.empty:
            with st.expander("Model trend", expanded=False):
                recent_history = history_frame.tail(20).copy()
                display_history = recent_history[["timestamp", "training_rows", "learned_rows", "accuracy", "spread_mae", "total_mae"]].copy()
                display_history["timestamp"] = display_history["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                latest_snapshot = recent_history.iloc[-1]
                previous_snapshot = recent_history.iloc[-2] if len(recent_history) > 1 else None
                trend_cols = st.columns(3)
                with trend_cols[0]:
                    st.metric(
                        "Accuracy trend",
                        f"{latest_snapshot['accuracy']:.3f}",
                        None if previous_snapshot is None else f"{latest_snapshot['accuracy'] - previous_snapshot['accuracy']:+.3f}",
                    )
                with trend_cols[1]:
                    st.metric(
                        "Spread MAE trend",
                        f"{latest_snapshot['spread_mae']:.2f}",
                        None if previous_snapshot is None else f"{previous_snapshot['spread_mae'] - latest_snapshot['spread_mae']:+.2f}",
                    )
                with trend_cols[2]:
                    st.metric(
                        "Total MAE trend",
                        f"{latest_snapshot['total_mae']:.2f}",
                        None if previous_snapshot is None else f"{previous_snapshot['total_mae'] - latest_snapshot['total_mae']:+.2f}",
                    )
                trend_chart = recent_history.set_index("timestamp")[["accuracy", "spread_mae", "total_mae"]]
                st.line_chart(trend_chart)
                st.dataframe(display_history, width="stretch")

        # --- Feature Importance ---
        st.write("### Feature Importance")
        st.caption("Model coefficient magnitudes — how much each input moves the prediction.")
        fi = feature_importance_data(models, sport)
        if not fi.empty:
            for model_name in fi["model"].unique():
                model_fi = fi[fi["model"] == model_name].copy()
                model_fi["abs_importance"] = model_fi["importance"].abs()
                model_fi = model_fi.sort_values("abs_importance", ascending=False)
                chart = model_fi.set_index("feature")[["importance"]]
                st.write(f"**{model_name}**")
                st.bar_chart(chart)
    except Exception as e:
        st.warning(f"Could not compute model accuracy: {e}")

    st.info(
        "Treat this as a decision-support tool, not an autopilot. Focus on closing line value, disciplined staking, and logged results over time."
    )


with tab_chat:
    # --- Betting Assistant Chat ---
    st.markdown("---")
    st.write("## Betting Assistant")
    st.caption("Ask anything about the current matchup, betting strategy, bankroll management, or value hunting.")

    if not GEMINI_API_KEY or genai is None:
        st.warning("Set GEMINI_API_KEY in your .env file to enable the betting assistant.")
    else:
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Build a system context from the current app state
        _chat_context = (
            f"You are an expert sports betting analyst assistant embedded in a betting analysis app. "
            f"Current sport: {sport}. Prediction target: {target}. "
            f"Selected matchup: {matchup_label(selected_market) if selected_market else 'None selected'}. "
            f"Home win probability: {result.get('home_win_prob', 'N/A')}. "
            f"Away win probability: {result.get('away_win_prob', 'N/A')}. "
            f"Predicted margin: {result.get('predicted_margin', 'N/A')}. "
            f"Predicted total: {result.get('predicted_total', 'N/A')}. "
            f"Bankroll: ${bankroll}. Flat bet: ${flat_bet}. "
            f"Answer concisely and practically. Focus on profitable betting habits, value, and discipline."
        )

        # Render chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask the betting assistant...")
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Build full conversation for the API
                    full_prompt = _chat_context + "\n\nConversation so far:\n"
                    for msg in st.session_state.chat_messages[:-1]:
                        full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
                    full_prompt += f"User: {user_input}\nAssistant:"

                    reply = maybe_generate_ai_summary(full_prompt)
                    if reply:
                        st.write(reply)
                        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                    else:
                        fallback = "AI assistant is unavailable right now (quota or key issue). Check your GEMINI_API_KEY."
                        st.warning(fallback)
                        st.session_state.chat_messages.append({"role": "assistant", "content": fallback})

        if st.session_state.chat_messages and st.button("Clear chat"):
            st.session_state.chat_messages = []
            st.rerun()

