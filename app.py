import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [class*="css"], [class*="stApp"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d !important;
    color: #ffffff !important;
}

/* ── Layout ── */
.main { background-color: #0d0d0d !important; }
.stApp { background-color: #0d0d0d !important; }
section[data-testid="stForm"] { max-width: 100%; }

/* ── Header ── */
.header-container {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 50;
    height: 4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1rem;
    background: linear-gradient(180deg, rgba(13,13,13,0.95) 0%, rgba(13,13,13,0.8) 100%);
    border-bottom: 1px solid #222222;
    backdrop-filter: blur(10px);
}

.header-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 3px;
    color: #F5A623;
    line-height: 1;
}

.header-btn {
    background: none;
    border: none;
    color: #F5A623;
    cursor: pointer;
    font-size: 1.3rem;
    opacity: 0.8;
    transition: opacity 0.2s;
}

.header-btn:hover { opacity: 1; }

/* ── Main Content ── */
.main-content {
    padding-top: 5rem;
    padding-bottom: 6rem;
    max-width: 640px;
    margin: 0 auto;
}

/* ── Footer Nav ── */
.footer-nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 50;
    height: 5rem;
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 0.5rem 1rem 0.5rem 1rem;
    border-top: 1px solid #222222;
    background: linear-gradient(180deg, rgba(13,13,13,0.8) 0%, rgba(13,13,13,0.95) 100%);
    backdrop-filter: blur(10px);
}

.footer-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
    font-size: 1.3rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    opacity: 0.5;
    transition: opacity 0.2s, color 0.2s;
    text-decoration: none;
    color: #666666;
}

.footer-btn.active {
    color: #F5A623;
    opacity: 1;
    background-color: rgba(245, 166, 35, 0.1);
    padding: 0.5rem 1.2rem;
    border-radius: 2rem;
}

.footer-btn:hover {
    color: #F5A623;
    opacity: 0.9;
}

/* ── Team Selection Card ── */
.team-card {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 0.75rem;
    padding: 1rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.teams-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.team-selector {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    background: #1a1a1a;
    border: 1px solid #222222;
    border-radius: 0.5rem;
    padding: 0.75rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

.team-selector:hover { border-color: #333333; }

.team-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888888;
}

.team-logo {
    width: 3.75rem;
    height: 3.75rem;
    border-radius: 50%;
    background: #000000;
    border: 1px solid #333333;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.team-logo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.team-abbr {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    color: #ffffff;
    line-height: 1;
    margin-top: 0.25rem;
}

.vs-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    color: #333333;
    line-height: 1;
}

/* ── Predict Button ── */
.predict-btn-container {
    display: flex;
    gap: 0.5rem;
}

.predict-btn {
    flex: 1;
    background: linear-gradient(180deg, #F5A623 0%, #e8890a 100%);
    color: #000000;
    border: none;
    border-radius: 0.5rem;
    padding: 1rem;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: opacity 0.2s;
    box-shadow: 0 0 15px rgba(245, 166, 35, 0.3);
    text-transform: uppercase;
}

.predict-btn:hover { opacity: 0.85; }

/* ── Result Box ── */
.result-box {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 0.75rem;
    overflow: hidden;
    margin-top: 1rem;
}

.result-header {
    background: #000000;
    border-bottom: 1px solid #222222;
    padding: 1rem;
    display: flex;
    justify-content: center;
    align-items: center;
}

.result-badge {
    background: #1a1a1a;
    border: 1px solid #F5A623;
    border-radius: 2rem;
    padding: 0.5rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 0 15px rgba(245, 166, 35, 0.2);
}

.result-badge-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    color: #F5A623;
    text-transform: uppercase;
}

/* ── Probability Section ── */
.prob-section {
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1rem;
}

.prob-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

.prob-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 1px;
    color: #888888;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.prob-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    line-height: 1;
    color: #F5A623;
}

.prob-value.away {
    color: #5B8CFF;
    text-align: right;
}

.prob-bar {
    width: 100%;
    height: 1rem;
    background: #1a1a1a;
    border-radius: 0.25rem;
    overflow: hidden;
    display: flex;
}

.prob-bar-fill {
    background: linear-gradient(90deg, #F5A623, #e8890a);
    height: 100%;
}

.prob-bar-fill.away {
    background: #5B8CFF;
}

/* ── Score Display ── */
.score-section {
    background: #111111;
    border-top: 1px solid #222222;
    padding: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
}

.score-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
}

.score-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 1px;
    color: #888888;
    text-transform: uppercase;
}

.score-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    color: #ffffff;
    line-height: 1;
}

.score-value.spread {
    color: #F5A623;
}

.divider-line {
    width: 1px;
    height: 2.5rem;
    background: #333333;
}

/* ── Streamlit Overrides ── */
.stButton > button {
    background: linear-gradient(180deg, #F5A623 0%, #e8890a 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 0 15px rgba(245, 166, 35, 0.3) !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stSelectbox label {
    color: #888888 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #222222 !important;
    border-radius: 0.5rem !important;
    color: #f0f0f0 !important;
}

.stSelectbox [data-baseweb="popover"] {
    background-color: #1a1a1a !important;
}

/* ── Schedule Card ── */
.game-card {
    background: #1a1a1a;
    border: 1px solid #222222;
    border-radius: 0.75rem;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
}

.game-info {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.game-teams {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex: 1;
}

.game-team {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.game-logo {
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    background: #000000;
    border: 1px solid #333333;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.game-logo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.game-text {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.game-name {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #e0e0e0;
}

.game-abbr {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    color: #666666;
}

.game-time {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: #666666;
    text-align: right;
    min-width: 50px;
}

.schedule-date {
    font-size: 0.65rem;
    color: #666666;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.5rem 0;
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e1e;
}

.no-games-msg {
    color: #666666;
    font-size: 0.75rem;
    text-align: center;
    padding: 2rem 1rem;
    border: 1px dashed #222222;
    border-radius: 0.75rem;
    background: #0a0a0a;
}

/* ── Stats Grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.stat-item {
    background: #000000;
    border-radius: 0.5rem;
    padding: 0.5rem 0.6rem;
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    border: 1px solid #1e1e1e;
}

.stat-key { color: #666666; font-weight: 500; }
.stat-val { color: #f0f0f0; font-weight: 700; }

/* ── Divider ── */
.divider { 
    border: none; 
    border-top: 1px solid #222222; 
    margin: 1rem 0;
}

/* ── Hide Streamlit Elements ── */
.viewerBadge_container__1QSob { display: none !important; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

/* ── Responsive ── */
@media (max-width: 640px) {
    .header-title { font-size: 1.5rem; }
    .prob-value { font-size: 1.5rem; }
    .score-value { font-size: 1.1rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Team Logos ────────────────────────────────────────────────────────────────
TEAM_IDS_ESPN = {
    "ATL": "1", "BOS": "2", "BKN": "17", "CHA": "30", "CHI": "4",
    "CLE": "5", "DAL": "6", "DEN": "7", "DET": "8", "GSW": "9",
    "HOU": "10", "IND": "11", "LAC": "12", "LAL": "13", "MEM": "29",
    "MIA": "14", "MIL": "15", "MIN": "16", "NOP": "3", "NYK": "18",
    "OKC": "25", "ORL": "19", "PHI": "20", "PHX": "21", "POR": "22",
    "SAC": "23", "SAS": "24", "TOR": "28", "UTA": "26", "WAS": "27"
}

def get_logo_url(abbr):
    if abbr in TEAM_IDS_ESPN:
        return f"https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{abbr.lower()}.png"
    return None

def logo_html(abbr, size=36):
    url = get_logo_url(abbr)
    name = TEAM_NAMES.get(abbr, abbr)
    if url:
        return f'<img src="{url}" width="{size}" height="{size}" style="object-fit:contain;vertical-align:middle;" title="{name}" onerror="this.style.display=\'none\'">'
    return f'<span style="font-size:0.65rem;color:#555;font-weight:600;">{abbr}</span>'

# ── Load models (V3 stacking ensemble) ───────────────────────────────────────
@st.cache_resource
def load_models():
    margin_model      = joblib.load("margin_model.pkl")        # XGBRegressor → dự đoán MARGIN
    total_model       = joblib.load("total_points_model.pkl")  # XGBRegressor → dự đoán TOTAL PTS
    xgb_clf           = joblib.load("xgb_clf.pkl")             # XGBClassifier
    calibrator        = joblib.load("calibrator.pkl")          # Platt calibration
    lgb_clf           = joblib.load("lgb_clf.pkl")             # LightGBM
    meta_model        = joblib.load("meta_model.pkl")          # Stacking meta (LogisticRegression)
    return margin_model, total_model, xgb_clf, calibrator, lgb_clf, meta_model

@st.cache_data
def load_data():
    df = pd.read_csv("nba_model_ready_v3.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

@st.cache_data
def load_config():
    with open("feature_config_v3.json") as f:
        return json.load(f)

margin_model, total_model, xgb_clf, calibrator, lgb_clf, meta_model = load_models()
df = load_data()
config = load_config()
FEATURES = config["feature_set_v3"]   # 21 features V3

# ── Team names ────────────────────────────────────────────────────────────────
TEAM_NAMES = {
    "ATL": "Atlanta Hawks",       "BKN": "Brooklyn Nets",        "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",   "CHI": "Chicago Bulls",        "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",    "DEN": "Denver Nuggets",       "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors","HOU": "Houston Rockets",     "IND": "Indiana Pacers",
    "LAC": "LA Clippers",         "LAL": "LA Lakers",            "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",          "MIL": "Milwaukee Bucks",      "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans","NYK": "New York Knicks",      "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",       "PHI": "Philadelphia 76ers",   "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",   "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",     "UTA": "Utah Jazz",            "WAS": "Washington Wizards"
}

NBA_ID_TO_ABBR = {
    1610612737:"ATL",1610612738:"BOS",1610612751:"BKN",1610612766:"CHA",
    1610612741:"CHI",1610612739:"CLE",1610612742:"DAL",1610612743:"DEN",
    1610612765:"DET",1610612744:"GSW",1610612745:"HOU",1610612754:"IND",
    1610612746:"LAC",1610612747:"LAL",1610612763:"MEM",1610612748:"MIA",
    1610612749:"MIL",1610612750:"MIN",1610612740:"NOP",1610612752:"NYK",
    1610612760:"OKC",1610612753:"ORL",1610612755:"PHI",1610612756:"PHX",
    1610612757:"POR",1610612758:"SAC",1610612759:"SAS",1610612761:"TOR",
    1610612762:"UTA",1610612764:"WAS"
}

all_teams = sorted(df["HOME_TEAM"].unique().tolist())
team_options = [f"{TEAM_NAMES.get(t, t)} ({t})" for t in all_teams]
abbr_map = {f"{TEAM_NAMES.get(t, t)} ({t})": t for t in all_teams}

# ── Feature builder (V3 — 21 features) ───────────────────────────────────────
def build_features(home_team, away_team):
    home_rows = df[df["HOME_TEAM"] == home_team].sort_values("GAME_DATE", ascending=False)
    away_rows = df[df["AWAY_TEAM"] == away_team].sort_values("GAME_DATE", ascending=False)
    if home_rows.empty or away_rows.empty:
        return None
    h = home_rows.iloc[0]
    a = away_rows.iloc[0]
    row = {
        "DIFF_PTS":          h["HOME_EMA_PTS"]         - a["AWAY_EMA_PTS"],
        "DIFF_FG_PCT":       h["HOME_EMA_FG_PCT"]       - a["AWAY_EMA_FG_PCT"],
        "DIFF_FG3_PCT":      h["HOME_EMA_FG3_PCT"]      - a["AWAY_EMA_FG3_PCT"],
        "DIFF_FT_PCT":       h["HOME_EMA_FT_PCT"]       - a["AWAY_EMA_FT_PCT"],
        "DIFF_OREB":         h["HOME_EMA_OREB"]         - a["AWAY_EMA_OREB"],
        "DIFF_DREB":         h["HOME_EMA_DREB"]         - a["AWAY_EMA_DREB"],
        "DIFF_AST":          h["HOME_EMA_AST"]          - a["AWAY_EMA_AST"],
        "DIFF_STL":          h["HOME_EMA_STL"]          - a["AWAY_EMA_STL"],
        "DIFF_BLK":          h["HOME_EMA_BLK"]          - a["AWAY_EMA_BLK"],
        "DIFF_TOV":          h["HOME_EMA_TOV"]          - a["AWAY_EMA_TOV"],
        "DIFF_WIN_PCT":      h["HOME_CURRENT_WIN_PCT"]  - a["AWAY_CURRENT_WIN_PCT"],
        "DIFF_WIN_STREAK":   h["HOME_WIN_STREAK"]       - a["AWAY_WIN_STREAK"],
        "DIFF_REST_DAYS":    h["HOME_REST_DAYS"]        - a["AWAY_REST_DAYS"],
        "DIFF_ELO":          h["HOME_ELO"]              - a["AWAY_ELO"],
        "DIFF_eFG_PCT":      h["HOME_EMA_eFG_PCT"]      - a["AWAY_EMA_eFG_PCT"],
        "DIFF_TO_RATIO":     h["HOME_EMA_TO_RATIO"]     - a["AWAY_EMA_TO_RATIO"],
        "DIFF_FT_RATE":      h["HOME_EMA_FT_RATE"]      - a["AWAY_EMA_FT_RATE"],
        "DIFF_OREB_PCT":     h["HOME_OREB_PCT"]         - a["AWAY_OREB_PCT"],
        "DIFF_PTS_ALLOWED":  h["HOME_EMA_PTS_ALLOWED"]  - a["AWAY_EMA_PTS_ALLOWED"],  # V3 mới
        "HOME_IS_B2B":       h["HOME_IS_B2B"],
        "AWAY_IS_B2B":       a["AWAY_IS_B2B"],
    }
    return pd.DataFrame([row])[FEATURES]

# ── Stacking predict ──────────────────────────────────────────────────────────
def predict_stacking(X):
    """
    Stacking ensemble: margin_model + calibrator + xgb_clf + lgb_clf → meta_model
    Trả về (p_home_win, pred_margin, pred_home_pts, pred_away_pts)
    """
    pred_margin = margin_model.predict(X)[0]
    pred_total  = total_model.predict(X)[0]

    p_score = calibrator.predict_proba([[pred_margin]])[0][1]
    p_clf   = xgb_clf.predict_proba(X)[0][1]
    p_lgb   = lgb_clf.predict_proba(X)[0][1]

    p_stacked = meta_model.predict_proba([[p_score, p_clf, p_lgb]])[0][1]

    pred_home_pts = (pred_total + pred_margin) / 2
    pred_away_pts = (pred_total - pred_margin) / 2

    return p_stacked, pred_margin, pred_home_pts, pred_away_pts

# ── Latest stats for display ──────────────────────────────────────────────────
def get_latest_stats(team, role):
    prefix = "HOME_" if role == "home" else "AWAY_"
    col    = "HOME_TEAM" if role == "home" else "AWAY_TEAM"
    rows = df[df[col] == team].sort_values("GAME_DATE", ascending=False)
    if rows.empty:
        return None, None
    latest = rows.iloc[0]
    stats = {
        "EMA_PTS":    latest[f"{prefix}EMA_PTS"],
        "EMA_FG_PCT": latest[f"{prefix}EMA_FG_PCT"],
        "EMA_FG3_PCT":latest[f"{prefix}EMA_FG3_PCT"],
        "WIN_PCT":    latest[f"{prefix}CURRENT_WIN_PCT"],
        "WIN_STREAK": latest[f"{prefix}WIN_STREAK"],
        "ELO":        latest[f"{prefix}ELO"],
        "REST_DAYS":  latest[f"{prefix}REST_DAYS"],
    }
    return stats, latest["GAME_DATE"].strftime("%d/%m/%Y")

# ── Fetch upcoming schedule ───────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_upcoming_schedule(days_ahead=7):
    """
    Lấy lịch thi đấu sắp tới từ multiple sources.
    Priority: balldontlie.io → ESPN API → manual fallback
    """
    games = []
    today = datetime.now()

    # Source 1: balldontlie.io (ưu tiên)
    try:
        start_date = today.strftime("%Y-%m-%d")
        end_date   = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = "https://api.balldontlie.io/v1/games"
        resp = requests.get(
            url, 
            params={"start_date": start_date, "end_date": end_date, "per_page": 100},
            headers={"Authorization": "0"}, 
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for g in data:
                h = g.get("home_team", {}).get("abbreviation", "").upper()
                a = g.get("visitor_team", {}).get("abbreviation", "").upper()
                d = g.get("date", "")[:10]
                game_time_str = g.get("date", "")[11:16] if len(g.get("date", "")) > 10 else "TBD"
                
                if h and a and d and h in TEAM_NAMES and a in TEAM_NAMES:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    status = g.get("status", "TBD")
                    
                    # Format thời gian
                    if status == "Final" or status == "Final/OT":
                        time_display = f"✓ {status}"
                    else:
                        time_display = game_time_str if game_time_str != "TBD" else "TBD"
                    
                    games.append({
                        "date": dt.strftime("%d/%m/%Y"),
                        "date_dt": dt,
                        "home": h,
                        "away": a,
                        "time": time_display,
                        "status": status
                    })
            
            if games:
                return sorted(games, key=lambda x: x["date_dt"])
    except requests.exceptions.RequestException:
        pass
    except Exception:
        pass

    # Source 2: ESPN API (fallback)
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        
        for delta in range(min(days_ahead, 14)):  # ESPN giới hạn 14 ngày
            try:
                date_str = (today + timedelta(days=delta)).strftime("%Y%m%d")
                resp = requests.get(
                    url, 
                    params={"dates": date_str},
                    headers={"User-Agent": "Mozilla/5.0"}, 
                    timeout=8
                )
                if resp.status_code != 200:
                    continue
                
                data = resp.json()
                for event in data.get("events", []):
                    comps = event.get("competitions", [{}])[0]
                    home_abbr, away_abbr = "", ""
                    
                    # Lấy thời gian từ event date và status
                    event_date_str = event.get("date", "")
                    if event_date_str:
                        try:
                            event_dt = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
                            game_time = event_dt.strftime("%H:%M")
                        except:
                            game_time = "TBD"
                    else:
                        game_time = "TBD"
                    
                    status_type = comps.get("status", {}).get("type", {})
                    status_detail = status_type.get("shortDetail", "TBD")
                    
                    # Parse team abbreviations
                    for c in comps.get("competitors", []):
                        abbr = c.get("team", {}).get("abbreviation", "").upper()
                        if c.get("homeAway") == "home":
                            home_abbr = abbr
                        else:
                            away_abbr = abbr
                    
                    # Validate teams exist in TEAM_NAMES
                    if home_abbr and away_abbr and home_abbr in TEAM_NAMES and away_abbr in TEAM_NAMES:
                        current_date_str = (today + timedelta(days=delta)).strftime("%d/%m/%Y")
                        
                        # Format time display
                        if "Final" in status_detail:
                            time_display = f"✓ {status_detail}"
                        else:
                            time_display = game_time if game_time != "TBD" else "TBD"
                        
                        games.append({
                            "date": current_date_str,
                            "date_dt": today + timedelta(days=delta),
                            "home": home_abbr,
                            "away": away_abbr,
                            "time": time_display,
                            "status": status_detail
                        })
            except (requests.exceptions.RequestException, ValueError):
                continue
        
        if games:
            return sorted(games, key=lambda x: x["date_dt"])
    except Exception:
        pass

    return games

# ── Session state ─────────────────────────────────────────────────────────────
if "quick_home" not in st.session_state:
    st.session_state["quick_home"] = None
if "quick_away" not in st.session_state:
    st.session_state["quick_away"] = None
if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = "matchups"

query_params = st.query_params
if "home" in query_params and "away" in query_params:
    st.session_state["quick_home"] = query_params["home"]
    st.session_state["quick_away"] = query_params["away"]

# ── Fixed Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <button class="header-btn" onclick="location.reload();">🔄</button>
    <div class="header-title">NBA PREDICTOR</div>
    <button class="header-btn">📅</button>
</div>
""", unsafe_allow_html=True)

# ── Main Content Container ────────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ── Team selectors ────────────────────────────────────────────────────────────
default_home_idx = 0
default_away_idx = 0

if st.session_state["quick_home"]:
    qh = st.session_state["quick_home"]
    key = f"{TEAM_NAMES.get(qh, qh)} ({qh})"
    if key in team_options:
        default_home_idx = team_options.index(key)

if st.session_state["quick_away"]:
    qa = st.session_state["quick_away"]
    key = f"{TEAM_NAMES.get(qa, qa)} ({qa})"
    if key in team_options:
        default_away_idx = team_options.index(key)

if default_home_idx == 0 and "LA Lakers (LAL)" in team_options:
    default_home_idx = team_options.index("LA Lakers (LAL)")
if default_away_idx == 0 and "Golden State Warriors (GSW)" in team_options:
    default_away_idx = team_options.index("Golden State Warriors (GSW)")

st.markdown('<p style="color:#888;font-size:0.8rem;text-align:center;margin-bottom:1rem;">Stacking Ensemble · XGBoost + LightGBM</p>', unsafe_allow_html=True)

st.markdown('<div class="team-card">', unsafe_allow_html=True)
st.markdown('<div class="teams-row">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown('<div class="team-selector">', unsafe_allow_html=True)
    st.markdown('<div class="team-label">🏠 Home</div>', unsafe_allow_html=True)
    home_sel = st.selectbox("Home", team_options, index=default_home_idx, label_visibility="collapsed", key="home_team")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div style="text-align:center;padding-top:1.5rem;"><div class="vs-label">VS</div></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="team-selector">', unsafe_allow_html=True)
    st.markdown('<div class="team-label">✈️ Away</div>', unsafe_allow_html=True)
    away_sel = st.selectbox("Away", team_options, index=default_away_idx, label_visibility="collapsed", key="away_team")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close teams-row
st.markdown('</div>', unsafe_allow_html=True)  # close team-card

home_abbr = abbr_map[home_sel]
away_abbr = abbr_map[away_sel]

# ── Logo display (Updated) ───────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-around;margin:1rem 0;">
    <div style="text-align:center;">
        <div class="team-logo">
            <img src="https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{}.png" alt="{}" onerror="this.style.display='none'">
        </div>
        <div class="team-abbr">{}</div>
    </div>
    <div style="text-align:center;color:#333;font-family:'Bebas Neue';font-size:1.8rem;letter-spacing:2px;">VS</div>
    <div style="text-align:center;">
        <div class="team-logo">
            <img src="https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{}.png" alt="{}" onerror="this.style.display='none'">
        </div>
        <div class="team-abbr">{}</div>
    </div>
</div>
""".format(home_abbr.lower(), home_abbr, home_abbr, away_abbr.lower(), away_abbr, away_abbr), unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict = st.button("🔥 DỰ ĐOÁN KẾT QUẢ", use_container_width=True, key="predict_btn")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict:
    if home_abbr == away_abbr:
        st.warning("Vui lòng chọn 2 đội khác nhau!")
    else:
        X = build_features(home_abbr, away_abbr)
        if X is None:
            st.error("Không đủ dữ liệu để dự đoán.")
        else:
            p_win, pred_margin, pred_home_pts, pred_away_pts = predict_stacking(X)
            home_prob  = p_win
            away_prob  = 1 - p_win
            winner     = home_abbr if home_prob >= 0.5 else away_abbr
            winner_name = TEAM_NAMES.get(winner, winner)
            winner_label = "🏠 Home Win" if home_prob >= 0.5 else "✈️ Away Win"

            home_stats, home_date = get_latest_stats(home_abbr, "home")
            away_stats, away_date = get_latest_stats(away_abbr, "away")

            # Result Box Header
            winner_emoji = "🏠" if home_prob >= 0.5 else "✈️"
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-header'>
                    <div class='result-badge'>
                        <span>{winner_emoji}</span>
                        <span class='result-badge-text'>{winner_label} — {winner_name}</span>
                    </div>
                </div>
                
                <div class='prob-section'>
                    <div class='prob-header'>
                        <div class='prob-item'>
                            <div class='prob-label'>Home Win Prob.</div>
                            <div class='prob-value'>{home_prob:.1%}</div>
                        </div>
                        <div class='prob-item'>
                            <div class='prob-label'>Away Win Prob.</div>
                            <div class='prob-value away'>{away_prob:.1%}</div>
                        </div>
                    </div>
                    <div class='prob-bar'>
                        <div class='prob-bar-fill' style='width: {home_prob*100:.1f}%;'></div>
                        <div class='prob-bar-fill away' style='width: {away_prob*100:.1f}%;'></div>
                    </div>
                </div>
                
                <div class='score-section'>
                    <div class='score-item'>
                        <div class='score-label'>Exp Pts</div>
                        <div class='score-value'>{pred_home_pts:.0f}</div>
                    </div>
                    <div class='divider-line'></div>
                    <div class='score-item'>
                        <div class='score-label'>Spread</div>
                        <div class='score-value spread'>{("+" if pred_margin > 0 else "") + f"{pred_margin:.1f}"}</div>
                    </div>
                    <div class='divider-line'></div>
                    <div class='score-item'>
                        <div class='score-label'>Exp Pts</div>
                        <div class='score-value'>{pred_away_pts:.0f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Team stats
            if home_stats and away_stats:
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#888;font-size:0.7rem;text-align:center;text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem;'>Recent Stats</p>",
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<p style='color:#F5A623;font-weight:600;margin-bottom:0.5rem;font-size:0.8rem;'>"
                                f"🏠 {TEAM_NAMES.get(home_abbr, 'HOME')}</p>",
                                unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='stat-grid'>
                        <div class='stat-item'><span class='stat-key'>PTS</span><span class='stat-val'>{home_stats['EMA_PTS']:.1f}</span></div>
                        <div class='stat-item'><span class='stat-key'>FG%</span><span class='stat-val'>{home_stats['EMA_FG_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>3P%</span><span class='stat-val'>{home_stats['EMA_FG3_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>WIN%</span><span class='stat-val'>{home_stats['WIN_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>ELO</span><span class='stat-val'>{home_stats['ELO']:.0f}</span></div>
                        <div class='stat-item'><span class='stat-key'>STREAK</span><span class='stat-val'>{int(home_stats['WIN_STREAK'])}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<p style='color:#5B8CFF;font-weight:600;margin-bottom:0.5rem;font-size:0.8rem;'>"
                                f"✈️ {TEAM_NAMES.get(away_abbr, 'AWAY')}</p>",
                                unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='stat-grid'>
                        <div class='stat-item'><span class='stat-key'>PTS</span><span class='stat-val'>{away_stats['EMA_PTS']:.1f}</span></div>
                        <div class='stat-item'><span class='stat-key'>FG%</span><span class='stat-val'>{away_stats['EMA_FG_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>3P%</span><span class='stat-val'>{away_stats['EMA_FG3_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>WIN%</span><span class='stat-val'>{away_stats['WIN_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>ELO</span><span class='stat-val'>{away_stats['ELO']:.0f}</span></div>
                        <div class='stat-item'><span class='stat-key'>STREAK</span><span class='stat-val'>{int(away_stats['WIN_STREAK'])}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

# ── Upcoming Schedule ─────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<div class='schedule-title'>📅 LỊCH THI ĐẤU SẮP TỚI</div>", unsafe_allow_html=True)

col_refresh, col_info = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Làm mới"):
        st.cache_data.clear()
        st.rerun()
with col_info:
    st.markdown("<p style='color:#555;font-size:0.8rem;margin-top:0.5rem;'>Cập nhật mỗi 30 phút</p>",
                unsafe_allow_html=True)

with st.spinner("⏳ Đang tải lịch thi đấu..."):
    upcoming = fetch_upcoming_schedule(days_ahead=7)

if not upcoming or len(upcoming) == 0:
    st.markdown("""
    <div class='no-games-msg'>
        ⚠️ Không tìm thấy lịch thi đấu<br>
        <span style='font-size:0.7rem;'>• Kiểm tra kết nối mạng<br>
        • API có thể quá tải - vui lòng thử lại sau</span>
    </div>
    """, unsafe_allow_html=True)
else:
    today_str    = datetime.now().strftime("%d/%m/%Y")
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")

    def date_label(d):
        if d == today_str:    return "🔴 HÔM NAY"
        if d == tomorrow_str: return "🟡 NGÀY MAI"
        date_obj = datetime.strptime(d, "%d/%m/%Y")
        weekday_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
        weekday = weekday_names[date_obj.weekday()]
        return f"📆 {weekday} - {d}"

    upcoming_sorted = sorted(upcoming, key=lambda x: x["date_dt"])
    current_date = None
    game_count = 0

    for game in upcoming_sorted:
        d = game.get("date", "")
        h = game.get("home", "").upper()
        a = game.get("away", "").upper()
        t = game.get("time", "TBD")
        
        # Validate teams before display
        if h not in TEAM_NAMES or a not in TEAM_NAMES:
            continue
        
        game_count += 1

        if d != current_date:
            current_date = d
            st.markdown(f"<div class='schedule-date-header'>{date_label(d)}</div>",
                        unsafe_allow_html=True)

        gcol1, gcol2 = st.columns([5, 1], gap="small")
        with gcol1:
            st.markdown(f"""
            <div class='game-card'>
                <div class='game-team'>
                    {logo_html(a, 32)}
                    <div>
                        <div class='team-name-sched'>{TEAM_NAMES.get(a, a)}</div>
                        <div class='team-abbr-sched'>{a} · AWAY</div>
                    </div>
                </div>
                <div class='vs-badge'>@</div>
                <div class='game-team'>
                    {logo_html(h, 32)}
                    <div>
                        <div class='team-name-sched'>{TEAM_NAMES.get(h, h)}</div>
                        <div class='team-abbr-sched'>{h} · HOME</div>
                    </div>
                </div>
                <div class='game-time'>{t}</div>
            </div>
            """, unsafe_allow_html=True)

        with gcol2:
            if st.button("⚡ DỰ ĐOÁN", key=f"sched_{h}_{a}_{d}_{game_count}"):
                h_key = f"{TEAM_NAMES.get(h, h)} ({h})"
                a_key = f"{TEAM_NAMES.get(a, a)} ({a})"
                if h_key in team_options and a_key in team_options:
                    st.session_state["quick_home"] = h
                    st.session_state["quick_away"] = a
                    st.rerun()

    if game_count == 0:
        st.markdown("""
        <div class='no-games-msg'>
            ❌ Không có trận đấu hợp lệ trong 7 ngày tới<br>
            <span style='font-size:0.7rem;'>Dữ liệu đội bóng có thể chưa được cập nhật</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:#333;font-size:0.72rem;text-align:center;margin-top:1rem;'>"
                    f"📊 Tổng {game_count} trận · Nguồn: balldontlie.io / ESPN API</p>",
                    unsafe_allow_html=True)

# ── Close main content ─────────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)

# ── Fixed Footer Navigation ────────────────────────────────────────────────────
st.markdown("""
<div class="footer-nav">
    <button class="footer-btn active" onclick="window.location.href='?tab=matchups'">
        <span style="font-size:1.3rem;">🏀</span>
        <span>Matchups</span>
    </button>
    <button class="footer-btn" onclick="window.location.href='?tab=analytics'">
        <span style="font-size:1.3rem;">📊</span>
        <span>Analytics</span>
    </button>
    <button class="footer-btn" onclick="window.location.href='?tab=history'">
        <span style="font-size:1.3rem;">📜</span>
        <span>History</span>
    </button>
    <button class="footer-btn" onclick="window.location.href='?tab=profile'">
        <span style="font-size:1.3rem;">👤</span>
        <span>Profile</span>
    </button>
</div>
""", unsafe_allow_html=True)
