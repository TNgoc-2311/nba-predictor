import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime, timedelta, timezone

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #0b0f1a;
    --surface: #111827;
    --surface-2: #0f172a;
    --border: #1f2a44;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #3b82f6;
    --accent-2: #38bdf8;
    --success: #22c55e;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background: radial-gradient(1200px 500px at 15% -10%, rgba(59,130,246,0.18) 0%, rgba(59,130,246,0) 55%),
                radial-gradient(900px 400px at 85% -20%, rgba(56,189,248,0.16) 0%, rgba(56,189,248,0) 60%),
                var(--bg);
}

.page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem 3rem 1.5rem;
}

.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    box-shadow: 0 20px 60px rgba(16, 24, 40, 0.08);
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-sub {
    color: var(--muted);
    margin-top: 0.35rem;
    font-size: 0.98rem;
}

.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #dbeafe;
    background: rgba(59, 130, 246, 0.15);
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.section {
    margin-top: 2rem;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    margin: 0 0 1rem 0;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 16px 32px rgba(16, 24, 40, 0.06);
}

.stSelectbox label, .stMarkdown p {
    color: var(--muted);
    font-size: 0.9rem;
}

div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    min-height: 44px;
}

.btn-primary > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.25);
}

.btn-primary > button:hover { transform: translateY(-1px); }

.btn-secondary > button {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.9rem !important;
    font-size: 0.85rem !important;
    width: 100% !important;
}

.result-box {
    background: var(--surface);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    border: 1px solid var(--border);
    text-align: center;
}

.result-winner {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    color: var(--text);
    letter-spacing: -0.01em;
}

.result-sub { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }

.prob-row {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    margin-top: 1.5rem;
    gap: 1rem;
}

.prob-card {
    background: var(--surface-2);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid var(--border);
}

.prob-pct { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; margin-bottom: 0.2rem; }
.prob-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

.score-box {
    background: var(--surface);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
    border: 1px solid var(--border);
    text-align: center;
}

.score-title { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
.score-val { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; color: var(--text); }
.score-sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 1rem;
}

.stat-item {
    background: var(--surface-2);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    border: 1px solid var(--border);
}

.stat-key { color: var(--muted); }
.stat-val { color: var(--text); font-weight: 600; }

.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
.last-game-note { font-size: 0.75rem; color: var(--muted); text-align: center; margin-top: 0.5rem; }

.schedule-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    color: var(--text);
    margin-bottom: 0.6rem;
}

.schedule-date-header {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.6rem;
    margin-top: 1rem;
}

.game-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
}

.game-team { display: flex; align-items: center; gap: 0.6rem; min-width: 160px; }
.team-name-sched { font-size: 0.86rem; font-weight: 600; color: var(--text); }
.team-abbr-sched { font-size: 0.7rem; color: var(--muted); }
.vs-badge { font-family: 'Space Grotesk', sans-serif; color: #94a3b8; font-size: 0.9rem; letter-spacing: 1px; padding: 0 0.5rem; }
.game-time { font-size: 0.75rem; color: var(--muted); text-align: right; min-width: 80px; }

.no-games-msg {
    color: var(--muted);
    font-size: 0.9rem;
    text-align: center;
    padding: 2rem;
    border: 1px dashed var(--border);
    border-radius: 12px;
    background: var(--surface);
}

.skeleton {
    background: linear-gradient(90deg, #0f172a 25%, #172036 50%, #0f172a 75%);
    background-size: 200% 100%;
    animation: shimmer 1.6s infinite;
    border-radius: 12px;
}

.skeleton-line {
    height: 14px;
    margin-bottom: 0.6rem;
}

.skeleton-card {
    height: 72px;
    margin-bottom: 0.6rem;
}

.insight-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
}

.insight-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
}

.insight-title { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
.insight-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; margin-top: 0.4rem; }
.insight-sub { color: var(--muted); font-size: 0.75rem; margin-top: 0.2rem; }

.history-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.history-item {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

.history-title { font-weight: 600; font-size: 0.9rem; }
.history-sub { color: var(--muted); font-size: 0.75rem; margin-top: 0.2rem; }

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

@media (max-width: 900px) {
    .hero { flex-direction: column; align-items: flex-start; }
    .prob-row { grid-template-columns: 1fr; }
    .insight-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .history-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 640px) {
    .page { padding: 1.5rem 1rem 2rem 1rem; }
    .hero-title { font-size: 2rem; }
    .game-card { flex-direction: column; align-items: flex-start; }
    .game-time { text-align: left; }
    .insight-grid { grid-template-columns: 1fr; }
    .history-grid { grid-template-columns: 1fr; }
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
ESPN_ABBR_MAP = {
    "SA": "SAS", "GS": "GSW", "NY": "NYK",
    "NO": "NOP", "UTH": "UTA",
}

VN_TZ = timezone(timedelta(hours=7))

@st.cache_data(ttl=3600)
def fetch_upcoming_schedule(days_ahead=7):
    games = []
    today = datetime.now(VN_TZ).replace(tzinfo=None)

    # Source 1: balldontlie.io
    try:
        start_date = today.strftime("%Y-%m-%d")
        end_date   = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = "https://api.balldontlie.io/v1/games"
        resp = requests.get(url, params={"start_date": start_date, "end_date": end_date, "per_page": 100},
                            headers={"Authorization": "0"}, timeout=10)
        if resp.status_code == 200:
            for g in resp.json().get("data", []):
                h = g.get("home_team", {}).get("abbreviation", "")
                a = g.get("visitor_team", {}).get("abbreviation", "")
                d = g.get("date", "")[:10]
                if h and a and d:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    status = g.get("status", "TBD")
                    games.append({"date": dt.strftime("%d/%m/%Y"), "date_dt": dt,
                                  "home": h, "away": a, "time": status})
            if games:
                return games
    except Exception:
        pass

    # Source 2: ESPN API
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        for delta in range(days_ahead):
            date_str = (today + timedelta(days=delta)).strftime("%Y%m%d")
            resp = requests.get(url, params={"dates": date_str},
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            if resp.status_code != 200:
                continue
            for event in resp.json().get("events", []):
                comps = event.get("competitions", [{}])[0]
                home_abbr, away_abbr = "", ""
                game_time = comps.get("status", {}).get("type", {}).get("shortDetail", "TBD")
                for c in comps.get("competitors", []):
                    abbr = c.get("team", {}).get("abbreviation", "")
                    if c.get("homeAway") == "home":
                        home_abbr = abbr
                    else:
                        away_abbr = abbr
                if home_abbr and away_abbr:
                    games.append({"date": (today + timedelta(days=delta)).strftime("%d/%m/%Y"),
                                  "date_dt": today + timedelta(days=delta),
                                  "home": home_abbr, "away": away_abbr, "time": game_time})
        return games
    except Exception:
        return []

# ── Session state ─────────────────────────────────────────────────────────────
if "quick_home" not in st.session_state:
    st.session_state["quick_home"] = None
if "quick_away" not in st.session_state:
    st.session_state["quick_away"] = None
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

query_params = st.query_params
if "home" in query_params and "away" in query_params:
    st.session_state["quick_home"] = query_params["home"]
    st.session_state["quick_away"] = query_params["away"]

st.markdown("<div class='page'>", unsafe_allow_html=True)

# ── UI Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div>
        <div class='hero-chip'>NBA Predictor · V3</div>
        <h1 class='hero-title'>NBA Prediction Studio</h1>
        <p class='hero-sub'>Win probability and score projection powered by a stacking ensemble (XGBoost + LightGBM + Platt Calibration).</p>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:0.9rem;color:#475467;'>Hourly refresh</div>
        <div style='font-size:0.75rem;color:#98a2b3;margin-top:0.2rem;'>ESPN + balldontlie.io</div>
    </div>
</div>
""", unsafe_allow_html=True)

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

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Matchup · Chọn trận</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Đội Nhà (Home)**")
    home_sel = st.selectbox("Home", team_options, index=default_home_idx, label_visibility="collapsed")
with col2:
    st.markdown("**Đội Khách (Away)**")
    away_sel = st.selectbox("Away", team_options, index=default_away_idx, label_visibility="collapsed")

home_abbr = abbr_map[home_sel]
away_abbr = abbr_map[away_sel]

# Logo display
lc1, lc2, lc3 = st.columns([2, 1, 2])
with lc1:
    st.markdown(f"<div style='text-align:center;padding:0.5rem 0;'>{logo_html(home_abbr, 56)}</div>",
                unsafe_allow_html=True)
with lc2:
    st.markdown("<div style='text-align:center;padding-top:1rem;color:#98a2b3;font-family:Space Grotesk;font-size:1rem;'>VS</div>",
                unsafe_allow_html=True)
with lc3:
    st.markdown(f"<div style='text-align:center;padding:0.5rem 0;'>{logo_html(away_abbr, 56)}</div>",
                unsafe_allow_html=True)

st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
predict = st.button("Run Prediction")
st.markdown("</div></div></div>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict:
    if home_abbr == away_abbr:
        st.warning("Please pick two different teams (Vui lòng chọn 2 đội khác nhau).")
    else:
        X = build_features(home_abbr, away_abbr)
        if X is None:
            st.error("Not enough data to run prediction (Không đủ dữ liệu để dự đoán).")
        else:
            p_win, pred_margin, pred_home_pts, pred_away_pts = predict_stacking(X)
            home_prob  = p_win
            away_prob  = 1 - p_win
            winner     = home_abbr if home_prob >= 0.5 else away_abbr
            winner_name = TEAM_NAMES.get(winner, winner)
            winner_label = " Home Win" if home_prob >= 0.5 else "Away Win"

            home_stats, home_date = get_latest_stats(home_abbr, "home")
            away_stats, away_date = get_latest_stats(away_abbr, "away")

            # Win probability result
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-sub'>Win Probability · Xác suất thắng</div>
                <div class='result-winner'>{winner_label} — {winner_name}</div>
                <div class='prob-row'>
                    <div class='prob-card'>
                        <div style='margin-bottom:0.4rem;'>{logo_html(home_abbr, 48)}</div>
                        <div class='prob-pct' style='color:var(--accent)'>{home_prob:.1%}</div>
                        <div class='prob-label'>{TEAM_NAMES.get(home_abbr, home_abbr)}</div>
                        <div class='prob-label' style='font-size:0.65rem'>HOME</div>
                    </div>
                    <div style='color:#98a2b3;font-size:1.1rem;font-weight:600;'>VS</div>
                    <div class='prob-card'>
                        <div style='margin-bottom:0.4rem;'>{logo_html(away_abbr, 48)}</div>
                        <div class='prob-pct' style='color:var(--accent-2)'>{away_prob:.1%}</div>
                        <div class='prob-label'>{TEAM_NAMES.get(away_abbr, away_abbr)}</div>
                        <div class='prob-label' style='font-size:0.65rem'>AWAY</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Score prediction
            sc1, sc2, sc3 = st.columns([2, 1, 2])
            with sc1:
                st.markdown(f"""
                <div class='score-box'>
                    <div class='score-title'>{logo_html(home_abbr, 20)} Home</div>
                    <div class='score-val'>{pred_home_pts:.0f}</div>
                    <div class='score-sub'>pts</div>
                </div>""", unsafe_allow_html=True)
            with sc2:
                margin_label = f"+{pred_margin:.1f}" if pred_margin > 0 else f"{pred_margin:.1f}"
                st.markdown(f"""
                <div class='score-box' style='padding-top:1.5rem;'>
                    <div class='score-title'>Margin · Chênh lệch</div>
                    <div class='score-val' style='font-size:1.4rem;'>{margin_label}</div>
                </div>""", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class='score-box'>
                    <div class='score-title'>{logo_html(away_abbr, 20)} Away</div>
                    <div class='score-val'>{pred_away_pts:.0f}</div>
                    <div class='score-sub'>pts</div>
                </div>""", unsafe_allow_html=True)

            # Team stats
            if home_stats and away_stats:
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#667085;font-size:0.8rem;text-align:center;'>Latest Form · Chỉ số gần nhất</p>",
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<p style='color:var(--accent);font-weight:600;margin-bottom:0.3rem;'>"
                                f"{logo_html(home_abbr, 20)} {TEAM_NAMES.get(home_abbr)} 🏠</p>",
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
                    <p class='last-game-note'>Last game: {home_date}</p>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<p style='color:var(--accent-2);font-weight:600;margin-bottom:0.3rem;'>"
                                f"{logo_html(away_abbr, 20)} {TEAM_NAMES.get(away_abbr)} ✈️</p>",
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
                    <p class='last-game-note'>Last game: {away_date}</p>
                    """, unsafe_allow_html=True)

            st.session_state["prediction_history"].insert(0, {
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "home": home_abbr,
                "away": away_abbr,
                "winner": winner,
                "home_prob": home_prob,
                "away_prob": away_prob,
                "margin": pred_margin,
            })

# ── Team Insights ─────────────────────────────────────────────────────────────
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Team Insights · Tổng quan đội</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

home_stats_latest, _ = get_latest_stats(home_abbr, "home")
away_stats_latest, _ = get_latest_stats(away_abbr, "away")

if home_stats_latest and away_stats_latest:
    st.markdown("""
    <div class='insight-grid'>
        <div class='insight-card'>
            <div class='insight-title'>Home ELO</div>
            <div class='insight-value'>{home_elo:.0f}</div>
            <div class='insight-sub'>{home_name}</div>
        </div>
        <div class='insight-card'>
            <div class='insight-title'>Away ELO</div>
            <div class='insight-value'>{away_elo:.0f}</div>
            <div class='insight-sub'>{away_name}</div>
        </div>
        <div class='insight-card'>
            <div class='insight-title'>Win % (Home vs Away)</div>
            <div class='insight-value'>{home_win:.1%} · {away_win:.1%}</div>
            <div class='insight-sub'>Season form</div>
        </div>
        <div class='insight-card'>
            <div class='insight-title'>EMA Points (Home vs Away)</div>
            <div class='insight-value'>{home_pts:.1f} · {away_pts:.1f}</div>
            <div class='insight-sub'>Offensive trend</div>
        </div>
    </div>
    """.format(
        home_elo=home_stats_latest["ELO"],
        away_elo=away_stats_latest["ELO"],
        home_win=home_stats_latest["WIN_PCT"],
        away_win=away_stats_latest["WIN_PCT"],
        home_pts=home_stats_latest["EMA_PTS"],
        away_pts=away_stats_latest["EMA_PTS"],
        home_name=TEAM_NAMES.get(home_abbr, home_abbr),
        away_name=TEAM_NAMES.get(away_abbr, away_abbr),
    ), unsafe_allow_html=True)
else:
    st.markdown("<p style='color:#98a2b3;'>No insight data available for the selected teams.</p>",
                unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ── Prediction History ────────────────────────────────────────────────────────
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Prediction History · Lịch sử dự đoán</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

history_items = st.session_state["prediction_history"][:6]
if not history_items:
    history_items = [
        {
            "home": "Dallas Mavericks",
            "away": "Denver Nuggets",
            "winner": "Denver Nuggets",
            "margin": -8.2,
            "timestamp": "12/05/2026 21:31",
        },
        {
            "home": "Atlanta Hawks",
            "away": "Golden State Warriors",
            "winner": "Atlanta Hawks",
            "margin": 5.9,
            "timestamp": "12/05/2026 21:31",
        },
        {
            "home": "Dallas Mavericks",
            "away": "Golden State Warriors",
            "winner": "Golden State Warriors",
            "margin": -0.7,
            "timestamp": "12/05/2026 21:30",
        },
    ]

def format_team_name(value):
    return TEAM_NAMES.get(value, value)

history_cards = []
for item in history_items:
    winner_name = format_team_name(item["winner"])
    home_name = format_team_name(item["home"])
    away_name = format_team_name(item["away"])
    history_cards.append(
        f"<div class='history-item'>"
        f"<div class='history-title'>{home_name} vs {away_name}</div>"
        f"<div class='history-sub'>Winner: {winner_name} · Margin {item['margin']:.1f}</div>"
        f"<div class='history-sub'>{item['timestamp']}</div>"
        "</div>"
    )

history_html = "<div class='history-grid'>" + "".join(history_cards) + "</div>"
st.markdown(history_html, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ── Upcoming Schedule ─────────────────────────────────────────────────────────
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='schedule-title'>Upcoming Schedule · Lịch thi đấu</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

placeholder = st.empty()
placeholder.markdown("""
<div class='skeleton skeleton-line'></div>
<div class='skeleton skeleton-card'></div>
<div class='skeleton skeleton-card'></div>
<div class='skeleton skeleton-card'></div>
""", unsafe_allow_html=True)

upcoming = fetch_upcoming_schedule(days_ahead=7)
placeholder.empty()

if not upcoming:
    st.markdown("""
    <div class='no-games-msg'>
        Không tìm thấy lịch thi đấu.<br>
        <span style='font-size:0.7rem;'>Hãy kiểm tra kết nối mạng.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    today_str    = datetime.now().strftime("%d/%m/%Y")
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")

    def date_label(d):
        if d == today_str:    return "Today · Hôm nay"
        if d == tomorrow_str: return "Tomorrow · Ngày mai"
        return f"{d}"

    upcoming_sorted = sorted(upcoming, key=lambda x: x["date_dt"])
    current_date = None

    for game in upcoming_sorted:
        d = game["date"]
        h = ESPN_ABBR_MAP.get(game["home"], game["home"])  
        a = ESPN_ABBR_MAP.get(game["away"], game["away"])  
        t = game["time"]

        if d != current_date:
            current_date = d
            st.markdown(f"<div class='schedule-date-header'>{date_label(d)}</div>",
                        unsafe_allow_html=True)

        gcol1, gcol2 = st.columns([5, 1])
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
            st.markdown("<div class='btn-secondary'>", unsafe_allow_html=True)
            if st.button("Quick Predict", key=f"sched_{h}_{a}_{d}"):
                h_fixed = ESPN_ABBR_MAP.get(h, h)
                a_fixed = ESPN_ABBR_MAP.get(a, a)
                h_key = f"{TEAM_NAMES.get(h_fixed, h_fixed)} ({h_fixed})"
                a_key = f"{TEAM_NAMES.get(a_fixed, a_fixed)} ({a_fixed})"
                if h_key in team_options and a_key in team_options:
                    st.session_state["quick_home"] = h_fixed
                    st.session_state["quick_away"] = a_fixed
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<p style='color:#98a2b3;font-size:0.72rem;text-align:right;margin-top:1rem;'>Source: ESPN API · Hourly refresh</p>",
                unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
