import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime, timedelta, timezone

# Định nghĩa múi giờ VN (UTC+7)
VN_TZ = timezone(timedelta(hours=7))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem !important;
    letter-spacing: 3px;
    color: #F5A623;
}

.stSelectbox label, .stMarkdown p {
    color: #aaaaaa;
    font-size: 0.85rem;
}

div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
}

.predict-btn > button {
    background: linear-gradient(135deg, #F5A623, #e8890a) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.3rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.predict-btn > button:hover { opacity: 0.85 !important; }

.result-box {
    background: #1a1a1a;
    border-radius: 14px;
    padding: 2rem;
    margin-top: 1.5rem;
    border: 1px solid #2a2a2a;
    text-align: center;
}

.result-winner {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: #F5A623;
    letter-spacing: 2px;
}

.result-sub { color: #888; font-size: 0.85rem; margin-top: 0.3rem; }

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.5rem;
    gap: 1rem;
}

.prob-card {
    flex: 1;
    background: #111;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #2a2a2a;
}

.prob-pct { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; margin-bottom: 0.2rem; }
.prob-label { font-size: 0.75rem; color: #777; text-transform: uppercase; letter-spacing: 1px; }

.score-box {
    background: #111;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
    border: 1px solid #2a2a2a;
    text-align: center;
}

.score-title { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
.score-val { font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; color: #f0f0f0; }
.score-sub { font-size: 0.75rem; color: #555; margin-top: 0.2rem; }

.margin-badge {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 0.8rem;
}

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 1rem;
}

.stat-item {
    background: #111;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    border: 1px solid #1e1e1e;
}

.stat-key { color: #666; }
.stat-val { color: #f0f0f0; font-weight: 600; }

.divider { border: none; border-top: 1px solid #222; margin: 1.5rem 0; }
.last-game-note { font-size: 0.75rem; color: #555; text-align: center; margin-top: 0.5rem; }

.schedule-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: #F5A623;
    letter-spacing: 2px;
    margin-bottom: 1rem;
}

.schedule-date-header {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

.game-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.game-team { display: flex; align-items: center; gap: 0.6rem; min-width: 160px; }
.team-name-sched { font-size: 0.82rem; font-weight: 600; color: #e0e0e0; }
.team-abbr-sched { font-size: 0.68rem; color: #555; }
.vs-badge { font-family: 'Bebas Neue', sans-serif; color: #333; font-size: 1rem; letter-spacing: 1px; padding: 0 0.5rem; }
.game-time { font-size: 0.75rem; color: #555; text-align: right; min-width: 60px; }

.no-games-msg {
    color: #444;
    font-size: 0.82rem;
    text-align: center;
    padding: 2rem;
    border: 1px dashed #222;
    border-radius: 10px;
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
            date_str = (datetime.now(VN_TZ).replace(tzinfo=None) + timedelta(days=delta)).strftime("%Y%m%d")
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
if "auto_predict" not in st.session_state:
    st.session_state["auto_predict"] = False

query_params = st.query_params
if "home" in query_params and "away" in query_params:
    st.session_state["quick_home"] = query_params["home"]
    st.session_state["quick_away"] = query_params["away"]

# ── UI Header ─────────────────────────────────────────────────────────────────
st.markdown("<h1>🏀 NBA PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;margin-top:-1rem;margin-bottom:2rem;'>Stacking Ensemble · XGBoost + LightGBM + Platt Calibration · V3</p>",
            unsafe_allow_html=True)

# ── Team selectors ────────────────────────────────────────────────────────────
if "home_sel" not in st.session_state:
    st.session_state["home_sel"] = "LA Lakers (LAL)" if "LA Lakers (LAL)" in team_options else team_options[0]
if "away_sel" not in st.session_state:
    st.session_state["away_sel"] = "Golden State Warriors (GSW)" if "Golden State Warriors (GSW)" in team_options else team_options[1]

if st.session_state["quick_home"]:
    qh = st.session_state["quick_home"]
    key = f"{TEAM_NAMES.get(qh, qh)} ({qh})"
    if key in team_options:
        st.session_state["home_sel"] = key

if st.session_state["quick_away"]:
    qa = st.session_state["quick_away"]
    key = f"{TEAM_NAMES.get(qa, qa)} ({qa})"
    if key in team_options:
        st.session_state["away_sel"] = key

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🏠 Đội Nhà (Home)**")
    home_sel = st.selectbox("Home", team_options,
                            key="home_sel",
                            label_visibility="collapsed")
with col2:
    st.markdown("**✈️ Đội Khách (Away)**")
    away_sel = st.selectbox("Away", team_options,
                            key="away_sel",
                            label_visibility="collapsed")

home_abbr = abbr_map[home_sel]
away_abbr = abbr_map[away_sel]

# Logo display
lc1, lc2, lc3 = st.columns([2, 1, 2])
with lc1:
    st.markdown(f"<div style='text-align:center;padding:0.5rem 0;'>{logo_html(home_abbr, 60)}</div>",
                unsafe_allow_html=True)
with lc2:
    st.markdown("<div style='text-align:center;padding-top:1rem;color:#333;font-family:Bebas Neue;font-size:1.2rem;'>VS</div>",
                unsafe_allow_html=True)
with lc3:
    st.markdown(f"<div style='text-align:center;padding:0.5rem 0;'>{logo_html(away_abbr, 60)}</div>",
                unsafe_allow_html=True)

st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
predict = st.button("DỰ ĐOÁN KẾT QUẢ") or st.session_state["auto_predict"]
st.markdown("</div>", unsafe_allow_html=True)
st.session_state["auto_predict"] = False

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

            # Win probability result
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-sub'>Dự đoán thắng</div>
                <div class='result-winner'>{winner_label} — {winner_name}</div>
                <div class='prob-row'>
                    <div class='prob-card'>
                        <div style='margin-bottom:0.4rem;'>{logo_html(home_abbr, 48)}</div>
                        <div class='prob-pct' style='color:#F5A623'>{home_prob:.1%}</div>
                        <div class='prob-label'>{TEAM_NAMES.get(home_abbr, home_abbr)}</div>
                        <div class='prob-label' style='font-size:0.65rem'>HOME</div>
                    </div>
                    <div style='color:#444;font-size:1.5rem;font-weight:bold;'>VS</div>
                    <div class='prob-card'>
                        <div style='margin-bottom:0.4rem;'>{logo_html(away_abbr, 48)}</div>
                        <div class='prob-pct' style='color:#5B8CFF'>{away_prob:.1%}</div>
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
                    <div class='score-title'>Margin</div>
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
                st.markdown("<p style='color:#555;font-size:0.8rem;text-align:center;'>CHỈ SỐ GẦN NHẤT</p>",
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<p style='color:#F5A623;font-weight:600;margin-bottom:0.3rem;'>"
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
                    <p class='last-game-note'>Trận gần nhất: {home_date}</p>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<p style='color:#5B8CFF;font-weight:600;margin-bottom:0.3rem;'>"
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
                    <p class='last-game-note'>Trận gần nhất: {away_date}</p>
                    """, unsafe_allow_html=True)

# ── Upcoming Schedule ─────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<div class='schedule-title'>📅 LỊCH THI ĐẤU SẮP TỚI</div>", unsafe_allow_html=True)

with st.spinner("Đang tải lịch thi đấu..."):
    upcoming = fetch_upcoming_schedule(days_ahead=7)

if not upcoming:
    st.markdown("""
    <div class='no-games-msg'>
        Không tìm thấy lịch thi đấu.<br>
        <span style='font-size:0.7rem;'>Hãy kiểm tra kết nối mạng.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    today_str    = datetime.now(VN_TZ).strftime("%d/%m/%Y")
    tomorrow_str = (datetime.now(VN_TZ) + timedelta(days=1)).strftime("%d/%m/%Y")

    def date_label(d):
        if d == today_str:    return "🔴 HÔM NAY"
        if d == tomorrow_str: return "🟡 NGÀY MAI"
        return f"📆 {d}"

    upcoming_sorted = sorted(upcoming, key=lambda x: x["date_dt"])
    current_date = None

    for game in upcoming_sorted:
        d, h, a, t = game["date"], game["home"], game["away"], game["time"]

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
            if st.button("⚡ DỰ ĐOÁN", key=f"sched_{h}_{a}_{d}"):
                h_key = f"{TEAM_NAMES.get(h, h)} ({h})"
                a_key = f"{TEAM_NAMES.get(a, a)} ({a})"
                if h_key in team_options and a_key in team_options:
                    st.session_state["quick_home"] = h
                    st.session_state["quick_away"] = a
                    st.session_state["auto_predict"] = True 
                    st.rerun()

    st.markdown("<p style='color:#333;font-size:0.72rem;text-align:right;margin-top:1rem;'>Nguồn: ESPN API · Cập nhật mỗi giờ</p>",
                unsafe_allow_html=True)
