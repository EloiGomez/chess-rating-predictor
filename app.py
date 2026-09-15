"""
app.py

Streamlit demo: predicts a Lichess player's Elo (bullet or blitz) from a
handful of their own games, either fetched live from the Lichess API by
username, or pasted/uploaded as a PGN file.

Reuses the exact same feature-extraction code as the training pipeline
(src/predict_from_pgn.py, which itself wraps analyze_with_stockfish.py and
extract_clock_features.py) and loads a model trained by train_baseline.py
--save-model-dir. Run with:

    streamlit run app.py
"""

import os
import sys

import chess.engine
import joblib
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from analyze_with_stockfish import STOCKFISH_PATH
from lichess_api import fetch_games_pgn, LichessApiError
from predict_from_pgn import (
    parse_games_for_player,
    analyze_player_games,
    aggregate_player_rows,
    build_model_input,
    MIN_GAMES_REQUIRED,
    MAX_GAMES_USED,
)

MODELS_DIR = "models"
# Must match the Stockfish depth the loaded models were TRAINED on --
# analyzing at a different depth would give the model ACPL/blunder
# numbers on a different scale than what it learned from.
ANALYSIS_DEPTH = 12

st.set_page_config(page_title="Chess Rating Predictor", page_icon="♞")


@st.cache_resource
def load_model(speed_category: str):
    path = os.path.join(MODELS_DIR, f"model_{speed_category}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def run_prediction(pgn_text: str, username: str, speed_category: str, max_games: int = MAX_GAMES_USED):
    games = parse_games_for_player(pgn_text, username)
    if not games:
        st.error(f"No games found for '{username}' in the provided PGN.")
        return

    bundle = load_model(speed_category)
    if bundle is None:
        st.error(
            f"No trained model found for '{speed_category}' yet. "
            f"Run: python src/train_baseline.py --speed-category {speed_category} --save-model-dir models"
        )
        return

    progress = st.progress(0.0, text="Starting Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        rows = []
        games_to_use = games[:max_games]
        for i, game in enumerate(games_to_use):
            progress.progress(
                (i + 1) / len(games_to_use),
                text=f"Analyzing game {i + 1}/{len(games_to_use)} with Stockfish (depth {ANALYSIS_DEPTH})...",
            )
            rows.extend(analyze_player_games([game], username, engine, ANALYSIS_DEPTH))
    finally:
        engine.quit()
        progress.empty()

    try:
        features = aggregate_player_rows(rows, speed_category)
    except ValueError as error:
        st.error(
            f"{error} Try fetching/uploading more {speed_category} games "
            f"(need at least {MIN_GAMES_REQUIRED})."
        )
        return

    X = build_model_input(features, bundle["feature_columns"], bundle["top_eco"])
    predicted_elo = bundle["model"].predict(X)[0]

    st.subheader(f"Predicted {speed_category} Elo: {predicted_elo:.0f}")
    st.caption(
        f"Based on {features['n_games']} game(s). Model's typical error on held-out players: "
        f"±{bundle['test_mae']:.0f} points (R²={bundle['test_r2']:.2f}, "
        f"trained on {bundle['n_players_trained_on']:,} players)."
    )

    if features["reported_elo"] is not None:
        st.metric(
            "Your actual Lichess rating (avg. over these games)",
            f"{features['reported_elo']:.0f}",
            delta=f"{predicted_elo - features['reported_elo']:+.0f} predicted - actual",
        )

    with st.expander("Features used for this prediction"):
        st.json({k: (round(v, 2) if isinstance(v, float) else v) for k, v in features.items()})


st.title("♞ Chess Rating Predictor")
st.write(
    "Estimates a Lichess player's rating from a handful of their own games, using Stockfish-derived "
    "move-quality features (not just the outcome of the games)."
)

speed_category = st.selectbox("Time control", ["bullet", "blitz"])

tab_fetch, tab_upload = st.tabs(["Fetch from Lichess", "Upload / paste PGN"])

with tab_fetch:
    username = st.text_input("Lichess username", key="fetch_username")
    max_games = st.slider("Games to fetch", MIN_GAMES_REQUIRED, MAX_GAMES_USED, 10, key="fetch_max")
    if st.button("Fetch and predict", key="fetch_button"):
        if not username:
            st.warning("Enter a username first.")
        else:
            try:
                with st.spinner(f"Fetching {username}'s recent {speed_category} games from Lichess..."):
                    pgn_text = fetch_games_pgn(username, speed_category, max_games)
                if not pgn_text.strip():
                    st.error(f"'{username}' has no rated {speed_category} games on Lichess.")
                else:
                    run_prediction(pgn_text, username, speed_category, max_games)
            except LichessApiError as error:
                st.error(str(error))

with tab_upload:
    st.caption(
        f"Export your games from lichess.org/@/{'{username}'}/download (tick \"Include clocks\") and "
        f"upload the .pgn here, or paste it directly. Needs at least {MIN_GAMES_REQUIRED} of your own "
        f"{speed_category} games in the same file."
    )
    upload_username = st.text_input("Your username (as it appears in the PGN's White/Black tags)", key="upload_username")
    uploaded_file = st.file_uploader("PGN file", type=["pgn"], key="pgn_file")
    pasted_pgn = st.text_area("...or paste PGN text here", height=150, key="pgn_paste")
    if st.button("Predict", key="upload_button"):
        pgn_text = uploaded_file.read().decode("utf-8", errors="replace") if uploaded_file else pasted_pgn
        if not upload_username:
            st.warning("Enter your username first.")
        elif not pgn_text.strip():
            st.warning("Upload a file or paste some PGN text first.")
        else:
            run_prediction(pgn_text, upload_username, speed_category)
