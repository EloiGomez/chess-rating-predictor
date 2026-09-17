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

import glob
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
# How many games the "Games to fetch" slider allows, above
# predict_from_pgn.MAX_GAMES_USED (15 -- the cap training data used).
# Averaging over more games than that only reduces noise further (see
# aggregate_player_rows's docstring), it's just slower here since app.py
# analyzes games one at a time with a single Stockfish instance rather
# than the training pipeline's parallel workers.
MAX_GAMES_FETCHABLE = 30

# Display names for each model type train_baseline.py --save-model-dir
# saves (one file per type: model_{category}_{slug}.joblib), so the
# selectbox can show a readable label without having to load every model
# file just to read its "model_name" field.
MODEL_LABELS = {
    "ridge": "Ridge (regularized linear)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting (usually strongest)",
}

st.set_page_config(page_title="Chess Rating Predictor", page_icon="♞")


def available_model_slugs(speed_category: str) -> list:
    """Which per-model-type files exist for this speed category (e.g.
    "ridge", "gradient_boosting"), so the UI can offer a choice. Empty on
    an older models/ directory that only has the single model_{category}.joblib
    file from before train_baseline.py started saving one file per type."""
    prefix = f"model_{speed_category}_"
    pattern = os.path.join(MODELS_DIR, f"{prefix}*.joblib")
    return sorted(
        os.path.splitext(os.path.basename(path))[0][len(prefix):]
        for path in glob.glob(pattern)
    )


@st.cache_resource
def load_model(speed_category: str, model_slug: str = None):
    filename = f"model_{speed_category}.joblib" if model_slug is None else f"model_{speed_category}_{model_slug}.joblib"
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def run_prediction(pgn_text: str, username: str, speed_category: str, model_slug: str = None,
                    max_games: int = MAX_GAMES_USED):
    games = parse_games_for_player(pgn_text, username)
    if not games:
        st.error(f"No games found for '{username}' in the provided PGN.")
        return

    bundle = load_model(speed_category, model_slug)
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

    # Only restrict to the player's single most common EXACT time control
    # if the loaded model was actually trained with that restriction
    # (see aggregate_by_player.py's restrict_to_main_time_control) --
    # otherwise this would compute averages a different way than whatever
    # model happens to be deployed was trained with, silently skewing
    # predictions.
    restrict_exact_time_control = "base_time" in bundle["feature_columns"]
    try:
        features = aggregate_player_rows(rows, speed_category, restrict_exact_time_control, max_games=max_games)
    except ValueError as error:
        st.error(
            f"{error} Try fetching/uploading more {speed_category} games "
            f"(need at least {MIN_GAMES_REQUIRED})."
        )
        return

    X = build_model_input(features, bundle["feature_columns"], bundle["top_eco"])
    predicted_elo = bundle["model"].predict(X)[0]
    mae = bundle["test_mae"]

    st.subheader(f"Predicted {speed_category} Elo: {predicted_elo:.0f}")
    st.write(
        f"Likely somewhere around **{predicted_elo - mae:.0f} – {predicted_elo + mae:.0f}** "
        f"(±{mae:.0f}, this model's typical error on players it wasn't trained on -- "
        "not a strict statistical interval, just a ballpark)."
    )
    st.caption(
        f"Based on {features['n_games']} game(s). Model: {bundle['model_name']} "
        f"(R²={bundle['test_r2']:.2f}, trained on {bundle['n_players_trained_on']:,} players)."
    )

    if features["reported_elo"] is not None:
        st.metric(
            "Your actual Lichess rating (avg. over these games)",
            f"{features['reported_elo']:.0f}",
            delta=f"{predicted_elo - features['reported_elo']:+.0f} predicted - actual",
        )

    # Not every feature this script CALCULATES is necessarily used by the
    # loaded model -- e.g. the game-phase and exact-time-control features
    # are always computed here, but only feed into predictions once a
    # model has actually been retrained with them (see train_baseline.py).
    # Splitting the two avoids the confusing impression that everything
    # shown was necessarily used to produce predicted_elo above.
    model_columns = set(bundle["feature_columns"])
    used = {k: v for k, v in features.items() if k in model_columns}
    not_used = {
        k: v for k, v in features.items()
        if k not in model_columns and k not in ("main_eco", "reported_elo")
    }
    eco_is_used = any(col.startswith("main_eco_grouped_") for col in model_columns)

    def _fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    with st.expander("Features used by this model"):
        st.json(_fmt(used))
        if eco_is_used:
            st.caption(f"Plus opening (main_eco={features['main_eco']}), one-hot encoded.")

    if not_used:
        with st.expander("Also computed, but NOT used by this model yet"):
            st.caption(
                "These are calculated for every prediction, but this particular model was trained "
                "before they existed as features -- they're shown for visibility, not used above."
            )
            st.json(_fmt(not_used))


st.title("♞ Chess Rating Predictor")
st.write(
    "Estimates a Lichess player's rating from a handful of their own games, using Stockfish-derived "
    "move-quality features (not just the outcome of the games)."
)

speed_category = st.selectbox("Time control", ["bullet", "blitz"])

model_slugs = available_model_slugs(speed_category)
if model_slugs:
    model_slug = st.selectbox(
        "Model", model_slugs, format_func=lambda slug: MODEL_LABELS.get(slug, slug),
        help="Different models trade off differently -- Gradient Boosting is usually the strongest, "
             "Ridge is simpler and more interpretable, Random Forest is in between.",
    )
else:
    # Older models/ directory with only the single model_{category}.joblib
    # file (from before train_baseline.py started saving one file per model
    # type) -- fall back to that, no choice to offer.
    model_slug = None

tab_fetch, tab_upload = st.tabs(["Fetch from Lichess", "Upload / paste PGN"])

with tab_fetch:
    username = st.text_input("Lichess username", key="fetch_username")
    max_games = st.slider("Games to fetch", MIN_GAMES_REQUIRED, MAX_GAMES_FETCHABLE, 10, key="fetch_max")
    if max_games > MAX_GAMES_USED:
        st.caption(
            f"More than {MAX_GAMES_USED} games goes beyond what the model's training data averaged over. "
            "This helps if you're a fairly consistent player, but Lichess returns your MOST RECENT games "
            "first, so a larger window also reaches further back in time -- if your level has been "
            "changing lately (improving or in a slump), older games pull the average away from your "
            "current strength rather than just reducing noise. Also slower to analyze here."
        )
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
                    run_prediction(pgn_text, username, speed_category, model_slug=model_slug, max_games=max_games)
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
            run_prediction(pgn_text, upload_username, speed_category, model_slug=model_slug)
