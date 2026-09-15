"""
predict_from_pgn.py

Shared feature-extraction logic for predicting a SINGLE player's Elo from
a handful of their own games. Reuses the exact same Stockfish analysis
(analyze_with_stockfish.py) and clock-time extraction
(extract_clock_features.py) the training pipeline uses, so a live
prediction is computed with the identical feature definitions the model
was trained on -- not a reimplementation that could quietly drift out of
sync with it.

Used by app.py (the Streamlit demo); not meant to be run directly.
"""

import io

import chess
import chess.engine
import chess.pgn
import pandas as pd

from analyze_with_stockfish import get_eval_sequence, compute_move_quality
from extract_clock_features import per_move_times
from aggregate_by_player import estimate_speed_category

MIN_GAMES_REQUIRED = 5
MAX_GAMES_USED = 15


def parse_games_for_player(pgn_text: str, username: str) -> list:
    """Parses a multi-game PGN blob and returns the chess.pgn.Game objects
    where `username` played (as either color) -- case-insensitively,
    since Lichess usernames aren't case sensitive but PGN headers store
    whatever casing was used when the account was created."""
    games = []
    stream = io.StringIO(pgn_text)
    username_lower = username.lower()
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        if username_lower not in (white.lower(), black.lower()):
            continue
        games.append(game)
    return games


def analyze_player_games(games: list, username: str, engine: chess.engine.SimpleEngine, depth: int) -> list:
    """Analyzes each game with Stockfish and returns one row per game with
    this player's OWN side's features -- same shape as one participation
    row in aggregate_by_player.py's long format."""
    username_lower = username.lower()
    rows = []

    for game in games:
        headers = game.headers
        white, black = headers.get("White", ""), headers.get("Black", "")
        side = "white" if white.lower() == username_lower else "black"
        opponent_side = "black" if side == "white" else "white"

        moves_uci = [move.uci() for move in game.mainline_moves()]
        if not moves_uci:
            continue

        evals = get_eval_sequence(moves_uci, engine, depth)
        quality = compute_move_quality(evals)

        own_elo_str = headers.get(f"{side.capitalize()}Elo", "")
        result = headers.get("Result", "")
        won = int((result == "1-0" and side == "white") or (result == "0-1" and side == "black"))

        white_times, black_times = per_move_times(game)
        own_times = white_times if side == "white" else black_times

        rows.append({
            "own_elo": int(own_elo_str) if own_elo_str.isdigit() else None,
            "own_acpl": quality[f"{side}_acpl"],
            "own_blunders": quality[f"{side}_blunders"],
            "own_mistakes": quality[f"{side}_mistakes"],
            "own_inaccuracies": quality[f"{side}_inaccuracies"],
            "opponent_acpl": quality[f"{opponent_side}_acpl"],
            "opponent_blunders": quality[f"{opponent_side}_blunders"],
            "opponent_mistakes": quality[f"{opponent_side}_mistakes"],
            "opponent_inaccuracies": quality[f"{opponent_side}_inaccuracies"],
            "won": won,
            "num_plies": len(evals) - 1,
            "eco": headers.get("ECO", ""),
            "speed_category": estimate_speed_category(headers.get("TimeControl", "")),
            "avg_time_per_move": (sum(own_times) / len(own_times)) if own_times else None,
        })

    return rows


def aggregate_player_rows(rows: list, speed_category: str) -> dict:
    """Averages this player's per-game rows into a single feature dict,
    mirroring aggregate_by_player.py's aggregate_players() for one ad-hoc
    player instead of a whole dataset. Restricted to `speed_category`
    (games of other formats, if any got mixed into the upload, are
    dropped) -- same rule the training data was built with."""
    df = pd.DataFrame(rows)
    df = df[df["speed_category"] == speed_category]
    df = df.head(MAX_GAMES_USED)

    if len(df) < MIN_GAMES_REQUIRED:
        raise ValueError(
            f"Only {len(df)} usable {speed_category} game(s) found -- need at least {MIN_GAMES_REQUIRED}."
        )

    eco_mode = df["eco"].mode()
    avg_acpl = df["own_acpl"].mean()
    avg_opponent_acpl = df["opponent_acpl"].mean()
    return {
        "n_games": len(df),
        "avg_num_plies": df["num_plies"].mean(),
        "avg_acpl": avg_acpl,
        "avg_blunders": df["own_blunders"].mean(),
        "avg_mistakes": df["own_mistakes"].mean(),
        "avg_inaccuracies": df["own_inaccuracies"].mean(),
        "avg_opponent_acpl": avg_opponent_acpl,
        "avg_opponent_blunders": df["opponent_blunders"].mean(),
        "avg_opponent_mistakes": df["opponent_mistakes"].mean(),
        "avg_opponent_inaccuracies": df["opponent_inaccuracies"].mean(),
        "acpl_diff": avg_acpl - avg_opponent_acpl,
        "win_rate": df["won"].mean(),
        "avg_time_per_move": df["avg_time_per_move"].mean(),
        "main_eco": eco_mode.iat[0] if not eco_mode.empty else "",
        "reported_elo": df["own_elo"].mean(),  # for display/comparison only -- NOT a model input
    }


def build_model_input(features: dict, feature_columns: list, top_eco: list) -> pd.DataFrame:
    """Turns the aggregated feature dict into a one-row DataFrame with
    EXACTLY the columns the saved model was trained on (same one-hot
    encoding, same "Other" grouping for an opening that wasn't common
    enough to get its own column at training time), so it can be fed
    straight into model.predict()."""
    eco_grouped = features["main_eco"] if features["main_eco"] in top_eco else "Other"

    row = {
        "n_games": features["n_games"],
        "avg_num_plies": features["avg_num_plies"],
        "avg_acpl": features["avg_acpl"],
        "avg_blunders": features["avg_blunders"],
        "avg_mistakes": features["avg_mistakes"],
        "avg_inaccuracies": features["avg_inaccuracies"],
        "avg_opponent_acpl": features["avg_opponent_acpl"],
        "avg_opponent_blunders": features["avg_opponent_blunders"],
        "avg_opponent_mistakes": features["avg_opponent_mistakes"],
        "avg_opponent_inaccuracies": features["avg_opponent_inaccuracies"],
        "acpl_diff": features["acpl_diff"],
        "win_rate": features["win_rate"],
        "avg_time_per_move": features["avg_time_per_move"],
        f"main_eco_grouped_{eco_grouped}": 1,
    }

    X = pd.DataFrame([row])
    # Any one-hot column the model expects but this single row doesn't
    # have (every OTHER opening code, plus main_speed_category dummies if
    # the model was trained mixed-format) is simply 0 -- reindex fills
    # those in and drops anything extra/unexpected.
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X
