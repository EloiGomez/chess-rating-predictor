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

from analyze_with_stockfish import get_eval_sequence, compute_move_quality, PHASES
from extract_clock_features import per_move_times, time_budget, parse_time_control
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

        evals, phases = get_eval_sequence(moves_uci, engine, depth)
        quality = compute_move_quality(evals, phases)

        own_elo_str = headers.get(f"{side.capitalize()}Elo", "")
        result = headers.get("Result", "")
        won = int((result == "1-0" and side == "white") or (result == "0-1" and side == "black"))

        white_times, black_times = per_move_times(game)
        own_times = white_times if side == "white" else black_times
        own_avg_time = (sum(own_times) / len(own_times)) if own_times else None

        budget = time_budget(headers.get("TimeControl", ""))
        per_move_share = budget / 40 if budget else None
        own_time_ratio = (own_avg_time / per_move_share) if (own_avg_time is not None and per_move_share) else None
        base_time, increment = parse_time_control(headers.get("TimeControl", ""))

        row = {
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
            "avg_time_per_move": own_avg_time,
            "time_ratio": own_time_ratio,
            "base_time": base_time,
            "increment": increment,
        }
        for phase in PHASES:
            row[f"own_{phase}_acpl"] = quality[f"{side}_{phase}_acpl"]
            row[f"own_{phase}_blunders"] = quality[f"{side}_{phase}_blunders"]
            row[f"own_{phase}_mistakes"] = quality[f"{side}_{phase}_mistakes"]
            row[f"own_{phase}_inaccuracies"] = quality[f"{side}_{phase}_inaccuracies"]
        rows.append(row)

    return rows


def aggregate_player_rows(rows: list, speed_category: str, restrict_exact_time_control: bool = False,
                           max_games: int = MAX_GAMES_USED) -> dict:
    """Averages this player's per-game rows into a single feature dict,
    mirroring aggregate_by_player.py's aggregate_players() for one ad-hoc
    player instead of a whole dataset. Restricted to `speed_category`
    (games of other formats, if any got mixed into the upload, are
    dropped).

    `restrict_exact_time_control` mirrors aggregate_by_player.py's
    restrict_to_main_time_control() -- further narrowing to the player's
    single most common EXACT time control (base_time, increment), not just
    the shared speed category. This MUST be off unless the loaded model was
    actually trained with that same restriction: turning it on
    unconditionally here computes averages a different way than whichever
    model happens to be loaded was trained with (e.g. a model trained by
    averaging over a player's mixed 180+0/180+2 blitz games would get fed,
    at prediction time, an average over only their 180+2 games -- a
    different quantity, not the noise-reduced version of the same one),
    which can silently skew predictions. app.py decides this by checking
    whether the loaded model's feature_columns include "base_time".

    `max_games` defaults to MAX_GAMES_USED (matching MAX_GAMES_PER_PLAYER,
    the cap training data used), but callers may raise it. This only
    reduces noise if the player's true strength was roughly CONSTANT over
    all those games -- caller beware: app.py's `rows` come from
    Lichess's most-recent-games-first API, so a larger max_games also
    reaches further back in time. For a player whose level has been
    trending up or down lately, that pulls the average toward their PAST
    strength rather than just averaging out noise around a stable one --
    confirmed in practice: a real player's prediction went from
    near-perfect at 5 games to off by -360 at 15, because their older
    games were weaker than their current form."""
    df = pd.DataFrame(rows)
    df = df[df["speed_category"] == speed_category]

    if restrict_exact_time_control and not df.empty:
        tc_counts = df.groupby(["base_time", "increment"]).size()
        if not tc_counts.empty:
            main_base_time, main_increment = tc_counts.idxmax()
            df = df[(df["base_time"] == main_base_time) & (df["increment"] == main_increment)]

    df = df.head(max_games)

    if len(df) < MIN_GAMES_REQUIRED:
        extra = (
            " Games at a different exact time control (even within the same speed category) "
            "aren't counted together." if restrict_exact_time_control else ""
        )
        raise ValueError(
            f"Only {len(df)} usable {speed_category} game(s) found -- need at least "
            f"{MIN_GAMES_REQUIRED}.{extra}"
        )

    eco_mode = df["eco"].mode()
    avg_acpl = df["own_acpl"].mean()
    avg_opponent_acpl = df["opponent_acpl"].mean()
    features = {
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
        "avg_time_ratio": df["time_ratio"].mean(),
        "base_time": df["base_time"].mean(),
        "increment": df["increment"].mean(),
        "main_eco": eco_mode.iat[0] if not eco_mode.empty else "",
        "reported_elo": df["own_elo"].mean(),  # for display/comparison only -- NOT a model input
    }
    for phase in PHASES:
        for stat in ("acpl", "blunders", "mistakes", "inaccuracies"):
            col = f"own_{phase}_{stat}"
            value = df[col].mean()
            # a short game might never reach middlegame/endgame, leaving
            # that phase's stats missing (NaN) for some or all of a
            # player's games -- fall back to their overall stat rather
            # than leaving NaN, since Ridge/RandomForestRegressor error on
            # missing values (only HistGradientBoostingRegressor tolerates
            # them natively).
            if pd.isna(value):
                value = features["avg_acpl" if stat == "acpl" else f"avg_{stat}"]
            features[f"avg_{phase}_{stat}"] = value
    return features


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
        "avg_time_ratio": features["avg_time_ratio"],
        "base_time": features["base_time"],
        "increment": features["increment"],
        f"main_eco_grouped_{eco_grouped}": 1,
    }
    for phase in PHASES:
        for stat in ("acpl", "blunders", "mistakes", "inaccuracies"):
            key = f"avg_{phase}_{stat}"
            row[key] = features[key]

    X = pd.DataFrame([row])
    # Any one-hot column the model expects but this single row doesn't
    # have (every OTHER opening code, plus main_speed_category dummies if
    # the model was trained mixed-format) is simply 0 -- reindex fills
    # those in and drops anything extra/unexpected.
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X
