"""
spot_check.py

Sanity-checks a trained model against RANDOM eligible players from the
full Lichess dump who were NOT part of the curated training set (the
formal held-out test set is still a random sample, but only from the
~3,000-3,400 players select_games_by_player.py actually picked -- this
instead draws from the full eligible pool, e.g. ~420K bullet players,
almost none of which the model has ever "seen" even indirectly).

Fetches each sampled player's recent games live from the Lichess API
(same mechanism app.py uses), analyzes them with Stockfish at the same
depth training used, and compares the prediction to their actual
(reported) Elo -- reusing predict_from_pgn.py so this is the exact same
code path a real user of the Streamlit app would go through.

Usage:
    python src/spot_check.py --speed-category bullet --n-players 15
"""

import argparse
import json
import random
import time

import chess.engine
import joblib
import pandas as pd

from analyze_with_stockfish import STOCKFISH_PATH
from lichess_api import fetch_games_pgn, LichessApiError
from predict_from_pgn import (
    parse_games_for_player,
    analyze_player_games,
    aggregate_player_rows,
    build_model_input,
    MIN_GAMES_REQUIRED,
)

ELIGIBLE_CACHE_PATH = "data/processed/eligible_players_by_category.json"
DEPTH = 12
GAMES_PER_PLAYER = 10
DELAY_BETWEEN_REQUESTS = 3  # seconds; respectful of Lichess's "one request at a time" limit


def pick_unseen_players(speed_category: str, n_players: int, already_used: set, seed: int) -> list:
    with open(ELIGIBLE_CACHE_PATH, encoding="utf-8") as f:
        eligible = set(json.load(f)[speed_category])
    candidates = list(eligible - already_used)
    random.Random(seed).shuffle(candidates)
    return candidates[: n_players * 3]  # oversample -- some will have too few usable games or fail to fetch


def main():
    parser = argparse.ArgumentParser(
        description="Spot-checks a trained model against random eligible players outside the training set."
    )
    parser.add_argument("--speed-category", required=True, choices=["bullet", "blitz"])
    parser.add_argument("--n-players", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trained_on = pd.read_csv(f"data/processed/players_{args.speed_category}.csv")
    already_used = set(trained_on["username"].astype(str))
    candidates = pick_unseen_players(args.speed_category, args.n_players, already_used, args.seed)
    print(f"Trying up to {len(candidates)} candidate players to find {args.n_players} usable ones...\n")

    bundle = joblib.load(f"models/model_{args.speed_category}.joblib")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    results = []
    try:
        for username in candidates:
            if len(results) >= args.n_players:
                break

            time.sleep(DELAY_BETWEEN_REQUESTS)
            try:
                pgn_text = fetch_games_pgn(username, args.speed_category, GAMES_PER_PLAYER)
            except LichessApiError as error:
                print(f"  [skip] {username}: {error}")
                continue
            if not pgn_text.strip():
                print(f"  [skip] {username}: no games returned")
                continue

            games = parse_games_for_player(pgn_text, username)
            if len(games) < MIN_GAMES_REQUIRED:
                print(f"  [skip] {username}: only {len(games)} games")
                continue

            rows = analyze_player_games(games, username, engine, DEPTH)
            try:
                features = aggregate_player_rows(rows, args.speed_category)
            except ValueError as error:
                print(f"  [skip] {username}: {error}")
                continue

            X = build_model_input(features, bundle["feature_columns"], bundle["top_eco"])
            predicted = bundle["model"].predict(X)[0]
            actual = features["reported_elo"]
            error = abs(predicted - actual)
            results.append((username, actual, predicted, error))
            print(f"  {username}: actual={actual:.0f}  predicted={predicted:.0f}  error={error:.0f}")
    finally:
        engine.quit()

    if not results:
        print("\nNo usable players found.")
        return

    df = pd.DataFrame(results, columns=["username", "actual", "predicted", "abs_error"])
    print(f"\n--- Spot-check results ({args.speed_category}, n={len(df)}) ---")
    print(f"Mean absolute error:   {df['abs_error'].mean():.1f}")
    print(f"Median absolute error: {df['abs_error'].median():.1f}")
    print(f"(for reference, the formal held-out test MAE was {bundle['test_mae']:.1f})")


if __name__ == "__main__":
    main()
