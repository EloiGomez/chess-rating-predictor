"""
train_baseline.py

Baseline model to predict a PLAYER's rating (Elo) from several of their
games (5-15), aggregated by aggregate_by_player.py: average PGN metadata
(opening, game length) and move-quality features extracted with Stockfish
(average ACPL, average blunders/mistakes/inaccuracies per game).

Predicting from several games instead of a single one is the central idea
of the project: the outcome of ONE game is very noisy (a bad day, an odd
opponent), but the average over several games of the same player is much
more stable.

Optionally restrict training to a single time-control speed with
--speed-category (e.g. "bullet", "blitz"). aggregate_by_player.py already
restricts each player to their own dominant format, so the dataset mixes
players of different formats together with main_speed_category as a
feature -- but a player's Elo dynamics (and how ACPL translates to skill)
genuinely differ by format, so a model trained on a single, large-enough
format can fit that format's relationship more sharply than one model
trying to cover all formats through a single dummy variable. Lichess data
skews heavily towards bullet and blitz; rapid and classical are usually
too small a slice to train (and evaluate) a dedicated model reliably --
check the "main_speed_category" counts printed at startup before trusting
a per-category run.

Usage:
    python src/aggregate_by_player.py --input data/processed/games_engine.csv \
                                       --output data/processed/players.csv
    python src/train_baseline.py
    python src/train_baseline.py --speed-category bullet
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DEFAULT_DATA_PATH = "data/processed/players.csv"
TOP_ECO_COUNT = 30

# Filename fragment for each model, when saving one file per model type
# (see main()'s --save-model-dir handling) so the Streamlit app can offer
# a choice between them instead of only ever deploying the single winner.
MODEL_SLUGS = {
    "Ridge (regularized linear)": "ridge",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}


def load_data(data_path: str, speed_category: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} players, {df.shape[1]} columns")
    print("Players per time-control speed:")
    print(df["main_speed_category"].value_counts().to_string())

    if speed_category:
        df = df[df["main_speed_category"] == speed_category].reset_index(drop=True)
        print(f"\nRestricted to speed_category='{speed_category}': {df.shape[0]} players")
        if df.shape[0] < 50:
            print("[warning] this is a small sample -- MAE/R^2 from it will be noisy, take them with a grain of salt.")

    print()
    print(df.head())
    return df


def build_features(df: pd.DataFrame, speed_category: str, top_eco=None):
    """Selects and encodes the model's input features.

    `top_eco` is normally computed from this dataset (the 30 most common
    openings; anything else gets grouped into "Other"), but a caller doing
    live inference on a handful of new games should instead pass in the
    top_eco list SAVED from training -- an opening code that happened to
    be common in one new game must still be grouped exactly the way the
    model learned it, not according to its own frequency in a 5-15-game
    sample.

    NOTE: avg_opponent_elo is deliberately EXCLUDED. Lichess pairs players
    with similarly-rated opponents, so a player's own Elo and their
    average opponent's Elo end up correlated at ~0.97 in this dataset --
    including it lets the model predict Elo almost entirely from Elo (via
    the opponent), which trivially inflates R^2 without the model learning
    anything from actual move quality. Keeping it out gives an honest read
    of how much the Stockfish-derived features alone can predict.

    The OPPONENT's move-quality features (avg_opponent_acpl etc.) don't
    have that problem -- Lichess pairs by rating, not by how cleanly the
    opponent happens to play in a given game, so they aren't a backdoor
    proxy for the target. They're included because a sloppy game from the
    opponent is a sign the position itself was messy, which puts the
    player's own ACPL in that same game into context. acpl_diff (own -
    opponent) makes that relative signal directly available to Ridge,
    which can't compute a difference between two features on its own the
    way a tree-based model can."""
    if top_eco is None:
        top_eco = df["main_eco"].value_counts().nlargest(TOP_ECO_COUNT).index
    df = df.copy()
    df["main_eco_grouped"] = df["main_eco"].where(df["main_eco"].isin(top_eco), "Other")

    numeric_features = [
        "n_games", "avg_num_plies", "avg_acpl", "avg_blunders",
        "avg_mistakes", "avg_inaccuracies", "win_rate",
    ]
    if "avg_opponent_acpl" in df.columns:
        numeric_features += [
            "avg_opponent_acpl", "avg_opponent_blunders",
            "avg_opponent_mistakes", "avg_opponent_inaccuracies", "acpl_diff",
        ]
    if "avg_time_per_move" in df.columns:
        numeric_features.append("avg_time_per_move")
    if "avg_time_ratio" in df.columns:
        # avg_time_per_move in raw seconds isn't directly comparable across
        # time controls that share a speed category but have different
        # clocks (e.g. 60+0 vs 120+1, both "bullet") -- caught from a real
        # player who plays 120+1 exclusively in an ~82%-60+0 bullet dataset
        # and was mispredicted by over 500 Elo. avg_time_ratio expresses
        # time usage as a fraction of that game's own budget instead (see
        # extract_clock_features.py's time_budget()).
        #
        # It's kept ALONGSIDE avg_time_per_move, not as a replacement: a
        # direct test of replacing it made the model meaningfully WORSE
        # (bullet R^2 0.889 -> 0.847) -- apparently raw play speed carries
        # real signal beyond "speed relative to this game's clock" (faster
        # players may just be objectively faster, in absolute terms, more
        # or less regardless of the specific time control). Keeping both
        # gives a small net improvement overall, though it doesn't fully
        # fix predictions for players who mostly play a minority time
        # control within their speed category -- the model still leans on
        # the raw feature, which is confounded for exactly those players.
        numeric_features.append("avg_time_ratio")
    if "base_time" in df.columns:
        # Each player is restricted (in aggregate_by_player.py) to games
        # played at a single exact time control, not just a shared speed
        # category -- a 3+0 blitz game and a 5+3 one are both "blitz" but
        # play very differently (much more thinking time per move in the
        # latter, in absolute terms). base_time/increment tell the model
        # exactly which time control this player's stats come from.
        numeric_features += ["base_time", "increment"]
    if "avg_opening_acpl" in df.columns:
        # Split move quality by game phase (opening: first 20 plies;
        # endgame: once both queens are off the board; middlegame:
        # everything between) instead of one ACPL number for the whole
        # game -- a player can be booked-up and precise in the opening but
        # shaky in endgame technique, or vice versa, and a single average
        # hides that. See analyze_with_stockfish.py's classify_phase().
        for phase in ("opening", "middlegame", "endgame"):
            numeric_features += [
                f"avg_{phase}_acpl", f"avg_{phase}_blunders",
                f"avg_{phase}_mistakes", f"avg_{phase}_inaccuracies",
            ]

    # main_speed_category has no variance once the dataset is restricted to
    # a single speed, so it wouldn't survive drop_first's collinearity
    # removal anyway -- excluding it explicitly is just clearer.
    categorical_features = ["main_eco_grouped"] if speed_category else ["main_eco_grouped", "main_speed_category"]

    features = df[categorical_features + numeric_features].copy()
    # One-hot encoding for the categorical columns only. drop_first=True
    # avoids the "dummy variable trap" (exact collinearity with the intercept).
    X = pd.get_dummies(features, columns=categorical_features, drop_first=True)
    y = df["elo"]
    return X, y, list(top_eco)


def main():
    parser = argparse.ArgumentParser(
        description="Trains a baseline Elo-prediction model on the player-aggregated dataset."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help=f"Input CSV (default {DEFAULT_DATA_PATH})")
    parser.add_argument(
        "--speed-category", default=None,
        help="Restrict training to one time-control speed (e.g. bullet, blitz, rapid, classical). "
             "Omit to train on all speeds mixed together (with speed as a feature).",
    )
    parser.add_argument(
        "--save-model-dir", default=None,
        help="If given, save the best-performing model here (joblib) for use by the prediction app -- "
             "retrained on ALL available data (not just the 80%% training split) after evaluation, since "
             "a deployed model should use every player it can.",
    )
    args = parser.parse_args()

    df = load_data(args.data, args.speed_category)
    X, y, top_eco = build_features(df, args.speed_category)

    # Each row is already a distinct player (not individual games), so this
    # split never mixes the same player's games between train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining:  {X_train.shape[0]} players")
    print(f"Test:      {X_test.shape[0]} players")

    # Reference point: how good would the model be if it just always
    # predicted the overall average rating?
    baseline_pred = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print(f"\n(Reference: always predicting the mean = MAE of {baseline_mae:.1f})")

    # Train and compare three models:
    # - Ridge: linear regression WITH regularization.
    # - Random Forest: non-linear, averages many independent trees (bagging)
    #   -- reduces variance/overfitting, can capture feature interactions.
    # - Gradient Boosting: also tree-based, but builds trees SEQUENTIALLY,
    #   each one correcting the previous ones' errors (reduces bias too).
    #   Usually the strongest option for tabular data like this. Uses
    #   scikit-learn's built-in HistGradientBoostingRegressor rather than a
    #   separate library (XGBoost/LightGBM) -- no new dependency, and it's
    #   a very competitive implementation of the same idea.
    models = {
        "Ridge (regularized linear)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    }

    print("\n--- Results ---")
    best_name, best_r2, best_mae = None, float("-inf"), None
    results = {}  # name -> (mae, r2), kept for saving every model, not just the winner
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = (mae, r2)

        print(f"\n{name}")
        print(f"  MAE (mean absolute error): {mae:.1f} Elo points")
        print(f"  R^2 (variance explained):  {r2:.3f}")

        if r2 > best_r2:
            best_name, best_r2, best_mae = name, r2, mae

    # Quick look at which features matter most, according to the Random Forest
    rf_model = models["Random Forest"]
    importances = sorted(
        zip(X.columns, rf_model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("\nMost influential features according to the Random Forest:")
    for feat, importance in importances[:10]:
        print(f"  {feat}: {importance:.3f}")

    if args.save_model_dir:
        os.makedirs(args.save_model_dir, exist_ok=True)
        label = args.speed_category or "mixed"

        # Save EVERY model, not just the winner -- lets the Streamlit app
        # offer a choice between them (e.g. to compare a simpler, more
        # interpretable Ridge against the strongest Gradient Boosting one)
        # instead of only ever being able to deploy whichever had the best
        # test R^2 on this particular split.
        for name, model in models.items():
            mae, r2 = results[name]
            print(f"\nRetraining {name} on all {len(X)} players for deployment...")
            model.fit(X, y)  # use every available player, not just the 80% split

            slug = MODEL_SLUGS[name]
            model_path = os.path.join(args.save_model_dir, f"model_{label}_{slug}.joblib")
            joblib.dump({
                "model": model,
                "model_name": name,
                "speed_category": args.speed_category,
                "feature_columns": list(X.columns),
                "top_eco": top_eco,
                "test_r2": r2,
                "test_mae": mae,
                "n_players_trained_on": len(X),
                "is_best": name == best_name,
            }, model_path)
            print(f"Saved to {model_path} (test R^2={r2:.3f}, test MAE={mae:.1f})")

        # Also save the winner under the plain model_{label}.joblib name --
        # kept for scripts (spot_check.py) and docs that expect "the"
        # deployed model without picking a specific one.
        best_path = os.path.join(args.save_model_dir, f"model_{label}.joblib")
        best_slug_path = os.path.join(args.save_model_dir, f"model_{label}_{MODEL_SLUGS[best_name]}.joblib")
        joblib.dump(joblib.load(best_slug_path), best_path)
        print(f"\nBest model ({best_name}) also saved to {best_path} for backward compatibility.")


if __name__ == "__main__":
    main()
