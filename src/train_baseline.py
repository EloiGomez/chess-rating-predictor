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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DEFAULT_DATA_PATH = "data/processed/players.csv"


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


def build_features(df: pd.DataFrame, speed_category: str):
    """Selects and encodes the model's input features.

    NOTE: avg_opponent_elo is deliberately EXCLUDED. Lichess pairs players
    with similarly-rated opponents, so a player's own Elo and their
    average opponent's Elo end up correlated at ~0.97 in this dataset --
    including it lets the model predict Elo almost entirely from Elo (via
    the opponent), which trivially inflates R^2 without the model learning
    anything from actual move quality. Keeping it out gives an honest read
    of how much the Stockfish-derived features alone can predict."""
    top_eco = df["main_eco"].value_counts().nlargest(30).index
    df["main_eco_grouped"] = df["main_eco"].where(df["main_eco"].isin(top_eco), "Other")

    numeric_features = [
        "n_games", "avg_num_plies", "avg_acpl", "avg_blunders",
        "avg_mistakes", "avg_inaccuracies", "win_rate",
    ]
    if "avg_time_per_move" in df.columns:
        numeric_features.append("avg_time_per_move")

    # main_speed_category has no variance once the dataset is restricted to
    # a single speed, so it wouldn't survive drop_first's collinearity
    # removal anyway -- excluding it explicitly is just clearer.
    categorical_features = ["main_eco_grouped"] if speed_category else ["main_eco_grouped", "main_speed_category"]

    features = df[categorical_features + numeric_features].copy()
    # One-hot encoding for the categorical columns only. drop_first=True
    # avoids the "dummy variable trap" (exact collinearity with the intercept).
    X = pd.get_dummies(features, columns=categorical_features, drop_first=True)
    y = df["elo"]
    return X, y


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
    args = parser.parse_args()

    df = load_data(args.data, args.speed_category)
    X, y = build_features(df, args.speed_category)

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

    # Train and compare two models:
    # - Ridge: linear regression WITH regularization.
    # - Random Forest: non-linear model, can capture interactions between
    #   features (e.g. "lots of blunders IN BULLET games" weighs
    #   differently than "lots of blunders in classical games").
    models = {
        "Ridge (regularized linear)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    print("\n--- Results ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n{name}")
        print(f"  MAE (mean absolute error): {mae:.1f} Elo points")
        print(f"  R^2 (variance explained):  {r2:.3f}")

    # Quick look at which features matter most, according to the Random Forest
    rf_model = models["Random Forest"]
    importances = sorted(
        zip(X.columns, rf_model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("\nMost influential features according to the Random Forest:")
    for feat, importance in importances[:10]:
        print(f"  {feat}: {importance:.3f}")


if __name__ == "__main__":
    main()
