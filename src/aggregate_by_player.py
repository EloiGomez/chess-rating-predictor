"""
aggregate_by_player.py

Converts the per-game CSV produced by analyze_with_stockfish.py (one row
per game) into a CSV with ONE ROW PER PLAYER, averaged over their games.
This is the central idea of the project: estimate a player's Elo from
several of their games (5-15, per select_games_by_player.py), not from a
single one -- the outcome of ONE game is very noisy (a bad day, an odd
opponent, a bullet game mixed with a classical one), but the average over
several games of the same player is much more stable and should be
considerably easier to predict.

Every game involves TWO players (White and Black). This script "unfolds"
each game into up to two records -- one per participating player --
keeping the columns that belong to THAT player (their own Elo, their own
ACPL, their own blunders...) and storing the opponent's Elo as an extra
feature. It then groups by username:

    - restricts each player to games played at THEIR single most frequent
      time-control speed (bullet/blitz/rapid/classical). Lichess gives
      every player a SEPARATE rating per speed category (someone can be
      2400 in bullet and 2000 in rapid), and select_games_by_player.py
      doesn't filter by speed, so a player's sampled games can mix
      formats. Averaging Elo (or ACPL -- bullet games are short and
      sloppy almost regardless of skill, classical games are long and
      careful almost regardless of skill) across formats would blend
      together numbers that don't mean the same thing.
    - drops players with fewer than --min-games games in that single
      format (not enough for a reliable average)
    - keeps at most --max-games games per player (in case the input CSV
      wasn't already capped by select_games_by_player.py)
    - averages the numeric features (ACPL, blunders, mistakes,
      inaccuracies, game length, opponent Elo, win rate) and keeps each
      player's most frequent opening as a categorical feature

The target Elo (column "elo") is also the average of the player's own
Elo across those games -- Lichess ratings fluctuate a bit from game to
game, so averaging gives a more stable estimate of "their level" than
just taking the rating from a single game.

Optionally, pass --clock-input with the output of
extract_clock_features.py to bring in time-per-move as an extra feature
(average seconds a player spends thinking per move, derived from the
PGN's clock comments -- a signal that's independent of the Stockfish
quality features). Both CSVs must come from processing the SAME input PGN
with the SAME game count, since they're merged by row position; this is
checked (White/Black Elo must match row-for-row) before merging, and the
whole clock feature is skipped with a warning if it doesn't line up.

Usage:
    python src/aggregate_by_player.py \
        --input data/processed/games_engine.csv \
        --output data/processed/players.csv \
        --clock-input data/processed/games_clock.csv \
        --min-games 5 --max-games 15
"""

import argparse

import pandas as pd

DEFAULT_MIN_GAMES = 5
DEFAULT_MAX_GAMES = 15


def estimate_speed_category(time_control: str) -> str:
    """Classifies the time control into bullet/blitz/rapid/classical,
    using the same formula Lichess uses internally: estimated time = base
    time + 40 * increment (in seconds). (Same logic as train_baseline.py --
    duplicated here because it's a small, pure function; not worth turning
    train_baseline.py into an importable module just for this.)"""
    try:
        base_str, increment_str = str(time_control).split("+")
        estimated_seconds = int(base_str) + 40 * int(increment_str)
    except (ValueError, AttributeError):
        return "unknown"

    if estimated_seconds < 30:
        return "ultrabullet"
    elif estimated_seconds < 180:
        return "bullet"
    elif estimated_seconds < 480:
        return "blitz"
    elif estimated_seconds < 1500:
        return "rapid"
    else:
        return "classical"


def merge_clock_features(games: pd.DataFrame, clock_path: str) -> pd.DataFrame:
    """Merges in white_avg_time_per_move / black_avg_time_per_move from
    extract_clock_features.py's output. The two files are matched by ROW
    POSITION (both come from processing the same PGN in the same order),
    so this checks White/Black Elo line up before trusting the merge --
    if they don't, the clock columns are skipped entirely rather than
    silently attaching the wrong game's time data to a player."""
    clock = pd.read_csv(clock_path)

    if len(clock) != len(games):
        print(f"[warning] --clock-input has {len(clock):,} rows but the games CSV has {len(games):,} -- "
              "they don't come from the same run, skipping clock features.")
        return games

    mismatched = (clock["white_elo"].astype(str).values != games["white_elo"].astype(str).values) | \
                 (clock["black_elo"].astype(str).values != games["black_elo"].astype(str).values)
    if mismatched.any():
        print(f"[warning] --clock-input rows don't line up with the games CSV ({mismatched.sum():,} mismatched "
              "Elo pairs) -- skipping clock features.")
        return games

    games = games.copy()
    games["white_avg_time_per_move"] = clock["white_avg_time_per_move"].values
    games["black_avg_time_per_move"] = clock["black_avg_time_per_move"].values
    return games


def to_long_format(games: pd.DataFrame) -> pd.DataFrame:
    """Converts the games DataFrame (one row per game, white_*/black_*
    columns) into a "long" DataFrame: one row per player participation in
    a game, with their own columns (without the white_/black_ prefix)
    plus the opponent's Elo."""
    has_clock = "white_avg_time_per_move" in games.columns and "black_avg_time_per_move" in games.columns

    sides = []
    for side, opponent, win_result in (("white", "black", "1-0"), ("black", "white", "0-1")):
        columns = {
            "username": games[side],
            "own_elo": games[f"{side}_elo"],
            "opponent_elo": games[f"{opponent}_elo"],
            "own_acpl": games[f"{side}_acpl"],
            "own_blunders": games[f"{side}_blunders"],
            "own_mistakes": games[f"{side}_mistakes"],
            "own_inaccuracies": games[f"{side}_inaccuracies"],
            "won": (games["result"] == win_result).astype(int),
            "num_plies": games["num_plies"],
            "eco": games["eco"],
            "speed_category": games["time_control"].apply(estimate_speed_category),
        }
        if has_clock:
            columns["own_avg_time_per_move"] = games[f"{side}_avg_time_per_move"]
        sides.append(pd.DataFrame(columns))

    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df[long_df["username"].astype(str).str.len() > 0]
    return long_df


def mode_or_empty(series: pd.Series):
    modes = series.mode()
    return modes.iat[0] if not modes.empty else ""


def restrict_to_main_speed_category(long_df: pd.DataFrame) -> pd.DataFrame:
    """Keeps, for each player, only the games played at THEIR single most
    frequent time-control speed (bullet/blitz/rapid/classical).

    Lichess gives each player a SEPARATE rating per speed category -- the
    same person can easily be 2400 in bullet, 2270 in blitz and 2000 in
    rapid. select_games_by_player.py doesn't filter by speed category, so
    a player's 5-15 sampled games can freely mix formats; averaging
    White/BlackElo across them would blend together ratings that don't
    mean the same thing into a number that corresponds to no real rating
    at all. It also confounds the Stockfish features: bullet games are
    short and sloppy (high ACPL) almost regardless of skill, classical
    games are long and careful (low ACPL) almost regardless of skill, so
    a mixed-format sample partly encodes "which format did they play"
    rather than "how well do they play". Restricting each player to a
    single format keeps every average internally consistent."""
    main_speed = long_df.groupby("username")["speed_category"].agg(mode_or_empty)
    long_df = long_df.join(main_speed.rename("main_speed_category"), on="username")
    return long_df[long_df["speed_category"] == long_df["main_speed_category"]].drop(columns="main_speed_category")


def aggregate_players(long_df: pd.DataFrame, min_games: int, max_games: int) -> pd.DataFrame:
    """Groups the long DataFrame by player: keeps only each player's
    dominant time-control format, drops players who don't reach min_games
    within that single format, keeps at most max_games games for each of
    them, and averages their features."""
    # First pass: everyone with enough games in ANY format combined, just
    # to have a reasonable pool to compute each player's dominant format
    # from (see restrict_to_main_speed_category).
    counts_any_format = long_df.groupby("username").size()
    candidates = counts_any_format[counts_any_format >= min_games].index
    single_format_df = restrict_to_main_speed_category(long_df[long_df["username"].isin(candidates)])

    # Order matters here: eligibility (min_games) must be decided on each
    # player's game count WITHIN their dominant format, before capping. If
    # capping to max_games happened FIRST and min_games were checked
    # afterwards, a player with plenty of games but max_games < min_games
    # would never qualify (and in general, capping first distorts who
    # counts as "eligible").
    counts = single_format_df.groupby("username").size()
    eligible = counts[counts >= min_games].index
    capped = single_format_df[single_format_df["username"].isin(eligible)] \
        .groupby("username", group_keys=False).head(max_games)

    numeric_agg = capped.groupby("username").agg(
        elo=("own_elo", "mean"),
        n_games=("own_elo", "size"),
        avg_acpl=("own_acpl", "mean"),
        avg_blunders=("own_blunders", "mean"),
        avg_mistakes=("own_mistakes", "mean"),
        avg_inaccuracies=("own_inaccuracies", "mean"),
        avg_opponent_elo=("opponent_elo", "mean"),
        win_rate=("won", "mean"),
        avg_num_plies=("num_plies", "mean"),
    )
    if "own_avg_time_per_move" in capped.columns:
        numeric_agg["avg_time_per_move"] = capped.groupby("username")["own_avg_time_per_move"].mean()

    categorical_agg = capped.groupby("username").agg(
        main_eco=("eco", mode_or_empty),
        # every remaining row for a player already shares the same
        # speed_category (that's what restrict_to_main_speed_category
        # enforced), so "first" is just reading it back, not re-computing
        # a mode.
        main_speed_category=("speed_category", "first"),
    )

    return numeric_agg.join(categorical_agg).reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregates a per-game CSV into a CSV with one row per player."
    )
    parser.add_argument("--input", required=True, help="Games CSV (output of analyze_with_stockfish.py)")
    parser.add_argument("--output", required=True, help="Output CSV, one row per player")
    parser.add_argument("--clock-input", default=None,
                         help="Optional: games CSV from extract_clock_features.py, same games/order, to add "
                              "average time-per-move as a feature")
    parser.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES,
                         help=f"Minimum games per player to include them (default {DEFAULT_MIN_GAMES})")
    parser.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES,
                         help=f"Maximum games to average per player (default {DEFAULT_MAX_GAMES})")
    args = parser.parse_args()

    games = pd.read_csv(args.input)
    if "white" not in games.columns or "black" not in games.columns:
        raise SystemExit(
            f"'{args.input}' has no 'white'/'black' columns -- it needs to be regenerated with the current "
            "version of analyze_with_stockfish.py (which now saves each player's username)."
        )

    if args.clock_input:
        games = merge_clock_features(games, args.clock_input)

    long_df = to_long_format(games)
    players = aggregate_players(long_df, args.min_games, args.max_games)

    players.to_csv(args.output, index=False)

    print(f"Input games: {len(games):,}")
    print(f"Participations (game x player): {len(long_df):,}")
    print(f"Players with >= {args.min_games} games: {len(players):,}")
    print(f"Games per player -- min: {players['n_games'].min()}, "
          f"mean: {players['n_games'].mean():.1f}, max: {players['n_games'].max()}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
