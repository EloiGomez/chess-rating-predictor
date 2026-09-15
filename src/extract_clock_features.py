"""
extract_clock_features.py

Extracts time-usage features from the Lichess clock comments
("{ [%clk 0:03:00] }") that follow each move in the "standard" dump
format -- a source of signal we have for free (it's already sitting in
the PGN text) and don't need a single extra Stockfish call for.

The idea: how long a player takes to think, and whether they tend to
drift into time trouble, is plausibly correlated with playing strength --
and it's completely independent of the ACPL/blunder features from
analyze_with_stockfish.py, so it's a genuinely new signal rather than a
reformulation of one we already have.

For each game, computes how many seconds each player spent per move
(previous clock reading - new clock reading + increment), then averages
that per player across their games -- same "own_"/long-format shape as
aggregate_by_player.py, so it merges into the same pipeline.

Also computes a NORMALIZED version, time_ratio = avg_time_per_move /
time_budget(time_control), where time_budget is the same base + 40 *
increment estimate select_games_by_player.py and aggregate_by_player.py
use for speed-category classification. Raw seconds-per-move isn't
comparable across time controls that both count as, say, "bullet" but
have very different clocks -- a 120+1 bullet game hands out roughly
double the time budget of a 60+0 one, so a player who exclusively plays
120+1 will show a higher avg_time_per_move than an equally-skilled 60+0
player for reasons that have nothing to do with skill. This was caught
from a real mispredicted player (~1890 actual vs ~1350 predicted) who
turned out to play 120+1 exclusively, in a bullet dataset that's ~82%
60+0. time_ratio expresses time usage as a fraction of the game's own
budget, so it's comparable across variants of the same speed category.

Output has one row per game (same order as the input PGN, none skipped),
so it lines up positionally with the games_engine.csv row order in the
same run -- aggregate_by_player.py's --clock-input option checks this
alignment (matching white/black Elo) before merging.

Usage:
    python src/extract_clock_features.py \
        --input data/processed/selected_games.pgn \
        --output data/processed/games_clock.csv
"""

import argparse
import re

import chess.pgn
import pandas as pd

CLOCK_RE = re.compile(r"\[%clk (\d+):(\d{2}):(\d{2})\]")


def parse_clock_seconds(comment: str):
    match = CLOCK_RE.search(comment or "")
    if not match:
        return None
    hours, minutes, seconds = (int(g) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def parse_time_control(time_control: str):
    """Returns (base_seconds, increment_seconds), or (None, None) if the
    time control isn't in the usual "base+increment" form (correspondence
    games use "-", for instance)."""
    try:
        base_str, increment_str = str(time_control).split("+")
        return int(base_str), int(increment_str)
    except (ValueError, AttributeError):
        return None, None


def time_budget(time_control: str):
    """Same estimate select_games_by_player.py/aggregate_by_player.py use
    for speed-category classification (base + 40 * increment) -- reused
    here as the denominator for time_ratio, so "spent X% of a typical
    allotment per move" means the same thing regardless of which specific
    base+increment combination a game used."""
    base, increment = parse_time_control(time_control)
    if base is None:
        return None
    return base + 40 * increment


def per_move_times(game: chess.pgn.Game) -> tuple:
    """Returns (white_seconds_per_move, black_seconds_per_move): lists of
    how many seconds each player spent on each of their moves, computed
    from consecutive clock readings."""
    base, increment = parse_time_control(game.headers.get("TimeControl", ""))
    white_times, black_times = [], []
    if base is None:
        return white_times, black_times

    prev_clock = {"white": base, "black": base}
    for ply, node in enumerate(game.mainline(), start=1):
        clk = parse_clock_seconds(node.comment)
        if clk is None:
            continue
        side = "white" if ply % 2 == 1 else "black"
        spent = max(0, prev_clock[side] - clk + increment)
        (white_times if side == "white" else black_times).append(spent)
        prev_clock[side] = clk

    return white_times, black_times


def extract_rows(input_path: str) -> list:
    rows = []
    with open(input_path, encoding="utf-8", errors="replace") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            headers = game.headers
            white_times, black_times = per_move_times(game)
            white_avg = (sum(white_times) / len(white_times)) if white_times else None
            black_avg = (sum(black_times) / len(black_times)) if black_times else None

            # "fair share" per move for THIS game's specific clock, assuming
            # ~40 moves/side (same assumption select_games_by_player.py's
            # speed-category estimate makes) -- dividing by this turns an
            # absolute seconds-per-move number into "typical pace" (~1.0),
            # "faster than typical" (<1), "slower" (>1), comparable across
            # different base+increment combinations within the same speed
            # category (e.g. 60+0 vs 120+1, both "bullet").
            budget = time_budget(headers.get("TimeControl", ""))
            per_move_share = budget / 40 if budget else None
            white_ratio = (white_avg / per_move_share) if (white_avg is not None and per_move_share) else None
            black_ratio = (black_avg / per_move_share) if (black_avg is not None and per_move_share) else None

            rows.append({
                "white": headers.get("White", ""),
                "black": headers.get("Black", ""),
                "white_elo": headers.get("WhiteElo", ""),
                "black_elo": headers.get("BlackElo", ""),
                "white_avg_time_per_move": white_avg,
                "black_avg_time_per_move": black_avg,
                "white_time_ratio": white_ratio,
                "black_time_ratio": black_ratio,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extracts per-player average time-per-move from Lichess clock comments."
    )
    parser.add_argument("--input", required=True, help="Input PGN (with %%clk comments)")
    parser.add_argument("--output", required=True, help="Output CSV, one row per game")
    args = parser.parse_args()

    rows = extract_rows(args.input)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    have_white = df["white_avg_time_per_move"].notna().sum()
    have_black = df["black_avg_time_per_move"].notna().sum()
    print(f"Games processed: {len(df):,}")
    print(f"Games with usable White clock data: {have_white:,}")
    print(f"Games with usable Black clock data: {have_black:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
