"""
parse_pgn.py

Reads a PGN file (chess games) and extracts, for each game, a set of
basic features into a CSV: both players' ratings, opening (ECO), number
of plies, result and time control.

This is the "MVP" version: it doesn't use Stockfish yet, only what's
already in the PGN headers and the move sequence. It's the first step to
validate that the full pipeline works end to end.

Usage:
    python src/parse_pgn.py --input data/raw/lichess_sample.pgn \
                             --output data/processed/games.csv \
                             --max-games 10000
"""

import argparse
import csv
import sys
from typing import Optional

import chess.pgn


def parse_games(input_path: str, output_path: str, max_games: Optional[int] = None) -> int:
    """Parses games from a PGN file and writes them as rows of a CSV.

    Returns the total number of games written.
    """
    count = 0

    with open(input_path, encoding="utf-8", errors="replace") as pgn_file, \
            open(output_path, "w", newline="", encoding="utf-8") as out_file:

        writer = csv.writer(out_file)
        writer.writerow([
            "white_elo",
            "black_elo",
            "eco",
            "opening",
            "result",
            "num_plies",
            "time_control",
        ])

        while True:
            if max_games is not None and count >= max_games:
                break

            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # end of file

            headers = game.headers

            white_elo = headers.get("WhiteElo", "")
            black_elo = headers.get("BlackElo", "")

            # skip games without a rating (bots, unrated casual games)
            if not white_elo.isdigit() or not black_elo.isdigit():
                continue

            eco = headers.get("ECO", "")
            opening = headers.get("Opening", "")
            result = headers.get("Result", "")
            time_control = headers.get("TimeControl", "")

            num_plies = sum(1 for _ in game.mainline_moves())

            writer.writerow([
                white_elo,
                black_elo,
                eco,
                opening,
                result,
                num_plies,
                time_control,
            ])

            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} games...", file=sys.stderr)

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Parses a PGN file into a CSV with basic per-game features."
    )
    parser.add_argument("--input", required=True, help="Path to the input .pgn file")
    parser.add_argument("--output", required=True, help="Path to the output .csv file")
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Maximum number of games to process (useful for quick tests)",
    )
    args = parser.parse_args()

    total = parse_games(args.input, args.output, args.max_games)
    print(f"Done. {total} games written to {args.output}")


if __name__ == "__main__":
    main()
