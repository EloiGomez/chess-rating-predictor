"""
benchmark_stockfish.py

Measures how long Stockfish takes to analyze a handful of real games,
using a single engine (no parallel workers -- see analyze_with_stockfish.py
for the parallel version), to estimate how the analysis depth trades off
against speed before committing to a full run at a new depth.

Usage:
    python src/benchmark_stockfish.py --depth 10
    python src/benchmark_stockfish.py --depth 15
"""

import argparse
import io
import time

import chess
import chess.engine
import chess.pgn
import zstandard

ZST_PATH = r"F:\lichess_data\lichess_db_standard_rated_2026-08.pgn.zst"

# Must match analyze_with_stockfish.py EXACTLY.
STOCKFISH_PATH = r"stockfish\stockfish-windows-x86-64-avx2.exe"
DEFAULT_DEPTH = 10

SAMPLE_GAMES = 20  # how many games to benchmark with


def open_pgn_stream(path):
    fh = open(path, "rb")
    dctx = zstandard.ZstdDecompressor()
    stream_reader = dctx.stream_reader(fh)
    return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")


def analyze_one_game(game, engine, depth):
    """Same logic as get_eval_sequence() in analyze_with_stockfish.py:
    evaluates the starting position and after every move."""
    board = game.board()
    n_positions = 0

    engine.analyse(board, chess.engine.Limit(depth=depth))
    n_positions += 1

    for move in game.mainline_moves():
        board.push(move)
        engine.analyse(board, chess.engine.Limit(depth=depth))
        n_positions += 1

    return n_positions


def main():
    parser = argparse.ArgumentParser(
        description="Benchmarks Stockfish analysis speed at a given depth on a handful of real games."
    )
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help=f"Analysis depth (default {DEFAULT_DEPTH})")
    parser.add_argument("--sample-games", type=int, default=SAMPLE_GAMES, help=f"Games to benchmark with (default {SAMPLE_GAMES})")
    args = parser.parse_args()

    print(f"Reading the first {args.sample_games} playable games from the dump...")
    games = []
    with open_pgn_stream(ZST_PATH) as pgn:
        while len(games) < args.sample_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if len(list(game.mainline_moves())) >= 10:  # discard near-empty games
                games.append(game)

    print(f"Games read: {len(games)}\n")

    print(f"Analyzing with Stockfish at depth {args.depth} (this is what we want to measure)...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    start = time.time()
    positions_analyzed = 0
    try:
        for game in games:
            positions_analyzed += analyze_one_game(game, engine, args.depth)
    finally:
        engine.quit()

    elapsed = time.time() - start
    seconds_per_position = elapsed / positions_analyzed
    seconds_per_game = elapsed / len(games)

    print("\n--- Benchmark results ---")
    print(f"Depth:             {args.depth}")
    print(f"Total time:        {elapsed:.1f} s for {positions_analyzed} positions across {len(games)} games")
    print(f"Per position:      {seconds_per_position:.3f} s")
    print(f"Per game (avg):    {seconds_per_game:.2f} s")

    print("\n--- Projection (single engine, no parallelism) ---")
    for hours in (1, 8, 24):
        games_that_fit = int((hours * 3600) / seconds_per_game)
        print(f"In {hours:>2} hours, roughly {games_that_fit:,} games fit")


if __name__ == "__main__":
    main()
