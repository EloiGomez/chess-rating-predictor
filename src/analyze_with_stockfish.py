"""
analyze_with_stockfish.py

Analyzes chess games with the Stockfish engine to extract move-quality
features: ACPL (Average Centipawn Loss) and number of blunders, mistakes
and inaccuracies, per player.

This is MUCH slower than parse_pgn.py, because it analyzes every position
of every game with the engine. A single Stockfish process only uses one
CPU core by default, so this script runs several engine instances in
parallel (one per worker process) to make use of all the cores available
on the machine -- with 12 cores and single-threaded games taking ~0.66s
each, running e.g. 10 games in parallel should bring the wall-clock time
down close to ~10x, not just save CPU time.

IMPORTANT: update STOCKFISH_PATH below with the path to your executable
(inside your stockfish/ folder, look for the .exe file).

Usage:
    python src/analyze_with_stockfish.py --input data/raw/lichess_sample.pgn --output data/processed/games_engine.csv --max-games 2000 --workers 6 --depth 12
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import chess
import chess.engine
import chess.pgn
import pandas as pd

LOG_DIR = "logs"


def make_logger():
    """Creates a logger that writes every line to the console AND to a
    timestamped UTF-8 file under logs/, flushing immediately. Without this,
    stdout gets block-buffered when it isn't connected to a terminal (e.g.
    when redirected to a file or captured by a background-task runner), so
    progress prints can sit invisible for minutes at a time."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"analyze_stockfish_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg: str, file=None) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=file, flush=True)
        print(line, file=log_file, flush=True)

    return log, log_path


# --- CONFIGURE THIS ---
# Path to the Stockfish executable. Adjust the filename to whatever you
# have in your stockfish/ folder.
STOCKFISH_PATH = r"stockfish\stockfish-windows-x86-64-avx2.exe"

# Default analysis depth per position (overridable with --depth). Higher =
# more accurate but much slower -- cost grows steeply, not linearly: on
# this project's machine, single-engine benchmarks went from 0.63s/game
# at depth 10, to 1.72s/game at depth 12 (2.7x), to 4.91s/game at depth 14
# (7.8x). Use benchmark_stockfish.py --depth N to measure before committing
# to a full run at a higher depth.
DEFAULT_DEPTH = 10

# Default number of parallel worker processes (each runs its own
# Stockfish instance). Stockfish is CPU-bound and gets essentially no
# benefit from hyperthreading, so oversubscribing beyond the number of
# PHYSICAL cores doesn't help -- os.cpu_count() reports LOGICAL cores,
# which is usually 2x the physical count on hyperthreaded CPUs, hence the
# // 2 here. (Measured on a 6-physical/12-logical-core machine: 6 and 10
# workers gave the same throughput, so there's no point spawning more.)
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)

# Standard thresholds (in centipawns) to classify move quality. These are
# a simplification of what Lichess uses internally (which is based on win
# probability, not just centipawns), but work fine as a first approximation.
INACCURACY_THRESHOLD = 50
MISTAKE_THRESHOLD = 100
BLUNDER_THRESHOLD = 300


def get_eval_sequence(moves_uci: list, engine: chess.engine.SimpleEngine, depth: int) -> list:
    """Returns Stockfish's evaluation (in centipawns, from White's
    perspective) at every position of the game: the starting position and
    after each move. Takes the moves as UCI strings (e.g. "e2e4") so this
    can be called from a worker process without needing to pickle a full
    chess.pgn.Game object."""
    board = chess.Board()
    evals = []

    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    evals.append(info["score"].white().score(mate_score=1000))

    for uci in moves_uci:
        board.push(chess.Move.from_uci(uci))
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        evals.append(info["score"].white().score(mate_score=1000))

    return evals


def compute_move_quality(evals: list) -> dict:
    """From the sequence of evaluations, computes ACPL and the number of
    blunders/mistakes/inaccuracies for each player."""
    white_losses = []
    black_losses = []

    for ply_index in range(1, len(evals)):
        before = evals[ply_index - 1]
        after = evals[ply_index]

        # odd ply_index (1, 3, 5...) = White moved; even = Black moved
        moved_white = (ply_index % 2 == 1)

        if moved_white:
            loss = max(0, before - after)
            white_losses.append(loss)
        else:
            loss = max(0, after - before)
            black_losses.append(loss)

    def summarize(losses):
        if not losses:
            return {"acpl": None, "blunders": 0, "mistakes": 0, "inaccuracies": 0}
        return {
            "acpl": sum(losses) / len(losses),
            "blunders": sum(1 for l in losses if l >= BLUNDER_THRESHOLD),
            "mistakes": sum(1 for l in losses if MISTAKE_THRESHOLD <= l < BLUNDER_THRESHOLD),
            "inaccuracies": sum(1 for l in losses if INACCURACY_THRESHOLD <= l < MISTAKE_THRESHOLD),
        }

    white_stats = summarize(white_losses)
    black_stats = summarize(black_losses)

    return {
        "white_acpl": white_stats["acpl"],
        "white_blunders": white_stats["blunders"],
        "white_mistakes": white_stats["mistakes"],
        "white_inaccuracies": white_stats["inaccuracies"],
        "black_acpl": black_stats["acpl"],
        "black_blunders": black_stats["blunders"],
        "black_mistakes": black_stats["mistakes"],
        "black_inaccuracies": black_stats["inaccuracies"],
    }


def analyze_one_game(headers: dict, moves_uci: list, engine: chess.engine.SimpleEngine, depth: int) -> dict:
    """Analyzes a single game (given as plain headers + a list of UCI
    moves) and returns one output row."""
    evals = get_eval_sequence(moves_uci, engine, depth)
    quality = compute_move_quality(evals)

    return {
        # usernames -- needed to be able to group games by player later on.
        "white": headers.get("White", ""),
        "black": headers.get("Black", ""),
        "white_elo": int(headers.get("WhiteElo", "0")),
        "black_elo": int(headers.get("BlackElo", "0")),
        "eco": headers.get("ECO", ""),
        "opening": headers.get("Opening", ""),
        "result": headers.get("Result", ""),
        "num_plies": len(evals) - 1,
        "time_control": headers.get("TimeControl", ""),
        **quality,
    }


def read_tasks(input_path: str, max_games: int) -> list:
    """Reads games from the input PGN and returns a list of
    (task_id, headers, moves_uci) tuples ready to be sent to worker
    processes. Only plain, picklable data (dicts, lists of strings) is
    kept -- not chess.pgn.Game objects."""
    tasks = []
    with open(input_path, encoding="utf-8", errors="replace") as pgn_file:
        while len(tasks) < max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # end of file

            headers = dict(game.headers)
            white_elo = headers.get("WhiteElo", "")
            black_elo = headers.get("BlackElo", "")
            if not white_elo.isdigit() or not black_elo.isdigit():
                continue

            moves_uci = [move.uci() for move in game.mainline_moves()]
            tasks.append((len(tasks), headers, moves_uci))

    return tasks


def worker_loop(stockfish_path: str, depth: int, task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Entry point for a worker process: opens its own Stockfish instance
    (starting an engine has real overhead, so this is done ONCE per
    worker, not once per game) and analyzes tasks until it receives the
    stop sentinel (None)."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            task_id, headers, moves_uci = task
            try:
                row = analyze_one_game(headers, moves_uci, engine, depth)
                result_queue.put((task_id, row, None))
            except Exception as error:
                result_queue.put((task_id, None, str(error)))
    finally:
        engine.quit()


def analyze_pgn(input_path: str, output_path: str, max_games: int, num_workers: int, depth: int, log) -> int:
    log(f"Reading up to {max_games} games from {input_path}...")
    tasks = read_tasks(input_path, max_games)
    log(f"{len(tasks)} games queued for analysis at depth {depth} with {num_workers} parallel worker(s).")

    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(num_workers):
        task_queue.put(None)  # one stop sentinel per worker

    workers = [
        mp.Process(target=worker_loop, args=(STOCKFISH_PATH, depth, task_queue, result_queue))
        for _ in range(num_workers)
    ]
    for w in workers:
        w.start()

    rows_by_id = {}
    start_time = time.time()
    n_done = 0
    n_errors = 0

    try:
        while n_done + n_errors < len(tasks):
            task_id, row, error = result_queue.get()
            if error is not None:
                n_errors += 1
                log(f"  [warning] game {task_id} failed and was skipped: {error}", file=sys.stderr)
            else:
                rows_by_id[task_id] = row
            n_done += 1

            if n_done % 50 == 0 or n_done + n_errors == len(tasks):
                elapsed = time.time() - start_time
                avg_per_game = elapsed / n_done
                log(
                    f"Analyzed {n_done}/{len(tasks)} games... "
                    f"({avg_per_game:.2f}s/game wall-clock, {elapsed:.0f}s elapsed)"
                )
    except KeyboardInterrupt:
        log("\nInterrupted by user -- saving whatever was analyzed so far...")
    finally:
        for w in workers:
            w.terminate()
        for w in workers:
            w.join()

    rows = [rows_by_id[i] for i in sorted(rows_by_id)]

    elapsed_total = time.time() - start_time
    avg_per_game = elapsed_total / len(rows) if rows else 0

    pd.DataFrame(rows).to_csv(output_path, index=False)

    log(f"\nTotal time: {elapsed_total:.1f}s for {len(rows)} games ({avg_per_game:.2f}s/game wall-clock average)")
    if avg_per_game > 0:
        for n in (2000, 3000):
            estimate_minutes = (avg_per_game * n) / 60
            log(f"  -> at this rate, {n} games would take ~{estimate_minutes:.0f} minutes")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyzes games with Stockfish to extract move-quality features, using several engine processes in parallel."
    )
    parser.add_argument("--input", required=True, help="Path to the input .pgn file")
    parser.add_argument("--output", required=True, help="Path to the output CSV")
    parser.add_argument("--max-games", type=int, default=2000, help="Maximum number of games to analyze")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel Stockfish processes (default {DEFAULT_WORKERS} on this machine)",
    )
    parser.add_argument(
        "--depth", type=int, default=DEFAULT_DEPTH,
        help=f"Analysis depth per position (default {DEFAULT_DEPTH}); cost grows steeply with depth, "
             "benchmark first with benchmark_stockfish.py --depth N",
    )
    args = parser.parse_args()

    if not os.path.exists(STOCKFISH_PATH):
        sys.exit(f"ERROR: Stockfish executable not found at {STOCKFISH_PATH}. Update STOCKFISH_PATH in this script.")

    log, log_path = make_logger()
    log(f"Log for this run: {log_path}")

    total = analyze_pgn(args.input, args.output, args.max_games, args.workers, args.depth, log)
    log(f"\nDone. {total} games analyzed and saved to {args.output}")


if __name__ == "__main__":
    main()
