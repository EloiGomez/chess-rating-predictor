"""
select_games_by_player.py

Scans a large Lichess dump (.pgn.zst) and selects a manageable subset of
games, grouped by player AND by time-control speed category (instead of
treating each game as an independent data point, and instead of mixing a
player's bullet games together with their blitz games).

Why split by speed category: Lichess gives each player a SEPARATE rating
per time-control speed (bullet/blitz/rapid/classical) -- someone can be
2400 in bullet and 2000 in rapid. If a player's sampled games mixed
formats, averaging their Elo (or their Stockfish move-quality features --
bullet games are short and sloppy almost regardless of skill, classical
games are long and careful almost regardless of skill) across formats
would blend together numbers that don't mean the same thing. So this
script builds a SEPARATE, dedicated pool of eligible players for each
category in TARGET_SPEED_CATEGORIES, and a separate output .pgn per
category, ready to be run independently through
analyze_with_stockfish.py -> aggregate_by_player.py -> train_baseline.py.

Filters out:
    - games without a valid Elo rating for either player
    - games with an "Anonymous" player (disconnected user)
    - unrated games (only "Rated ..." events)
    - abandoned or very short games
    - games whose time-control speed isn't in TARGET_SPEED_CATEGORIES

And only keeps players who reach MIN_GAMES_PER_PLAYER valid games IN THAT
CATEGORY -- if a player only has 1 bullet game in the sample, there's no
way to compute a reliable average of their bullet strength from it. To
stop a very active player from dominating the sample, each player is
capped at MAX_GAMES_PER_PLAYER per category.

This does TWO full passes over the file (pass 1 to find out which
(player, category) pairs are "eligible" across the WHOLE file, pass 2 to
collect their games), covering ALL target categories in each single pass
-- so adding a second or third category doesn't cost a second or third
2-hour pass 1, it's all gathered in the same scan:

    - Pass 1 uses chess.pgn.read_headers() instead of read_game(). This
      python-chess function is built exactly for this: it reads the
      headers and SKIPS the rest of the game without tokenizing moves,
      parsing SAN, or building a board. By far the biggest time saver
      (pass 1 used to rebuild the full move tree of every game just to
      read its headers).

    - Pass 2 no longer uses chess.pgn.read_game() at all. A "standard"
      Lichess dump is very regular PGN (headers + move text), so
      read_raw_game() below splits both blocks by hand, without
      validating or parsing a single move. We trust the dump is already
      valid PGN -- if it weren't, analyze_with_stockfish.py would fail
      anyway later on when re-parsing the output .pgn with
      chess.pgn.read_game(), so no real error detection is lost. The scan
      that finds where the move text block starts/ends reuses the exact
      same regex and state machine that python-chess uses internally for
      its own "skip mode" (SKIP_MOVETEXT_REGEX), so it correctly handles
      the clock comments ("{ [%clk 0:03:00] }") that the "standard" dump
      includes without confusing them with the end of the game.

    Pass 2 stops once EVERY category in TARGET_SPEED_CATEGORIES has
    TARGET_PLAYERS_PER_CATEGORY players who reached the minimum (counting
    only games that passed every filter, not raw games read -- an early
    version of this script stopped by counting raw games instead, which
    was a bug: with very active eligible players, a tiny early slice of
    the file could already fill a raw-game quota while the vast majority
    of those players hadn't accumulated their minimum yet, so pruning
    left far fewer players than expected).

    Pass 2 also samples EACH category's players in a STRATIFIED way across
    Elo bands (ELO_BUCKET_WIDTH-point buckets), instead of just taking
    whichever eligible players happen to show up first while scanning.
    Without this, the selection ends up shaped like Lichess's overall
    rating distribution -- heavily concentrated around the median, with
    almost no players at the tails -- because prolific players near the
    median contribute a disproportionate share of raw games in the file.
    A model trained on that skew learns the middle of the range well and
    is unreliable for anyone far from it (a 3200-rated player looks
    nothing like anyone the model has seen). Pass 1 also tracks each
    player's mean Elo (not just their game count) so pass 2 can target a
    roughly even number of qualified players FROM EACH BUCKET (capped by
    how many are actually available in that bucket -- sparse extreme
    buckets will naturally end up smaller, that's expected, not an error).

Pass 1 cache (eligible players per category). If ELIGIBLE_CACHE_PATH
already exists, pass 1 is SKIPPED entirely and the cached result is used
instead -- that way, if pass 2 fails or gets interrupted, or you want to
add a category later, the whole file doesn't need to be re-read from
scratch every time. Delete that file by hand to force pass 1 to run again
(for example, if you change MIN_GAMES_PER_PLAYER or
TARGET_SPEED_CATEGORIES).

Designed to be left running unattended: it writes its own UTF-8 log to
logs/ (independent of how console output gets redirected, which on
PowerShell often mangles encodings) and, if interrupted mid-pass-2
(Ctrl+C, session closing, etc.), it still saves whatever had already been
selected instead of losing everything.

Usage:
    python src/select_games_by_player.py
"""

import io
import json
import math
import os
import re
import sys
import time

import chess.pgn
import zstandard

# --- Configuration ---------------------------------------------------------

ZST_PATH = r"F:\lichess_data\lichess_db_standard_rated_2026-08.pgn.zst"
OUTPUT_PGN_TEMPLATE = "data/processed/selected_games_{category}.pgn"
ELIGIBLE_CACHE_PATH = "data/processed/eligible_players_by_category.json"
LOG_DIR = "logs"

# Which time-control speeds to build a dedicated, non-mixed dataset for.
# Classical is essentially absent from Lichess's population and rapid is
# usually too small a slice to be worth a dedicated pool -- add them here
# if that's not true for your dump.
TARGET_SPEED_CATEGORIES = ["bullet", "blitz"]

MIN_GAMES_PER_PLAYER = 5    # below this, not worth aggregating
MAX_GAMES_PER_PLAYER = 15   # stops one very active player from dominating the sample

# Pass 2 stops, for each category, as soon as this many players have
# reached the minimum (not by counting raw games read -- see the
# docstring above). With MAX_GAMES_PER_PLAYER as a cap, each category's
# final dataset will have at most TARGET_PLAYERS_PER_CATEGORY * 15 games.
TARGET_PLAYERS_PER_CATEGORY = 3_000

# Width, in Elo points, of the bands pass 2 tries to sample evenly from
# (see the docstring above). Finer buckets (lower width) give more
# granular coverage of the rating range, at the cost of a lower per-bucket
# target (target_players_per_category / n_buckets) -- which bites hardest
# in the already-sparse extreme buckets.
ELO_BUCKET_WIDTH = 100

MIN_PLIES = 10              # discards near-empty / aborted games
LOG_EVERY_PASS1 = 2_000_000  # how often (in headers read) to print progress
LOG_EVERY_PASS2 = 200_000    # how often (in games read) to print progress

RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
_MOVE_NUMBER_RE = re.compile(r"^\d+\.(?:\.\.)?$")
_NAG_LIKE_RE = re.compile(r"^(\$\d+|[!?]{1,2})$")
_COMMENT_RE = re.compile(r"\{[^{}]*\}")


# --- Logging helpers --------------------------------------------------------

def make_logger():
    """Creates a logger that writes every line to the console AND to a
    timestamped UTF-8 file under logs/. Meant for reviewing progress after
    a long, unattended run without depending on how stdout was
    redirected."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"select_games_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        print(line, file=log_file, flush=True)

    return log, log_path


# --- Reading the .pgn.zst ---------------------------------------------------

def open_pgn_stream(path):
    """Opens a .pgn.zst as a text stream, ready to be read game by game."""
    fh = open(path, "rb")
    dctx = zstandard.ZstdDecompressor()
    stream_reader = dctx.stream_reader(fh)
    return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")


def read_raw_game(stream):
    """Reads ONE game from a PGN stream without building a board or
    parsing a single move: it splits the headers block (using the same
    regex python-chess uses, chess.pgn.TAG_REGEX) from the move-text
    block by hand, keeping the latter as plain text.

    To find where the move text ends, it reuses the same regex and state
    machine python-chess uses in its own "skip mode"
    (chess.pgn.SKIP_MOVETEXT_REGEX): it stays aware of "{ ... }" comments
    that might contain a blank line, so it doesn't mistake a multi-line
    comment for the end of the game.

    Returns (headers: dict, movetext: str), or None once the stream ends.
    """
    line = stream.readline()
    if not line:
        return None
    line = line.lstrip("\ufeff")

    # Skip blank lines or file-level comments before the game starts.
    while line and (line.isspace() or line.startswith("%") or line.startswith(";")):
        line = stream.readline()
    if not line:
        return None

    headers = {}
    consecutive_empty_lines = 0
    while line:
        # Same as python-chess's own parser: up to one blank line INSIDE
        # the headers block is tolerated before giving up -- that's
        # exactly the one separating headers from move text. If it isn't
        # consumed here, the loop below mistakes it for the blank line
        # that ends the game, and the whole move text is lost.
        if consecutive_empty_lines < 1 and line.isspace():
            consecutive_empty_lines += 1
            line = stream.readline()
            continue
        if not line.startswith("["):
            break
        consecutive_empty_lines = 0
        match = chess.pgn.TAG_REGEX.match(line)
        if match:
            headers[match.group(1)] = match.group(2)
        line = stream.readline()

    movetext_lines = []
    in_comment = False
    while line:
        if not in_comment:
            if line.isspace():
                break
            if line.startswith("%"):
                line = stream.readline()
                continue

        movetext_lines.append(line)
        for match in chess.pgn.SKIP_MOVETEXT_REGEX.finditer(line):
            token = match.group(0)
            if token == "{":
                in_comment = True
            elif not in_comment and token == ";":
                break
            elif token == "}":
                in_comment = False

        line = stream.readline()

    return headers, "".join(movetext_lines)


def count_plies(movetext: str) -> int:
    """Counts the plies (half-moves) in a raw move-text block, without
    parsing SAN: strips "{...}" comments, the final result token and move
    numbers, and counts what's left.

    This is an approximation (it doesn't validate that moves are legal),
    but it's enough for the MIN_PLIES filter -- and the Lichess dumps are
    already validated anyway."""
    text = _COMMENT_RE.sub(" ", movetext)
    plies = 0
    for token in text.split():
        if token in RESULT_TOKENS:
            continue
        if _MOVE_NUMBER_RE.match(token) or _NAG_LIKE_RE.match(token):
            continue
        plies += 1
    return plies


def estimate_speed_category(time_control: str) -> str:
    """Classifies a time control into bullet/blitz/rapid/classical, using
    the same formula Lichess uses internally: estimated time = base time
    + 40 * increment (in seconds). (Same logic as aggregate_by_player.py
    and train_baseline.py -- duplicated here because it's a small, pure
    function; not worth centralizing across scripts just for this.)"""
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


def elo_bucket(elo: float) -> str:
    """Labels an Elo value with its ELO_BUCKET_WIDTH-point band, e.g.
    1450 -> "1400-1600"."""
    lower = int(elo // ELO_BUCKET_WIDTH) * ELO_BUCKET_WIDTH
    return f"{lower}-{lower + ELO_BUCKET_WIDTH}"


# --- Filters -----------------------------------------------------------

def passes_header_filters(headers) -> bool:
    """Header-based filters for a game (works the same with a pass-1
    chess.pgn.Headers object or a pass-2 plain dict). Does NOT check the
    time-control speed category -- callers check that separately, since
    what counts as "wanted" depends on which category the game belongs
    to."""
    white, black = headers.get("White", ""), headers.get("Black", "")
    if white == "Anonymous" or black == "Anonymous":
        return False

    white_elo, black_elo = headers.get("WhiteElo", "?"), headers.get("BlackElo", "?")
    if not white_elo.isdigit() or not black_elo.isdigit():
        return False

    event = headers.get("Event", "")
    if "Rated" not in event:
        return False

    termination = headers.get("Termination", "")
    if termination == "Abandoned":
        return False

    return True


# --- Pass 1: count valid games per (player, category) -----------------------

def first_pass_count_players(path, target_categories, log):
    """Scans the whole file reading only headers (chess.pgn.read_headers,
    which skips the rest of the game without tokenizing moves) and counts
    how many valid games each player has IN EACH TARGET CATEGORY, plus the
    sum of their Elo across those games (so a mean Elo -- used to bucket
    them for stratified sampling in pass 2 -- can be computed afterwards)."""
    counts = {cat: {} for cat in target_categories}
    elo_sums = {cat: {} for cat in target_categories}
    n_read = 0
    start = time.time()

    with open_pgn_stream(path) as pgn:
        while True:
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break
            n_read += 1
            if n_read % LOG_EVERY_PASS1 == 0:
                elapsed = time.time() - start
                log(f"  [pass 1] {n_read:,} headers read... ({n_read / elapsed:,.0f} games/s)")

            if not passes_header_filters(headers):
                continue

            category = estimate_speed_category(headers.get("TimeControl", ""))
            if category not in counts:
                continue

            player_counts = counts[category]
            player_elo_sums = elo_sums[category]
            for player, elo_str in (
                (headers["White"], headers["WhiteElo"]),
                (headers["Black"], headers["BlackElo"]),
            ):
                player_counts[player] = player_counts.get(player, 0) + 1
                player_elo_sums[player] = player_elo_sums.get(player, 0) + int(elo_str)

    elapsed = time.time() - start
    rate = n_read / elapsed if elapsed else 0
    log(f"Pass 1 complete: {n_read:,} games read in total ({elapsed:.0f}s, {rate:,.0f} games/s).")
    return counts, elo_sums


def build_elo_buckets(eligible_players, player_mean_elo, target_players_per_category, log):
    """Assigns each eligible player to an Elo bucket and works out, per
    category, how many qualified players pass 2 should aim to collect FROM
    EACH bucket: the target split evenly across that category's populated
    buckets, capped at how many eligible players actually exist in a given
    bucket (a sparse bucket just ends up smaller -- there's no way to
    manufacture players who aren't in the data)."""
    player_bucket = {}
    bucket_target = {}

    for cat, players in eligible_players.items():
        player_bucket[cat] = {p: elo_bucket(player_mean_elo[cat][p]) for p in players}

        pool_size = {}
        for bucket in player_bucket[cat].values():
            pool_size[bucket] = pool_size.get(bucket, 0) + 1

        n_buckets = len(pool_size)
        even_share = math.ceil(target_players_per_category / n_buckets) if n_buckets else 0
        bucket_target[cat] = {bucket: min(even_share, size) for bucket, size in pool_size.items()}

        achievable = sum(bucket_target[cat].values())
        log(f"[{cat}] Elo buckets (target {even_share}/bucket, capped by availability):")
        for bucket in sorted(bucket_target[cat], key=lambda b: int(b.split("-")[0])):
            log(f"    {bucket:>11}: {bucket_target[cat][bucket]:>5,} target (of {pool_size[bucket]:,} eligible)")
        log(f"[{cat}] Achievable total across all buckets: {achievable:,} "
            f"(requested {target_players_per_category:,})")

    return player_bucket, bucket_target


def _player_wanted(player, counts_for_cat, bucket_of, qualified_by_bucket, bucket_target):
    """Whether a game should be counted towards this player: keeps
    accumulating an already-qualified player up to MAX_GAMES_PER_PLAYER
    regardless of their bucket's status (their games are "sunk cost", more
    of them only helps), but for a player NOT YET qualified, only starts
    counting if their bucket hasn't already reached its target -- no point
    accumulating a new player from an already-satisfied bucket, we'd just
    prune them at the end anyway."""
    if player not in counts_for_cat:
        return False
    current = counts_for_cat[player]
    if current >= MAX_GAMES_PER_PLAYER:
        return False
    if current >= MIN_GAMES_PER_PLAYER:
        return True
    bucket = bucket_of[player]
    return len(qualified_by_bucket[bucket]) < bucket_target[bucket]


# --- Pass 2: select games from the eligible players -------------------------

def second_pass_select_games(path, eligible_players, player_bucket, bucket_target, log):
    """Scans the file again with read_raw_game() (without building a
    single chess.pgn.Game object) and, for each (category, eligible
    player), saves up to MAX_GAMES_PER_PLAYER games that pass the filters
    in that category -- sampling players FROM EACH ELO BUCKET (see
    build_elo_buckets) rather than just whoever shows up first.

    Stops as soon as EVERY bucket of EVERY category has reached its target
    number of qualified players (MIN_GAMES_PER_PLAYER ACCEPTED games, not
    raw games read), or once the end of the file is reached."""
    categories = list(eligible_players.keys())
    selected_count = {cat: {p: 0 for p in eligible_players[cat]} for cat in categories}
    qualified_by_bucket = {cat: {b: set() for b in bucket_target[cat]} for cat in categories}
    selected_games = {cat: [] for cat in categories}
    n_read = 0
    start = time.time()

    def all_categories_done():
        return all(
            len(qualified_by_bucket[cat][b]) >= target
            for cat in categories
            for b, target in bucket_target[cat].items()
        )

    # Interruptions/errors are caught HERE (not in main()) so that if
    # something cuts pass 2 short, whatever is already in selected_games
    # (a local variable of this function) is still returned instead of
    # being lost while unwinding the stack.
    try:
        with open_pgn_stream(path) as pgn:
            while True:
                game = read_raw_game(pgn)
                if game is None:
                    break
                headers, movetext = game
                n_read += 1
                if n_read % LOG_EVERY_PASS2 == 0:
                    elapsed = time.time() - start
                    progress = ", ".join(
                        f"{cat}: {sum(len(s) for s in qualified_by_bucket[cat].values()):,}/"
                        f"{sum(bucket_target[cat].values()):,}"
                        for cat in categories
                    )
                    log(f"  [pass 2] {n_read:,} games read ({progress}) ({n_read / elapsed:,.0f} games/s)")

                category = estimate_speed_category(headers.get("TimeControl", ""))
                if category not in eligible_players or not passes_header_filters(headers):
                    continue

                counts_for_cat = selected_count[category]
                bucket_of = player_bucket[category]
                qualified_for_cat = qualified_by_bucket[category]
                target_for_cat = bucket_target[category]

                white, black = headers.get("White", ""), headers.get("Black", "")
                white_wanted = _player_wanted(white, counts_for_cat, bucket_of, qualified_for_cat, target_for_cat)
                black_wanted = _player_wanted(black, counts_for_cat, bucket_of, qualified_for_cat, target_for_cat)

                if not (white_wanted or black_wanted):
                    continue

                if count_plies(movetext) < MIN_PLIES:
                    continue

                selected_games[category].append((headers, movetext))
                if white_wanted:
                    counts_for_cat[white] += 1
                    if counts_for_cat[white] == MIN_GAMES_PER_PLAYER:
                        qualified_for_cat[bucket_of[white]].add(white)
                if black_wanted:
                    counts_for_cat[black] += 1
                    if counts_for_cat[black] == MIN_GAMES_PER_PLAYER:
                        qualified_for_cat[bucket_of[black]].add(black)

                if all_categories_done():
                    break
    except KeyboardInterrupt:
        log("\n  [pass 2] interrupted by the user -- keeping what was selected up to this point.")
    except Exception:
        import traceback
        log("\n  [pass 2] unexpected error -- keeping what was selected up to this point:")
        log(traceback.format_exc())

    elapsed = time.time() - start
    rate = n_read / elapsed if elapsed else 0
    summary = ", ".join(
        f"{cat}: {sum(len(s) for s in qualified_by_bucket[cat].values()):,} players" for cat in categories
    )
    log(f"Pass 2 complete: {n_read:,} games read ({summary}) ({elapsed:.0f}s, {rate:,.0f} games/s).")
    return selected_games, selected_count


def prune_players_below_minimum(games, selected_count, log, category):
    """Pass 2 can stop with some players "in progress" (fewer than
    MIN_GAMES_PER_PLAYER accepted games). These are removed so the final
    .pgn only contains players with enough games."""
    final_eligible = {p for p, c in selected_count.items() if c >= MIN_GAMES_PER_PLAYER}

    kept_games = [
        (headers, movetext) for headers, movetext in games
        if headers.get("White", "") in final_eligible or headers.get("Black", "") in final_eligible
    ]
    log(
        f"[{category}] After final pruning: {len(final_eligible):,} valid players, "
        f"{len(kept_games):,} games kept (out of {len(games):,})."
    )
    return kept_games


def write_games(games, path, log, category):
    with open(path, "w", encoding="utf-8") as f:
        for headers, movetext in games:
            for tag, value in headers.items():
                f.write(f'[{tag} "{value}"]\n')
            f.write("\n")
            f.write(movetext.rstrip("\n"))
            f.write("\n\n")
    log(f"[{category}] Saved to {path} ({len(games):,} games).")


def main():
    log, log_path = make_logger()
    log(f"Log for this run: {log_path}")
    log(f"Target categories: {', '.join(TARGET_SPEED_CATEGORIES)}")

    if not os.path.exists(ZST_PATH):
        log(f"ERROR: input file not found at {ZST_PATH}")
        log("Download the dump from https://database.lichess.org/ and adjust ZST_PATH if needed.")
        sys.exit(1)

    games = None
    selected_count = None
    try:
        if os.path.exists(ELIGIBLE_CACHE_PATH):
            log(f"--- Found pass-1 cache at {ELIGIBLE_CACHE_PATH}, skipping pass 1 ---")
            with open(ELIGIBLE_CACHE_PATH, encoding="utf-8") as f:
                cached = json.load(f)
            player_mean_elo = {cat: cached.get(cat, {}) for cat in TARGET_SPEED_CATEGORIES}
            eligible_players = {cat: set(player_mean_elo[cat].keys()) for cat in TARGET_SPEED_CATEGORIES}
        else:
            log("--- Pass 1: counting valid games per player and category ---")
            counts, elo_sums = first_pass_count_players(ZST_PATH, TARGET_SPEED_CATEGORIES, log)

            eligible_players = {
                cat: {p for p, c in counts[cat].items() if c >= MIN_GAMES_PER_PLAYER}
                for cat in TARGET_SPEED_CATEGORIES
            }
            player_mean_elo = {
                cat: {p: elo_sums[cat][p] / counts[cat][p] for p in eligible_players[cat]}
                for cat in TARGET_SPEED_CATEGORIES
            }

            with open(ELIGIBLE_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(player_mean_elo, f)
            log(f"Saved eligible-players cache to {ELIGIBLE_CACHE_PATH}")

        for cat in TARGET_SPEED_CATEGORIES:
            log(f"[{cat}] Players with >= {MIN_GAMES_PER_PLAYER} valid games: {len(eligible_players[cat]):,}")

        log("\n--- Building Elo buckets for stratified sampling ---")
        player_bucket, bucket_target = build_elo_buckets(
            eligible_players, player_mean_elo, TARGET_PLAYERS_PER_CATEGORY, log
        )

        log("\n--- Pass 2: selecting games from those players ---")
        games, selected_count = second_pass_select_games(
            ZST_PATH, eligible_players, player_bucket, bucket_target, log
        )
    except KeyboardInterrupt:
        log("\nInterrupted by the user during pass 1 (nothing to save yet).")
    except Exception:
        import traceback
        log("\nUnexpected error during pass 1:")
        log(traceback.format_exc())

    if not games:
        log("No games were selected to save. Done.")
        return

    log("\n--- Final pruning ---")
    for cat in TARGET_SPEED_CATEGORIES:
        cat_games = prune_players_below_minimum(games[cat], selected_count[cat], log, cat)
        output_path = OUTPUT_PGN_TEMPLATE.format(category=cat)
        write_games(cat_games, output_path, log, cat)

    log("\nYou can now run analyze_with_stockfish.py on each category's .pgn as --input.")


if __name__ == "__main__":
    main()
