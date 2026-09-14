"""
extract_sample.py

Decompresses a Lichess .pgn.zst file "on the fly" (streaming) and saves
only the first N games into a regular, manageable .pgn file. It never
writes the full decompressed file to disk, which for a single Lichess
month can be hundreds of GB.

Requires: pip install zstandard

Usage:
    python src/extract_sample.py --input E:\\lichess_data\\lichess_db_standard_rated_2024-01.pgn.zst ^
                                  --output data/raw/lichess_sample.pgn ^
                                  --max-games 20000
"""

import argparse
import io

import chess.pgn
import zstandard as zstd


def extract_games(input_path: str, output_path: str, max_games: int) -> int:
    """Reads games directly from the compressed stream and writes the
    first `max_games` to a plain-text .pgn file."""
    dctx = zstd.ZstdDecompressor()
    games_written = 0

    with open(input_path, "rb") as compressed_file, \
            dctx.stream_reader(compressed_file) as reader, \
            io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as text_stream, \
            open(output_path, "w", encoding="utf-8") as out_file:

        while games_written < max_games:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break  # the compressed file ended before reaching the limit

            # re-write the already-parsed game as valid PGN
            print(game, file=out_file, end="\n\n")
            games_written += 1

            if games_written % 1000 == 0:
                print(f"Extracted {games_written} games...")

    return games_written


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the first N games from a Lichess .pgn.zst file without decompressing the whole thing."
    )
    parser.add_argument("--input", required=True, help="Path to the original .pgn.zst file")
    parser.add_argument("--output", required=True, help="Path to the output .pgn file (much smaller)")
    parser.add_argument("--max-games", type=int, default=20000, help="Number of games to extract")
    args = parser.parse_args()

    total = extract_games(args.input, args.output, args.max_games)
    print(f"Done. {total} games written to {args.output}")


if __name__ == "__main__":
    main()
