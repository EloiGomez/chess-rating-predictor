"""
extract_sample.py

Descomprime un archivo .pgn.zst de Lichess "al vuelo" (streaming) y guarda
solo las primeras N partidas en un .pgn normal y manejable. Nunca escribe
a disco el archivo descomprimido completo, que para un mes de Lichess
puede pesar cientos de GB.

Requiere: pip install zstandard

Uso:
    python src/extract_sample.py --input E:\\lichess_data\\lichess_db_standard_rated_2024-01.pgn.zst ^
                                  --output data/raw/lichess_sample.pgn ^
                                  --max-games 20000
"""

import argparse
import io

import chess.pgn
import zstandard as zstd


def extract_games(input_path: str, output_path: str, max_games: int) -> int:
    """Lee partidas directamente del stream comprimido y escribe las
    primeras `max_games` a un .pgn de texto plano."""
    dctx = zstd.ZstdDecompressor()
    games_written = 0

    with open(input_path, "rb") as compressed_file, \
            dctx.stream_reader(compressed_file) as reader, \
            io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as text_stream, \
            open(output_path, "w", encoding="utf-8") as out_file:

        while games_written < max_games:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break  # se acabo el archivo comprimido antes de llegar al limite

            # re-escribimos la partida ya parseada como PGN valido
            print(game, file=out_file, end="\n\n")
            games_written += 1

            if games_written % 1000 == 0:
                print(f"Extraidas {games_written} partidas...")

    return games_written


def main():
    parser = argparse.ArgumentParser(
        description="Extrae las primeras N partidas de un .pgn.zst de Lichess sin descomprimir todo el archivo."
    )
    parser.add_argument("--input", required=True, help="Ruta al archivo .pgn.zst original")
    parser.add_argument("--output", required=True, help="Ruta al .pgn de salida (mucho mas chico)")
    parser.add_argument("--max-games", type=int, default=20000, help="Numero de partidas a extraer")
    args = parser.parse_args()

    total = extract_games(args.input, args.output, args.max_games)
    print(f"Listo. {total} partidas escritas en {args.output}")


if __name__ == "__main__":
    main()
