"""
parse_pgn.py

Lee un archivo PGN (partidas de ajedrez) y extrae, por cada partida, un
conjunto de features basicos a un CSV: rating de ambos jugadores, apertura
(ECO), numero de jugadas, resultado y control de tiempo.

Esta es la version "MVP": no usa Stockfish todavia, solo lo que ya viene
en las cabeceras del PGN y en la secuencia de jugadas. Es el primer paso
para validar que el pipeline completo funciona de punta a punta.

Uso:
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
    """Parsea partidas de un archivo PGN y las escribe como filas de un CSV.

    Devuelve el numero total de partidas escritas.
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
                break  # fin del archivo

            headers = game.headers

            white_elo = headers.get("WhiteElo", "")
            black_elo = headers.get("BlackElo", "")

            # nos saltamos partidas sin rating (bots, partidas casuales sin clasificar)
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
                print(f"Procesadas {count} partidas...", file=sys.stderr)

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Parsea un archivo PGN a un CSV con features basicos por partida."
    )
    parser.add_argument("--input", required=True, help="Ruta al archivo .pgn de entrada")
    parser.add_argument("--output", required=True, help="Ruta al archivo .csv de salida")
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Numero maximo de partidas a procesar (util para pruebas rapidas)",
    )
    args = parser.parse_args()

    total = parse_games(args.input, args.output, args.max_games)
    print(f"Listo. {total} partidas escritas en {args.output}")


if __name__ == "__main__":
    main()
