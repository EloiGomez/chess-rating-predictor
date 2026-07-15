# Chess Rating Predictor

Proyecto de Machine Learning que intenta predecir el rating (Elo) aproximado
de un jugador de ajedrez a partir de una partida en formato PGN.

Este es el punto de partida (MVP): parsea partidas y extrae features básicos
(sin usar todavía un motor como Stockfish). Es la base sobre la que se irán
agregando features más avanzados y, después, el modelo de predicción.

## Estructura del proyecto

```
chess-rating-predictor/
├── data/
│   ├── raw/            # PGNs originales (no se suben a git, salvo sample.pgn)
│   └── processed/      # CSVs generados por los scripts
├── src/
│   └── parse_pgn.py    # Parsea un .pgn a un .csv con features básicos
├── requirements.txt
└── README.md
```

## Setup en PyCharm

1. Abre la carpeta del proyecto en PyCharm (`File > Open`).
2. Crea un entorno virtual si PyCharm no lo hace automáticamente:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # en Windows: .venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Probar rápido con el PGN de muestra

El repo incluye `data/raw/sample.pgn` con 3 partidas de ejemplo, para
verificar que todo funciona sin tener que descargar nada:

```bash
python src/parse_pgn.py --input data/raw/sample.pgn --output data/processed/sample_games.csv
```

Esto debería generar `data/processed/sample_games.csv` con una fila por
partida (rating de blancas y negras, apertura, número de jugadas, resultado).

## Usar datos reales de Lichess

1. Ve a [database.lichess.org](https://database.lichess.org/) y descarga uno
   de los archivos mensuales (vienen comprimidos en `.zst`). Para empezar,
   no hace falta el mes completo: puedes cortar el archivo después de
   descomprimirlo.
2. Descomprime el archivo. Si tienes `zstd` instalado:
   ```bash
   zstd -d lichess_db_standard_rated_YYYY-MM.pgn.zst
   ```
   Si no, puedes instalar la librería `zstandard` de Python y descomprimir
   desde un script.
3. Mueve el `.pgn` resultante a `data/raw/`.
4. Corre el parser, limitando el número de partidas para no tardar horas:
   ```bash
   python src/parse_pgn.py --input data/raw/tu_archivo.pgn --output data/processed/games.csv --max-games 20000
   ```

## Próximos pasos

- [ ] Entrenar un modelo baseline (regresión lineal / Random Forest) con los
      features actuales, para validar el pipeline de punta a punta.
- [ ] Agregar features de calidad de jugada usando Stockfish local
      (precisión, número de blunders, cambios de evaluación).
- [ ] Comparar modelos (scikit-learn vs XGBoost).
- [ ] Empaquetar en una demo simple con Streamlit: subes un PGN, te devuelve
      el rating estimado.
